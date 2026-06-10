"""CLI: walk-forward OOS evaluation for a corpus + one-or-more weight sources.

Loads `BacktestRecord`s from a glob of analysis_mvp JSONs, attaches
forward returns via yfinance (one call per symbol, identical to
`backtest.py`), generates rolling 18-month-train / 3-month-test /
1-month-step windows, then evaluates each requested weight source against
the same set of windows.

Weight sources (specified via `--weight-source`, may be repeated):

- **`v1.4`** — snapshot of `DEFAULT_FACTOR_WEIGHTS` pre-Section-27. The
  only delta vs current v1.5 is `options_iv_term_structure: 0.00` (was
  0.04 in v1.5). Snapshot is inlined below with a comment.
- **`v1.5`** — current `DEFAULT_FACTOR_WEIGHTS` imported from
  `scoring.py`.
- **`ic_signed_rolling`** — for each window, refits weights via
  `summarize_factors` on the train slice then derives signed weights via
  `ic_signed_weights(min_abs_ic=0.05, min_n=20)`. This is the proper OOS
  protocol.
- **`custom_json`** — loads a fixed weight vector from `--weights-json`.

Outputs (per weight source, suffix in filename):

- `summary.json` — combined per-source summaries (also written to
  `summary_<source>.json` per source).
- `summary.md` — one section per source, one table per horizon.
- `windows.csv` — one row per (source, window) with per-horizon stats.

The one-line stdout summary uses the LAST requested source (typically
the most-recent / OOS one).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    BacktestRecord,
    explode_records_to_factors,
    ic_signed_weights,
    summarize_factors,
)
from tradingagents.analysis_only.scoring import (  # noqa: E402
    DEFAULT_FACTOR_WEIGHTS,
)
from tradingagents.analysis_only.walk_forward import (  # noqa: E402
    evaluate_window,
    generate_windows,
    render_walk_forward_markdown,
    summarize_walk_forward,
)


# v1.4 weight snapshot — exactly current `DEFAULT_FACTOR_WEIGHTS` with
# `options_iv_term_structure` zeroed (v1.5's only delta per handoff
# Section 27). Reconstructed at runtime so a future refactor of the
# v1.5 dict doesn't silently drift this snapshot.
def _build_v1_4_weights() -> dict[str, float]:
    snap = dict(DEFAULT_FACTOR_WEIGHTS)
    # Section 27: v1.5 commits options_iv_term_structure 0.00 → 0.04
    # (with score sign inverted). v1.4 was 0.00.
    snap["options_iv_term_structure"] = 0.00
    return snap


def _make_weight_fn(source: str, *, custom_weights: dict[str, float] | None,
                    horizon: str, min_abs_ic: float, min_n: int):
    """Return a `weight_fn` matching the requested source name.

    Returns a `Callable[[list[BacktestRecord]], dict[str, float] | None]`.
    """
    if source == "v1.4":
        weights = _build_v1_4_weights()
        return lambda _train: dict(weights)
    if source == "v1.5":
        weights = dict(DEFAULT_FACTOR_WEIGHTS)
        return lambda _train: dict(weights)
    if source == "ic_signed_rolling":
        def _fit(train: list[BacktestRecord]) -> dict[str, float] | None:
            by_factor = explode_records_to_factors(train)
            if not by_factor:
                return None
            summary = summarize_factors(by_factor, return_fields=[horizon])
            w = ic_signed_weights(
                summary, horizon=horizon, min_abs_ic=min_abs_ic, min_n=min_n
            )
            return w or None
        return _fit
    if source == "per_horizon_rolling":
        # Sentinel: the per-horizon-rolling source is dispatched specially
        # in main() because it returns DIFFERENT weights per horizon.
        # Returning None here triggers the dispatch.
        return None
    if source == "custom_json":
        if not custom_weights:
            raise SystemExit("--weight-source custom_json requires --weights-json")
        weights = dict(custom_weights)
        return lambda _train: dict(weights)
    raise SystemExit(
        f"Unknown --weight-source: {source!r}. "
        "Must be one of: v1.4, v1.5, ic_signed_rolling, custom_json, "
        "per_horizon_json."
    )


def _load_per_horizon_weights(path: str) -> dict[str, dict[str, float]]:
    """Load the JSON written by fit_per_horizon_weights.py."""
    payload = json.loads(Path(path).read_text())
    if "weights_by_horizon" not in payload:
        raise SystemExit(
            f"{path}: payload missing required key 'weights_by_horizon'"
        )
    weights_by_horizon = payload.get("weights_by_horizon")
    if not isinstance(weights_by_horizon, dict):
        raise SystemExit(
            f"{path}: 'weights_by_horizon' must be a dict, got {type(weights_by_horizon).__name__}"
        )
    return {
        h: {k: float(v) for k, v in (w or {}).items()}
        for h, w in weights_by_horizon.items()
    }


def _fit_per_horizon_weights_for_window(
    train: list[BacktestRecord], *, horizons: list[str],
    min_abs_ic: float, min_n: int,
) -> dict[str, dict[str, float]]:
    """Per-horizon IC-signed weights computed from a single training window.

    Returns `{horizon: {factor: ic_weight}}`. Used by per_horizon_rolling
    to get strict OOS weights (no information from test or future windows).
    """
    by_factor = explode_records_to_factors(train)
    if not by_factor:
        return {h: {} for h in horizons}
    summary = summarize_factors(by_factor, return_fields=list(horizons))
    return {
        h: ic_signed_weights(
            summary, horizon=h, min_abs_ic=min_abs_ic, min_n=min_n,
        )
        for h in horizons
    }


def _evaluate_window_per_horizon(
    records: list[BacktestRecord],
    win,
    *,
    per_horizon_weights: dict[str, dict[str, float]],
    horizons: list[str],
) -> dict:
    """Run evaluate_window once per horizon, each with its own weight vector.

    Merges the per_horizon sub-dicts. `weights_used` becomes a dict of dicts
    so downstream CSV / summary code can still render a "n_nonzero" count
    (it picks the largest by total absolute weight as a proxy).
    """
    merged_per_horizon: dict[str, dict] = {}
    weights_by_h: dict[str, dict[str, float]] = {}
    train_n = test_n = 0
    for horizon in horizons:
        weights = per_horizon_weights.get(horizon) or {}
        weights_by_h[horizon] = dict(weights)
        weight_fn = (lambda w=weights: (lambda _train: dict(w) if w else None))()
        result = evaluate_window(
            records, win, weight_fn=weight_fn, horizons=[horizon],
        )
        merged_per_horizon[horizon] = result["per_horizon"][horizon]
        train_n = result["train_n"]
        test_n = result["test_n"]
    # Pick the densest weight vector as the canonical `weights_used` so the
    # downstream CSV/summary's n_nonzero column is informative; full per-h
    # vectors are preserved under `weights_used_per_horizon`.
    canonical = max(
        weights_by_h.values(), key=lambda d: sum(1 for v in d.values() if v),
        default={},
    )
    return {
        "window": win.as_dict(),
        "train_n": train_n,
        "test_n": test_n,
        "weights_used": dict(canonical),
        "weights_used_per_horizon": weights_by_h,
        "per_horizon": merged_per_horizon,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reports-glob", action="append",
        help=(
            "Glob for report JSONs. May be passed multiple times. "
            "Defaults to reports/analysis_mvp/*.json when omitted."
        ),
    )
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--test-months", type=int, default=3)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument(
        "--horizons", nargs="+", default=["ret_5d", "ret_20d", "ret_60d"],
    )
    p.add_argument(
        "--fit-horizon", default="ret_60d",
        help="Horizon used by ic_signed_rolling to derive signed weights.",
    )
    p.add_argument(
        "--min-abs-ic", type=float, default=0.05,
        help="ic_signed_rolling: minimum |IC| to keep a factor.",
    )
    p.add_argument(
        "--min-n", type=int, default=20,
        help="ic_signed_rolling: minimum paired observations per factor.",
    )
    p.add_argument(
        "--weight-source", nargs="+", required=True,
        choices=["v1.4", "v1.5", "ic_signed_rolling", "custom_json",
                 "per_horizon_json", "per_horizon_rolling"],
        help="One or more weight sources to evaluate against the same windows.",
    )
    p.add_argument(
        "--weights-json", default=None,
        help="Path to weight JSON for --weight-source custom_json.",
    )
    p.add_argument(
        "--per-horizon-weights-json", default=None,
        help=(
            "Path to per-horizon weights JSON (output of "
            "fit_per_horizon_weights.py) for --weight-source per_horizon_json."
        ),
    )
    p.add_argument(
        "--output-dir", default="backtest/results/walk_forward",
    )
    p.add_argument(
        "--baseline-hit-rate", type=float, default=0.5,
        help="Passive baseline test_hit fraction threshold.",
    )
    p.add_argument(
        "--horizons-int", nargs="+", type=int, default=[5, 20, 60],
        help=(
            "Forward-return horizons in trading days for the yfinance "
            "fetch. Should map to the --horizons string list."
        ),
    )
    return p.parse_args()


def _ret_field_to_days(field: str) -> int:
    # "ret_60d" → 60
    try:
        return int(field.replace("ret_", "").replace("d", ""))
    except ValueError as e:
        raise SystemExit(f"Cannot parse horizon {field!r}") from e


def _records_date_span(records: list[BacktestRecord]) -> tuple[str, str]:
    dates = sorted(r.as_of_date for r in records)
    return dates[0], dates[-1]


def _write_windows_csv(
    out_path: Path,
    source_to_window_stats: dict[str, list[dict]],
    horizons: list[str],
) -> None:
    fieldnames = ["weight_source", "train_start", "train_end", "test_start",
                  "test_end", "train_n", "test_n", "weights_n_nonzero"]
    for h in horizons:
        fieldnames += [
            f"{h}_train_hit",
            f"{h}_test_hit",
            f"{h}_overfit_gap",
            f"{h}_n_train_with_return",
            f"{h}_n_test_with_return",
            f"{h}_bullish_train_hit",
            f"{h}_bullish_test_hit",
            f"{h}_bullish_overfit_gap",
            f"{h}_n_bullish_test",
            f"{h}_bearish_test_hit",
            f"{h}_n_bearish_test",
        ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for source, stats_list in source_to_window_stats.items():
            for ws in stats_list:
                win = ws["window"]
                weights = ws.get("weights_used") or {}
                n_nz = sum(1 for v in weights.values() if v)
                row = [
                    source,
                    win["train_start"], win["train_end"],
                    win["test_start"], win["test_end"],
                    ws["train_n"], ws["test_n"], n_nz,
                ]
                for h in horizons:
                    block = (ws.get("per_horizon") or {}).get(h) or {}
                    row += [
                        block.get("train_hit"),
                        block.get("test_hit"),
                        block.get("overfit_gap"),
                        block.get("n_train_with_return"),
                        block.get("n_test_with_return"),
                        block.get("bullish_train_hit"),
                        block.get("bullish_test_hit"),
                        block.get("bullish_overfit_gap"),
                        block.get("n_bullish_test"),
                        block.get("bearish_test_hit"),
                        block.get("n_bearish_test"),
                    ]
                w.writerow(row)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_globs = args.reports_glob or ["reports/analysis_mvp/*.json"]
    seen: set[str] = set()
    paths: list[str] = []
    for pattern in reports_globs:
        for p in sorted(glob.glob(pattern)):
            if p not in seen:
                paths.append(p)
                seen.add(p)
    if not paths:
        raise SystemExit(f"No reports matched: {reports_globs}")
    print(f"Loading {len(paths)} reports...")
    t0 = time.time()
    records = load_records(
        paths,
        horizons=args.horizons_int,
        capture_factor_scores=True,
        capture_market_context=False,
        capture_llm_critic=False,
        benchmark_symbol=None,
    )
    print(f"Loaded {len(records)} records in {time.time() - t0:.1f}s.")

    custom_weights = None
    if args.weights_json:
        custom_weights = json.loads(Path(args.weights_json).read_text())
        # Ensure it's a flat dict[str, float].
        if not isinstance(custom_weights, dict):
            raise SystemExit("--weights-json must be a flat factor→weight dict")
    per_horizon_weights = None
    if "per_horizon_json" in args.weight_source:
        if not args.per_horizon_weights_json:
            raise SystemExit(
                "--weight-source per_horizon_json requires "
                "--per-horizon-weights-json <path>"
            )
        per_horizon_weights = _load_per_horizon_weights(args.per_horizon_weights_json)

    corpus_min, corpus_max = _records_date_span(records)
    windows = generate_windows(
        corpus_min_date=corpus_min,
        corpus_max_date=corpus_max,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    print(
        f"Corpus span {corpus_min} → {corpus_max}; "
        f"generated {len(windows)} windows "
        f"(train {args.train_months}mo / test {args.test_months}mo / "
        f"step {args.step_months}mo)."
    )
    if not windows:
        raise SystemExit("No walk-forward windows fit the corpus span.")

    combined_summary: dict[str, dict] = {}
    source_to_window_stats: dict[str, list[dict]] = {}
    md_sections: list[str] = []
    last_source_summary = None

    for source in args.weight_source:
        if source == "per_horizon_json":
            print(f"[{source}] evaluating {len(windows)} windows "
                  f"(per-horizon static weights)...")
            ts = time.time()
            per_window: list[dict] = []
            for win in windows:
                ws = _evaluate_window_per_horizon(
                    records, win,
                    per_horizon_weights=per_horizon_weights,
                    horizons=list(args.horizons),
                )
                per_window.append(ws)
        elif source == "per_horizon_rolling":
            print(f"[{source}] evaluating {len(windows)} windows "
                  f"(per-horizon rolling fit, strict OOS)...")
            ts = time.time()
            per_window = []
            for win in windows:
                train_slice = [
                    r for r in records
                    if win.train_start <= r.as_of_date <= win.train_end
                ]
                window_weights = _fit_per_horizon_weights_for_window(
                    train_slice,
                    horizons=list(args.horizons),
                    min_abs_ic=args.min_abs_ic,
                    min_n=args.min_n,
                )
                ws = _evaluate_window_per_horizon(
                    records, win,
                    per_horizon_weights=window_weights,
                    horizons=list(args.horizons),
                )
                per_window.append(ws)
        else:
            weight_fn = _make_weight_fn(
                source,
                custom_weights=custom_weights,
                horizon=args.fit_horizon,
                min_abs_ic=args.min_abs_ic,
                min_n=args.min_n,
            )
            print(f"[{source}] evaluating {len(windows)} windows...")
            ts = time.time()
            per_window = []
            for win in windows:
                ws = evaluate_window(
                    records, win, weight_fn=weight_fn, horizons=args.horizons,
                )
                per_window.append(ws)
        summary = summarize_walk_forward(
            per_window, horizons=args.horizons,
            baseline_hit_rate=args.baseline_hit_rate,
        )
        print(
            f"[{source}] done in {time.time() - ts:.1f}s. "
            f"n_windows={summary['n_windows']}"
        )
        combined_summary[source] = summary
        source_to_window_stats[source] = per_window

        # Per-source summary.json so it can be diffed/inspected
        # individually (matches the "summary.json" naming in the plan).
        per_source_path = out_dir / f"summary_{source.replace('.', '_')}.json"
        per_source_path.write_text(
            json.dumps(
                {
                    "weight_source": source,
                    "summary": summary,
                    "per_window": per_window,
                },
                indent=2,
            )
            + "\n"
        )
        md_sections.append(
            render_walk_forward_markdown(
                summary,
                baseline_hit_rate=args.baseline_hit_rate,
                title=f"Walk-forward OOS — `{source}`",
            )
        )
        last_source_summary = (source, summary)

    # Combined artifacts.
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "windows": [w.as_dict() for w in windows],
                "horizons": list(args.horizons),
                "train_months": args.train_months,
                "test_months": args.test_months,
                "step_months": args.step_months,
                "baseline_hit_rate": args.baseline_hit_rate,
                "by_source": combined_summary,
            },
            indent=2,
        )
        + "\n"
    )
    md_header = (
        "# Walk-forward OOS evaluation\n\n"
        f"- Corpus: `{', '.join(reports_globs)}` "
        f"(span `{corpus_min}` → `{corpus_max}`, {len(records)} records)\n"
        f"- Windows: train={args.train_months}mo, "
        f"test={args.test_months}mo, step={args.step_months}mo → "
        f"**{len(windows)}** windows\n"
        f"- Weight sources evaluated: "
        f"{', '.join('`' + s + '`' for s in args.weight_source)}\n"
        f"- Baseline test_hit threshold: **{args.baseline_hit_rate:.2f}**\n\n"
        "---\n\n"
    )
    (out_dir / "summary.md").write_text(md_header + "\n\n".join(md_sections))
    _write_windows_csv(
        out_dir / "windows.csv", source_to_window_stats, list(args.horizons)
    )

    # One-line stdout summary per source so all numbers land in the log.
    h = "ret_60d"
    for src, summary in combined_summary.items():
        stats = (summary.get("per_horizon") or {}).get(h) or {}
        med = stats.get("median_test_hit")
        gap = stats.get("median_overfit_gap")
        bmed = stats.get("median_bullish_test_hit")
        bgap = stats.get("median_bullish_overfit_gap")
        med_str = f"{med * 100:.2f}%" if med is not None else "—"
        gap_str = f"{gap * 100:.2f}pp" if gap is not None else "—"
        bmed_str = f"{bmed * 100:.2f}%" if bmed is not None else "—"
        bgap_str = f"{bgap * 100:.2f}pp" if bgap is not None else "—"
        print(
            f"n_windows={summary.get('n_windows', 0)} "
            f"source={src} "
            f"median_test_hit_60d={med_str} "
            f"median_overfit_gap_60d={gap_str} "
            f"median_bullish_test_hit_60d={bmed_str} "
            f"median_bullish_overfit_gap_60d={bgap_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
