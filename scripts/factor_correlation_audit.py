"""Phase 2 acceptance gate: pairwise factor correlation + VIF audit.

Loads the entire `--reports-glob` corpus, extracts every factor's score
column, and prints:
- Pairwise Spearman correlation matrix.
- Variance Inflation Factor (VIF) per factor (cheap estimate via 1/(1-R^2)
  from a sequential leave-one-out regression).
- A "flagged" list of pairs with |rho| > --max-rho (default 0.7) that
  need merging or orthogonalization before Phase 3 walk-forward.

Writes:
- backtest/results/factor_correlation_matrix.json
- backtest/results/factor_correlation_audit.md
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradingagents.analysis_only.scoring import factor_correlation_matrix  # noqa: E402


def _load_factor_rows(report_paths: list[str]) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for path in report_paths:
        try:
            payload = json.loads(Path(path).read_text())
        except Exception:
            continue
        model_scoring = (payload.get("key_features") or {}).get(
            "model_scoring"
        ) or {}
        for_factors = model_scoring.get("factor_scores") or []
        if not for_factors:
            continue
        row: dict[str, float | None] = {}
        for f in for_factors:
            if not f.get("data_available"):
                continue
            name = f.get("factor")
            score = f.get("score")
            if name is None or score is None:
                continue
            try:
                row[str(name)] = float(score)
            except (TypeError, ValueError):
                continue
        if row:
            rows.append(row)
    return rows


def _vif_approx(
    rows: list[dict[str, float | None]],
    *,
    min_paired: int = 30,
) -> dict[str, float | None]:
    """Approximate VIF per factor.

    For factor f, fit a least-squares regression of f's score on the
    average of every other factor's score (per record) and compute
    VIF = 1 / (1 - R^2). This is the cheap diagnostic — a full VIF
    requires multi-variable OLS but that pulls in numpy/scipy and the
    pairwise correlation matrix already catches the worst pairs.
    """
    all_factors = sorted({f for row in rows for f in row.keys()})
    out: dict[str, float | None] = {}
    for f in all_factors:
        ys: list[float] = []
        xs: list[float] = []
        for row in rows:
            target = row.get(f)
            others = [v for k, v in row.items() if k != f and v is not None]
            if target is None or len(others) < 2:
                continue
            ys.append(target)
            xs.append(statistics.fmean(others))
        if len(ys) < min_paired:
            out[f] = None
            continue
        mx = statistics.fmean(xs)
        my = statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
        if sxx == 0 or syy == 0:
            out[f] = None
            continue
        r = sxy / (sxx * syy) ** 0.5
        r2 = r * r
        if r2 >= 0.999:
            out[f] = float("inf")
        else:
            out[f] = round(1.0 / (1.0 - r2), 3)
    return out


def _render_md(
    matrix: dict[str, dict[str, float | None]],
    *,
    vif: dict[str, float | None],
    flagged: list[tuple[str, str, float]],
    n_records: int,
    max_rho: float,
) -> str:
    lines = [
        "# Factor correlation audit (Phase 2)",
        "",
        f"- Records evaluated: **{n_records}**",
        f"- |rho| threshold for flagging: **{max_rho:.2f}**",
        f"- Factors with VIF >= 5 are flagged separately.",
        "",
        "## Flagged correlated pairs",
        "",
    ]
    if not flagged:
        lines.append("_None — all pairs within threshold._")
    else:
        lines.append("| Factor A | Factor B | Spearman rho |")
        lines.append("|---|---|---:|")
        for a, b, rho in flagged:
            lines.append(f"| {a} | {b} | {rho:+.3f} |")
    lines += ["", "## Variance Inflation Factor (approx, vs mean-of-others)", ""]
    lines.append("| Factor | VIF |")
    lines.append("|---|---:|")
    for f in sorted(vif):
        v = vif[f]
        cell = "n/a" if v is None else (
            "inf" if v == float("inf") else f"{v:.2f}"
        )
        lines.append(f"| {f} | {cell} |")
    lines += ["", "## Full pairwise Spearman matrix", ""]
    factors = sorted(matrix)
    header = "| factor | " + " | ".join(factors) + " |"
    sep = "|---" + "|---" * len(factors) + "|"
    lines.append(header)
    lines.append(sep)
    for fa in factors:
        cells = []
        for fb in factors:
            v = matrix.get(fa, {}).get(fb)
            cells.append("n/a" if v is None else f"{v:+.2f}")
        lines.append(f"| {fa} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-glob", default="reports/analysis_mvp/*.json",
    )
    parser.add_argument(
        "--output-dir", default="backtest/results",
    )
    parser.add_argument(
        "--max-rho", type=float, default=0.7,
        help="Flag any pair with |rho| above this threshold.",
    )
    parser.add_argument(
        "--min-paired", type=int, default=30,
        help="Minimum non-None observations needed for a correlation cell.",
    )
    parser.add_argument(
        "--fail-on-flag",
        action="store_true",
        help=(
            "Exit non-zero if any pair exceeds --max-rho. Set during the "
            "Phase 2 acceptance check after merging/orthogonalization is "
            "supposed to be done."
        ),
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2

    print(f"Loading factor rows from {len(paths)} reports...")
    rows = _load_factor_rows(paths)
    if not rows:
        print("[fail] no factor rows extracted", file=sys.stderr)
        return 2
    print(f"Extracted {len(rows)} factor rows.")

    matrix = factor_correlation_matrix(rows, min_paired=args.min_paired)
    vif = _vif_approx(rows, min_paired=args.min_paired)

    flagged: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = set()
    for fa, row in matrix.items():
        for fb, v in row.items():
            if fa == fb or v is None:
                continue
            pair = frozenset({fa, fb})
            if pair in seen:
                continue
            seen.add(pair)
            if abs(v) > args.max_rho:
                flagged.append((fa, fb, float(v)))
    flagged.sort(key=lambda t: abs(t[2]), reverse=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factor_correlation_matrix.json").write_text(
        json.dumps(matrix, indent=2)
    )
    (out_dir / "factor_correlation_audit.md").write_text(
        _render_md(
            matrix,
            vif=vif,
            flagged=flagged,
            n_records=len(rows),
            max_rho=args.max_rho,
        )
    )
    print(f"Wrote: {out_dir / 'factor_correlation_matrix.json'}")
    print(f"Wrote: {out_dir / 'factor_correlation_audit.md'}")

    print()
    print(f"Flagged pairs (|rho| > {args.max_rho}): {len(flagged)}")
    for a, b, rho in flagged[:20]:
        print(f"  {rho:+.3f}  {a}  <->  {b}")

    high_vif = [(f, v) for f, v in vif.items() if v is not None and v >= 5.0]
    if high_vif:
        print()
        print("High-VIF factors (>=5):")
        for f, v in sorted(high_vif, key=lambda t: t[1], reverse=True):
            print(f"  {v:>6.2f}  {f}")

    if args.fail_on_flag and flagged:
        print(
            f"\nFAIL: {len(flagged)} pair(s) exceed |rho| > {args.max_rho}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
