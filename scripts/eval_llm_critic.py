"""Phase 6: evaluate the LLM critic against the corpus.

Computes critic metrics against direction-signed forward returns:
  1. Standalone Spearman IC of each critic field.
  2. Incremental Spearman IC of each critic field vs the residual of
     `signed_forward_return ~ composite_score`.
  3. Legacy veto efficacy when old v1.0 veto fields are present.

Ships the result as `backtest/results/critic_eval.{json,md}` plus
stdout. Phase 6 gate:
  - v1.1-compatible expected-sign incremental IC > 0.03 on walk-forward
    OOS at ret_20d or ret_60d, with at least 500 critic-covered records.
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

from backtest import load_records  # noqa: E402
from tradingagents.analysis_only.backtest import (  # noqa: E402
    walk_forward_backtest,
)
from tradingagents.analysis_only.scoring import (  # noqa: E402
    _spearman_pairwise,
)


PHASE6_GATE_HORIZONS = ("ret_20d", "ret_60d")


def _olst_residuals(
    xs: list[float], ys: list[float]
) -> list[float] | None:
    """Univariate-OLS residuals of y on x; None if degenerate."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    beta = sxy / sxx
    alpha = my - beta * mx
    return [ys[i] - (alpha + beta * xs[i]) for i in range(n)]


def _olst_residuals_two(
    x1: list[float], x2: list[float], y: list[float],
) -> list[float] | None:
    """Bivariate-OLS residuals of y on (x1, x2); None if degenerate.

    Closed-form for 2 predictors. Solves the 2x2 normal equations.
    """
    n = len(y)
    if n < 4 or len(x1) != n or len(x2) != n:
        return None
    m1 = sum(x1) / n
    m2 = sum(x2) / n
    my = sum(y) / n
    s11 = sum((a - m1) ** 2 for a in x1)
    s22 = sum((a - m2) ** 2 for a in x2)
    s12 = sum((x1[i] - m1) * (x2[i] - m2) for i in range(n))
    s1y = sum((x1[i] - m1) * (y[i] - my) for i in range(n))
    s2y = sum((x2[i] - m2) * (y[i] - my) for i in range(n))
    det = s11 * s22 - s12 * s12
    if det == 0:
        return None
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    a = my - b1 * m1 - b2 * m2
    return [y[i] - (a + b1 * x1[i] + b2 * x2[i]) for i in range(n)]


def _direction_signed_return(record, horizon: str) -> float | None:
    ret = record.forward_returns.get(horizon)
    if ret is None:
        return None
    if record.direction == "bullish":
        return float(ret)
    if record.direction == "bearish":
        return -float(ret)
    return None


def _stats_for_critic_field(
    records: list,
    *,
    field: str,
    horizon: str,
) -> dict:
    paired: list[tuple[float, float, float]] = []  # (critic, composite, ret)
    for r in records:
        if not r.llm_critic:
            continue
        v = r.llm_critic.get(field)
        if v is None:
            continue
        comp = r.composite_score
        ret = _direction_signed_return(r, horizon)
        if comp is None or ret is None:
            continue
        paired.append((float(v), float(comp), float(ret)))
    if len(paired) < 10:
        return {
            "n": len(paired),
            "ic_standalone": None,
            "ic_incremental": None,
        }
    critics = [p[0] for p in paired]
    comps = [p[1] for p in paired]
    rets = [p[2] for p in paired]
    standalone = _spearman_pairwise(critics, rets)
    residuals = _olst_residuals(comps, rets)
    incremental = (
        _spearman_pairwise(critics, residuals) if residuals else None
    )
    return {
        "n": len(paired),
        "ic_standalone": standalone,
        "ic_incremental": incremental,
    }


def _stats_for_disagreement_field(
    records: list,
    *,
    field: str,
    horizon: str,
) -> dict:
    """Phase 7 incremental IC of a disagreement metric.

    Three IC numbers per field:
      - standalone IC vs forward return
      - incremental IC after regressing out composite_score
      - incremental IC after regressing out (composite_score, critic_invalidation_prob_30d)
    The last one is the strict Phase 7 gate metric — disagreement
    must add information *on top of* the per-report critic itself.
    """
    paired: list[tuple[float, float, float, float]] = []
    for r in records:
        if not r.llm_disagreement:
            continue
        v = r.llm_disagreement.get(field)
        if v is None:
            continue
        comp = r.composite_score
        ret = _direction_signed_return(r, horizon)
        if comp is None or ret is None:
            continue
        critic_inv = (
            (r.llm_critic or {}).get("invalidation_prob_30d")
        )
        paired.append((
            float(v), float(comp),
            float(critic_inv) if critic_inv is not None else 0.0,
            float(ret),
        ))
    if len(paired) < 10:
        return {
            "n": len(paired),
            "ic_standalone": None,
            "ic_incremental_vs_composite": None,
            "ic_incremental_vs_composite_and_critic": None,
        }
    dis_vals = [p[0] for p in paired]
    comps = [p[1] for p in paired]
    critics = [p[2] for p in paired]
    rets = [p[3] for p in paired]
    standalone = _spearman_pairwise(dis_vals, rets)
    res_comp = _olst_residuals(comps, rets)
    inc_vs_comp = (
        _spearman_pairwise(dis_vals, res_comp) if res_comp else None
    )
    res_both = _olst_residuals_two(comps, critics, rets)
    inc_vs_both = (
        _spearman_pairwise(dis_vals, res_both) if res_both else None
    )
    return {
        "n": len(paired),
        "ic_standalone": standalone,
        "ic_incremental_vs_composite": inc_vs_comp,
        "ic_incremental_vs_composite_and_critic": inc_vs_both,
    }


def _veto_efficacy(records: list, *, horizon: str) -> dict:
    """How much realized loss does the veto remove vs baseline?

    For each vetoed record, the baseline assumes the trade was taken
    in the analyst's stated direction (bullish = long, bearish = short,
    neutral = no position). The veto avoids the trade entirely
    (returns 0). The metric we report is the mean *signed-direction
    return* on vetoed records: a negative mean means the veto saved
    money; a positive mean means the veto cost money.
    """
    vetoed: list[float] = []
    not_vetoed: list[float] = []
    for r in records:
        if not r.llm_critic:
            continue
        ret = r.forward_returns.get(horizon)
        if ret is None:
            continue
        if r.direction == "bullish":
            signed = float(ret)
        elif r.direction == "bearish":
            signed = -float(ret)
        else:
            continue
        if r.llm_critic.get("veto"):
            vetoed.append(signed)
        else:
            not_vetoed.append(signed)
    out = {
        "n_vetoed": len(vetoed),
        "n_not_vetoed": len(not_vetoed),
        "vetoed_mean_signed_ret": (
            round(statistics.fmean(vetoed), 6) if vetoed else None
        ),
        "vetoed_hit_rate": (
            round(sum(1 for v in vetoed if v > 0) / len(vetoed), 4)
            if vetoed else None
        ),
        "not_vetoed_mean_signed_ret": (
            round(statistics.fmean(not_vetoed), 6) if not_vetoed else None
        ),
        "not_vetoed_hit_rate": (
            round(sum(1 for v in not_vetoed if v > 0) / len(not_vetoed), 4)
            if not_vetoed else None
        ),
    }
    # Loss-reduction: among vetoed records that would have lost money,
    # how much loss did we remove? Compare avg negative return in
    # vetoed vs non-vetoed losing trades.
    losing_vetoed = [v for v in vetoed if v < 0]
    losing_not_vetoed = [v for v in not_vetoed if v < 0]
    if losing_vetoed and losing_not_vetoed:
        avg_loss_vetoed = statistics.fmean(losing_vetoed)
        avg_loss_not_vetoed = statistics.fmean(losing_not_vetoed)
        if avg_loss_not_vetoed == 0:
            loss_reduction = None
        else:
            loss_reduction = round(
                1.0 - (avg_loss_vetoed / avg_loss_not_vetoed), 4
            )
        out["loss_reduction_ratio"] = loss_reduction
    else:
        out["loss_reduction_ratio"] = None
    return out


def _phase6_gate_evaluation(
    horizons: dict,
    *,
    min_n: int,
    threshold: float,
) -> tuple[bool, list[dict], dict | None, list[str]]:
    candidates: list[dict] = []
    for horizon in PHASE6_GATE_HORIZONS:
        fields = horizons.get(horizon) or {}
        inv = fields.get("invalidation_prob_30d") or {}
        conf = fields.get("confidence_adjustment") or {}
        inv_ic = inv.get("ic_incremental")
        conf_ic = conf.get("ic_incremental")
        if inv.get("n", 0) >= min_n and inv_ic is not None:
            candidates.append({
                "horizon": horizon,
                "field": "invalidation_prob_30d",
                "n": inv.get("n"),
                "ic_incremental": inv_ic,
                "expected_signed_ic": -float(inv_ic),
                "expected_direction": "negative",
            })
        if conf.get("n", 0) >= min_n and conf_ic is not None:
            candidates.append({
                "horizon": horizon,
                "field": "confidence_adjustment",
                "n": conf.get("n"),
                "ic_incremental": conf_ic,
                "expected_signed_ic": float(conf_ic),
                "expected_direction": "positive",
            })
    best = max(candidates, key=lambda c: c["expected_signed_ic"], default=None)
    gate_pass = bool(best is not None and best["expected_signed_ic"] > threshold)
    if best is None:
        notes = [
            "no ret_20d/ret_60d critic field met the minimum OOS sample count "
            f"({min_n}) with a computable incremental IC",
        ]
    else:
        notes = [
            f"expected-sign incremental IC gate (>{threshold}): "
            f"{'PASS' if gate_pass else 'FAIL'}",
            (
                f"best={best['horizon']} {best['field']} "
                f"n={best['n']} ic={best['ic_incremental']} "
                f"expected_signed_ic={best['expected_signed_ic']:.4f}"
            ),
            "legacy veto efficacy is reported for old records but is not part "
            "of the v1.1 gate",
        ]
    return gate_pass, candidates, best, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    parser.add_argument(
        "--output-json", default="backtest/results/critic_eval.json"
    )
    parser.add_argument(
        "--output-md", default="backtest/results/critic_eval.md"
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--primary-horizon", default="ret_20d")
    parser.add_argument(
        "--walk-forward", action="store_true", default=True,
        help="Phase 6 gate is OOS; this is on by default.",
    )
    parser.add_argument(
        "--no-walk-forward", dest="walk_forward", action="store_false",
    )
    parser.add_argument("--wf-refit-freq-weeks", type=int, default=4)
    parser.add_argument("--wf-train-window-weeks", type=int, default=52)
    parser.add_argument("--wf-gap-weeks", type=int, default=4)
    parser.add_argument("--wf-first-refit-after-weeks", type=int, default=26)
    parser.add_argument("--wf-min-n", type=int, default=50)
    parser.add_argument("--wf-min-abs-ic", type=float, default=0.05)
    parser.add_argument("--incremental-ic-gate", type=float, default=0.03)
    parser.add_argument("--min-critic-oos-n", type=int, default=500)
    parser.add_argument(
        "--disagreement-ic-gate", type=float, default=0.02,
        help="Phase 7 gate: incremental IC of llm_disagreement vs composite + critic.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2
    print(f"Loading {len(paths)} reports + critic blocks...")
    records = load_records(
        paths,
        horizons=args.horizons,
        capture_factor_scores=True,
        capture_market_context=False,
        capture_llm_critic=True,
        benchmark_symbol=None,
    )
    n_with_critic = sum(1 for r in records if r.llm_critic)
    n_with_disagreement = sum(1 for r in records if r.llm_disagreement)
    print(f"Records with critic block: {n_with_critic}/{len(records)}")
    print(f"Records with disagreement block (multi-model): "
          f"{n_with_disagreement}/{len(records)}")

    if args.walk_forward:
        records, _ = walk_forward_backtest(
            records,
            refit_freq_weeks=args.wf_refit_freq_weeks,
            train_window_weeks=args.wf_train_window_weeks,
            gap_weeks=args.wf_gap_weeks,
            first_refit_after_weeks=args.wf_first_refit_after_weeks,
            horizon=args.primary_horizon,
            min_abs_ic=args.wf_min_abs_ic,
            min_n=args.wf_min_n,
        )
        print(f"OOS records (walk-forward): {len(records)}")

    summary: dict = {
        "n_records": len(records),
        "n_with_critic": sum(1 for r in records if r.llm_critic),
        "n_with_disagreement": sum(
            1 for r in records if r.llm_disagreement
        ),
        "walk_forward": args.walk_forward,
        "horizons": {},
        "veto": {},
        "disagreement": {},
        "gates": {
            "incremental_ic_threshold": args.incremental_ic_gate,
            "min_critic_oos_n": args.min_critic_oos_n,
            "phase6_horizons": list(PHASE6_GATE_HORIZONS),
            "disagreement_ic_threshold": args.disagreement_ic_gate,
        },
    }
    horizons_str = [f"ret_{h}d" for h in args.horizons]
    for h in horizons_str:
        summary["horizons"][h] = {
            "invalidation_prob_30d": _stats_for_critic_field(
                records, field="invalidation_prob_30d", horizon=h,
            ),
            "confidence_adjustment": _stats_for_critic_field(
                records, field="confidence_adjustment", horizon=h,
            ),
        }
        summary["veto"][h] = _veto_efficacy(records, horizon=h)
        summary["disagreement"][h] = {
            "confidence_adjustment_stdev": _stats_for_disagreement_field(
                records,
                field="confidence_adjustment_stdev", horizon=h,
            ),
            "blindspots_jaccard_mean": _stats_for_disagreement_field(
                records,
                field="blindspots_jaccard_mean", horizon=h,
            ),
            "invalidation_prob_30d_stdev": _stats_for_disagreement_field(
                records,
                field="invalidation_prob_30d_stdev", horizon=h,
            ),
        }

    # Phase 6 gate evaluation. v1.1 critic removed the stale veto bit, so the
    # production gate uses expected-sign incremental IC only.
    gate_pass, phase6_candidates, best, gate_notes = _phase6_gate_evaluation(
        summary["horizons"],
        min_n=args.min_critic_oos_n,
        threshold=args.incremental_ic_gate,
    )
    summary["phase6_gate_candidates"] = phase6_candidates
    summary["phase6_best_candidate"] = best
    summary["gate_pass"] = gate_pass
    summary["gate_notes"] = gate_notes

    # Phase 7 disagreement gate evaluation (gated on n_with_disagreement > 0)
    primary = args.primary_horizon
    dis_primary = summary["disagreement"].get(primary, {})
    if summary["n_with_disagreement"] == 0:
        summary["disagreement_gate_pass"] = None
        summary["disagreement_gate_notes"] = [
            "no records have a multi-model disagreement block; skip "
            "Phase 7 gate"
        ]
    else:
        best_ic = 0.0
        for field_stats in dis_primary.values():
            ic = field_stats.get("ic_incremental_vs_composite_and_critic")
            if ic is not None and abs(ic) > best_ic:
                best_ic = abs(ic)
        dis_passes = best_ic > args.disagreement_ic_gate
        summary["disagreement_gate_pass"] = bool(dis_passes)
        summary["disagreement_gate_notes"] = [
            f"disagreement incremental IC vs composite+critic "
            f"(>{args.disagreement_ic_gate}): "
            f"{'PASS' if dis_passes else 'FAIL'} (best |IC|={best_ic:.4f})"
        ]

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    out_md = Path(args.output_md)
    md_lines = [
        "# LLM Critic Evaluation",
        "",
        f"- Records: {summary['n_records']}  "
        f"(with critic block: {summary['n_with_critic']})",
        f"- Walk-forward OOS: {summary['walk_forward']}",
        "",
        "## Standalone + incremental IC vs forward return",
        "",
        "| horizon | field | n | IC standalone | IC incremental |",
        "| ------- | ----- | -:| -------------:| --------------:|",
    ]
    for h, fields in summary["horizons"].items():
        for fname, st in fields.items():
            md_lines.append(
                f"| {h} | {fname} | {st.get('n')} | "
                f"{st.get('ic_standalone')} | "
                f"{st.get('ic_incremental')} |"
            )
    md_lines += [
        "",
        "## Veto efficacy",
        "",
        "| horizon | n_vetoed | n_not_vetoed | vetoed mean | not_vetoed mean | loss_reduction |",
        "| ------- | -------: | -----------: | ----------: | --------------: | -------------: |",
    ]
    for h, v in summary["veto"].items():
        md_lines.append(
            f"| {h} | {v['n_vetoed']} | {v['n_not_vetoed']} | "
            f"{v['vetoed_mean_signed_ret']} | "
            f"{v['not_vetoed_mean_signed_ret']} | "
            f"{v['loss_reduction_ratio']} |"
        )
    md_lines += [
        "",
        "## LLM disagreement (Phase 7)",
        "",
        "| horizon | field | n | IC standalone | inc IC vs comp | inc IC vs comp+critic |",
        "| ------- | ----- | -:| -------------:| --------------:| ---------------------:|",
    ]
    for h, fields in summary["disagreement"].items():
        for fname, st in fields.items():
            md_lines.append(
                f"| {h} | {fname} | {st.get('n')} | "
                f"{st.get('ic_standalone')} | "
                f"{st.get('ic_incremental_vs_composite')} | "
                f"{st.get('ic_incremental_vs_composite_and_critic')} |"
            )
    md_lines += [
        "",
        f"## Phase 6 gate: {'PASS' if summary['gate_pass'] else 'FAIL'}",
        "",
    ]
    for note in summary["gate_notes"]:
        md_lines.append(f"- {note}")
    md_lines += [
        "",
        f"## Phase 7 gate: "
        f"{'PASS' if summary['disagreement_gate_pass'] else ('SKIP' if summary['disagreement_gate_pass'] is None else 'FAIL')}",
        "",
    ]
    for note in summary["disagreement_gate_notes"]:
        md_lines.append(f"- {note}")
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    print(f"Phase 6 gate: {'PASS' if summary['gate_pass'] else 'FAIL'}")
    for note in summary["gate_notes"]:
        print(f"  - {note}")
    print()
    label = (
        "SKIP" if summary["disagreement_gate_pass"] is None
        else "PASS" if summary["disagreement_gate_pass"] else "FAIL"
    )
    print(f"Phase 7 gate: {label}")
    for note in summary["disagreement_gate_notes"]:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
