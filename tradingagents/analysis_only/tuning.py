"""Pure helpers for cheap-knob model tuning.

The tuner replays existing report JSON artifacts. It does not regenerate
historical reports, call data providers, or mutate production config files.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from portfolio.simulator import (
    SimulationConfig,
    SimulationResult,
    WeeklyObservation,
    run_simulation,
)
from portfolio.sizing import SizingConfig
from tradingagents.analysis_only.backtest import (
    BacktestRecord,
    rebuild_records_with_weights,
    summarize_all,
)


@dataclass(frozen=True)
class DateSlice:
    name: str
    date_from: str
    date_to: str


@dataclass(frozen=True)
class TuningBounds:
    bullish_threshold: tuple[float, float]
    bearish_threshold: tuple[float, float]
    neutral_bands: tuple[float, ...]
    weight_multiplier: tuple[float, float]
    policies: tuple[str, ...]
    max_per_name: tuple[float, float]
    max_long_exposure: tuple[float, float]
    top_n: tuple[int, int]
    stale_signal_decay: tuple[float, float]


@dataclass(frozen=True)
class TuningGates:
    min_bullish_20d_count: int = 30
    min_avg_long_exposure: float = 0.10
    max_drawdown_floor: float = -0.25
    require_excess_cagr_positive: bool = True
    enable_bearish: bool = False


@dataclass(frozen=True)
class TuningConfig:
    seed: int
    max_random_candidates: int
    refine_top_n: int
    refine_variants_per_candidate: int
    report_glob: str
    base_weights_path: str
    benchmark: str
    horizons: tuple[int, ...]
    search_slice: DateSlice
    holdout_slice: DateSlice
    bounds: TuningBounds
    gates: TuningGates
    universe: tuple[str, ...]
    initial_capital: float = 100_000.0
    cost_per_side_bps: float = 5.0
    preserve_zero_weights: bool = True


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    source: str
    weights: dict[str, float]
    bullish_threshold: float
    bearish_threshold: float
    neutral_band: float
    policy: str
    max_per_name: float
    max_long_exposure: float
    top_n: int
    stale_signal_decay: float
    enable_bearish: bool = False

    def sizing_config(self, universe: Iterable[str]) -> SizingConfig:
        return SizingConfig(
            policy=self.policy,
            max_per_name=self.max_per_name,
            max_long_exposure=self.max_long_exposure,
            min_position_weight=0.02,
            enable_bearish=self.enable_bearish,
            stale_signal_decay=self.stale_signal_decay,
            composite_threshold=self.bullish_threshold,
            top_n=self.top_n,
            universe=tuple(universe),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "weights": self.weights,
            "bullish_threshold": self.bullish_threshold,
            "bearish_threshold": self.bearish_threshold,
            "neutral_band": self.neutral_band,
            "policy": self.policy,
            "max_per_name": self.max_per_name,
            "max_long_exposure": self.max_long_exposure,
            "top_n": self.top_n,
            "stale_signal_decay": self.stale_signal_decay,
            "enable_bearish": self.enable_bearish,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CandidateConfig
    slice_name: str
    rejected: bool
    rejection_reasons: tuple[str, ...]
    score: float
    bullish_20d_hit_rate: float | None
    bullish_20d_mean_return: float | None
    bullish_20d_count: int
    portfolio_cagr: float | None
    benchmark_cagr: float | None
    excess_cagr: float | None
    sharpe: float | None
    benchmark_sharpe: float | None
    sharpe_spread: float | None
    max_drawdown: float | None
    turnover: float | None
    avg_long_exposure: float | None
    avg_max_position: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "slice": self.slice_name,
            "rejected": self.rejected,
            "rejection_reasons": list(self.rejection_reasons),
            "score": self.score,
            "bullish_20d_hit_rate": self.bullish_20d_hit_rate,
            "bullish_20d_mean_return": self.bullish_20d_mean_return,
            "bullish_20d_count": self.bullish_20d_count,
            "portfolio_cagr": self.portfolio_cagr,
            "benchmark_cagr": self.benchmark_cagr,
            "excess_cagr": self.excess_cagr,
            "sharpe": self.sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "sharpe_spread": self.sharpe_spread,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "avg_long_exposure": self.avg_long_exposure,
            "avg_max_position": self.avg_max_position,
        }


def load_tuning_config(path: str | Path) -> TuningConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    slices = raw.get("slices") or {}
    bounds = raw.get("bounds") or {}
    sizing = bounds.get("sizing") or {}
    gates = raw.get("gates") or {}
    search = slices.get("search") or {}
    holdout = slices.get("holdout") or {}
    return TuningConfig(
        seed=int(raw.get("seed", 42)),
        max_random_candidates=int(raw.get("max_random_candidates", 500)),
        refine_top_n=int(raw.get("refine_top_n", 25)),
        refine_variants_per_candidate=int(raw.get("refine_variants_per_candidate", 10)),
        report_glob=str(raw.get("report_glob", "reports/analysis_mvp/*.json")),
        base_weights_path=str(raw.get("base_weights_path", "configs/proposed_weights_v1.json")),
        benchmark=str(raw.get("benchmark", "SPY")),
        horizons=tuple(int(v) for v in raw.get("horizons", [5, 20, 60])),
        search_slice=DateSlice(
            name="search",
            date_from=str(search.get("date_from", "2025-11-21")),
            date_to=str(search.get("date_to", "2026-02-27")),
        ),
        holdout_slice=DateSlice(
            name="holdout",
            date_from=str(holdout.get("date_from", "2026-02-28")),
            date_to=str(holdout.get("date_to", "2026-05-22")),
        ),
        bounds=TuningBounds(
            bullish_threshold=tuple(bounds.get("bullish_threshold", [0.10, 0.25])),
            bearish_threshold=tuple(bounds.get("bearish_threshold", [0.15, 0.40])),
            neutral_bands=tuple(float(v) for v in bounds.get("neutral_bands", [0.01, 0.02, 0.03, 0.05])),
            weight_multiplier=tuple(bounds.get("weight_multiplier", [0.50, 1.50])),
            policies=tuple(sizing.get("policies", ["equal_weight_bullish", "top_n_bullish", "confidence_weighted"])),
            max_per_name=tuple(sizing.get("max_per_name", [0.08, 0.20])),
            max_long_exposure=tuple(sizing.get("max_long_exposure", [0.40, 0.90])),
            top_n=tuple(int(v) for v in sizing.get("top_n", [3, 8])),
            stale_signal_decay=tuple(sizing.get("stale_signal_decay", [0.25, 1.0])),
        ),
        gates=TuningGates(
            min_bullish_20d_count=int(gates.get("min_bullish_20d_count", 30)),
            min_avg_long_exposure=float(gates.get("min_avg_long_exposure", 0.10)),
            max_drawdown_floor=float(gates.get("max_drawdown_floor", -0.25)),
            require_excess_cagr_positive=bool(gates.get("require_excess_cagr_positive", True)),
            enable_bearish=bool(gates.get("enable_bearish", False)),
        ),
        universe=tuple(str(v).upper() for v in raw.get("universe", [])),
        initial_capital=float(raw.get("initial_capital", 100_000.0)),
        cost_per_side_bps=float(raw.get("cost_per_side_bps", 5.0)),
        preserve_zero_weights=bool(raw.get("preserve_zero_weights", True)),
    )


def load_base_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text())
    weights: dict[str, float] = {}
    for key, value in payload.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, (int, float)):
            weights[str(key)] = float(value)
    if not weights:
        raise ValueError(f"no numeric factor weights found in {path}")
    return weights


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return clean
    return {k: round(v / total, 8) for k, v in clean.items()}


def _sample_weights(
    rng: random.Random,
    base_weights: dict[str, float],
    *,
    multiplier_bounds: tuple[float, float],
    preserve_zero_weights: bool,
) -> dict[str, float]:
    lo, hi = multiplier_bounds
    sampled: dict[str, float] = {}
    for factor, base in base_weights.items():
        if preserve_zero_weights and base == 0:
            sampled[factor] = 0.0
            continue
        sampled[factor] = float(base) * rng.uniform(float(lo), float(hi))
    return normalize_weights(sampled)


def generate_random_candidates(
    *,
    config: TuningConfig,
    base_weights: dict[str, float],
) -> list[CandidateConfig]:
    rng = random.Random(config.seed)
    candidates = []
    for i in range(config.max_random_candidates):
        candidates.append(
            CandidateConfig(
                candidate_id=f"rand_{i:04d}",
                source="random",
                weights=_sample_weights(
                    rng,
                    base_weights,
                    multiplier_bounds=config.bounds.weight_multiplier,
                    preserve_zero_weights=config.preserve_zero_weights,
                ),
                bullish_threshold=round(rng.uniform(*config.bounds.bullish_threshold), 3),
                bearish_threshold=round(rng.uniform(*config.bounds.bearish_threshold), 3),
                neutral_band=float(rng.choice(config.bounds.neutral_bands)),
                policy=str(rng.choice(config.bounds.policies)),
                max_per_name=round(rng.uniform(*config.bounds.max_per_name), 3),
                max_long_exposure=round(rng.uniform(*config.bounds.max_long_exposure), 3),
                top_n=rng.randint(*config.bounds.top_n),
                stale_signal_decay=round(rng.uniform(*config.bounds.stale_signal_decay), 3),
                enable_bearish=config.gates.enable_bearish,
            )
        )
    return candidates


def refine_candidates(
    *,
    config: TuningConfig,
    top_candidates: list[CandidateConfig],
    start_index: int,
) -> list[CandidateConfig]:
    rng = random.Random(config.seed + 10_000)
    refined: list[CandidateConfig] = []
    for base_idx, base in enumerate(top_candidates[: config.refine_top_n]):
        for j in range(config.refine_variants_per_candidate):
            weights = {}
            for factor, value in base.weights.items():
                if config.preserve_zero_weights and value == 0:
                    weights[factor] = 0.0
                else:
                    weights[factor] = value * rng.uniform(0.85, 1.15)
            idx = start_index + len(refined)
            refined.append(
                CandidateConfig(
                    candidate_id=f"refine_{idx:04d}",
                    source=f"refined:{base.candidate_id}:{base_idx}:{j}",
                    weights=normalize_weights(weights),
                    bullish_threshold=_clamp_round(
                        base.bullish_threshold + rng.uniform(-0.025, 0.025),
                        config.bounds.bullish_threshold,
                    ),
                    bearish_threshold=_clamp_round(
                        base.bearish_threshold + rng.uniform(-0.04, 0.04),
                        config.bounds.bearish_threshold,
                    ),
                    neutral_band=float(rng.choice(config.bounds.neutral_bands)),
                    policy=base.policy,
                    max_per_name=_clamp_round(
                        base.max_per_name + rng.uniform(-0.025, 0.025),
                        config.bounds.max_per_name,
                    ),
                    max_long_exposure=_clamp_round(
                        base.max_long_exposure + rng.uniform(-0.08, 0.08),
                        config.bounds.max_long_exposure,
                    ),
                    top_n=max(config.bounds.top_n[0], min(config.bounds.top_n[1], base.top_n + rng.choice([-1, 0, 1]))),
                    stale_signal_decay=_clamp_round(
                        base.stale_signal_decay + rng.uniform(-0.15, 0.15),
                        config.bounds.stale_signal_decay,
                    ),
                    enable_bearish=config.gates.enable_bearish,
                )
            )
    return refined


def _clamp_round(value: float, bounds: tuple[float, float]) -> float:
    return round(max(float(bounds[0]), min(float(bounds[1]), value)), 3)


def filter_records_by_slice(
    records: Iterable[BacktestRecord],
    date_slice: DateSlice,
) -> list[BacktestRecord]:
    return [
        r for r in records
        if date_slice.date_from <= r.as_of_date <= date_slice.date_to
    ]


def records_to_observations(
    records: list[BacktestRecord],
    *,
    universe: Iterable[str],
) -> dict[date, dict[str, WeeklyObservation]]:
    """Group candidate records into weekly observations with carry-forward."""
    universe_set = {s.upper() for s in universe}
    by_ticker_week: dict[str, dict[date, BacktestRecord]] = {}
    for r in records:
        if universe_set and r.symbol not in universe_set:
            continue
        try:
            d = datetime.strptime(r.as_of_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        by_ticker_week.setdefault(r.symbol, {})[d] = r

    all_weeks = sorted({w for entries in by_ticker_week.values() for w in entries})
    out: dict[date, dict[str, WeeklyObservation]] = {w: {} for w in all_weeks}
    for ticker, entries in by_ticker_week.items():
        latest: BacktestRecord | None = None
        latest_week: date | None = None
        for w in all_weeks:
            if w in entries:
                latest = entries[w]
                latest_week = w
            if latest is None or latest_week is None:
                continue
            out[w][ticker] = WeeklyObservation(
                ticker=ticker,
                week=w,
                direction=latest.direction,
                composite=latest.composite_score,
                confidence=latest.confidence,
                composite_age_weeks=max(0, (w - latest_week).days // 7),
            )
    return out


def evaluate_candidate(
    *,
    candidate: CandidateConfig,
    records: list[BacktestRecord],
    prices: pd.DataFrame,
    benchmark: SimulationResult,
    date_slice: DateSlice,
    config: TuningConfig,
) -> CandidateEvaluation:
    rebuilt = rebuild_records_with_weights(
        records,
        weights=candidate.weights,
        bullish_threshold=candidate.bullish_threshold,
        bearish_threshold=candidate.bearish_threshold,
        neutral_band=candidate.neutral_band,
    )
    summary = summarize_all(
        rebuilt,
        return_fields=["ret_20d"],
        neutral_band=candidate.neutral_band,
    )
    bullish_stats = (
        summary.get("by_horizon", {})
        .get("ret_20d", {})
        .get("by_direction", {})
        .get("bullish", {})
    )
    observations = records_to_observations(rebuilt, universe=config.universe)
    weeks = sorted(observations)
    sim = run_simulation(
        weeks=weeks,
        observations=observations,
        prices=prices,
        sizing_config=candidate.sizing_config(config.universe),
        sim_config=SimulationConfig(
            initial_capital=config.initial_capital,
            cost_per_side_bps=config.cost_per_side_bps,
            benchmark=config.benchmark,
        ),
        policy_name=candidate.candidate_id,
    )
    return score_candidate(
        candidate=candidate,
        slice_name=date_slice.name,
        bullish_stats=bullish_stats,
        simulation=sim,
        benchmark=benchmark,
        gates=config.gates,
    )


def score_candidate(
    *,
    candidate: CandidateConfig,
    slice_name: str,
    bullish_stats: dict[str, Any],
    simulation: SimulationResult,
    benchmark: SimulationResult,
    gates: TuningGates,
) -> CandidateEvaluation:
    metrics = simulation.metrics or {}
    bench = benchmark.metrics or {}
    hit_rate = _as_float(bullish_stats.get("hit_rate"))
    mean_return = _as_float(bullish_stats.get("mean_forward_return"))
    count = int(bullish_stats.get("count_with_return") or 0)
    cagr = _as_float(metrics.get("cagr"))
    bench_cagr = _as_float(bench.get("cagr"))
    sharpe = _as_float(metrics.get("sharpe_annualized"))
    bench_sharpe = _as_float(bench.get("sharpe_annualized"))
    max_dd = _as_float(metrics.get("max_drawdown"))
    turnover = _as_float(metrics.get("total_one_way_turnover"))
    avg_long = _as_float(metrics.get("avg_long_exposure"))
    avg_max_position = _avg_max_position(simulation)
    excess_cagr = (
        cagr - bench_cagr
        if cagr is not None and bench_cagr is not None
        else None
    )
    sharpe_spread = (
        sharpe - bench_sharpe
        if sharpe is not None and bench_sharpe is not None
        else None
    )

    score = 0.0
    score += 2.0 * (hit_rate or 0.0)
    score += 1.5 * (mean_return or 0.0)
    score += 0.35 * (excess_cagr or 0.0)
    score += 0.20 * (sharpe_spread or 0.0)
    score += 1.0 * (max_dd or 0.0)
    score -= 0.03 * (turnover or 0.0)
    score -= 0.75 * max(0.0, (avg_max_position or 0.0) - candidate.max_per_name)

    reasons = []
    if count < gates.min_bullish_20d_count:
        reasons.append(f"bullish_20d_count<{gates.min_bullish_20d_count}")
    if avg_long is None or avg_long < gates.min_avg_long_exposure:
        reasons.append(f"avg_long_exposure<{gates.min_avg_long_exposure:.2f}")
    if avg_long is not None and avg_long > candidate.max_long_exposure + 0.01:
        reasons.append("avg_long_exposure>candidate_max_long_exposure")
    if max_dd is not None and max_dd < gates.max_drawdown_floor:
        reasons.append(f"max_drawdown<{gates.max_drawdown_floor:.2f}")
    if gates.require_excess_cagr_positive and (excess_cagr is None or excess_cagr <= 0):
        reasons.append("excess_cagr<=0")
    if candidate.enable_bearish:
        reasons.append("bearish_enabled")

    return CandidateEvaluation(
        candidate=candidate,
        slice_name=slice_name,
        rejected=bool(reasons),
        rejection_reasons=tuple(reasons),
        score=round(score, 6),
        bullish_20d_hit_rate=hit_rate,
        bullish_20d_mean_return=mean_return,
        bullish_20d_count=count,
        portfolio_cagr=cagr,
        benchmark_cagr=bench_cagr,
        excess_cagr=excess_cagr,
        sharpe=sharpe,
        benchmark_sharpe=bench_sharpe,
        sharpe_spread=sharpe_spread,
        max_drawdown=max_dd,
        turnover=turnover,
        avg_long_exposure=avg_long,
        avg_max_position=avg_max_position,
    )


def _avg_max_position(simulation: SimulationResult) -> float:
    vals = []
    for state in simulation.states:
        weights = [max(0.0, float(v)) for v in state.target_weights.values()]
        vals.append(max(weights) if weights else 0.0)
    return float(statistics.fmean(vals)) if vals else 0.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_benchmark(
    *,
    weeks: list[date],
    prices: pd.DataFrame,
    config: TuningConfig,
) -> SimulationResult:
    return run_simulation(
        weeks=weeks,
        observations={w: {} for w in weeks},
        prices=prices,
        sizing_config=SizingConfig(
            policy="equal_weight_bullish",
            max_per_name=1.0,
            max_long_exposure=1.0,
            min_position_weight=0.0,
            universe=(config.benchmark,),
        ),
        sim_config=SimulationConfig(
            initial_capital=config.initial_capital,
            cost_per_side_bps=config.cost_per_side_bps,
            benchmark=config.benchmark,
        ),
        policy_name=f"{config.benchmark}_baseline",
        benchmark_only=True,
    )


def write_tuning_outputs(
    *,
    output_dir: str | Path,
    search_evals: list[CandidateEvaluation],
    holdout_evals: dict[str, CandidateEvaluation],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        search_evals,
        key=lambda e: (e.rejected, -e.score),
    )
    accepted = [e for e in ranked if not e.rejected]
    rejected = [e for e in ranked if e.rejected]
    best = accepted[0] if accepted else ranked[0]

    rows = [_leaderboard_row(e, holdout_evals.get(e.candidate.candidate_id)) for e in ranked]
    (out / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    with (out / "leaderboard.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    (out / "best_search_config.json").write_text(
        json.dumps(best.candidate.to_dict(), indent=2, sort_keys=True)
    )
    (out / "top_10_configs.json").write_text(
        json.dumps([e.candidate.to_dict() for e in accepted[:10]], indent=2)
    )
    (out / "rejected_configs.json").write_text(
        json.dumps([e.to_dict() for e in rejected], indent=2)
    )
    (out / "best_holdout_report.md").write_text(
        render_best_holdout_report(best, holdout_evals.get(best.candidate.candidate_id))
    )


def _leaderboard_row(
    search_eval: CandidateEvaluation,
    holdout_eval: CandidateEvaluation | None,
) -> dict[str, Any]:
    c = search_eval.candidate
    row = {
        "candidate_id": c.candidate_id,
        "source": c.source,
        "rejected": search_eval.rejected,
        "rejection_reasons": ";".join(search_eval.rejection_reasons),
        "search_score": search_eval.score,
        "search_bullish_20d_hit_rate": search_eval.bullish_20d_hit_rate,
        "search_bullish_20d_count": search_eval.bullish_20d_count,
        "search_excess_cagr": search_eval.excess_cagr,
        "search_sharpe_spread": search_eval.sharpe_spread,
        "search_max_drawdown": search_eval.max_drawdown,
        "search_turnover": search_eval.turnover,
        "search_avg_long_exposure": search_eval.avg_long_exposure,
        "search_avg_max_position": search_eval.avg_max_position,
        "policy": c.policy,
        "bullish_threshold": c.bullish_threshold,
        "bearish_threshold": c.bearish_threshold,
        "neutral_band": c.neutral_band,
        "max_per_name": c.max_per_name,
        "max_long_exposure": c.max_long_exposure,
        "top_n": c.top_n,
        "stale_signal_decay": c.stale_signal_decay,
    }
    if holdout_eval is not None:
        row.update({
            "holdout_score": holdout_eval.score,
            "holdout_rejected": holdout_eval.rejected,
            "holdout_bullish_20d_hit_rate": holdout_eval.bullish_20d_hit_rate,
            "holdout_bullish_20d_count": holdout_eval.bullish_20d_count,
            "holdout_excess_cagr": holdout_eval.excess_cagr,
            "holdout_sharpe_spread": holdout_eval.sharpe_spread,
            "holdout_max_drawdown": holdout_eval.max_drawdown,
        })
    return row


def render_best_holdout_report(
    search_eval: CandidateEvaluation,
    holdout_eval: CandidateEvaluation | None,
) -> str:
    c = search_eval.candidate
    lines = [
        "# Best Search Candidate Holdout Report",
        "",
        f"- Candidate: `{c.candidate_id}`",
        f"- Policy: `{c.policy}`",
        f"- Thresholds: bullish `{c.bullish_threshold:.3f}`, bearish `{c.bearish_threshold:.3f}`, neutral band `{c.neutral_band:.3f}`",
        f"- Sizing: max/name `{c.max_per_name:.3f}`, max long `{c.max_long_exposure:.3f}`, top_n `{c.top_n}`, stale decay `{c.stale_signal_decay:.3f}`",
        "",
        "## Search Slice",
        "",
        _render_eval_line(search_eval),
    ]
    if holdout_eval is not None:
        lines += [
            "",
            "## Holdout Slice",
            "",
            _render_eval_line(holdout_eval),
            "",
            "## Stability",
            "",
            _render_stability_line(search_eval, holdout_eval),
        ]
    lines += [
        "",
        "## Weights",
        "",
        "```json",
        json.dumps(c.weights, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _render_eval_line(e: CandidateEvaluation) -> str:
    status = "REJECTED" if e.rejected else "accepted"
    return (
        f"- Status: **{status}**; score `{e.score:.4f}`; "
        f"bullish 20d hit `{_pct(e.bullish_20d_hit_rate)}` "
        f"(n={e.bullish_20d_count}); excess CAGR `{_pct(e.excess_cagr)}`; "
        f"Sharpe spread `{_signed(e.sharpe_spread)}`; max DD `{_pct(e.max_drawdown)}`."
    )


def _render_stability_line(search_eval: CandidateEvaluation, holdout_eval: CandidateEvaluation) -> str:
    hit_delta = _delta(holdout_eval.bullish_20d_hit_rate, search_eval.bullish_20d_hit_rate)
    excess_delta = _delta(holdout_eval.excess_cagr, search_eval.excess_cagr)
    return f"- Holdout minus search: hit-rate `{_signed(hit_delta)}`, excess CAGR `{_signed(excess_delta)}`."


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b
