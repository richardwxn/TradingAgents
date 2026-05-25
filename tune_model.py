"""Tune cheap model knobs against existing historical report artifacts."""

from __future__ import annotations

import argparse
import glob
from datetime import timedelta

from backtest import load_records
from portfolio_simulate import _fetch_weekly_close, _project_to_weeks
from tradingagents.analysis_only.tuning import (
    evaluate_candidate,
    filter_records_by_slice,
    generate_random_candidates,
    load_base_weights,
    load_tuning_config,
    records_to_observations,
    refine_candidates,
    run_benchmark,
    write_tuning_outputs,
)
from tradingagents.analysis_only.agent_review import (
    run_tuning_agent_review,
    write_agent_review_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tuning.yaml")
    parser.add_argument("--output-dir", default="backtest/results/tuning")
    parser.add_argument(
        "--enable-agent-review",
        action="store_true",
        help=(
            "Run the full TradingAgents graph on representative holdout "
            "contexts and write agent_review.json/md."
        ),
    )
    parser.add_argument("--agent-review-top-n", type=int, default=5)
    parser.add_argument("--agent-review-max-contexts", type=int, default=5)
    parser.add_argument("--agent-review-llm-provider", default="openai")
    parser.add_argument("--agent-review-quick-model", default="gpt-5.4-mini")
    parser.add_argument(
        "--agent-review-deep-model",
        default=None,
        help="Defaults to --agent-review-quick-model when omitted.",
    )
    parser.add_argument("--agent-review-base-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tuning_config(args.config)
    base_weights = load_base_weights(config.base_weights_path)
    report_paths = sorted(glob.glob(config.report_glob))
    if not report_paths:
        raise SystemExit(f"No reports matched: {config.report_glob}")

    print(f"Loading {len(report_paths)} reports and forward returns...")
    records = load_records(
        report_paths,
        horizons=list(config.horizons),
        capture_factor_scores=True,
        benchmark_symbol=config.benchmark,
    )
    if not records:
        raise SystemExit("No usable records loaded.")

    universe = list(config.universe) or sorted({r.symbol for r in records})
    search_records = filter_records_by_slice(records, config.search_slice)
    holdout_records = filter_records_by_slice(records, config.holdout_slice)
    if not search_records:
        raise SystemExit("No search-slice records loaded.")
    if not holdout_records:
        raise SystemExit("No holdout-slice records loaded.")

    all_observation_weeks = sorted(
        set(records_to_observations(search_records, universe=universe))
        | set(records_to_observations(holdout_records, universe=universe))
    )
    start = min(all_observation_weeks)
    end = max(all_observation_weeks)
    print(f"Fetching weekly prices once ({start} -> {end}) for {len(universe)} symbols + {config.benchmark}...")
    daily_prices = _fetch_weekly_close(
        sorted(set(universe) | {config.benchmark}),
        start=start,
        end=end + timedelta(days=7),
    )
    if daily_prices.empty:
        raise SystemExit("No portfolio price data available.")
    weekly_prices = _project_to_weeks(daily_prices, all_observation_weeks)

    search_weeks = sorted(records_to_observations(search_records, universe=universe))
    holdout_weeks = sorted(records_to_observations(holdout_records, universe=universe))
    search_benchmark = run_benchmark(
        weeks=search_weeks,
        prices=weekly_prices,
        config=config,
    )
    holdout_benchmark = run_benchmark(
        weeks=holdout_weeks,
        prices=weekly_prices,
        config=config,
    )

    print(f"Generating {config.max_random_candidates} random candidates...")
    random_candidates = generate_random_candidates(config=config, base_weights=base_weights)
    random_evals = [
        evaluate_candidate(
            candidate=c,
            records=search_records,
            prices=weekly_prices,
            benchmark=search_benchmark,
            date_slice=config.search_slice,
            config=config,
        )
        for c in random_candidates
    ]
    ranked_random = sorted(random_evals, key=lambda e: (e.rejected, -e.score))
    top_for_refine = [e.candidate for e in ranked_random if not e.rejected][: config.refine_top_n]
    if not top_for_refine:
        top_for_refine = [e.candidate for e in ranked_random[: config.refine_top_n]]

    print(
        f"Refining {len(top_for_refine)} candidates x "
        f"{config.refine_variants_per_candidate} variants..."
    )
    refined_candidates = refine_candidates(
        config=config,
        top_candidates=top_for_refine,
        start_index=len(random_candidates),
    )
    refined_evals = [
        evaluate_candidate(
            candidate=c,
            records=search_records,
            prices=weekly_prices,
            benchmark=search_benchmark,
            date_slice=config.search_slice,
            config=config,
        )
        for c in refined_candidates
    ]
    search_evals = sorted(
        random_evals + refined_evals,
        key=lambda e: (e.rejected, -e.score),
    )

    print(f"Evaluating holdout for {len(search_evals)} candidates...")
    holdout_evals = {
        e.candidate.candidate_id: evaluate_candidate(
            candidate=e.candidate,
            records=holdout_records,
            prices=weekly_prices,
            benchmark=holdout_benchmark,
            date_slice=config.holdout_slice,
            config=config,
        )
        for e in search_evals
    }

    write_tuning_outputs(
        output_dir=args.output_dir,
        search_evals=search_evals,
        holdout_evals=holdout_evals,
    )

    if args.enable_agent_review:
        print(
            "Running TradingAgents review "
            f"(top_n={args.agent_review_top_n}, "
            f"contexts={args.agent_review_max_contexts})..."
        )
        review = run_tuning_agent_review(
            search_evals=search_evals,
            holdout_evals=holdout_evals,
            holdout_records=holdout_records,
            top_n=args.agent_review_top_n,
            max_contexts=args.agent_review_max_contexts,
            provider=args.agent_review_llm_provider,
            quick_model=args.agent_review_quick_model,
            deep_model=args.agent_review_deep_model,
            base_url=args.agent_review_base_url,
        )
        write_agent_review_outputs(output_dir=args.output_dir, block=review)

    best = next((e for e in search_evals if not e.rejected), search_evals[0])
    holdout = holdout_evals.get(best.candidate.candidate_id)
    print("\n=== best search candidate ===")
    print(f"candidate: {best.candidate.candidate_id}")
    print(f"score: {best.score:.4f}")
    print(f"search bullish 20d hit: {(best.bullish_20d_hit_rate or 0) * 100:.2f}%")
    if holdout:
        print(f"holdout bullish 20d hit: {(holdout.bullish_20d_hit_rate or 0) * 100:.2f}%")
        print(f"holdout excess CAGR: {(holdout.excess_cagr or 0) * 100:.2f}%")
    print(f"Wrote tuning artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
