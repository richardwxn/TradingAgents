from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import logging

from tradingagents.analysis_only import AnalysisOnlyMVP, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis-only MVP for a ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. NVDA)")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--horizon",
        default="swing_1_4_weeks",
        help="Analysis horizon label",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/analysis_mvp",
        help="Directory to write JSON report",
    )
    parser.add_argument(
        "--data-provider",
        default="polygon",
        choices=["auto", "yfinance", "openbb", "polygon"],
        help="Market data provider preference",
    )
    parser.add_argument(
        "--disable-options-scan",
        action="store_true",
        help="Disable unusual options activity scan",
    )
    parser.add_argument(
        "--min-unusual-option-notional",
        type=float,
        default=500000.0,
        help="Minimum estimated contract notional for unusual options filter",
    )
    parser.add_argument(
        "--min-option-volume-oi-ratio",
        type=float,
        default=3.0,
        help="Minimum volume/open-interest ratio for unusual options filter",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step pipeline progress",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path for run logs",
    )
    parser.add_argument(
        "--competitors",
        default="",
        help="Comma-separated competitor tickers (e.g. AMD,INTC,AVGO)",
    )
    parser.add_argument(
        "--enable-llm-insights",
        action="store_true",
        help="Enable optional LLM synthesis for industry/news/peer insights",
    )
    parser.add_argument(
        "--enable-narrative",
        action="store_true",
        help=(
            "Use the LLM to rewrite the thesis + bull/bear bullets as prose. "
            "Cheaper than --enable-llm-insights; falls back to templated text "
            "on LLM error."
        ),
    )
    parser.add_argument(
        "--enable-llm-critic",
        action="store_true",
        help=(
            "Phase 6: run the adversarial LLM critic and attach the "
            "validated `llm_critic` block to the report. Off by default."
        ),
    )
    parser.add_argument(
        "--enable-tradingagents-review",
        action="store_true",
        help=(
            "Run the full TradingAgents graph for this ticker/date and attach "
            "a structured review block to key_features.tradingagents_review."
        ),
    )
    parser.add_argument(
        "--confidence-calibration-path",
        default="configs/confidence_calibration.json",
        help=(
            "Phase 5: path to the isotonic calibration JSON. When the file "
            "exists, the composite -> realized hit-rate map replaces the "
            "heuristic confidence formula."
        ),
    )
    parser.add_argument(
        "--regime-weights-path",
        default="configs/regime_weights.json",
        help=(
            "Phase 4: path to the regime-conditional factor-weights JSON. "
            "Optional; pipeline falls back to default weights when absent."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "google", "anthropic", "xai", "openrouter", "ollama"],
        help="LLM provider for optional synthesis",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4-mini",
        help="LLM model id for optional synthesis",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional custom base URL for LLM provider",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing a sibling Markdown report",
    )
    parser.add_argument(
        "--no-json-stdout",
        action="store_true",
        help="Do not echo the full JSON report to stdout",
    )
    parser.add_argument(
        "--state-store",
        default="state/analysis_state.sqlite",
        help=(
            "Path to sqlite state store; enables the delta_since_last_report "
            "block. Use --no-state to disable."
        ),
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Disable state store load/save and the delta block.",
    )
    parser.add_argument(
        "--portfolio-path",
        default="configs/portfolio_snapshot.json",
        help="Optional JSON portfolio snapshot for position-aware action plans.",
    )
    return parser.parse_args()


def _build_logger(log_file: str | None) -> logging.Logger | None:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analysis_mvp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def main() -> None:
    args = parse_args()
    logger = _build_logger(args.log_file)
    competitors = [
        t.strip().upper()
        for t in args.competitors.split(",")
        if t.strip()
    ]
    state_path = None if args.no_state else args.state_store
    mvp = AnalysisOnlyMVP(
        horizon=args.horizon,
        data_provider=args.data_provider,
        options_enabled=not args.disable_options_scan,
        min_unusual_option_notional=args.min_unusual_option_notional,
        min_option_volume_oi_ratio=args.min_option_volume_oi_ratio,
        competitors=competitors,
        enable_llm_insights=args.enable_llm_insights,
        enable_narrative=args.enable_narrative,
        enable_llm_critic=args.enable_llm_critic,
        enable_tradingagents_review=args.enable_tradingagents_review,
        confidence_calibration_path=args.confidence_calibration_path,
        regime_weights_path=args.regime_weights_path,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        portfolio_path=args.portfolio_path,
        state_store_path=state_path,
        verbose=args.verbose,
        logger=logger,
    )
    report = mvp.run(symbol=args.ticker, as_of_date=args.date)
    saved_path = mvp.save_report(report, output_dir=Path(args.output_dir))

    print(f"Saved JSON report: {saved_path.resolve()}")
    if not args.no_markdown:
        md_path = saved_path.with_suffix(".md")
        md_path.write_text(render_markdown(report.to_json_dict()))
        print(f"Saved Markdown report: {md_path.resolve()}")
    if not args.no_json_stdout:
        print(json.dumps(report.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
