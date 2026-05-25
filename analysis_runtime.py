from __future__ import annotations

import argparse
import json

from tradingagents.analysis_only.config import load_config
from tradingagents.analysis_only.runtime import AnalysisRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delta-based watchlist runtime for analysis-only mode."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml or config.json",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously based on runtime.interval_minutes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = AnalysisRuntime(config)
    if args.loop:
        runtime.run_loop()
        return
    result = runtime.run_once()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

