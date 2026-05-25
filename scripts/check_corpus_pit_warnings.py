"""Phase 1 acceptance gate: report PIT-warning rate across the corpus.

Counts the fraction of report JSONs in `--reports-glob` whose
`data_quality.pit_warnings` is non-empty *after* dropping known-benign
categories (live ticker.info sector/industry labels and peer
fundamentals that are stable but flagged for transparency). The plan
requires >=95% of reports to be PIT-clean on the strict subset before
downstream phases (IC, walk-forward, regime) can be trusted.

Exit codes:
  0  >= --min-clean-pct of reports are clean AND count >= --min-count
  1  acceptance failed
  2  no reports matched the glob
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path


def _load_report(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# Stable labels that come from `ticker.info`. Flagged for transparency,
# not actual lookahead. These categories never enter the factor composite.
BENIGN_CATEGORIES = {
    "industry_context.sector_labels",
    "competitor_analysis.peer_fundamentals",
}

# Live-loaded sections that are intentionally used for narrative/context
# only — they appear in the report body but the values do NOT feed
# factor_scores or composite_score. Leaks here pollute the *story* the
# report tells, not the backtest signal. The strict Phase 1 gate ignores
# these by default. Audit any new entry against `pipeline._build_report`
# before adding it.
CONTEXT_ONLY_CATEGORIES = {
    "news",
    "event_timeline",
    "analyst_consensus",
    "industry_news_context",
    "earnings_calendar.forward_eps_estimates",
    "option_strategy_chain",
    "options_iv",
}

# Categories that CAN contaminate the composite. The strict gate counts
# only these against the clean fraction. Currently the only score-relevant
# section that leaks is `options_flow` (feeds the `options_net_flow`
# factor, weight 0.05 in DEFAULT_FACTOR_WEIGHTS).
SCORE_RELEVANT_CATEGORIES = {
    "options_flow",
    "fundamentals",
}


def _categorize(warning: str) -> str:
    return str(warning).split(":", 1)[0].strip() or str(warning)


def _classify(category: str) -> str:
    if category in BENIGN_CATEGORIES:
        return "benign"
    if category in CONTEXT_ONLY_CATEGORIES:
        return "context"
    if category in SCORE_RELEVANT_CATEGORIES:
        return "score_relevant"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-glob", default="reports/analysis_mvp/*.json",
    )
    parser.add_argument(
        "--min-clean-pct", type=float, default=0.95,
        help="Minimum fraction of reports clean on the strict subset.",
    )
    parser.add_argument(
        "--min-count", type=int, default=1750,
        help="Minimum total report count (Phase 1 gate: 1,750).",
    )
    parser.add_argument(
        "--show-warnings-sample", type=int, default=10,
        help="Print up to N example warnings per category.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["score_relevant", "score_plus_context", "all"],
        default="score_relevant",
        help=(
            "Which warning categories count against the gate. "
            "score_relevant (default) = only sections that feed the "
            "composite (options_flow, fundamentals). "
            "score_plus_context = also count narrative leaks "
            "(news, analyst_consensus, etc.). "
            "all = every pit_warning entry."
        ),
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2

    total = 0
    unreadable = 0
    clean_by_mode: dict[str, int] = {
        "score_relevant": 0,
        "score_plus_context": 0,
        "all": 0,
    }
    by_warning: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    examples: dict[str, list[tuple[str, str]]] = {}
    by_ticker_total: Counter[str] = Counter()
    by_ticker_clean: Counter[str] = Counter()

    for path in paths:
        total += 1
        payload = _load_report(path)
        if payload is None:
            unreadable += 1
            continue
        symbol = (payload.get("symbol") or "").upper()
        by_ticker_total[symbol] += 1
        dq = payload.get("data_quality") or {}
        warnings = dq.get("pit_warnings") or []
        cats = [_categorize(w) for w in warnings]
        classes = {_classify(c) for c in cats}
        if not warnings:
            for mode in clean_by_mode:
                clean_by_mode[mode] += 1
            if args.gate_mode == "score_relevant":
                by_ticker_clean[symbol] += 1
        else:
            if "score_relevant" not in classes:
                clean_by_mode["score_relevant"] += 1
                if args.gate_mode == "score_relevant":
                    by_ticker_clean[symbol] += 1
            if not (classes & {"score_relevant", "context"}):
                clean_by_mode["score_plus_context"] += 1
                if args.gate_mode == "score_plus_context":
                    by_ticker_clean[symbol] += 1
        for cat in cats:
            by_warning[cat] += 1
            by_class[_classify(cat)] += 1
            examples.setdefault(cat, []).append((symbol, cat))

    clean = clean_by_mode[args.gate_mode]
    pct_clean = clean / total if total else 0.0

    print("=== corpus PIT-warnings audit ===")
    print(f"Reports total:      {total}")
    print(f"Unreadable:         {unreadable}")
    for mode, n in clean_by_mode.items():
        marker = " <-- gate" if mode == args.gate_mode else ""
        print(f"Clean ({mode}): {n} ({(n / total if total else 0):.2%}){marker}")
    print(f"Warnings by class:  {dict(by_class)}")
    print()
    print("Per-ticker coverage:")
    for sym in sorted(by_ticker_total):
        n = by_ticker_total[sym]
        c = by_ticker_clean[sym]
        ratio = (c / n) if n else 0.0
        print(f"  {sym:<6} {n:>4} reports  {c:>4} clean  {ratio:6.1%}")

    if by_warning:
        print()
        print("Top warning categories (class shown in brackets):")
        for key, count in by_warning.most_common(25):
            print(f"  {count:>5} [{_classify(key):<14}] {key}")
            for sym, w in examples.get(key, [])[: args.show_warnings_sample]:
                print(f"          - {sym}: {w[:120]}")

    failures: list[str] = []
    if pct_clean < args.min_clean_pct:
        failures.append(
            f"pit_clean_pct {pct_clean:.4f} < required {args.min_clean_pct:.4f}"
        )
    if total < args.min_count:
        failures.append(f"total reports {total} < required {args.min_count}")

    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
