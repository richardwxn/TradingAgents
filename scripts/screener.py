"""Nasdaq screener CLI — Phase B1 (Unit 3).

Runs `AnalysisOnlyMVP` over a universe yaml in a threadpool, extracts a
compact `ScreenerCandidate` per ticker, and writes a ranked summary
(`ranked.json` + `ranked.md`) to `reports/screener/YYYY-MM-DD/`.

Phase B1 ships with a hardcoded Nasdaq 100 list in
`configs/screener_universe_nasdaq100.yaml`. Unit 4 (Phase B2) will swap
that for a Polygon-derived Nasdaq Composite filtered list and add
cohort-aware scoring.

Per-ticker reports are NOT persisted — the screener is in-memory only so
it doesn't pollute the Phase 2 corpus under `reports/analysis_mvp/`.

Usage:
    python scripts/screener.py \\
        --universe-yaml configs/screener_universe_nasdaq100.yaml \\
        --date 2026-05-22 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_cls
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_universe_yaml(
    path: Path,
) -> tuple[list[str], dict[str, dict], dict]:
    """Read the screener universe yaml.

    Returns `(tickers, per_ticker_metadata, raw_yaml_dict)`.

    Handles both:
    - Phase B1 flat shape: `tickers: [AAPL, AMZN, ...]` →
      per_ticker_metadata is empty dict per ticker.
    - Phase B2 structured shape (Unit 4):
      `tickers: [{symbol: AAPL, sector: Tech-MegaCap,
                  market_cap_usd: ..., adv_usd: ..., sic_code: ...}, ...]` →
      per_ticker_metadata = {SYMBOL: {sector: ..., market_cap_usd: ...,
                                       adv_usd: ..., sic_code: ...}}.
    """
    import yaml  # local import

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    raw_tickers = data.get("tickers") or []
    out: list[str] = []
    meta: dict[str, dict] = {}
    for entry in raw_tickers:
        if isinstance(entry, str):
            sym = entry.strip().upper()
            entry_meta: dict = {}
        elif isinstance(entry, dict):
            sym = str(entry.get("symbol") or "").strip().upper()
            entry_meta = {
                k: v
                for k, v in entry.items()
                if k != "symbol" and v is not None
            }
        else:
            sym = ""
            entry_meta = {}
        if sym and sym not in out:
            out.append(sym)
            meta[sym] = entry_meta
    return out, meta, data


def _load_sector_map(path: Path) -> dict[str, str]:
    """Load `sectors:` block from configs/universe.yaml.

    Returns SYMBOL -> sector_label. Empty dict on parse failure so the
    screener still runs (everything buckets under "Unknown").
    """
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    raw = data.get("sectors") or {}
    return {str(k).upper(): str(v) for k, v in raw.items() if v}


def _load_core_tickers(path: Path) -> set[str]:
    """Load just the `core:` cohort from configs/universe.yaml."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return set()
    return {str(t).strip().upper() for t in (data.get("core") or [])}


def _screen_one(args: tuple) -> dict:
    """Worker: build an AnalysisOnlyMVP, run it, return a compact result.

    Worker is defined at module level so it's picklable (matches the
    `scripts/generate_corpus.py` precedent — same yfinance/Polygon retry
    semantics, but the screener doesn't write per-ticker reports to disk).

    `args` is a tuple of:
        (ticker, date_str, sector_label, market_cap, adv_usd, cohort_aware,
         max_retries)
    """
    (
        ticker,
        date_str,
        sector_label,
        market_cap,
        adv_usd,
        cohort_aware,
        max_retries,
    ) = args
    result: dict = {
        "ticker": ticker,
        "date": date_str,
        "status": "unknown",
        "elapsed_s": None,
        "error": None,
        "attempts": 0,
        "candidate_dict": None,
    }

    try:
        from tradingagents.analysis_only.pipeline import AnalysisOnlyMVP
        from tradingagents.analysis_only.screener import (
            candidate_to_dict,
            cohort_for_sector,
            extract_candidate_from_report,
            rescore_report_for_cohort,
        )
    except Exception as exc:
        result["status"] = "import_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    started = time.time()
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            mvp = AnalysisOnlyMVP(
                data_provider="polygon",
                options_enabled=False,
                enable_news_fetching=False,
                enable_filings_fetching=False,
                enable_llm_insights=False,
                enable_narrative=False,
                enable_llm_critic=False,
                enable_tradingagents_review=False,
                # Unit 2 perf flags — essential at Nasdaq-Composite scale.
                enable_intraday_context=False,
                enable_peer_competitor_analysis=False,
                state_store_path=None,  # screener doesn't write history
                verbose=False,
                logger=logging.getLogger("screener.worker"),
            )
            report = mvp.run(symbol=ticker, as_of_date=date_str)
            report_dict = report.to_json_dict()
            cohort = (
                cohort_for_sector(sector_label) if cohort_aware else "tech"
            )
            if cohort_aware:
                # Always pass through rescore so the tech-weights composite
                # is recorded on the report dict for the markdown column,
                # even when cohort == "tech" (no-op other than capture).
                rescore_report_for_cohort(report_dict, cohort=cohort)
            # Carry sector tag through via the local sector_map; we still
            # pass `sector_map` containing the single ticker so the helper
            # uses it. Cleaner than mutating report_dict.
            candidate = extract_candidate_from_report(
                report_dict,
                sector_map=({ticker: sector_label} if sector_label else {}),
            )
            # Universe-yaml-sourced enrichments (Unit 4).
            if market_cap is not None and candidate.market_cap is None:
                candidate.market_cap = float(market_cap)
            if adv_usd is not None:
                candidate.adv_usd = float(adv_usd)
            candidate.cohort = cohort
            result["candidate_dict"] = candidate_to_dict(candidate)
            result["status"] = "ok" if attempt == 1 else f"ok_retry_{attempt}"
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(5 * (3 ** (attempt - 1)))

    if last_error is not None:
        result["status"] = "error"
        result["error"] = (
            f"{type(last_error).__name__}: {last_error}\n"
            + traceback.format_exc(limit=4)
        )
    result["elapsed_s"] = round(time.time() - started, 2)
    return result


def _attach_tradingagents_reviews(
    candidates: list,
    *,
    as_of_date: str,
    top_n: int,
    provider: str,
    quick_model: str,
    deep_model: str | None,
    base_url: str | None,
) -> None:
    if top_n <= 0:
        return
    try:
        from tradingagents.analysis_only.agent_review import run_report_agent_review
    except Exception as exc:
        for c in candidates[:top_n]:
            c.tradingagents_review_status = "import_error"
            c.review_summary = f"{type(exc).__name__}: {exc}"
        return

    reviewed = 0
    for c in candidates:
        if reviewed >= top_n:
            break
        if str(c.direction).lower() != "bullish":
            continue
        context = {
            "symbol": c.symbol,
            "as_of_date": as_of_date,
            "direction": c.direction,
            "confidence": c.confidence,
            "composite_score": c.composite_score,
            "top_factors": [
                {
                    "factor": name,
                    "weighted_score": weighted,
                    "rationale": rationale,
                }
                for name, weighted, rationale in c.top_factors
            ],
            "sector": c.sector,
            "pit_status_summary": c.pit_status_summary,
        }
        try:
            block = run_report_agent_review(
                symbol=c.symbol,
                as_of_date=as_of_date,
                report_context=context,
                provider=provider,
                quick_model=quick_model,
                deep_model=deep_model or quick_model,
                base_url=base_url,
            )
            gate = block.get("gate") or {}
            c.tradingagents_review_status = str(block.get("status") or "unknown")
            c.review_risk_veto = bool(gate.get("risk_veto"))
            c.review_summary = str(gate.get("reason") or "")
            c.review_missing_evidence = list(gate.get("missing_evidence") or [])
        except Exception as exc:
            c.tradingagents_review_status = "runtime_error"
            c.review_risk_veto = None
            c.review_summary = f"{type(exc).__name__}: {exc}"
            c.review_missing_evidence = []
        reviewed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe-yaml",
        required=True,
        help="Path to the screener universe yaml (flat ticker list or "
        "structured Unit-4 shape).",
    )
    parser.add_argument(
        "--date",
        default=date_cls.today().isoformat(),
        help="Scan date (ISO). Default: today.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size. Default: 4.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Cap for the top-overall table. Default: 50.",
    )
    parser.add_argument(
        "--top-n-per-sector",
        type=int,
        default=5,
        help="Cap per sector bucket. Default: 5.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/screener",
        help="Output dir (date subdir appended). Default: reports/screener.",
    )
    parser.add_argument(
        "--exclude-core",
        type=lambda v: str(v).lower() not in ("false", "0", "no"),
        default=True,
        help="Filter out tickers already in configs/universe.yaml::core "
        "(default True). Pass --exclude-core=false to disable.",
    )
    parser.add_argument(
        "--core-universe-yaml",
        default="configs/universe.yaml",
        help="Path used to look up the `core:` cohort + `sectors:` map.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Per-job retry attempts (5s/15s backoff). Default: 2.",
    )
    parser.add_argument(
        "--cohort-aware",
        nargs="?",
        const=True,
        type=lambda v: str(v).lower() not in ("false", "0", "no"),
        default=True,
        help="Re-derive composite under cohort-aware weights (non-tech "
        "sectors use universal factors only). Default True. "
        "Pass --cohort-aware=false to disable; pass --cohort-aware (no "
        "value) for the same as default-true.",
    )
    parser.add_argument(
        "--enable-tradingagents-review",
        action="store_true",
        help="Run TradingAgents deep-dive review on top bullish candidates "
        "after cheap screener ranking. Default: disabled.",
    )
    parser.add_argument(
        "--tradingagents-review-top-n",
        type=int,
        default=5,
        help="Number of top bullish candidates to review when enabled. Default: 5.",
    )
    parser.add_argument(
        "--tradingagents-review-provider",
        default="openai",
        help="LLM provider for TradingAgents review. Default: openai.",
    )
    parser.add_argument(
        "--tradingagents-review-quick-model",
        default="gpt-5.4-mini",
        help="Quick model for TradingAgents review. Default: gpt-5.4-mini.",
    )
    parser.add_argument(
        "--tradingagents-review-deep-model",
        default=None,
        help="Deep model for TradingAgents review. Defaults to quick model.",
    )
    parser.add_argument(
        "--tradingagents-review-base-url",
        default=None,
        help="Optional base URL for the review LLM provider.",
    )
    parser.add_argument(
        "--errors-log",
        default="",
        help="Optional JSONL file to append per-job errors to. Defaults to "
        "<output-dir>/<date>/errors.jsonl.",
    )
    args = parser.parse_args()

    universe_path = Path(args.universe_yaml)
    if not universe_path.exists():
        print(f"ERROR: universe yaml not found: {universe_path}", file=sys.stderr)
        return 2

    tickers, universe_meta, _raw = _load_universe_yaml(universe_path)
    universe_size = len(tickers)
    if universe_size == 0:
        print(f"ERROR: no tickers loaded from {universe_path}", file=sys.stderr)
        return 2

    core_path = Path(args.core_universe_yaml)
    # The structured universe yaml is authoritative for sector (it carries
    # the SIC-derived mapping for the broader Nasdaq Composite). Fall back
    # to the curated 26-name configs/universe.yaml::sectors map for any
    # ticker whose universe-yaml entry doesn't carry a sector.
    core_sector_map = _load_sector_map(core_path)
    core_tickers = _load_core_tickers(core_path) if args.exclude_core else set()

    def _resolve_sector(symbol: str) -> str:
        meta = universe_meta.get(symbol) or {}
        sector = meta.get("sector")
        if sector:
            return str(sector)
        return core_sector_map.get(symbol, "")

    pre_exclude_tickers = list(tickers)
    if args.exclude_core:
        tickers = [t for t in tickers if t not in core_tickers]
    excluded_count = len(pre_exclude_tickers) - len(tickers)

    out_dir = Path(args.output_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_log_path = Path(args.errors_log) if args.errors_log else (
        out_dir / "errors.jsonl"
    )

    print(f"Universe: {universe_size} tickers from {universe_path}")
    print(
        f"After --exclude-core (excluded {excluded_count}): "
        f"{len(tickers)} tickers"
    )
    print(f"Scan date: {args.date}  workers: {args.workers}  output: {out_dir}")
    print(f"Errors log: {errors_log_path}\n")

    started_all = time.time()
    jobs = [
        (
            t,
            args.date,
            _resolve_sector(t),
            (universe_meta.get(t) or {}).get("market_cap_usd"),
            (universe_meta.get(t) or {}).get("adv_usd"),
            args.cohort_aware,
            args.max_retries,
        )
        for t in tickers
    ]

    results: list[dict] = []
    completed = 0
    errors = 0
    print(f"{'#':>4} {'ticker':<8} {'status':<14} {'elapsed':>8} {'composite':>10}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(_screen_one, job): job for job in jobs}
        for fut in as_completed(future_map):
            result = fut.result()
            completed += 1
            elapsed_s = result.get("elapsed_s") or 0.0
            comp = (
                result.get("candidate_dict", {}) or {}
            ).get("composite_score") if result.get("candidate_dict") else None
            comp_str = f"{comp:+.3f}" if isinstance(comp, (int, float)) else "—"
            print(
                f"{completed:>4} {result['ticker']:<8} {result['status']:<14} "
                f"{elapsed_s:>7.1f}s {comp_str:>10}"
            )
            if result["status"].startswith("ok"):
                results.append(result)
            else:
                errors += 1
                with errors_log_path.open("a") as fh:
                    fh.write(json.dumps({k: v for k, v in result.items()
                                          if k != "candidate_dict"}) + "\n")

    scan_elapsed_s = time.time() - started_all

    # Build candidate list. Lazy-import here so the CLI is robust to
    # missing optional deps when --help is invoked.
    from tradingagents.analysis_only.screener import (
        ScreenerCandidate,
        candidate_to_dict,
        rank_candidates,
        rank_per_sector,
        render_screener_markdown,
    )

    candidates: list[ScreenerCandidate] = []
    for r in results:
        cd = r.get("candidate_dict") or {}
        if not cd:
            continue
        tech_wts_raw = cd.get("composite_score_tech_weights")
        candidates.append(
            ScreenerCandidate(
                symbol=str(cd.get("symbol") or "").upper(),
                composite_score=float(cd.get("composite_score") or 0.0),
                direction=str(cd.get("direction") or "neutral"),
                confidence=float(cd.get("confidence") or 0.0),
                top_factors=[
                    (
                        str(tf.get("factor") or ""),
                        float(tf.get("weighted_score") or 0.0),
                        str(tf.get("rationale") or ""),
                    )
                    for tf in (cd.get("top_factors") or [])
                ],
                sector=str(cd.get("sector") or "Unknown"),
                market_cap=cd.get("market_cap"),
                adv_usd=cd.get("adv_usd"),
                next_earnings_in_calendar_days=cd.get(
                    "next_earnings_in_calendar_days"
                ),
                pit_status_summary=str(cd.get("pit_status_summary") or ""),
                composite_score_tech_weights=(
                    float(tech_wts_raw) if tech_wts_raw is not None else None
                ),
                cohort=cd.get("cohort"),
                tradingagents_review_status=cd.get("tradingagents_review_status"),
                review_risk_veto=cd.get("review_risk_veto"),
                review_summary=cd.get("review_summary"),
                review_missing_evidence=list(cd.get("review_missing_evidence") or []),
            )
        )

    top_overall = rank_candidates(candidates, top_n=args.top_n)
    if args.enable_tradingagents_review:
        _attach_tradingagents_reviews(
            top_overall,
            as_of_date=args.date,
            top_n=args.tradingagents_review_top_n,
            provider=args.tradingagents_review_provider,
            quick_model=args.tradingagents_review_quick_model,
            deep_model=args.tradingagents_review_deep_model,
            base_url=args.tradingagents_review_base_url,
        )
    per_sector = rank_per_sector(candidates, top_n_per_sector=args.top_n_per_sector)

    ranked_json_payload = {
        "as_of_date": args.date,
        "universe_size": universe_size,
        "candidates_evaluated": len(candidates),
        "excluded_core_count": excluded_count,
        "errors": errors,
        "scan_elapsed_s": round(scan_elapsed_s, 2),
        "workers": args.workers,
        "top_n": args.top_n,
        "top_n_per_sector": args.top_n_per_sector,
        "cohort_aware": bool(args.cohort_aware),
        "tradingagents_review_enabled": bool(args.enable_tradingagents_review),
        "tradingagents_review_top_n": args.tradingagents_review_top_n,
        "top_overall": [candidate_to_dict(c) for c in top_overall],
        "per_sector": {
            sector: [candidate_to_dict(c) for c in picks]
            for sector, picks in per_sector.items()
        },
    }
    json_path = out_dir / "ranked.json"
    json_path.write_text(json.dumps(ranked_json_payload, indent=2) + "\n")

    md = render_screener_markdown(
        top_overall=top_overall,
        per_sector_top=per_sector,
        as_of_date=args.date,
        universe_size=universe_size,
        scan_elapsed_s=scan_elapsed_s,
        candidates_evaluated=len(candidates),
        excluded_core_count=excluded_count,
        top_n=args.top_n,
        top_n_per_sector=args.top_n_per_sector,
        cohort_aware=bool(args.cohort_aware),
    )
    md_path = out_dir / "ranked.md"
    md_path.write_text(md)

    print(
        f"\nDone. evaluated={len(candidates)}  errors={errors}  "
        f"elapsed={int(scan_elapsed_s // 60)}m{int(scan_elapsed_s % 60):02d}s"
    )
    print(f"Outputs:\n  {json_path}\n  {md_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
