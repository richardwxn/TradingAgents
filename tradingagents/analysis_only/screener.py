"""Pure helpers for the Nasdaq screener (Phase B1).

Phase B1 (Unit 3) emits a ranked top-N + per-sector breakdown for the
hardcoded Nasdaq 100 universe. The actual `AnalysisOnlyMVP` instantiation +
threadpool live in `scripts/screener.py`; this module is import-pure (no
yfinance / Polygon side effects) so it can be unit-tested in isolation.

The screener consumes the JSON-serializable report dict (the shape produced
by `AnalysisOnlyMVP.run(...).to_json_dict()`) and turns each into a compact
`ScreenerCandidate`. Ranking + markdown rendering operate on these
dataclasses, never on the full report shape.

Unit 4 (Phase B2) extends this module with cohort-aware scoring and reads
sector/market_cap/adv from a richer universe yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Sector buckets we recognize (from configs/universe.yaml::sectors). Tickers
# not in that map bucket under UNKNOWN_SECTOR so the per-sector view still
# surfaces them — they're a fact-of-life for the Nasdaq 100 superset of
# the curated 26-name core universe.
UNKNOWN_SECTOR = "Unknown"


# Sector → cohort mapping for cohort-aware scoring (Unit 4 / Phase B2).
#
# Rationale:
#   - The handoff Section 22 cohort IC analysis split factor IC by tech
#     (24-name core) vs cross-sector (6-name canary). Several factors
#     have sign-disagreeing IC across the split — those weights are
#     calibrated for tech-momentum behavior and should NOT be applied
#     when scoring a defensive / cyclical / financial name.
#   - This map decides which factor set a sector gets scored against.
#     "tech" → full `DEFAULT_FACTOR_WEIGHTS`. "non_tech" → universal-only
#     (see `tradingagents.analysis_only.scoring.UNIVERSAL_FACTOR_NAMES`).
#
# Boundary call (subjective; documented for review):
#   - Aerospace + Financial-Data + Specialty-Materials are tagged "tech"
#     here because the 26-name core universe treats them as AI-infra
#     adjacent (RKLB satellites, FIG data, GLW specialty glass for AI
#     datacenter). A more conservative pass could bucket them non_tech.
#   - Energy-Nuclear is tagged "non_tech": even though LEU is in the
#     tech-infra-adjacent narrative, the price action / IC profile is
#     defensives-like. We bucket as non_tech to be safe.
#   - "Other" is non_tech by default — the conservative default for
#     anything our SIC→sector mapper didn't classify.
_TECH_SECTORS: frozenset[str] = frozenset({
    "Semiconductors",
    "Tech-MegaCap",
    "Software",
    "Networking",
    "Photonics",
    "Specialty-Materials",
    "Aerospace",
    "Financial-Data",
})
_NON_TECH_SECTORS: frozenset[str] = frozenset({
    "Financials",
    "Energy",
    "Healthcare",
    "Consumer-Staples",
    "Utilities",
    "Consumer-Discretionary",
    "Energy-Nuclear",
    "Other",
})


def cohort_for_sector(sector: str | None) -> str:
    """Map a sector label → "tech" or "non_tech" for cohort-aware scoring.

    Unknown / unmapped sectors default to "non_tech" (conservative — the
    universal factor set is a strictly-smaller-and-safer signal set so
    falling back to it is the right side of a wrong-default tradeoff).
    """
    s = (sector or "").strip()
    if s in _TECH_SECTORS:
        return "tech"
    return "non_tech"


@dataclass
class ScreenerCandidate:
    """Compact per-ticker result emitted by the screener.

    `top_factors` is a list of `(name, weighted_score, rationale)` triples
    ranked by absolute weighted contribution. The screener captures the
    top 3 so the markdown report can show *why* a name made the cut at a
    glance without forcing the reader to open the full JSON.
    """

    symbol: str
    composite_score: float
    direction: str
    confidence: float
    top_factors: list[tuple[str, float, str]] = field(default_factory=list)
    sector: str = UNKNOWN_SECTOR
    market_cap: float | None = None
    adv_usd: float | None = None
    next_earnings_in_calendar_days: int | None = None
    pit_status_summary: str = ""
    # Unit 4: when `--cohort-aware` rewrites `composite_score` for a
    # non_tech name, the original tech-weights composite is preserved
    # here so the markdown render can show both columns. For tech
    # names (or when cohort-aware is disabled) this equals
    # `composite_score`. None when not populated.
    composite_score_tech_weights: float | None = None
    cohort: str | None = None
    tradingagents_review_status: str | None = None
    review_risk_veto: bool | None = None
    review_summary: str | None = None
    review_missing_evidence: list[str] = field(default_factory=list)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort cast that swallows None / strings / NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _next_earnings_in_calendar_days(
    company_events: dict[str, Any] | None, as_of_date: str
) -> int | None:
    """Calendar-day delta from `as_of_date` to next_earnings_date.

    Returns None when either side is missing or unparseable. Negative deltas
    (earnings already announced) are returned as-is so the screener can flag
    fresh-print stale-earnings situations downstream.
    """
    if not company_events:
        return None
    next_date = company_events.get("next_earnings_date")
    if not next_date:
        return None
    try:
        as_of = datetime.fromisoformat(str(as_of_date)).date()
        ed = datetime.fromisoformat(str(next_date)).date()
    except (ValueError, TypeError):
        return None
    return (ed - as_of).days


def _summarize_pit_status(pit_status: dict[str, Any] | None) -> str:
    """One-line summary of any non-PIT sections in the report.

    `pit_status` is a flat dict of section -> status emitted by
    `AnalysisOnlyMVP`. We return a short comma-separated list of any
    `non_pit_*` sections so the screener markdown can warn the reader when
    a name's ranking relies on live snapshots vs PIT data.
    """
    if not pit_status:
        return ""
    bad = sorted(
        section
        for section, status in pit_status.items()
        if isinstance(status, str) and status.startswith("non_pit_")
    )
    if not bad:
        return "pit"
    # Compact: drop common prefixes for readability.
    short = [s.split(".")[-1] for s in bad[:3]]
    if len(bad) > 3:
        return f"non-pit: {', '.join(short)} +{len(bad) - 3} more"
    return f"non-pit: {', '.join(short)}"


def rescore_report_for_cohort(
    report: dict[str, Any],
    *,
    cohort: str,
) -> dict[str, Any]:
    """Re-derive `composite_score` / `direction` / `confidence` under cohort weights.

    Mutates the report dict in-place:
    - Preserves the original composite as `key_features.model_scoring.
      composite_score_tech_weights`.
    - Recomputes `composite_score` + `direction` + `confidence` using
      `resolve_factor_weights(cohort=cohort)` + `compute_composite(...)`
      + `direction_for_composite(...)` + `confidence_for(...)`.
    - Also overwrites each `factor_scores[i].weighted_score` so the
      "top factors" extraction in `extract_candidate_from_report`
      reflects what actually drove the cohort composite.

    Returns the same dict for chaining. No-op (other than recording
    tech-weights composite) when cohort == "tech" or factor_scores is
    empty.

    NOTE: This re-runs the composite arithmetic only — none of the
    underlying factor evidence is recomputed. Each factor's raw
    `score` is unchanged; only its `weight` (and therefore
    `weighted_score`) is reapplied under the cohort weight vector.
    """
    # Local imports to keep the screener module's import surface light.
    from tradingagents.analysis_only.scoring import (
        compute_composite,
        confidence_for,
        direction_for_composite,
        resolve_factor_weights,
    )

    model_scoring = (report.setdefault("key_features", {})
                     .setdefault("model_scoring", {}))
    factor_scores = model_scoring.get("factor_scores") or []
    original_composite = _coerce_float(model_scoring.get("composite_score"), 0.0)

    # Always preserve the original composite for the markdown
    # `Composite (tech wts)` column even when cohort is tech (so the
    # column always has a value).
    model_scoring["composite_score_tech_weights"] = round(original_composite, 4)

    if cohort == "tech" or not factor_scores:
        return report

    cohort_weights = resolve_factor_weights(cohort=cohort)
    # Reapply weights per-factor. Factors not in the weight table get
    # weight 0 (which mirrors how `DEFAULT_FACTOR_WEIGHTS` treats
    # unknown names today).
    for fs in factor_scores:
        name = str(fs.get("factor") or "")
        new_w = float(cohort_weights.get(name, 0.0))
        fs["weight"] = round(new_w, 6)
        raw_score = fs.get("score")
        if fs.get("data_available") and raw_score is not None:
            try:
                fs["weighted_score"] = round(float(raw_score) * new_w, 6)
            except (TypeError, ValueError):
                fs["weighted_score"] = 0.0
        else:
            fs["weighted_score"] = 0.0

    composite_payload = compute_composite(factor_scores, cohort_weights)
    new_composite = composite_payload["composite_score"]
    coverage = composite_payload.get("coverage", 0.0)

    model_scoring["composite_score"] = new_composite
    model_scoring["pillar_scores"] = composite_payload.get("pillar_scores", {})
    model_scoring["coverage"] = coverage
    model_scoring["active_weight"] = composite_payload.get("active_weight", 0.0)
    model_scoring["total_weight"] = composite_payload.get("total_weight", 0.0)

    report["direction"] = direction_for_composite(new_composite)
    report["confidence"] = confidence_for(new_composite, coverage)
    return report


def extract_candidate_from_report(
    report: dict[str, Any],
    *,
    sector_map: dict[str, str] | None = None,
) -> ScreenerCandidate:
    """Parse an `AnalysisOnlyMVP.to_json_dict()` dict → `ScreenerCandidate`.

    `sector_map` maps SYMBOL -> sector_label (typically loaded from
    `configs/universe.yaml::sectors`). When the ticker isn't in the map
    we bucket as `UNKNOWN_SECTOR`. Phase B1 falls back here for the
    Nasdaq-100 superset; Unit 4 will source sector from a structured
    universe yaml directly.
    """
    sector_map = sector_map or {}
    symbol = str(report.get("symbol") or "").upper()
    model_scoring = report.get("key_features", {}).get("model_scoring", {}) or {}
    composite = _coerce_float(model_scoring.get("composite_score"), default=0.0)
    # `composite_score_tech_weights` is populated by `rescore_report_for_cohort`
    # (Unit 4). When absent, fall back to the composite itself so the
    # markdown column always renders.
    tech_weights_composite_raw = model_scoring.get("composite_score_tech_weights")
    tech_weights_composite: float | None = (
        _coerce_float(tech_weights_composite_raw, default=composite)
        if tech_weights_composite_raw is not None
        else None
    )
    direction = str(report.get("direction") or "neutral")
    confidence = _coerce_float(report.get("confidence"), default=0.0)

    factor_scores = (
        (report.get("key_features", {}).get("model_scoring", {}) or {})
        .get("factor_scores")
        or []
    )
    # Rank factors by |weighted_score|. Ties broken by raw weighted score
    # (more positive first) so visually the top three feel coherent with the
    # direction.
    ranked = sorted(
        (
            (
                str(fs.get("factor") or ""),
                _coerce_float(fs.get("weighted_score"), 0.0),
                str(fs.get("rationale") or ""),
            )
            for fs in factor_scores
            if fs.get("factor")
        ),
        key=lambda triple: (-abs(triple[1]), -triple[1]),
    )
    top_factors = [t for t in ranked[:3] if abs(t[1]) > 1e-6]

    sector = sector_map.get(symbol, UNKNOWN_SECTOR)
    fundamentals = report.get("key_features", {}).get("fundamental", {}) or {}
    market_cap = _coerce_float(fundamentals.get("market_cap"), default=0.0) or None

    company_events = report.get("key_features", {}).get("company_events", {}) or {}
    next_e_days = _next_earnings_in_calendar_days(
        company_events, str(report.get("as_of_date") or "")
    )

    pit_status = report.get("key_features", {}).get("pit_status", {}) or {}
    pit_summary = _summarize_pit_status(pit_status)

    return ScreenerCandidate(
        symbol=symbol,
        composite_score=composite,
        direction=direction,
        confidence=confidence,
        top_factors=top_factors,
        sector=sector,
        market_cap=market_cap,
        adv_usd=None,  # Unit 4 will source from universe yaml.
        next_earnings_in_calendar_days=next_e_days,
        pit_status_summary=pit_summary,
        composite_score_tech_weights=tech_weights_composite,
        tradingagents_review_status=(
            ((report.get("key_features") or {}).get("tradingagents_review") or {})
            .get("status")
        ),
        review_risk_veto=(
            (((report.get("key_features") or {}).get("tradingagents_review") or {})
             .get("gate") or {})
            .get("risk_veto")
        ),
        review_summary=(
            (((report.get("key_features") or {}).get("tradingagents_review") or {})
             .get("gate") or {})
            .get("reason")
        ),
        review_missing_evidence=list(
            ((((report.get("key_features") or {}).get("tradingagents_review") or {})
              .get("gate") or {})
             .get("missing_evidence") or [])
        ),
    )


def _effective_rank_score(c: ScreenerCandidate) -> float:
    """Return the composite to rank by.

    In cohort-aware mode (`--cohort-aware`), `composite_score` for non-tech
    sectors is recomputed under the narrow universal-factor weight set,
    which saturates at the trend-factor cap (e.g. +0.832 on every bullish
    non-tech name — Section 27 finding). That makes the cohort-aware
    composite useless for ranking; ~25 names tie at the cap.

    `composite_score_tech_weights` preserves the original full-weight v1.4
    composite even in cohort-aware mode, with real spread between names.
    So when it's populated, prefer it as the rank key. When it's None
    (non-cohort-aware run), fall back to `composite_score`.

    See plan Future-work #9. Original cohort-aware composite remains in
    `composite_score` for display.
    """
    if c.composite_score_tech_weights is not None:
        return c.composite_score_tech_weights
    return c.composite_score


def rank_candidates(
    candidates: list[ScreenerCandidate], *, top_n: int = 50
) -> list[ScreenerCandidate]:
    """Sort by effective composite desc, cap at `top_n`.

    Uses `_effective_rank_score` to pick tech-weights composite in
    cohort-aware mode (avoids the universal-factor cap saturating the
    top of the list), falling back to `composite_score` otherwise.

    Note: we sort by absolute composite NOT (so a strong-bearish name with
    composite=-0.6 doesn't outrank a strong-bullish name with +0.3) — the
    screener's job is to surface high-conviction *bullish* candidates for
    swing entry. Bears would need a different sort, deferred.

    Ties broken by confidence desc, then symbol alpha for determinism.
    """
    sorted_c = sorted(
        candidates,
        key=lambda c: (-_effective_rank_score(c), -c.confidence, c.symbol),
    )
    return sorted_c[:top_n]


def rank_per_sector(
    candidates: list[ScreenerCandidate],
    *,
    top_n_per_sector: int = 5,
) -> dict[str, list[ScreenerCandidate]]:
    """Group candidates by sector, return top-N per bucket.

    Returns `dict[sector_label, list[ScreenerCandidate]]`. The "Unknown"
    bucket is populated for tickers not in the sector map — these are
    the bulk of the Nasdaq 100 names outside the 26-name core universe.
    Unit 4 will solve the broader sector-mapping problem properly.

    Output keys are sorted: known sectors alpha, then "Unknown" last.
    """
    buckets: dict[str, list[ScreenerCandidate]] = {}
    for c in candidates:
        buckets.setdefault(c.sector or UNKNOWN_SECTOR, []).append(c)
    out: dict[str, list[ScreenerCandidate]] = {}
    known = sorted(s for s in buckets if s != UNKNOWN_SECTOR)
    ordered = known + ([UNKNOWN_SECTOR] if UNKNOWN_SECTOR in buckets else [])
    for s in ordered:
        ranked = sorted(
            buckets[s],
            key=lambda c: (-_effective_rank_score(c), -c.confidence, c.symbol),
        )
        out[s] = ranked[:top_n_per_sector]
    return out


def _fmt_factor(triple: tuple[str, float, str]) -> str:
    """Render `(name, weighted, rationale)` as a short markdown cell."""
    name, weighted, rationale = triple
    sign = "+" if weighted >= 0 else ""
    return f"{name} ({sign}{weighted:.3f})"


def _fmt_market_cap(mc: float | None) -> str:
    if mc is None or mc <= 0:
        return "—"
    if mc >= 1e12:
        return f"${mc / 1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc / 1e9:.1f}B"
    if mc >= 1e6:
        return f"${mc / 1e6:.0f}M"
    return f"${mc:,.0f}"


def _fmt_earnings(days: int | None) -> str:
    if days is None:
        return "—"
    if days < 0:
        return f"{days}d (past)"
    if days == 0:
        return "today"
    return f"{days}d"


def _fmt_tech_wt_composite(c: ScreenerCandidate) -> str:
    """Render the tech-weights composite as a signed 3dp string, or em-dash."""
    if c.composite_score_tech_weights is None:
        return "—"
    return f"{c.composite_score_tech_weights:+.3f}"


def _fmt_review(c: ScreenerCandidate) -> str:
    if not c.tradingagents_review_status:
        return "—"
    veto = " veto" if c.review_risk_veto else ""
    summary = (c.review_summary or "").strip()
    if summary:
        return f"{c.tradingagents_review_status}{veto}: {summary}"
    return f"{c.tradingagents_review_status}{veto}"


def render_screener_markdown(
    top_overall: list[ScreenerCandidate],
    per_sector_top: dict[str, list[ScreenerCandidate]],
    *,
    as_of_date: str,
    universe_size: int,
    scan_elapsed_s: float,
    candidates_evaluated: int | None = None,
    excluded_core_count: int | None = None,
    top_n: int | None = None,
    top_n_per_sector: int | None = None,
    cohort_aware: bool | None = None,
) -> str:
    """Render the full screener markdown.

    Includes:
    - Scan-stats header (universe size, elapsed, top-N filter, exclude-core,
      cohort-aware caveat).
    - Top-N overall table with cohort-aware `Composite` + tech-weights
      `Composite (tech wts)` columns side-by-side (Unit 4) plus
      direction + confidence + sector + market cap + top-3 factor names.
    - Per-sector top-K tables.
    """
    lines: list[str] = []
    lines.append(f"# Screener — {as_of_date}")
    lines.append("")
    lines.append("## Scan stats")
    lines.append("")
    lines.append(f"- **Universe size:** {universe_size} tickers")
    if candidates_evaluated is not None:
        lines.append(
            f"- **Candidates evaluated (post-filters):** {candidates_evaluated}"
        )
    if excluded_core_count is not None:
        lines.append(f"- **Excluded as already-in-core:** {excluded_core_count}")
    if top_n is not None:
        lines.append(f"- **Top-N overall:** {top_n}")
    if top_n_per_sector is not None:
        lines.append(f"- **Top-N per sector:** {top_n_per_sector}")
    if cohort_aware is not None:
        lines.append(
            f"- **Cohort-aware scoring:** "
            f"{'enabled' if cohort_aware else 'disabled'}"
        )
    lines.append(f"- **Scan elapsed:** {scan_elapsed_s:.1f}s")
    lines.append("")
    lines.append(
        "> **Caveat — screener composites are NOT directly comparable to "
        "`analysis_mvp.py` composites.** The screener disables heavyweight "
        "pipeline phases (news, filings, intraday, peers, options, LLM) "
        "for throughput. `compute_composite` renormalizes against active "
        "weights, so disabling factors with net-negative weighted "
        "contribution amplifies the remaining factors (~1.2x for the "
        "current weight vector). Use the **ranking**, not the absolute "
        "composite, for cross-reference with the weekly reports."
    )
    if cohort_aware:
        lines.append("")
        lines.append(
            "> **Cohort-aware scoring (`--cohort-aware`):** non-tech "
            "sectors are scored against the universal-factor set only "
            "(handoff Section 22). The `Composite (tech wts)` column "
            "preserves the original tech-weights composite for reference; "
            "for tech-cohort names the two columns are equal."
        )
    lines.append("")

    lines.append(f"## Top {len(top_overall)} overall (by composite)")
    lines.append("")
    if not top_overall:
        lines.append("_No candidates produced — check error log._")
        lines.append("")
    else:
        lines.append(
            "| # | Symbol | Composite | Composite (tech wts) | Direction | "
            "Confidence | Sector | Market cap | Next earnings | Top factors | "
            "PIT | TradingAgents review |"
        )
        lines.append(
            "|---|---|---:|---:|---|---:|---|---:|---|---|---|---|"
        )
        for i, c in enumerate(top_overall, start=1):
            tf = " · ".join(_fmt_factor(t) for t in c.top_factors) or "—"
            lines.append(
                f"| {i} | **{c.symbol}** | {c.composite_score:+.3f} | "
                f"{_fmt_tech_wt_composite(c)} | "
                f"{c.direction} | {c.confidence:.2f} | "
                f"{c.sector} | {_fmt_market_cap(c.market_cap)} | "
                f"{_fmt_earnings(c.next_earnings_in_calendar_days)} | "
                f"{tf} | {c.pit_status_summary} |"
                f" {_fmt_review(c)} |"
            )
        lines.append("")

    lines.append("## Per-sector top picks")
    lines.append("")
    if not per_sector_top:
        lines.append("_No sector buckets populated._")
        lines.append("")
    else:
        for sector, picks in per_sector_top.items():
            lines.append(f"### {sector}")
            lines.append("")
            if not picks:
                lines.append("_No candidates in this sector._")
                lines.append("")
                continue
            lines.append(
                "| # | Symbol | Composite | Composite (tech wts) | "
                "Direction | Confidence | Top factors | TradingAgents review |"
            )
            lines.append("|---|---|---:|---:|---|---:|---|---|")
            for i, c in enumerate(picks, start=1):
                tf = " · ".join(_fmt_factor(t) for t in c.top_factors) or "—"
                lines.append(
                    f"| {i} | **{c.symbol}** | {c.composite_score:+.3f} | "
                    f"{_fmt_tech_wt_composite(c)} | "
                    f"{c.direction} | {c.confidence:.2f} | {tf} | "
                    f"{_fmt_review(c)} |"
                )
            lines.append("")
    return "\n".join(lines)


def candidate_to_dict(c: ScreenerCandidate) -> dict[str, Any]:
    """JSON-serializable form for `ranked.json`."""
    return {
        "symbol": c.symbol,
        "composite_score": round(c.composite_score, 4),
        "composite_score_tech_weights": (
            round(c.composite_score_tech_weights, 4)
            if c.composite_score_tech_weights is not None
            else None
        ),
        "cohort": c.cohort,
        "direction": c.direction,
        "confidence": round(c.confidence, 4),
        "top_factors": [
            {
                "factor": name,
                "weighted_score": round(weighted, 4),
                "rationale": rationale,
            }
            for name, weighted, rationale in c.top_factors
        ],
        "sector": c.sector,
        "market_cap": c.market_cap,
        "adv_usd": c.adv_usd,
        "next_earnings_in_calendar_days": c.next_earnings_in_calendar_days,
        "pit_status_summary": c.pit_status_summary,
        "tradingagents_review_status": c.tradingagents_review_status,
        "review_risk_veto": c.review_risk_veto,
        "review_summary": c.review_summary,
        "review_missing_evidence": list(c.review_missing_evidence),
    }
