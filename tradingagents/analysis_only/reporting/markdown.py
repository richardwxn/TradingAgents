"""Render an analysis-only JSON report into Markdown.

The renderer is pure (input dict -> string). Charts are deferred; this is the
text-only baseline so reports are humanly readable without spelunking the JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json


def render_markdown_file(
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render `json_path` to a sibling `.md` (or `output_path`) and return it."""
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text())
    md = render_markdown(payload)
    target = Path(output_path) if output_path else json_path.with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md)
    return target


def render_markdown(payload: dict[str, Any]) -> str:
    sections = [
        _render_header(payload),
        _render_topline(payload),
        _render_decision_summary(payload),
        _render_delta(payload),
        _render_cases(payload),
        _render_price_levels(payload),
        _render_forecasts(payload),
        _render_price_target(payload),
        _render_scorecard(payload),
        _render_pillars(payload),
        _render_fundamentals(payload),
        _render_options(payload),
        _render_options_iv(payload),
        _render_option_strategies(payload),
        _render_earnings_calendar(payload),
        _render_analyst_consensus(payload),
        _render_market_industry(payload),
        _render_industry_news(payload),
        _render_competitors(payload),
        _render_events(payload),
        _render_risks(payload),
        _render_llm_insights(payload),
        _render_tradingagents_review(payload),
        _render_data_quality(payload),
    ]
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# ---------- section renderers ----------


def _render_header(p: dict[str, Any]) -> str:
    symbol = p.get("symbol", "?")
    as_of = p.get("as_of_date", "?")
    horizon = (p.get("horizon") or "").replace("_", " ")
    generated = p.get("generated_at_utc", "?")
    lines = [
        f"# {symbol} — Analysis Report",
        "",
        f"- **As of:** {as_of}",
        f"- **Horizon:** {horizon}",
        f"- **Generated (UTC):** {generated}",
    ]
    return "\n".join(lines)


def _render_topline(p: dict[str, Any]) -> str:
    kf = p.get("key_features") or {}
    tech = kf.get("technical") or {}
    scoring = kf.get("model_scoring") or {}
    composite = scoring.get("composite_score")
    direction = (p.get("direction") or "neutral").lower()
    confidence = p.get("confidence")
    close = tech.get("close")
    lines = [
        "## Top line",
        "",
        f"- **Direction:** {direction}",
        f"- **Confidence:** {_fmt_num(confidence)}",
        f"- **Composite score:** {_fmt_num(composite)}",
        f"- **Last close:** {_fmt_price(close)}",
    ]
    thesis = p.get("thesis")
    if thesis:
        source = ((p.get("key_features") or {}).get("narrative_source")
                  or "templated")
        suffix = "" if source == "templated" else f" _(source: {source})_"
        lines += ["", f"**Thesis.**{suffix} " + str(thesis).strip()]
    return "\n".join(lines)


def _render_decision_summary(p: dict[str, Any]) -> str:
    decision = (p.get("key_features") or {}).get("decision_summary") or {}
    if not decision or decision.get("status") != "ok":
        return ""
    entry = decision.get("entry") or {}
    exit_plan = decision.get("exit") or {}
    zone = entry.get("preferred_buy_zone") or {}
    lines = [
        "## Simple action",
        "",
        f"- **Action:** {decision.get('label', decision.get('action', 'watch'))}",
        f"- **Current price:** {_fmt_price(decision.get('current_price'))}",
        f"- **Estimated win probability:** {_fmt_pct(decision.get('estimated_win_probability'))}",
        f"- **Decision confidence:** {_fmt_pct(decision.get('confidence'))}",
        f"- **Base target / upside:** {_fmt_price(decision.get('base_target'))} "
        f"({_fmt_pct(decision.get('base_upside_pct'))})",
        "",
        f"**Plain read:** {decision.get('summary', '')}",
        "",
        "| Level | Price |",
        "|---|---|",
        f"| Starter buy at/below | {_fmt_price(entry.get('starter_buy_at_or_below'))} |",
        f"| Preferred buy zone | {_band(zone.get('low'), zone.get('high'))} |",
        f"| Add below | {_fmt_price(entry.get('add_below'))} |",
        f"| Take profit 1 | {_fmt_price(exit_plan.get('take_profit_1'))} |",
        f"| Take profit 2 | {_fmt_price(exit_plan.get('take_profit_2'))} |",
        f"| Stop / invalidate | {_fmt_price(exit_plan.get('stop_loss'))} |",
    ]
    rationale = decision.get("rationale") or []
    if rationale:
        lines += ["", "**Why**", ""]
        lines.extend(f"- {_escape_pipes(str(item))}" for item in rationale)
    caveats = decision.get("caveats") or []
    if caveats:
        lines += ["", "**Caveats**", ""]
        lines.extend(f"- {_escape_pipes(str(item))}" for item in caveats)
    return "\n".join(lines)


def _render_delta(p: dict[str, Any]) -> str:
    delta = ((p.get("key_features") or {}).get("delta_since_last_report")
             or {})
    status = delta.get("status")
    if status == "state_store_disabled":
        return ""
    if status == "first_report":
        return "## Delta since last report\n\n- No prior run on record for this symbol."
    if status != "ok":
        return ""
    lines = ["## Delta since last report"]
    lines.append("")
    lines.append(f"- **Previous run:** {delta.get('previous_run_at_utc')}")
    lines.append(
        f"- **Previous as-of:** {delta.get('previous_as_of_date')} "
        f"(Δ {_fmt_int(delta.get('as_of_date_delta_days'))} days)"
    )
    if delta.get("direction_changed"):
        lines.append(
            f"- **Direction changed:** {delta.get('direction_transition')}"
        )
    else:
        lines.append(
            f"- **Direction:** unchanged ({delta.get('previous_direction')})"
        )
    lines.append(
        f"- **Composite score Δ:** {_fmt_signed(delta.get('composite_score_delta'))} "
        f"(from {_fmt_signed(delta.get('previous_composite_score'))})"
    )
    lines.append(
        f"- **Confidence Δ:** {_fmt_signed(delta.get('confidence_delta'))} "
        f"(from {_fmt_num(delta.get('previous_confidence'))})"
    )
    lines.append(
        f"- **Close Δ:** {_fmt_signed(delta.get('close_delta'))} "
        f"({_fmt_pct(delta.get('close_pct_change'))} from "
        f"{_fmt_price(delta.get('previous_close'))})"
    )
    lines.append(
        "- **Options unusual count Δ:** "
        f"{_fmt_signed(delta.get('options_unusual_count_delta'), digits=0)}"
    )
    flips = delta.get("factor_bucket_flips") or []
    if flips:
        lines += [
            "",
            "**Factors that flipped bucket**",
            "",
            "| Factor | From → To | Rationale |",
            "|---|---|---|",
        ]
        for f in flips:
            lines.append(
                "| {factor} | {f} → {t} | {r} |".format(
                    factor=f.get("factor", "?"),
                    f=f.get("from_bucket", "?"),
                    t=f.get("to_bucket", "?"),
                    r=_escape_pipes(str(f.get("rationale", ""))),
                )
            )
    movers = delta.get("top_factor_movers") or []
    if movers:
        lines += [
            "",
            "**Top factor movers (by Δ weighted score)**",
            "",
            "| Factor | Pillar | Δ weighted | From → To |",
            "|---|---|---|---|",
        ]
        for m in movers:
            lines.append(
                "| {factor} | {p} | {d} | {f} → {t} |".format(
                    factor=m.get("factor", "?"),
                    p=m.get("pillar", "?"),
                    d=_fmt_signed(m.get("weighted_score_delta")),
                    f=_fmt_signed(m.get("from")),
                    t=_fmt_signed(m.get("to")),
                )
            )
    return "\n".join(lines)


def _render_cases(p: dict[str, Any]) -> str:
    bull = p.get("bull_case") or []
    bear = p.get("bear_case") or []
    if not bull and not bear:
        return ""
    lines = ["## Bull / Bear case"]
    if bull:
        lines.append("\n**Bull case**\n")
        lines.extend(f"- {_clean_case_string(b)}" for b in bull)
    if bear:
        lines.append("\n**Bear case**\n")
        lines.extend(f"- {_clean_case_string(b)}" for b in bear)
    return "\n".join(lines)


def _render_price_levels(p: dict[str, Any]) -> str:
    kf = p.get("key_features") or {}
    tech = kf.get("technical") or {}
    forecast = (kf.get("price_range_forecast") or {}).get("1w") or {}
    rows = [
        ("Last close", tech.get("close"), "price"),
        ("SMA 20", tech.get("sma_20"), "price"),
        ("SMA 50", tech.get("sma_50"), "price"),
        ("SMA 200", tech.get("sma_200"), "price"),
        ("RSI(14)", tech.get("rsi_14"), "num"),
        ("MACD hist", tech.get("macd_hist"), "num"),
        ("ATR(14)", tech.get("atr_14"), "price"),
        ("20d realized vol", tech.get("volatility_20d"), "pct"),
        ("1w forecast center", forecast.get("center_price"), "price"),
        ("1w 60% band", _band(forecast.get("lower_60"), forecast.get("upper_60")), "raw"),
        ("1w 80% band", _band(forecast.get("lower_80"), forecast.get("upper_80")), "raw"),
    ]
    has_any = any(_is_present(v) for _, v, _ in rows)
    if not has_any:
        return ""
    lines = ["## Key prices & levels", "", "| Metric | Value |", "|---|---|"]
    for label, value, kind in rows:
        if not _is_present(value):
            continue
        lines.append(f"| {label} | {_fmt_by_kind(value, kind)} |")
    return "\n".join(lines)


def _render_forecasts(p: dict[str, Any]) -> str:
    forecasts = (p.get("key_features") or {}).get("price_range_forecast") or {}
    if not forecasts:
        return ""
    lines = [
        "## Price range forecast",
        "",
        "| Horizon | Center | 60% band | 80% band | 95% band | 80% width |",
        "|---|---|---|---|---|---|",
    ]
    for label, block in forecasts.items():
        lines.append(
            "| {label} | {center} | {b60} | {b80} | {b95} | {width} |".format(
                label=label,
                center=_fmt_price(block.get("center_price")),
                b60=_band(block.get("lower_60"), block.get("upper_60")),
                b80=_band(block.get("lower_80"), block.get("upper_80")),
                b95=_band(block.get("lower_95"), block.get("upper_95")),
                width=_fmt_pct(block.get("range_width_pct_80")),
            )
        )
    weekly_assumptions = (forecasts.get("1w") or {}).get("assumptions") or {}
    if weekly_assumptions:
        iv = weekly_assumptions.get("implied_vol_annual")
        rv = weekly_assumptions.get("realized_vol_daily")
        atr = weekly_assumptions.get("atr_daily_pct")
        evt = weekly_assumptions.get("event_risk_multiplier")
        lines += [
            "",
            "_Assumptions (1w):_ "
            f"realized_vol_daily={_fmt_pct(rv)}, "
            f"atr_daily_pct={_fmt_pct(atr)}, "
            f"implied_vol_annual={_fmt_pct(iv)}, "
            f"event_risk_multiplier={_fmt_num(evt)}.",
        ]
    return "\n".join(lines)


def _render_price_target(p: dict[str, Any]) -> str:
    target = (p.get("key_features") or {}).get("price_target") or {}
    if not target or target.get("status") != "ok":
        return ""
    lines = [
        "## Price target scenarios",
        "",
        "| Horizon | Bear | Base | Bull | Base upside | Confidence | Coverage |",
        "|---|---|---|---|---|---|---|",
        "| {h} | {bear} | {base} | {bull} | {up} | {conf} | {cov} |".format(
            h=target.get("time_horizon", "?"),
            bear=_fmt_price(target.get("bear")),
            base=_fmt_price(target.get("base")),
            bull=_fmt_price(target.get("bull")),
            up=_fmt_pct(target.get("base_upside_pct")),
            conf=_fmt_num(target.get("confidence")),
            cov=_fmt_num(target.get("coverage")),
        ),
        "",
        f"- Method: {target.get('method', 'unknown')}",
        f"- Scenario spread: {_fmt_pct(target.get('scenario_spread_pct'))}",
    ]
    sources = target.get("source_weights") or []
    if sources:
        lines += [
            "",
            "**Source blend**",
            "",
            "| Source | Target | Weight | PIT | Detail |",
            "|---|---|---|---|---|",
        ]
        for source in sources:
            lines.append(
                "| {name} | {target} | {weight} | {pit} | {detail} |".format(
                    name=_escape_pipes(str(source.get("name", "?"))),
                    target=_fmt_price(source.get("target")),
                    weight=_fmt_num(source.get("weight")),
                    pit=_escape_pipes(str(source.get("pit_status", "?"))),
                    detail=_escape_pipes(str(source.get("detail", ""))),
                )
            )
    drivers = target.get("drivers") or []
    if drivers:
        lines += ["", "**Drivers**", ""]
        lines.extend(f"- {_escape_pipes(str(item))}" for item in drivers)
    risks = target.get("risks") or []
    if risks:
        lines += ["", "**Target risks**", ""]
        lines.extend(f"- {_escape_pipes(str(item))}" for item in risks)
    missing = target.get("missing_data") or []
    if missing:
        lines += ["", "**Missing / weak data**", ""]
        lines.extend(f"- {_escape_pipes(str(item))}" for item in missing)
    extra = target.get("extra_data") or {}
    if extra:
        lines += [
            "",
            "**Extra data used**",
            "",
            "| Data | Value |",
            "|---|---|",
            "| Analyst targets | {v} |".format(
                v=_escape_pipes(str(extra.get("analyst_targets") or {}))
            ),
            "| Next-quarter EPS estimate | {v} ({pit}) |".format(
                v=_fmt_num(extra.get("next_quarter_eps_estimate")),
                pit=_escape_pipes(str(extra.get("forward_eps_pit_status"))),
            ),
            "| Annualized forward EPS estimate | {v} ({pit}) |".format(
                v=_fmt_num(extra.get("annualized_forward_eps_estimate")),
                pit=_escape_pipes(str(extra.get("forward_eps_pit_status"))),
            ),
            "| Peer forward P/E median | {v} |".format(
                v=_fmt_num(extra.get("peer_forward_pe_median"))
            ),
            "| Peer trailing P/E median | {v} |".format(
                v=_fmt_num(extra.get("peer_trailing_pe_median"))
            ),
            "| Options ATM IV 30d | {v} |".format(
                v=_fmt_pct(extra.get("options_atm_iv_30d"))
            ),
            "| Industry news signals | {v} |".format(
                v=_escape_pipes(str(extra.get("industry_news_signals") or {}))
            ),
        ]
    return "\n".join(lines)


def _render_scorecard(p: dict[str, Any]) -> str:
    scoring = (p.get("key_features") or {}).get("model_scoring") or {}
    factors = scoring.get("factor_scores") or []
    if not factors:
        return ""
    lines = [
        "## Factor scorecard",
        "",
        "| Factor | Pillar | Score | Weight | Weighted | Bucket | Rationale |",
        "|---|---|---|---|---|---|---|",
    ]
    sorted_factors = sorted(
        factors,
        key=lambda f: abs(f.get("weighted_score") or 0.0),
        reverse=True,
    )
    for f in sorted_factors:
        lines.append(
            "| {factor} | {pillar} | {score} | {weight} | {weighted} | "
            "{bucket} | {rationale} |".format(
                factor=str(f.get("factor", "?")),
                pillar=str(f.get("pillar", "?")),
                score=_fmt_signed(f.get("score")),
                weight=_fmt_num(f.get("weight")),
                weighted=_fmt_signed(f.get("weighted_score")),
                bucket=str(f.get("bucket", "?")),
                rationale=_escape_pipes(str(f.get("rationale", ""))),
            )
        )
    return "\n".join(lines)


def _render_pillars(p: dict[str, Any]) -> str:
    scoring = (p.get("key_features") or {}).get("model_scoring") or {}
    pillars = scoring.get("pillar_scores") or {}
    composite = scoring.get("composite_score")
    if not pillars and composite is None:
        return ""
    lines = ["## Pillar scores", "", "| Pillar | Score |", "|---|---|"]
    for pillar, score in pillars.items():
        lines.append(f"| {pillar} | {_fmt_signed(score)} |")
    if composite is not None:
        lines.append(f"| **composite** | **{_fmt_signed(composite)}** |")
    return "\n".join(lines)


def _render_fundamentals(p: dict[str, Any]) -> str:
    fund = (p.get("key_features") or {}).get("fundamental") or {}
    if not fund:
        return ""
    rows = [
        ("Market cap", fund.get("market_cap"), "money"),
        ("Enterprise value", fund.get("enterprise_value"), "money"),
        ("Trailing P/E", fund.get("trailing_pe"), "num"),
        ("Forward P/E", fund.get("forward_pe"), "num"),
        ("Price / book", fund.get("price_to_book"), "num"),
        ("Price / sales", fund.get("price_to_sales"), "num"),
        ("Gross margin", fund.get("gross_margins"), "pct"),
        ("Operating margin", fund.get("operating_margins"), "pct"),
        ("Profit margin", fund.get("profit_margins"), "pct"),
        ("ROE", fund.get("return_on_equity"), "pct"),
        ("ROA", fund.get("return_on_assets"), "pct"),
        ("Debt / equity", fund.get("debt_to_equity"), "num"),
        ("Current ratio", fund.get("current_ratio"), "num"),
        ("Revenue growth (yoy)", fund.get("revenue_growth"), "pct"),
        ("Earnings growth (yoy)", fund.get("earnings_growth"), "pct"),
        ("Revenue QoQ", fund.get("revenue_qoq_growth"), "pct"),
        ("Net income QoQ", fund.get("net_income_qoq_growth"), "pct"),
        ("Op income QoQ", fund.get("operating_income_qoq_growth"), "pct"),
        ("FCF QoQ", fund.get("free_cashflow_qoq_growth"), "pct"),
        ("Beta", fund.get("beta"), "num"),
    ]
    if not any(_is_present(v) for _, v, _ in rows):
        return ""
    lines = ["## Fundamentals", "", "| Metric | Value |", "|---|---|"]
    for label, value, kind in rows:
        if not _is_present(value):
            continue
        lines.append(f"| {label} | {_fmt_by_kind(value, kind)} |")
    return "\n".join(lines)


def _render_options(p: dict[str, Any]) -> str:
    opts = (p.get("key_features") or {}).get("options_flow") or {}
    status = opts.get("scan_status")
    if not status or status == "disabled":
        return ""
    lines = [
        "## Options flow",
        "",
        f"- **Scan status:** {status}",
        f"- **Expiries scanned:** {_fmt_int(opts.get('expiries_scanned'))}",
        f"- **Contracts scanned:** {_fmt_int(opts.get('contracts_scanned'))}",
        f"- **Unusual count:** {_fmt_int(opts.get('unusual_count'))}",
        f"- **Call notional:** {_fmt_money(opts.get('call_notional'))}",
        f"- **Put notional:** {_fmt_money(opts.get('put_notional'))}",
        "- **Net (call − put) notional:** "
        + _fmt_money(opts.get("net_call_put_notional")),
        f"- **ATM IV (30d, annualized):** {_fmt_pct(opts.get('atm_iv_30d'))}",
    ]
    top = opts.get("top_unusual") or []
    if top:
        lines += [
            "",
            "**Top unusual contracts**",
            "",
            "| Type | Expiry | Strike | Volume | OI | Vol/OI | Est notional |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in top[:10]:
            lines.append(
                "| {t} | {e} | {k} | {v} | {oi} | {voi} | {n} |".format(
                    t=row.get("type", "?"),
                    e=row.get("expiry", "?"),
                    k=_fmt_price(row.get("strike")),
                    v=_fmt_int(row.get("volume")),
                    oi=_fmt_int(row.get("open_interest")),
                    voi=_fmt_num(row.get("volume_oi_ratio")),
                    n=_fmt_money(row.get("estimated_notional")),
                )
            )
    return "\n".join(lines)


def _render_options_iv(p: dict[str, Any]) -> str:
    iv = (p.get("key_features") or {}).get("options_iv") or {}
    status = iv.get("status")
    if not status or status == "unavailable":
        reason = iv.get("reason")
        if reason and status == "unavailable":
            return f"## Options IV surface\n\n- _Unavailable: {reason}_"
        return ""
    lines = [
        "## Options IV surface",
        "",
        f"- **ATM IV (30d):** {_fmt_pct(iv.get('atm_iv_30d'))}",
        f"- **ATM IV (60d):** {_fmt_pct(iv.get('atm_iv_60d'))}",
        f"- **ATM IV (90d):** {_fmt_pct(iv.get('atm_iv_90d'))}",
    ]
    slope = iv.get("term_structure_slope_30_to_60")
    if slope is not None:
        shape = "backwardation" if iv.get("term_structure_is_backwardation") else "contango"
        lines.append(
            f"- **Term structure (30→60d):** {_fmt_pct(slope)} ({shape})"
        )
    skew = iv.get("skew_25d_30d")
    if skew is not None:
        lines.append(
            "- **25Δ skew (30d, put − call):** "
            f"{_fmt_pct(skew)} "
            f"(put {_fmt_pct(iv.get('skew_25d_30d_put_iv'))}, "
            f"call {_fmt_pct(iv.get('skew_25d_30d_call_iv'))})"
        )
    rv = iv.get("realized_vol_annual_20d")
    if rv is not None:
        lines.append(f"- **Realized vol (20d, annualized):** {_fmt_pct(rv)}")
    ratio = iv.get("implied_realized_ratio")
    signal = iv.get("implied_realized_signal")
    if ratio is not None:
        lines.append(
            f"- **Implied/realized ratio:** {_fmt_num(ratio)} ({signal or '?'})"
        )
    move = iv.get("earnings_implied_move")
    if move is not None:
        lines.append(
            f"- **Earnings implied move:** {_fmt_pct(move)} "
            f"(expiry {iv.get('earnings_implied_move_expiry') or '?'})"
        )
    history_status = iv.get("iv_history_status")
    if history_status == "ok":
        lines.append(
            f"- **IV rank (252d):** {_fmt_pct(iv.get('iv_rank_252d'))} "
            f"· **IV percentile:** {_fmt_pct(iv.get('iv_percentile_252d'))} "
            f"(n={_fmt_int(iv.get('iv_history_observations'))})"
        )
    elif history_status == "insufficient_history":
        lines.append(
            "- **IV rank/percentile:** _building history "
            f"(n={_fmt_int(iv.get('iv_history_observations'))})_"
        )
    pit = iv.get("pit_status")
    if pit:
        lines.append(f"- _PIT status: {pit}_")
    return "\n".join(lines)


def _render_option_strategies(p: dict[str, Any]) -> str:
    opts = (p.get("key_features") or {}).get("option_strategies") or {}
    strategies = opts.get("strategies") or []
    if not strategies and opts.get("status") in (None, "disabled"):
        return ""
    lines = [
        "## Option strategy candidates",
        "",
        f"- **Status:** {opts.get('status', 'unknown')}",
        f"- **Chain source:** {opts.get('source', 'unknown')}",
        f"- **PIT status:** {opts.get('pit_status', 'unknown')}",
        f"- **Recommended candidate:** {opts.get('recommended') or 'none'}",
    ]
    if opts.get("capital_warning"):
        lines.append("- Buying power warning: short-put margin usage is already high.")
    if opts.get("earnings_warning"):
        lines.append("- Earnings warning: near-term event risk can dominate option pricing.")
    if not strategies:
        reason = opts.get("reason") or "No strategies available."
        lines.append(f"- {reason}")
        return "\n".join(lines)
    lines += [
        "",
        "| Strategy | Verdict | Contract / spread | Premium / debit | Breakeven / cushion | Max loss | Max profit | Est. POP | Reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in strategies:
        contract = _strategy_contract_label(item)
        premium = item.get("debit")
        premium_label = "debit" if premium is not None else "premium"
        if premium is None:
            premium = item.get("premium")
        lines.append(
            "| {label} | {verdict} | {contract} | {premium_label}: {premium} | "
            "{breakeven} | {max_loss} | {max_profit} | {pop} | {reason} |".format(
                label=_escape_pipes(str(item.get("label") or item.get("type") or "?")),
                verdict=_escape_pipes(str(item.get("verdict") or "?")),
                contract=_escape_pipes(contract),
                premium_label=premium_label,
                premium=_fmt_price(premium),
                breakeven=_strategy_breakeven_text(item),
                max_loss=_fmt_money(item.get("max_loss")),
                max_profit=_fmt_money(item.get("max_profit")),
                pop=_fmt_pct(item.get("estimated_pop")),
                reason=_escape_pipes(str(item.get("reason") or "")),
            )
        )
    return "\n".join(lines)


def _strategy_contract_label(item: dict[str, Any]) -> str:
    expiry = item.get("expiry") or "?"
    dte = _fmt_int(item.get("dte"))
    if item.get("long_strike") is not None or item.get("short_strike") is not None:
        return (
            f"{expiry} ({dte} DTE) "
            f"{_fmt_price(item.get('long_strike'))}/"
            f"{_fmt_price(item.get('short_strike'))}"
        )
    return (
        f"{expiry} ({dte} DTE) "
        f"{_fmt_price(item.get('strike'))} {item.get('option_type') or ''}"
    ).strip()


def _strategy_breakeven_text(item: dict[str, Any]) -> str:
    if item.get("type") == "sell_call":
        cost_basis = item.get("cost_basis_breakeven")
        current_ref = item.get("premium_adjusted_reference_price")
        if _is_present(cost_basis):
            text = f"basis {_fmt_price(cost_basis)}"
            if _is_present(current_ref):
                text += f"; current-premium {_fmt_price(current_ref)}"
            return text
        if _is_present(current_ref):
            return f"current-premium {_fmt_price(current_ref)}"
    return _fmt_price(item.get("breakeven"))


def _render_earnings_calendar(p: dict[str, Any]) -> str:
    cal = (p.get("key_features") or {}).get("earnings_calendar") or {}
    if cal.get("status") != "ok":
        return ""
    lines = ["## Earnings calendar", ""]
    next_date = cal.get("next_earnings_date")
    days = cal.get("next_earnings_in_calendar_days")
    if next_date:
        lines.append(
            f"- **Next earnings:** {next_date} (in {_fmt_int(days)} days)"
        )
        if cal.get("earnings_in_30_days"):
            lines.append("- Earnings inside the 1m forecast window.")
        elif cal.get("earnings_in_90_days"):
            lines.append("- Earnings inside the 3m forecast window.")
    upcoming = cal.get("upcoming_earnings") or []
    if upcoming:
        lines += [
            "",
            "**Upcoming**",
            "",
            "| Date | EPS estimate | Reported EPS | Surprise % |",
            "|---|---|---|---|",
        ]
        for row in upcoming[:4]:
            lines.append(
                "| {d} | {e} | {r} | {s} |".format(
                    d=row.get("date", "?"),
                    e=_fmt_num(row.get("eps_estimate")),
                    r=_fmt_num(row.get("reported_eps")),
                    s=_fmt_num(row.get("surprisepct")),
                )
            )
    past = cal.get("past_earnings") or []
    if past:
        lines += [
            "",
            "**Recent prints**",
            "",
            "| Date | EPS estimate | Reported EPS | Surprise % |",
            "|---|---|---|---|",
        ]
        for row in past[:4]:
            lines.append(
                "| {d} | {e} | {r} | {s} |".format(
                    d=row.get("date", "?"),
                    e=_fmt_num(row.get("eps_estimate")),
                    r=_fmt_num(row.get("reported_eps")),
                    s=_fmt_num(row.get("surprisepct")),
                )
            )
    fwd_status = cal.get("forward_eps_estimates_pit_status")
    if fwd_status and fwd_status != "live":
        lines += [
            "",
            f"> _Forward EPS estimates are live consensus; PIT status: {fwd_status}._",
        ]
    return "\n".join(lines)


def _render_analyst_consensus(p: dict[str, Any]) -> str:
    cons = (p.get("key_features") or {}).get("analyst_consensus") or {}
    if cons.get("status") != "ok":
        return ""
    targets = cons.get("price_targets") or {}
    ratings = cons.get("ratings_distribution") or {}
    lines = ["## Analyst consensus", ""]
    if targets:
        lines += [
            "**Price targets**",
            "",
            "| Mean | Median | Low | High |",
            "|---|---|---|---|",
            "| {m} | {md} | {l} | {h} |".format(
                m=_fmt_price(targets.get("mean")),
                md=_fmt_price(targets.get("median")),
                l=_fmt_price(targets.get("low")),
                h=_fmt_price(targets.get("high")),
            ),
        ]
        upside = cons.get("implied_upside_pct_vs_mean_target")
        if upside is not None:
            lines += [
                "",
                f"- Implied upside vs mean target: {_fmt_pct(upside)}",
            ]
    if ratings:
        lines += [
            "",
            "**Ratings distribution**",
            "",
            "| Strong buy | Buy | Hold | Sell | Strong sell | Total | % positive |",
            "|---|---|---|---|---|---|---|",
            "| {sb} | {b} | {h} | {s} | {ss} | {tot} | {pct} |".format(
                sb=_fmt_int(ratings.get("strongBuy")),
                b=_fmt_int(ratings.get("buy")),
                h=_fmt_int(ratings.get("hold")),
                s=_fmt_int(ratings.get("sell")),
                ss=_fmt_int(ratings.get("strongSell")),
                tot=_fmt_int(cons.get("analyst_count")),
                pct=_fmt_pct(cons.get("positive_ratings_pct")),
            ),
        ]
    pit_status = cons.get("consensus_pit_status")
    if pit_status and pit_status != "live":
        lines += [
            "",
            f"> _Consensus snapshot is live; PIT status: {pit_status}._",
        ]
    return "\n".join(lines)


def _render_market_industry(p: dict[str, Any]) -> str:
    kf = p.get("key_features") or {}
    market = kf.get("market_context") or {}
    industry = kf.get("industry_context") or {}
    intraday = kf.get("intraday_context") or {}
    if not (market or industry or intraday):
        return ""
    lines = ["## Market, industry & intraday"]
    if market:
        lines += [
            "",
            "**Market regime**",
            f"- SPY 20d return: {_fmt_pct(market.get('spy_return_20d'))}",
            f"- SPY > 50DMA: {market.get('spy_above_50dma')}",
            f"- VIX level: {_fmt_num(market.get('vix_level'))}",
            f"- VIX 20d return: {_fmt_pct(market.get('vix_return_20d'))}",
            f"- 10Y yield 20d return: {_fmt_pct(market.get('tnx_return_20d'))}",
            f"- CNN Fear & Greed: {_fmt_num(market.get('fear_greed_score'))} "
            f"({market.get('fear_greed_rating') or '—'})",
            f"- Fear & Greed as of: {market.get('fear_greed_as_of') or '—'}",
            f"- Fear & Greed PIT status: "
            f"{market.get('fear_greed_pit_status') or '—'}",
        ]
    if industry:
        lines += [
            "",
            "**Sector / industry**",
            f"- Sector: {industry.get('sector') or '—'} "
            f"(ETF {industry.get('sector_etf') or '—'})",
            f"- Industry: {industry.get('industry') or '—'}",
            f"- Sector 20d return: {_fmt_pct(industry.get('sector_return_20d'))}",
            f"- Benchmark ({industry.get('benchmark') or '—'}) 20d: "
            f"{_fmt_pct(industry.get('benchmark_return_20d'))}",
            f"- Sector relative 20d: "
            f"{_fmt_pct(industry.get('sector_relative_return_20d'))}",
            f"- Sector above 50DMA: {industry.get('sector_above_50dma')}",
            f"- Stock/sector 60d corr: "
            f"{_fmt_num(industry.get('stock_sector_corr_60d'))}",
        ]
    if intraday and intraday.get("status") == "ok":
        lines += [
            "",
            "**Intraday triggers**",
            f"- Latest close: {_fmt_price(intraday.get('latest_close'))}",
            f"- Intraday RSI(14): {_fmt_num(intraday.get('intraday_rsi_14'))}",
            f"- VWAP proxy: {_fmt_price(intraday.get('vwap_proxy'))}",
            f"- Prev day high / low: "
            f"{_fmt_price(intraday.get('prev_day_high'))} / "
            f"{_fmt_price(intraday.get('prev_day_low'))}",
            f"- Broke prev-day high: {intraday.get('break_above_prev_day_high')}",
            f"- Broke prev-day low: {intraday.get('break_below_prev_day_low')}",
        ]
    return "\n".join(lines)


def _render_competitors(p: dict[str, Any]) -> str:
    comp = (p.get("key_features") or {}).get("competitor_analysis") or {}
    rows = comp.get("peer_metrics") or []
    if not rows:
        return ""
    lines = [
        "## Competitors",
        "",
        "| Ticker | Close | 5d | 20d | 60d | Trailing P/E | Rev growth | "
        "Profit margin | Market cap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {t} | {c} | {r5} | {r20} | {r60} | {pe} | {rg} | {pm} | {mc} |".format(
                t=r.get("ticker", "?"),
                c=_fmt_price(r.get("close")),
                r5=_fmt_pct(r.get("return_5d")),
                r20=_fmt_pct(r.get("return_20d")),
                r60=_fmt_pct(r.get("return_60d")),
                pe=_fmt_num(r.get("trailing_pe")),
                rg=_fmt_pct(r.get("revenue_growth")),
                pm=_fmt_pct(r.get("profit_margins")),
                mc=_fmt_money(r.get("market_cap")),
            )
        )
    summary = comp.get("summary") or {}
    if summary:
        lines += [
            "",
            "**Peer-relative summary**",
            f"- Peers: {summary.get('peer_count')}",
            f"- 20d return vs peer median: "
            f"{_fmt_pct(summary.get('return_20d_vs_peers'))}",
            f"- Trailing P/E vs peer median: "
            f"{_fmt_num(summary.get('trailing_pe_vs_peers'))}",
        ]
    return "\n".join(lines)


def _render_industry_news(p: dict[str, Any]) -> str:
    ctx = (p.get("key_features") or {}).get("industry_news_context") or {}
    if not ctx or ctx.get("status") != "ok":
        return ""
    lines = [
        "## Industry news themes",
        "",
        f"- Status: {ctx.get('status')}",
        f"- PIT status: {ctx.get('pit_status')}",
        f"- Tickers scanned: {', '.join(ctx.get('tickers_scanned') or [])}",
        f"- Summary: {ctx.get('semantic_summary') or '—'}",
    ]
    themes = ctx.get("ranked_themes") or []
    if themes:
        lines += [
            "",
            "| Theme | Count | Examples |",
            "|---|---:|---|",
        ]
        for item in themes[:8]:
            examples = "; ".join(item.get("examples") or [])
            lines.append(
                "| {theme} | {count} | {examples} |".format(
                    theme=_escape_pipes(str(item.get("theme", "?"))),
                    count=item.get("count", 0),
                    examples=_escape_pipes(examples),
                )
            )
    headlines = ctx.get("top_headlines") or []
    if headlines:
        lines += [
            "",
            "**Top related headlines**",
            "",
        ]
        for item in headlines[:8]:
            source = item.get("source_ticker") or "?"
            title = item.get("title") or ""
            lines.append(f"- `{source}` {_escape_pipes(str(title))}")
    return "\n".join(lines)


def _render_events(p: dict[str, Any]) -> str:
    kf = p.get("key_features") or {}
    events = kf.get("event_timeline") or []
    company_events = kf.get("company_events") or {}
    next_earnings = company_events.get("next_earnings_date")
    if not events and not next_earnings:
        return ""
    lines = ["## Events & catalysts"]
    if next_earnings:
        lines.append("")
        lines.append(f"- **Next earnings:** {next_earnings}")
    if events:
        lines += [
            "",
            "| Source | Type | Direction | Relevance | Horizon | Title |",
            "|---|---|---|---|---|---|",
        ]
        for e in events[:12]:
            lines.append(
                "| {s} | {t} | {d} | {r} | {h} | {ti} |".format(
                    s=e.get("source", "?"),
                    t=e.get("event_type", "?"),
                    d=e.get("likely_direction", "?"),
                    r=_fmt_num(e.get("relevance")),
                    h=e.get("time_horizon", "?"),
                    ti=_escape_pipes(str(e.get("title", ""))),
                )
            )
    return "\n".join(lines)


def _render_risks(p: dict[str, Any]) -> str:
    risks = p.get("risk_flags") or []
    invalidations = p.get("invalidation_conditions") or []
    if not risks and not invalidations:
        return ""
    lines = ["## Risks & invalidation"]
    if risks:
        lines.append("\n**Risk flags**\n")
        lines.extend(f"- {r}" for r in risks)
    if invalidations:
        lines.append("\n**Invalidation conditions**\n")
        lines.extend(f"- {c}" for c in invalidations)
    return "\n".join(lines)


def _render_llm_insights(p: dict[str, Any]) -> str:
    insights = (p.get("key_features") or {}).get("llm_insights") or {}
    if not insights or not insights.get("enabled"):
        return ""
    status = insights.get("status")
    if status != "ok":
        return f"## LLM insights\n\n- Status: {status}"
    analysis = insights.get("analysis") or {}
    lines = ["## LLM insights"]
    lines.append("")
    lines.append(f"- Provider: {insights.get('provider')} ({insights.get('model')})")
    lines.append(f"- Direction: {analysis.get('direction')}")
    lines.append(f"- Confidence: {_fmt_num(analysis.get('confidence'))}")
    if analysis.get("summary"):
        lines += ["", "**Summary**\n", analysis["summary"]]
    for label, key in (
        ("Industry angle", "industry_angle"),
        ("Company event angle", "company_event_angle"),
        ("Competitor angle", "competitor_angle"),
    ):
        val = analysis.get(key)
        if val:
            lines += ["", f"**{label}**\n", val]
    for label, key in (
        ("Key catalysts", "key_catalysts"),
        ("Key risks", "key_risks"),
        ("What to watch next", "what_to_watch_next"),
        ("Missing cases", "missing_cases"),
    ):
        items = analysis.get(key) or []
        if items:
            lines.append(f"\n**{label}**\n")
            lines.extend(f"- {x}" for x in items)
    return "\n".join(lines)


def _render_tradingagents_review(p: dict[str, Any]) -> str:
    review = (p.get("key_features") or {}).get("tradingagents_review") or {}
    if not review or not review.get("enabled"):
        return ""
    status = review.get("status")
    if status != "ok":
        return f"## TradingAgents review\n\n- Status: {status}"
    analysis = review.get("analysis") or {}
    lines = ["## TradingAgents review", ""]
    lines.append(f"- Provider: {review.get('provider')} ({review.get('model')})")

    for title, key, lead, body in (
        ("Factor hypotheses", "factor_hypotheses", "name", "rationale"),
        (
            "Candidate risk critiques",
            "candidate_risk_critiques",
            "candidate_id",
            "concern",
        ),
        ("Overfit explanations", "overfit_explanations", "candidate_id", "evidence"),
        (
            "Next data/features",
            "feature_recommendations",
            "feature_or_dataset",
            "reason",
        ),
    ):
        items = analysis.get(key) or []
        if not items:
            continue
        lines += ["", f"**{title}**", ""]
        for item in items:
            lines.append(f"- **{item.get(lead)}:** {item.get(body)}")
    return "\n".join(lines)


def _render_data_quality(p: dict[str, Any]) -> str:
    dq = p.get("data_quality") or {}
    pit_status = (p.get("key_features") or {}).get("pit_status") or {}
    if not dq and not pit_status:
        return ""
    lines = ["## Data quality"]
    if dq:
        lines += [
            "",
            f"- As-of mode: **{dq.get('as_of_mode', 'unknown')}**",
            f"- Price rows: {dq.get('price_rows')}",
            f"- News items (post-PIT filter): {dq.get('news_items')}",
            f"- Fundamentals present: {dq.get('has_fundamentals')}",
            f"- Intraday: {dq.get('intraday_status')}",
            f"- Filings: {dq.get('filings_status')}",
            f"- Options scan: {dq.get('options_scan_status')} "
            f"(unusual={dq.get('options_unusual_count')})",
            f"- Data provider: requested={dq.get('data_provider_requested')}, "
            f"resolved={dq.get('data_provider_resolved')}",
            f"- Scoring coverage: {_fmt_num(dq.get('scoring_coverage'))}",
        ]
        warnings = dq.get("pit_warnings") or []
        if warnings:
            lines += [
                "",
                "> **PIT warnings:** sections using realtime data while "
                "`as_of_date` is historical (results may not reproduce):",
                "",
            ]
            lines.extend(f"> - {w}" for w in warnings)
    if pit_status:
        lines += ["", "**Section PIT status**\n"]
        lines.append("| Section | Status |")
        lines.append("|---|---|")
        for section, status in sorted(pit_status.items()):
            lines.append(f"| {section} | {status} |")
    return "\n".join(lines)


# ---------- formatting helpers ----------


def _is_present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v != ""
    return True


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.{digits}f}"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_signed(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:+,.{digits}f}"


def _fmt_price(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"${f:,.2f}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f * 100:,.2f}%"


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    abs_f = abs(f)
    sign = "-" if f < 0 else ""
    if abs_f >= 1e12:
        return f"{sign}${abs_f / 1e12:,.2f}T"
    if abs_f >= 1e9:
        return f"{sign}${abs_f / 1e9:,.2f}B"
    if abs_f >= 1e6:
        return f"{sign}${abs_f / 1e6:,.2f}M"
    if abs_f >= 1e3:
        return f"{sign}${abs_f / 1e3:,.2f}K"
    return f"{sign}${abs_f:,.2f}"


def _fmt_by_kind(v: Any, kind: str) -> str:
    if kind == "price":
        return _fmt_price(v)
    if kind == "pct":
        return _fmt_pct(v)
    if kind == "money":
        return _fmt_money(v)
    if kind == "raw":
        return str(v) if v is not None else "—"
    return _fmt_num(v)


def _band(lower: Any, upper: Any) -> str:
    if lower is None or upper is None:
        return "—"
    return f"{_fmt_price(lower)} – {_fmt_price(upper)}"


def _escape_pipes(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _clean_case_string(s: str) -> str:
    """Strip the `factor_name: rationale. (weighted=+0.060)` formatting noise."""
    if not s:
        return s
    text = str(s)
    if ":" in text:
        prefix, rest = text.split(":", 1)
        text = rest.strip()
    if "(weighted=" in text:
        text = text.split("(weighted=", 1)[0].strip()
    return text


def _iter_present(rows: Iterable[tuple[str, Any, str]]):
    for label, value, kind in rows:
        if _is_present(value):
            yield label, value, kind
