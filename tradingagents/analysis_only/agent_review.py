"""TradingAgents graph-backed review helpers for analysis-only workflows.

This module keeps the full TradingAgents graph optional and bounded.  It
selects a small set of representative ticker/date contexts, runs the
framework graph for those contexts, then asks an LLM for structured review
outputs that can be stored beside tuning artifacts or attached to a report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REVIEW_PROMPT_VERSION = "v1.0"


DEFAULT_REVIEW_GATE: dict[str, Any] = {
    "status": "unavailable",
    "agree_with_signal": None,
    "risk_veto": False,
    "confidence_adjustment": 0.0,
    "sizing_multiplier": 1.0,
    "ticket_gate": "allow",
    "missing_evidence": [],
    "execution_caveats": [],
    "reason": "TradingAgents review unavailable.",
}


def _strip_code_fences(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return s


def normalize_review_gate(
    review: dict[str, Any] | None,
    *,
    report_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic gate from a TradingAgents review block.

    The gate is deliberately derived rather than purely LLM-authored so
    older review payloads and partial graph failures remain useful.  It is
    advisory by default; callers decide whether to apply sizing/ticket effects.
    """
    gate = dict(DEFAULT_REVIEW_GATE)
    gate["missing_evidence"] = []
    gate["execution_caveats"] = []
    if not isinstance(review, dict) or review.get("status") != "ok":
        if isinstance(review, dict) and review.get("status"):
            gate["status"] = str(review.get("status"))
            gate["reason"] = f"TradingAgents review status={review.get('status')}."
        return gate

    ctx = ((review.get("graph_contexts") or [{}])[0]) or {}
    analysis = review.get("analysis") or {}
    report_context = report_context or {}
    quant = _gate_stance(report_context.get("direction"))
    graph = _gate_stance(
        ctx.get("processed_decision") or ctx.get("final_trade_decision")
    )
    agree = bool(quant and graph and quant == graph) if quant and graph else None

    risk_items = analysis.get("candidate_risk_critiques") or []
    high_risk_items = [
        item for item in risk_items
        if str(item.get("severity", "")).lower() == "high"
    ]
    medium_risk_items = [
        item for item in risk_items
        if str(item.get("severity", "")).lower() == "medium"
    ]
    missing = [
        str(item.get("feature_or_dataset") or item.get("reason") or "").strip()
        for item in (analysis.get("feature_recommendations") or [])
        if str(item.get("priority", "")).lower() == "high"
    ]
    caveats = [
        str(item.get("concern") or item.get("mitigation") or "").strip()
        for item in risk_items
    ]
    if ctx.get("status") and ctx.get("status") != "ok":
        caveats.append(f"Graph context status={ctx.get('status')}.")

    risk_veto = bool(
        (quant == "bullish" and graph == "bearish")
        or (quant == "bullish" and high_risk_items)
    )
    manual_review = bool(
        not risk_veto
        and (
            agree is False
            or high_risk_items
            or len(medium_risk_items) >= 2
            or len(missing) >= 2
            or (ctx.get("status") and ctx.get("status") != "ok")
        )
    )
    if risk_veto:
        ticket_gate = "block_buy_add"
        sizing_multiplier = 0.0
        confidence_adjustment = -0.10
        reason = "TradingAgents risk review vetoes new long exposure."
    elif manual_review:
        ticket_gate = "manual_review"
        sizing_multiplier = 0.5
        confidence_adjustment = -0.05
        reason = "TradingAgents review requires manual confirmation."
    elif agree is True and graph == "bullish":
        ticket_gate = "allow"
        sizing_multiplier = 1.0
        confidence_adjustment = 0.03
        reason = "TradingAgents review agrees with the bullish signal."
    else:
        ticket_gate = "allow"
        sizing_multiplier = 1.0
        confidence_adjustment = 0.0
        reason = "TradingAgents review does not add a blocking caveat."

    return {
        "status": "ok",
        "agree_with_signal": agree,
        "risk_veto": risk_veto,
        "confidence_adjustment": confidence_adjustment,
        "sizing_multiplier": sizing_multiplier,
        "ticket_gate": ticket_gate,
        "missing_evidence": [m for m in missing if m][:8],
        "execution_caveats": [c for c in caveats if c][:8],
        "reason": reason,
    }


def _gate_stance(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in {"bullish", "buy", "strong buy", "add"} or "buy" in raw:
        return "bullish"
    if raw in {"bearish", "sell", "strong sell", "exit"} or "sell" in raw:
        return "bearish"
    if raw in {"neutral", "hold", "watch", "wait"} or "hold" in raw:
        return "neutral"
    return None


def _build_review_schema():
    from pydantic import BaseModel, Field, field_validator

    def _bounded_string_list(value: Any, max_items: int) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, Iterable):
            raw_items = list(value)
        else:
            raw_items = [value]
        return [
            str(item).strip()
            for item in raw_items
            if str(item).strip()
        ][:max_items]

    def _bounded_object_list(value: Any, max_items: int) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value[:max_items]
        if isinstance(value, tuple):
            return list(value)[:max_items]
        return [value][:max_items]

    def _lower_label(value: Any) -> str:
        return str(value).strip().lower()

    class FactorHypothesis(BaseModel):
        name: str = Field(min_length=3, max_length=120)
        rationale: str = Field(min_length=5, max_length=1200)
        required_data: list[str] = Field(default_factory=list, max_length=8)
        proposed_factor_formula: str = Field(min_length=3, max_length=1000)
        expected_direction: str = Field(pattern="^(bullish|bearish|context|unknown)$")
        validation_test: str = Field(min_length=5, max_length=1000)

        @field_validator("required_data", mode="before")
        @classmethod
        def _normalize_required_data(cls, value: Any) -> list[str]:
            return _bounded_string_list(value, 8)

        @field_validator("expected_direction", mode="before")
        @classmethod
        def _normalize_expected_direction(cls, value: Any) -> str:
            label = _lower_label(value)
            return {
                "mixed": "context",
                "neutral": "context",
                "conditional": "context",
                "unclear": "unknown",
            }.get(label, label)

    class CandidateRiskCritique(BaseModel):
        candidate_id: str = Field(min_length=1, max_length=80)
        concern: str = Field(min_length=5, max_length=1000)
        affected_knobs: list[str] = Field(default_factory=list, max_length=12)
        failure_mode: str = Field(min_length=5, max_length=1000)
        severity: str = Field(pattern="^(low|medium|high)$")
        mitigation: str = Field(min_length=5, max_length=1000)

        @field_validator("affected_knobs", mode="before")
        @classmethod
        def _normalize_affected_knobs(cls, value: Any) -> list[str]:
            return _bounded_string_list(value, 12)

        @field_validator("severity", mode="before")
        @classmethod
        def _normalize_severity(cls, value: Any) -> str:
            return _lower_label(value)

    class OverfitExplanation(BaseModel):
        candidate_id: str = Field(min_length=1, max_length=80)
        evidence: str = Field(min_length=5, max_length=1200)
        suspected_overfit_mechanism: str = Field(min_length=5, max_length=1000)
        confidence: float = Field(ge=0.0, le=1.0)
        suggested_gate: str = Field(min_length=5, max_length=1000)

    class FeatureRecommendation(BaseModel):
        feature_or_dataset: str = Field(min_length=3, max_length=160)
        reason: str = Field(min_length=5, max_length=1200)
        priority: str = Field(pattern="^(low|medium|high)$")
        expected_signal_type: str = Field(min_length=3, max_length=240)
        validation_plan: str = Field(min_length=5, max_length=1000)

        @field_validator("priority", mode="before")
        @classmethod
        def _normalize_priority(cls, value: Any) -> str:
            return _lower_label(value)

    class AgentReviewOutput(BaseModel):
        factor_hypotheses: list[FactorHypothesis] = Field(default_factory=list, max_length=8)
        candidate_risk_critiques: list[CandidateRiskCritique] = Field(default_factory=list, max_length=12)
        overfit_explanations: list[OverfitExplanation] = Field(default_factory=list, max_length=12)
        feature_recommendations: list[FeatureRecommendation] = Field(default_factory=list, max_length=8)

        @field_validator("factor_hypotheses", "feature_recommendations", mode="before")
        @classmethod
        def _normalize_short_sections(cls, value: Any) -> list[Any]:
            return _bounded_object_list(value, 8)

        @field_validator("candidate_risk_critiques", "overfit_explanations", mode="before")
        @classmethod
        def _normalize_long_sections(cls, value: Any) -> list[Any]:
            return _bounded_object_list(value, 12)

    return AgentReviewOutput


def _signed_return(direction: str | None, ret: float | None) -> float | None:
    if ret is None:
        return None
    if direction == "bullish":
        return float(ret)
    if direction == "bearish":
        return -float(ret)
    return None


def _record_key(record: Any) -> tuple[str, str]:
    return (str(record.symbol).upper(), str(record.as_of_date))


def select_representative_contexts(
    original_records: Iterable[Any],
    rebuilt_records: Iterable[Any],
    *,
    max_contexts: int = 5,
    return_field: str = "ret_20d",
) -> list[dict[str, Any]]:
    """Select deterministic ticker/date contexts for graph review.

    Contexts are deduped and ordered by the plan categories:
    largest signed winner, largest signed loser, highest absolute composite,
    most recent active signal, and largest original-vs-rebuilt direction
    change. Remaining slots are filled by absolute composite.
    """
    if max_contexts <= 0:
        return []
    originals = {_record_key(r): r for r in original_records}
    rebuilt = list(rebuilt_records)
    pairs = [(originals.get(_record_key(r)), r) for r in rebuilt]

    def context_for(orig: Any | None, rec: Any, reason: str) -> dict[str, Any]:
        ret = (rec.forward_returns or {}).get(return_field)
        return {
            "symbol": rec.symbol,
            "as_of_date": rec.as_of_date,
            "selection_reasons": [reason],
            "original_direction": getattr(orig, "direction", None),
            "rebuilt_direction": rec.direction,
            "composite_score": rec.composite_score,
            "forward_return": ret,
            "signed_return": _signed_return(rec.direction, ret),
            "source_path": rec.source_path,
        }

    selected: dict[tuple[str, str], dict[str, Any]] = {}

    def add(orig: Any | None, rec: Any | None, reason: str) -> None:
        if rec is None or len(selected) >= max_contexts:
            return
        key = _record_key(rec)
        if key in selected:
            if reason not in selected[key]["selection_reasons"]:
                selected[key]["selection_reasons"].append(reason)
            return
        selected[key] = context_for(orig, rec, reason)

    active = [(o, r) for o, r in pairs if r.direction in {"bullish", "bearish"}]
    signed = [
        (o, r, _signed_return(r.direction, (r.forward_returns or {}).get(return_field)))
        for o, r in active
    ]
    signed = [(o, r, s) for o, r, s in signed if s is not None]
    if signed:
        o, r, _ = max(signed, key=lambda item: (item[2], item[1].as_of_date, item[1].symbol))
        add(o, r, "largest_signed_winner")
        o, r, _ = min(signed, key=lambda item: (item[2], item[1].as_of_date, item[1].symbol))
        add(o, r, "largest_signed_loser")

    scored = [
        (o, r)
        for o, r in pairs
        if r.composite_score is not None
    ]
    if scored:
        o, r = max(
            scored,
            key=lambda item: (
                abs(float(item[1].composite_score or 0.0)),
                item[1].as_of_date,
                item[1].symbol,
            ),
        )
        add(o, r, "highest_abs_composite")

    if active:
        o, r = max(active, key=lambda item: (item[1].as_of_date, item[1].symbol))
        add(o, r, "most_recent_active_signal")

    changed = [
        (o, r)
        for o, r in pairs
        if o is not None and getattr(o, "direction", None) != r.direction
    ]
    if changed:
        o, r = max(
            changed,
            key=lambda item: (
                abs(float(item[1].composite_score or 0.0) - float(getattr(item[0], "composite_score", 0.0) or 0.0)),
                item[1].as_of_date,
                item[1].symbol,
            ),
        )
        add(o, r, "largest_direction_change")

    for o, r in sorted(
        scored,
        key=lambda item: (
            abs(float(item[1].composite_score or 0.0)),
            item[1].as_of_date,
            item[1].symbol,
        ),
        reverse=True,
    ):
        if len(selected) >= max_contexts:
            break
        add(o, r, "abs_composite_fill")

    return list(selected.values())[:max_contexts]


def candidate_review_payload(
    search_evals: Iterable[Any],
    holdout_evals: dict[str, Any],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    ranked = sorted(search_evals, key=lambda e: (e.rejected, -e.score))
    out: list[dict[str, Any]] = []
    for e in ranked[: max(0, top_n)]:
        c = e.candidate
        h = holdout_evals.get(c.candidate_id)
        out.append({
            "candidate": c.to_dict(),
            "search": e.to_dict(),
            "holdout": h.to_dict() if h is not None else None,
        })
    return out


def _review_prompt_payload(
    *,
    graph_contexts: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
    report_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "graph_contexts": graph_contexts,
        "tuner_candidates": candidates or [],
        "report_context": report_context or {},
    }


def build_review_prompt(
    *,
    graph_contexts: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
    report_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = _review_prompt_payload(
        graph_contexts=graph_contexts,
        candidates=candidates,
        report_context=report_context,
    )
    prompt = (
        "You are a TradingAgents meta-reviewer. You receive full graph outputs "
        "from analyst, researcher, trader, and risk-management agents, plus "
        "optional analysis-only tuner candidates. Use the graph debates as "
        "context, but return strict JSON only. Do not give trading advice; "
        "focus on improving the analysis system.\n\n"
        "Required JSON keys:\n"
        "- factor_hypotheses: at most 8 objects with name, rationale, "
        "required_data, proposed_factor_formula, expected_direction "
        "(bullish|bearish|context|unknown), validation_test.\n"
        "- candidate_risk_critiques: at most 12 objects with candidate_id, "
        "concern, affected_knobs, failure_mode, severity (low|medium|high), "
        "mitigation. Empty list is allowed when no candidates are provided.\n"
        "- overfit_explanations: at most 12 objects with candidate_id, evidence, "
        "suspected_overfit_mechanism, confidence (0-1), suggested_gate. Empty "
        "list is allowed when no candidates are provided.\n"
        "- feature_recommendations: at most 8 objects with feature_or_dataset, "
        "reason, priority (low|medium|high), expected_signal_type, "
        "validation_plan.\n\n"
        "Prefer concise, testable suggestions. Tie critiques to concrete "
        "candidate knobs and search-vs-holdout behavior when available.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True)}"
    )
    return prompt, payload


def report_context_from_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build the report-review context from a saved analysis report payload."""
    key_features = report.get("key_features") or {}
    scoring = key_features.get("model_scoring") or {}
    factor_scores = scoring.get("factor_scores") or []
    top_factors = sorted(
        (f for f in factor_scores if isinstance(f, dict) and f.get("data_available")),
        key=lambda f: abs(f.get("weighted_score") or 0.0),
        reverse=True,
    )[:12]
    return {
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "direction": report.get("direction"),
        "confidence": report.get("confidence"),
        "composite_score": scoring.get("composite_score"),
        "coverage": (report.get("data_quality") or {}).get("scoring_coverage"),
        "pillar_scores": scoring.get("pillar_scores") or {},
        "top_factors": top_factors,
        "technicals": key_features.get("technical") or {},
        "fundamentals": key_features.get("fundamental") or {},
        "options_flow": key_features.get("options_flow") or {},
        "market_context": key_features.get("market_context") or {},
        "industry_context": key_features.get("industry_context") or {},
        "competitor_summary": (
            (key_features.get("competitor_analysis") or {}).get("summary") or {}
        ),
        "risk_flags": report.get("risk_flags") or [],
        "earnings_calendar": key_features.get("earnings_calendar") or {},
        "data_quality": report.get("data_quality") or {},
    }


def _extract_graph_state(
    *,
    context: dict[str, Any],
    state: dict[str, Any],
    decision: str | None,
) -> dict[str, Any]:
    invest = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    return {
        "symbol": context.get("symbol"),
        "as_of_date": context.get("as_of_date"),
        "selection_reasons": context.get("selection_reasons") or [],
        "status": "ok",
        "market_report": state.get("market_report"),
        "sentiment_report": state.get("sentiment_report"),
        "news_report": state.get("news_report"),
        "fundamentals_report": state.get("fundamentals_report"),
        "investment_debate": {
            "bull_history": invest.get("bull_history"),
            "bear_history": invest.get("bear_history"),
            "judge_decision": invest.get("judge_decision"),
        },
        "trader_plan": state.get("trader_investment_plan") or state.get("investment_plan"),
        "risk_debate": {
            "aggressive_history": risk.get("aggressive_history"),
            "neutral_history": risk.get("neutral_history"),
            "conservative_history": risk.get("conservative_history"),
            "judge_decision": risk.get("judge_decision"),
        },
        "final_trade_decision": state.get("final_trade_decision"),
        "processed_decision": decision,
    }


def _graph_config(
    *,
    provider: str | None,
    quick_model: str | None,
    deep_model: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    if provider:
        cfg["llm_provider"] = provider
    if quick_model:
        cfg["quick_think_llm"] = quick_model
    if deep_model:
        cfg["deep_think_llm"] = deep_model
    if base_url:
        cfg["backend_url"] = base_url
    return cfg


def run_graph_contexts(
    contexts: list[dict[str, Any]],
    *,
    provider: str | None = None,
    quick_model: str | None = None,
    deep_model: str | None = None,
    base_url: str | None = None,
    selected_analysts: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not contexts:
        return []
    cfg = _graph_config(
        provider=provider,
        quick_model=quick_model,
        deep_model=deep_model,
        base_url=base_url,
    )
    out: list[dict[str, Any]] = []
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts or [
                "market", "social", "news", "fundamentals",
            ],
            debug=False,
            config=cfg,
        )
    except Exception as exc:
        return [
            {
                "symbol": c.get("symbol"),
                "as_of_date": c.get("as_of_date"),
                "selection_reasons": c.get("selection_reasons") or [],
                "status": "graph_init_error",
                "error": str(exc),
            }
            for c in contexts
        ]

    for context in contexts:
        try:
            state, decision = graph.propagate(
                str(context["symbol"]),
                str(context["as_of_date"]),
            )
            out.append(_extract_graph_state(
                context=context,
                state=state,
                decision=str(decision) if decision is not None else None,
            ))
        except Exception as exc:
            out.append({
                "symbol": context.get("symbol"),
                "as_of_date": context.get("as_of_date"),
                "selection_reasons": context.get("selection_reasons") or [],
                "status": "graph_call_error",
                "error": str(exc),
            })
    return out


def generate_structured_review(
    *,
    graph_contexts: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
    report_context: dict[str, Any] | None = None,
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    base_url: str | None = None,
) -> dict[str, Any]:
    prompt, payload = build_review_prompt(
        graph_contexts=graph_contexts,
        candidates=candidates,
        report_context=report_context,
    )
    block: dict[str, Any] = {
        "enabled": True,
        "status": "unknown",
        "provider": provider,
        "model": model,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "graph_contexts": graph_contexts,
        "input_summary": {
            "graph_context_count": len(graph_contexts),
            "candidate_count": len(candidates or []),
            "has_report_context": bool(report_context),
        },
    }
    try:
        from tradingagents.llm_clients.factory import create_llm_client

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=0.0,
        )
        llm = client.get_llm()
    except Exception as exc:
        block["status"] = "llm_init_error"
        block["error"] = str(exc)
        return block

    try:
        response = llm.invoke(prompt)
        cleaned = _strip_code_fences(str(getattr(response, "content", response)))
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        block["status"] = "llm_json_parse_error"
        block["error"] = str(exc)
        block["raw_response"] = cleaned[:2000] if "cleaned" in locals() else ""
        return block
    except Exception as exc:
        block["status"] = "llm_call_error"
        block["error"] = str(exc)
        return block

    try:
        from pydantic import ValidationError

        schema = _build_review_schema()
        validated = schema.model_validate(parsed)
    except ValidationError as exc:
        block["status"] = "llm_schema_validation_error"
        block["error"] = str(exc)
        block["raw_response"] = json.dumps(parsed)[:2000]
        return block

    block["status"] = "ok"
    block["analysis"] = validated.model_dump()
    block["input_payload"] = payload
    return block


def run_tuning_agent_review(
    *,
    search_evals: list[Any],
    holdout_evals: dict[str, Any],
    holdout_records: list[Any],
    top_n: int = 5,
    max_contexts: int = 5,
    provider: str = "openai",
    quick_model: str = "gpt-5.4-mini",
    deep_model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    from tradingagents.analysis_only.backtest import rebuild_records_with_weights

    ranked = sorted(search_evals, key=lambda e: (e.rejected, -e.score))
    best = next((e for e in ranked if not e.rejected), ranked[0] if ranked else None)
    if best is None:
        return {
            "enabled": True,
            "status": "no_candidates",
            "provider": provider,
            "model": quick_model,
        }
    rebuilt_holdout = rebuild_records_with_weights(
        holdout_records,
        weights=best.candidate.weights,
        bullish_threshold=best.candidate.bullish_threshold,
        bearish_threshold=best.candidate.bearish_threshold,
        neutral_band=best.candidate.neutral_band,
    )
    contexts = select_representative_contexts(
        holdout_records,
        rebuilt_holdout,
        max_contexts=max_contexts,
    )
    graph_contexts = run_graph_contexts(
        contexts,
        provider=provider,
        quick_model=quick_model,
        deep_model=deep_model or quick_model,
        base_url=base_url,
    )
    candidates = candidate_review_payload(
        search_evals,
        holdout_evals,
        top_n=top_n,
    )
    block = generate_structured_review(
        graph_contexts=graph_contexts,
        candidates=candidates,
        provider=provider,
        model=quick_model,
        base_url=base_url,
    )
    block["review_type"] = "tuning"
    block["selected_contexts"] = contexts
    block["best_candidate_id"] = best.candidate.candidate_id
    return block


def run_report_agent_review(
    *,
    symbol: str,
    as_of_date: str,
    report_context: dict[str, Any],
    provider: str = "openai",
    quick_model: str = "gpt-5.4-mini",
    deep_model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    contexts = [{
        "symbol": symbol,
        "as_of_date": as_of_date,
        "selection_reasons": ["single_report"],
    }]
    graph_contexts = run_graph_contexts(
        contexts,
        provider=provider,
        quick_model=quick_model,
        deep_model=deep_model or quick_model,
        base_url=base_url,
    )
    block = generate_structured_review(
        graph_contexts=graph_contexts,
        report_context=report_context,
        provider=provider,
        model=quick_model,
        base_url=base_url,
    )
    block["review_type"] = "report"
    block["gate"] = normalize_review_gate(block, report_context=report_context)
    return block


def render_agent_review_markdown(block: dict[str, Any]) -> str:
    lines = ["# TradingAgents Review", ""]
    lines.append(f"- Status: **{block.get('status', 'unknown')}**")
    lines.append(f"- Provider/model: `{block.get('provider')}` / `{block.get('model')}`")
    if block.get("best_candidate_id"):
        lines.append(f"- Best candidate context: `{block.get('best_candidate_id')}`")
    graph_contexts = block.get("graph_contexts") or []
    if graph_contexts:
        lines += ["", "## Graph Contexts", ""]
        lines.append("| Symbol | Date | Status | Reasons |")
        lines.append("|---|---|---|---|")
        for c in graph_contexts:
            reasons = ", ".join(c.get("selection_reasons") or [])
            lines.append(
                f"| {c.get('symbol')} | {c.get('as_of_date')} | "
                f"{c.get('status')} | {reasons} |"
            )
    if block.get("status") != "ok":
        if block.get("error"):
            lines += ["", f"Error: `{block.get('error')}`"]
        return "\n".join(lines) + "\n"

    analysis = block.get("analysis") or {}
    sections = [
        ("Factor Hypotheses", "factor_hypotheses", "name", "rationale"),
        ("Candidate Risk Critiques", "candidate_risk_critiques", "candidate_id", "concern"),
        ("Overfit Explanations", "overfit_explanations", "candidate_id", "evidence"),
        ("Feature Recommendations", "feature_recommendations", "feature_or_dataset", "reason"),
    ]
    for title, key, lead, body in sections:
        items = analysis.get(key) or []
        lines += ["", f"## {title}", ""]
        if not items:
            lines.append("_None returned._")
            continue
        for item in items:
            lines.append(f"- **{item.get(lead)}:** {item.get(body)}")
    return "\n".join(lines) + "\n"


def write_agent_review_outputs(
    *,
    output_dir: str | Path,
    block: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_review.json").write_text(json.dumps(block, indent=2, sort_keys=True))
    (out / "agent_review.md").write_text(render_agent_review_markdown(block))
