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


def _strip_code_fences(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return s


def _build_review_schema():
    from pydantic import BaseModel, Field

    class FactorHypothesis(BaseModel):
        name: str = Field(min_length=3, max_length=120)
        rationale: str = Field(min_length=5, max_length=1200)
        required_data: list[str] = Field(default_factory=list, max_length=8)
        proposed_factor_formula: str = Field(min_length=3, max_length=1000)
        expected_direction: str = Field(pattern="^(bullish|bearish|context|unknown)$")
        validation_test: str = Field(min_length=5, max_length=1000)

    class CandidateRiskCritique(BaseModel):
        candidate_id: str = Field(min_length=1, max_length=80)
        concern: str = Field(min_length=5, max_length=1000)
        affected_knobs: list[str] = Field(default_factory=list, max_length=12)
        failure_mode: str = Field(min_length=5, max_length=1000)
        severity: str = Field(pattern="^(low|medium|high)$")
        mitigation: str = Field(min_length=5, max_length=1000)

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

    class AgentReviewOutput(BaseModel):
        factor_hypotheses: list[FactorHypothesis] = Field(default_factory=list, max_length=8)
        candidate_risk_critiques: list[CandidateRiskCritique] = Field(default_factory=list, max_length=12)
        overfit_explanations: list[OverfitExplanation] = Field(default_factory=list, max_length=12)
        feature_recommendations: list[FeatureRecommendation] = Field(default_factory=list, max_length=8)

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
        "- factor_hypotheses: list of objects with name, rationale, "
        "required_data, proposed_factor_formula, expected_direction "
        "(bullish|bearish|context|unknown), validation_test.\n"
        "- candidate_risk_critiques: list of objects with candidate_id, "
        "concern, affected_knobs, failure_mode, severity (low|medium|high), "
        "mitigation. Empty list is allowed when no candidates are provided.\n"
        "- overfit_explanations: list of objects with candidate_id, evidence, "
        "suspected_overfit_mechanism, confidence (0-1), suggested_gate. Empty "
        "list is allowed when no candidates are provided.\n"
        "- feature_recommendations: list of objects with feature_or_dataset, "
        "reason, priority (low|medium|high), expected_signal_type, "
        "validation_plan.\n\n"
        "Prefer concise, testable suggestions. Tie critiques to concrete "
        "candidate knobs and search-vs-holdout behavior when available.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True)}"
    )
    return prompt, payload


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
