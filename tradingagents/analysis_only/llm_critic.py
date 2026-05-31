"""Phase 6: LLM critic as a logged, backtested feature.

Given a finished `AnalysisOnlyMVP` report payload, send a frozen prompt
to an LLM at temperature=0 and ask it to flag factor blind-spots, an
invalidation probability over 30 days, an asymmetric confidence
adjustment (clamped to [-0.2, 0.0]), and a hard veto bit. The output
is validated against `CriticOutput` (Pydantic) and stored on the
report JSON under a new `llm_critic` block, together with the model
id and a SHA-1 hash of the prompt so the call is fully reproducible.

The critic is logged for *every* report unconditionally. It is only
wired into sizing / confidence after Phase 6's acceptance gate
(incremental IC > 0.03 OOS + veto efficacy >= 20% loss reduction)
passes in `scripts/eval_llm_critic.py`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


CRITIC_PROMPT_VERSION = "v1.1"

# Frozen prompt. Edit only by bumping CRITIC_PROMPT_VERSION; any change
# invalidates historical backfills.
#
# v1.1 (2026-05-25): dropped the `veto` boolean. The v1.0 backfill ran
# 1,420 critic calls on the corpus and showed veto rate ~65% with
# vetoed trades *out*performing non-vetoed trades at ret_5d/ret_20d
# (loss_reduction_ratio negative). The veto bit was anti-signal — the
# adversarial framing biased gpt-5.4-mini toward vetoing whenever it
# could find a thread to pull, regardless of decisiveness. The
# continuous `confidence_adjustment` field already encodes the same
# information on a calibrated scale; removing the bit costs nothing
# and stops it from polluting any future sizing logic that consumed
# it. v1.0 critic blocks on disk remain readable for backwards
# compatibility; the eval skips veto stats on v1.1 records since the
# field is absent.
CRITIC_PROMPT_TEMPLATE = (
    "You are an adversarial buy-side equity research critic. You receive "
    "a structured analysis payload that an analyst has already produced "
    "for a single ticker. Your job is to challenge it, not extend it. "
    "Be terse, mechanical, and avoid flattery.\n\n"
    "Output STRICT JSON only. No prose. No markdown. No code fences. "
    "Keys (all required):\n"
    "- factor_blindspots: list[str], 0-5 short phrases naming factors "
    "or evidence the analyst undercounted, double-counted, or missed. "
    "Examples: 'ignored peer-group de-rating', 'momentum / RSI "
    "double-count', 'no earnings-event blackout adjustment'. Empty "
    "list is allowed and preferred when nothing is wrong.\n"
    "- invalidation_prob_30d: float in [0, 1]. The probability the "
    "analyst's stated direction will be invalidated by realized price "
    "action within 30 calendar days, given only the supplied evidence. "
    "Calibrate against a base-rate of 0.4-0.5; do not anchor to 0.5.\n"
    "- confidence_adjustment: float in [-0.2, 0.0]. Asymmetric — you "
    "may only LOWER confidence. 0.0 means no change. -0.2 is reserved "
    "for cases where the analyst's evidence is materially "
    "contradicted by their own snapshot.\n\n"
    "INPUT_JSON:\n{payload_json}"
)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _build_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full report into the critic's input payload.

    Deterministic; if this changes, bump `CRITIC_PROMPT_VERSION`.
    """
    summary = report.get("summary") or {}
    factor_scores = report.get("factor_scores") or []
    top_factors = sorted(
        (f for f in factor_scores if f.get("data_available")),
        key=lambda f: abs(f.get("weighted_score") or 0.0),
        reverse=True,
    )[:10]
    return {
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "direction": summary.get("direction"),
        "confidence": summary.get("confidence"),
        "composite_score": summary.get("composite_score"),
        "pillar_scores": report.get("pillar_scores") or {},
        "coverage": summary.get("coverage"),
        "top_factors": [
            {
                "factor": f.get("factor"),
                "pillar": f.get("pillar"),
                "bucket": f.get("bucket"),
                "score": f.get("score"),
                "weighted_score": f.get("weighted_score"),
                "rationale": f.get("rationale"),
            }
            for f in top_factors
        ],
        "snapshot": {
            k: (report.get("technicals") or {}).get(k)
            for k in (
                "close", "rsi_14", "macd_hist", "return_20d",
                "volatility_20d", "atr_14",
            )
        }
        | {
            k: (report.get("fundamentals") or {}).get(k)
            for k in (
                "trailing_pe", "forward_pe", "revenue_growth",
                "profit_margins",
            )
        },
        "options_flow": {
            k: (report.get("options_flow") or {}).get(k)
            for k in (
                "unusual_count", "net_call_put_notional", "atm_iv_30d",
                "iv_rank", "iv_skew",
            )
        },
        "risk_flags": report.get("risk_flags") or [],
        "earnings_calendar": report.get("earnings_calendar") or {},
        "data_quality": (report.get("data_quality") or {}).get(
            "pit_warnings"
        ) or [],
    }


def build_critic_prompt(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render the frozen prompt with a deterministic payload."""
    payload = _build_payload(report)
    prompt = CRITIC_PROMPT_TEMPLATE.format(
        payload_json=json.dumps(payload, sort_keys=True)
    )
    return prompt, payload


def _build_critic_schema():
    """Local import so the module can be loaded without pydantic at runtime.

    v1.1: dropped the `veto` / `veto_reason` fields. The Pydantic config
    `extra="ignore"` keeps the schema tolerant of v1.0-trained models
    that still emit those keys — Pydantic drops them silently rather
    than raising, so a stale model never blocks a v1.1 backfill.
    """
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    class CriticOutput(BaseModel):
        model_config = ConfigDict(extra="ignore")

        factor_blindspots: list[str] = Field(default_factory=list, max_length=5)
        invalidation_prob_30d: float = Field(ge=0.0, le=1.0)
        confidence_adjustment: float = Field(ge=-0.2, le=0.0)

        @model_validator(mode="before")
        @classmethod
        def _accept_common_key_typos(cls, data):
            if isinstance(data, dict):
                data = dict(data)
                typo = "invalidiation_prob_30d"
                if "invalidation_prob_30d" not in data and typo in data:
                    data["invalidation_prob_30d"] = data[typo]
            return data

    return CriticOutput


def _strip_code_fences(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return s


def run_critic(
    report: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run the critic against a single report and return its `llm_critic` block.

    The block is intentionally JSON-serializable and includes:
      - status: "ok" | "llm_init_error" | "llm_call_error" | "..."
      - provider, model, prompt_version, prompt_hash
      - output: validated CriticOutput dict (only on status="ok")
      - raw_response: trimmed raw text (only on schema-error paths)

    The caller is responsible for persisting this block onto the report.
    """
    prompt, _payload = build_critic_prompt(report)
    block: dict[str, Any] = {
        "status": "unknown",
        "provider": provider,
        "model": model,
        "prompt_version": CRITIC_PROMPT_VERSION,
        "prompt_hash": _prompt_hash(prompt),
        "temperature": temperature,
    }
    try:
        from tradingagents.llm_clients.factory import create_llm_client

        client = create_llm_client(
            provider=provider, model=model, base_url=base_url,
            temperature=temperature,
        )
        llm = client.get_llm()
    except Exception as exc:
        block["status"] = "llm_init_error"
        block["error"] = str(exc)
        return block

    try:
        response = llm.invoke(prompt)
        content = str(getattr(response, "content", response))
    except Exception as exc:
        block["status"] = "llm_call_error"
        block["error"] = str(exc)
        return block

    cleaned = _strip_code_fences(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        block["status"] = "llm_json_parse_error"
        block["error"] = str(exc)
        block["raw_response"] = cleaned[:2000]
        return block

    try:
        from pydantic import ValidationError

        schema = _build_critic_schema()
        validated = schema.model_validate(parsed)
    except ValidationError as exc:
        block["status"] = "llm_schema_validation_error"
        block["error"] = str(exc)
        block["raw_response"] = cleaned[:2000]
        return block

    block["status"] = "ok"
    block["output"] = validated.model_dump()
    return block


# ---------- Phase 7: multi-model disagreement ----------


def _pairwise_jaccard_distance(
    sets: list[set[str]],
) -> float | None:
    """Mean pairwise Jaccard distance over n>=2 sets.

    Jaccard distance = 1 - |A & B| / |A | B|. Empty-vs-empty pairs
    contribute 0 (perfect agreement on "nothing was wrong").
    """
    if len(sets) < 2:
        return None
    pairs: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                pairs.append(0.0)
                continue
            inter = a & b
            pairs.append(1.0 - len(inter) / len(union))
    if not pairs:
        return None
    return round(sum(pairs) / len(pairs), 4)


def _normalize_blindspot(text: str) -> str:
    """Lowercase + strip + collapse whitespace so 'Ignored peer' == 'ignored  peer'."""
    return " ".join(text.lower().split())


def compute_llm_disagreement(
    per_model_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce >=2 critic outputs into a disagreement summary.

    Inputs are the validated `CriticOutput.model_dump()` dicts (one per
    model). Returns a dict with:
      - n_models: int
      - confidence_adjustment_stdev: float | None
      - confidence_adjustment_mean: float | None
      - blindspots_jaccard_mean: float | None (mean pairwise Jaccard distance)
      - invalidation_prob_30d_stdev: float | None
      - veto_agreement_rate: float | None (fraction of models that veto'd
        when at least one did; 0 if none did)

    Returns an empty-ish dict when fewer than 2 valid outputs are passed.
    """
    import statistics

    valid = [o for o in per_model_outputs if isinstance(o, dict)]
    out: dict[str, Any] = {
        "n_models": len(valid),
        "confidence_adjustment_stdev": None,
        "confidence_adjustment_mean": None,
        "invalidation_prob_30d_stdev": None,
        "invalidation_prob_30d_mean": None,
        "blindspots_jaccard_mean": None,
        "veto_agreement_rate": None,
    }
    if len(valid) < 2:
        return out
    conf_adj = [
        float(o["confidence_adjustment"]) for o in valid
        if isinstance(o.get("confidence_adjustment"), (int, float))
    ]
    inv_p = [
        float(o["invalidation_prob_30d"]) for o in valid
        if isinstance(o.get("invalidation_prob_30d"), (int, float))
    ]
    if len(conf_adj) >= 2:
        out["confidence_adjustment_stdev"] = round(
            statistics.pstdev(conf_adj), 4
        )
        out["confidence_adjustment_mean"] = round(
            statistics.fmean(conf_adj), 4
        )
    if len(inv_p) >= 2:
        out["invalidation_prob_30d_stdev"] = round(
            statistics.pstdev(inv_p), 4
        )
        out["invalidation_prob_30d_mean"] = round(statistics.fmean(inv_p), 4)
    sets = [
        {_normalize_blindspot(b) for b in (o.get("factor_blindspots") or [])}
        for o in valid
    ]
    out["blindspots_jaccard_mean"] = _pairwise_jaccard_distance(sets)
    vetoes = [bool(o.get("veto")) for o in valid]
    if any(vetoes):
        out["veto_agreement_rate"] = round(
            sum(1 for v in vetoes if v) / len(vetoes), 4
        )
    else:
        out["veto_agreement_rate"] = 0.0
    return out


def critic_output_or_none(critic_block: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the validated `output` dict if present, else None.

    Helper for backtest/extraction code so callers don't have to
    repeat the status check.
    """
    if not critic_block:
        return None
    if critic_block.get("status") != "ok":
        return None
    return critic_block.get("output") or None
