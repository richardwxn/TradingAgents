"""Scoring core for the debate-grounding ablation.

Pure functions (no LLM, no network) that take per-(ticker,date) outcomes —
the grounded debate rating, the control (ungrounded) rating, the quant
model's direction, and the realized forward return — and produce the
comparison metrics: decision hit-rate, return-capture, agreement with the
quant signal, and the key metric, OVERRIDE-CORRECTNESS (on names where the
debate disagrees with the model, who was right).

Kept separate from the graph-running shell (`scripts/eval_debate_grounding.py`)
so the metrics are fully unit-testable without running a debate.
"""

from __future__ import annotations

from typing import Any, Sequence

# 5-tier rating -> directional stance.
_RATING_DIRECTION = {
    "buy": "bullish",
    "overweight": "bullish",
    "hold": "neutral",
    "underweight": "bearish",
    "sell": "bearish",
}

# Neutral is "right" only when the move was small (matches the analysis-only
# and calibration hit convention).
_NEUTRAL_BAND = 0.02


def rating_to_direction(rating: str | None) -> str:
    """Map a 5-tier rating (Buy/Overweight/Hold/Underweight/Sell) to a stance."""
    return _RATING_DIRECTION.get(str(rating or "").strip().lower(), "neutral")


def decision_hit(direction: str, realized_return: float | None) -> int | None:
    """1 if the directional call was correct against the realized return."""
    if realized_return is None:
        return None
    if direction == "bullish":
        return 1 if realized_return > 0 else 0
    if direction == "bearish":
        return 1 if realized_return < 0 else 0
    if direction == "neutral":
        return 1 if abs(realized_return) < _NEUTRAL_BAND else 0
    return None


def return_capture(direction: str, realized_return: float | None) -> float | None:
    """Signed return the stance captured (long a winner / short a loser = +)."""
    if realized_return is None:
        return None
    if direction == "bullish":
        return realized_return
    if direction == "bearish":
        return -realized_return
    return 0.0


def _rate(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _arm(pairs: Sequence[dict[str, Any]], direction_key: str) -> dict[str, Any]:
    """Aggregate hit-rate + return-capture for one decision source."""
    hits: list[int] = []
    caps: list[float] = []
    for p in pairs:
        ret = p.get("realized_return")
        d = p.get(direction_key)
        h = decision_hit(d, ret)
        c = return_capture(d, ret)
        if h is not None:
            hits.append(h)
        if c is not None:
            caps.append(c)
    return {
        "n": len(hits),
        "hit_rate": _rate(hits),
        "mean_return_capture": _mean(caps),
    }


def score_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the ablation comparison.

    Each pair: ``{symbol, as_of_date, realized_return, quant_direction,
    grounded_rating, control_rating}``. Ratings are mapped to directions
    internally. Returns grounded / control / quant arms, agreement rates, and
    override-correctness on disagreement subsets.
    """
    enriched: list[dict[str, Any]] = []
    for p in pairs:
        e = dict(p)
        e["grounded_direction"] = rating_to_direction(p.get("grounded_rating"))
        e["control_direction"] = rating_to_direction(p.get("control_rating"))
        e["quant_direction"] = str(p.get("quant_direction") or "neutral").lower()
        enriched.append(e)

    grounded = _arm(enriched, "grounded_direction")
    control = _arm(enriched, "control_direction")
    quant = _arm(enriched, "quant_direction")

    def agreement(direction_key: str) -> float | None:
        flags = [
            1 if e[direction_key] == e["quant_direction"] else 0
            for e in enriched
        ]
        return _rate(flags)

    def override_stats(direction_key: str) -> dict[str, Any]:
        # On names where the debate disagrees with the quant signal: when the
        # realized return is known, did the debate or the quant model win?
        debate_wins: list[int] = []
        quant_wins: list[int] = []
        for e in enriched:
            if e[direction_key] == e["quant_direction"]:
                continue
            ret = e.get("realized_return")
            dh = decision_hit(e[direction_key], ret)
            qh = decision_hit(e["quant_direction"], ret)
            if dh is None or qh is None:
                continue
            debate_wins.append(dh)
            quant_wins.append(qh)
        return {
            "n_disagreements": len(debate_wins),
            "debate_win_rate": _rate(debate_wins),
            "quant_win_rate": _rate(quant_wins),
        }

    return {
        "n_pairs": len(enriched),
        "grounded": grounded,
        "control": control,
        "quant": quant,
        "agreement_grounded_vs_quant": agreement("grounded_direction"),
        "agreement_control_vs_quant": agreement("control_direction"),
        "override_grounded": override_stats("grounded_direction"),
        "override_control": override_stats("control_direction"),
        "pairs": enriched,
    }


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def _sig(v: float | None) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "n/a"


# --------------------------------------------------------------------------
# LLM-judge process score (cheap proxy): did the grounded debate actually
# engage with the quant signal, or ignore it? Prompt/parse are pure; the LLM
# call lives in the CLI shell so this stays testable.
# --------------------------------------------------------------------------

JUDGE_PROMPT_VERSION = "v1.0"

_JUDGE_PROMPT = (
    "You are auditing a multi-agent investment debate transcript. The debaters "
    "were given a quantitative model signal to engage with. Judge ONLY whether "
    "they used it — not whether the decision was correct.\n\n"
    "Output STRICT JSON only. Keys (all required):\n"
    "- used_quant_signal: bool — did any debater explicitly reference the "
    "quant model's composite, factors, confidence, or signal?\n"
    "- engaged_divergence: bool — if the signal flagged a 20d-vs-60d horizon "
    "divergence, did the debate address it? (true if no divergence was present)\n"
    "- score: float in [0,1] — overall quality of engagement with the signal.\n\n"
    "TRANSCRIPT:\n{transcript}"
)


def build_judge_prompt(transcript: str, *, max_chars: int = 12_000) -> str:
    return _JUDGE_PROMPT.format(transcript=(transcript or "")[:max_chars])


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the judge's JSON; tolerant of code fences. Returns a status block."""
    import json

    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").replace("json", "", 1).strip()
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {"status": "parse_error", "raw": (text or "")[:500]}
    return {
        "status": "ok",
        "used_quant_signal": bool(data.get("used_quant_signal")),
        "engaged_divergence": bool(data.get("engaged_divergence")),
        "score": float(data["score"]) if isinstance(data.get("score"), (int, float)) else None,
    }


def aggregate_judge(verdicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [v for v in verdicts if v.get("status") == "ok"]
    if not ok:
        return {"n": 0, "used_rate": None, "divergence_rate": None, "mean_score": None}
    return {
        "n": len(ok),
        "used_rate": _rate([1 if v["used_quant_signal"] else 0 for v in ok]),
        "divergence_rate": _rate([1 if v["engaged_divergence"] else 0 for v in ok]),
        "mean_score": _mean([v["score"] for v in ok if v["score"] is not None]),
    }


def render_debate_eval_markdown(result: dict[str, Any]) -> str:
    """Render the ablation comparison as a markdown report."""
    g, c, q = result["grounded"], result["control"], result["quant"]
    og, oc = result["override_grounded"], result["override_control"]
    lines = [
        "# Debate-grounding ablation",
        "",
        f"Pairs evaluated: **{result['n_pairs']}**. Compares the multi-agent "
        "decision WITH the Quant Analyst node (grounded) vs WITHOUT (control), "
        "and vs the quant model alone, against realized forward returns.",
        "",
        "## Decision quality",
        "",
        "| source | n | hit-rate | mean return-capture |",
        "|---|---:|---:|---:|",
        f"| grounded debate | {g['n']} | {_pct(g['hit_rate'])} | {_sig(g['mean_return_capture'])} |",
        f"| control debate | {c['n']} | {_pct(c['hit_rate'])} | {_sig(c['mean_return_capture'])} |",
        f"| quant model alone | {q['n']} | {_pct(q['hit_rate'])} | {_sig(q['mean_return_capture'])} |",
        "",
        "## Agreement with the quant signal",
        "",
        f"- grounded vs quant: {_pct(result['agreement_grounded_vs_quant'])}",
        f"- control vs quant: {_pct(result['agreement_control_vs_quant'])}",
        "",
        "## Override-correctness (the value test)",
        "",
        "On names where the debate DISAGREES with the quant model, who was "
        "right against realized returns? If the grounded debate's overrides "
        "beat the model, grounding adds value beyond echoing the signal.",
        "",
        "| | disagreements | debate win-rate | quant win-rate |",
        "|---|---:|---:|---:|",
        f"| grounded | {og['n_disagreements']} | {_pct(og['debate_win_rate'])} | {_pct(og['quant_win_rate'])} |",
        f"| control | {oc['n_disagreements']} | {_pct(oc['debate_win_rate'])} | {_pct(oc['quant_win_rate'])} |",
    ]
    judge = result.get("judge")
    if judge and judge.get("n"):
        lines += [
            "",
            "## Process check (LLM judge on grounded transcripts)",
            "",
            f"- transcripts judged: {judge['n']}",
            f"- referenced the quant signal: {_pct(judge['used_rate'])}",
            f"- engaged the horizon divergence: {_pct(judge['divergence_rate'])}",
            f"- mean engagement score: {_sig(judge['mean_score'])}",
        ]
    return "\n".join(lines)
