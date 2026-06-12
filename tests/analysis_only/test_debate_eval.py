"""Tests for the debate-grounding ablation scoring core + orchestration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tradingagents.analysis_only.debate_eval import (
    aggregate_judge,
    build_judge_prompt,
    decision_hit,
    parse_judge_response,
    rating_to_direction,
    render_debate_eval_markdown,
    return_capture,
    score_pairs,
)


# ---------- primitives ----------

@pytest.mark.parametrize("rating,expected", [
    ("Buy", "bullish"), ("Overweight", "bullish"), ("Hold", "neutral"),
    ("Underweight", "bearish"), ("Sell", "bearish"), ("", "neutral"), (None, "neutral"),
])
def test_rating_to_direction(rating, expected):
    assert rating_to_direction(rating) == expected


def test_decision_hit():
    assert decision_hit("bullish", 0.05) == 1
    assert decision_hit("bullish", -0.05) == 0
    assert decision_hit("bearish", -0.05) == 1
    assert decision_hit("neutral", 0.01) == 1
    assert decision_hit("neutral", 0.05) == 0
    assert decision_hit("bullish", None) is None


def test_return_capture():
    assert return_capture("bullish", 0.05) == 0.05
    assert return_capture("bearish", -0.05) == 0.05  # correctly short a loser
    assert return_capture("neutral", 0.05) == 0.0
    assert return_capture("bullish", None) is None


# ---------- aggregation ----------

def _pair(sym, ret, quant, grounded, control):
    return {
        "symbol": sym, "as_of_date": "2025-01-01", "realized_return": ret,
        "quant_direction": quant, "grounded_rating": grounded,
        "control_rating": control,
    }


def test_score_pairs_hit_and_capture():
    pairs = [
        # grounded right (bull, +), control wrong (sell on a winner), quant right
        _pair("A", 0.10, "bullish", "Buy", "Sell"),
        # grounded wrong (bull on loser), control right (sell), quant wrong
        _pair("B", -0.10, "bullish", "Buy", "Sell"),
    ]
    res = score_pairs(pairs)
    assert res["n_pairs"] == 2
    assert res["grounded"]["hit_rate"] == 0.5   # 1 of 2
    assert res["control"]["hit_rate"] == 0.5
    assert res["quant"]["hit_rate"] == 0.5
    # grounded capture: +0.10 then -0.10 -> mean 0.0
    assert res["grounded"]["mean_return_capture"] == 0.0


def test_override_correctness_rewards_good_overrides():
    # On both names the debate (grounded) OVERRIDES the quant signal and is
    # right; quant is wrong. Grounding's overrides add value.
    pairs = [
        _pair("A", -0.08, "bullish", "Sell", "Buy"),   # debate bearish & right
        _pair("B", 0.08, "bearish", "Buy", "Sell"),    # debate bullish & right
    ]
    res = score_pairs(pairs)
    og = res["override_grounded"]
    assert og["n_disagreements"] == 2
    assert og["debate_win_rate"] == 1.0
    assert og["quant_win_rate"] == 0.0


def test_agreement_rates():
    pairs = [
        _pair("A", 0.05, "bullish", "Buy", "Buy"),     # grounded agrees, control agrees
        _pair("B", 0.05, "bullish", "Sell", "Buy"),    # grounded disagrees, control agrees
    ]
    res = score_pairs(pairs)
    assert res["agreement_grounded_vs_quant"] == 0.5
    assert res["agreement_control_vs_quant"] == 1.0


def test_render_markdown_contains_sections():
    res = score_pairs([_pair("A", 0.05, "bullish", "Buy", "Hold")])
    md = render_debate_eval_markdown(res)
    assert "Debate-grounding ablation" in md
    assert "Override-correctness" in md
    assert "Decision quality" in md


# ---------- judge (pure prompt/parse) ----------

def test_judge_prompt_truncates():
    p = build_judge_prompt("x" * 50000, max_chars=100)
    assert "x" * 100 in p and "x" * 101 not in p


def test_parse_judge_ok_and_fenced():
    v = parse_judge_response('```json\n{"used_quant_signal": true, "engaged_divergence": false, "score": 0.8}\n```')
    assert v["status"] == "ok"
    assert v["used_quant_signal"] is True
    assert v["engaged_divergence"] is False
    assert v["score"] == 0.8


def test_parse_judge_bad_json():
    assert parse_judge_response("not json")["status"] == "parse_error"


def test_aggregate_judge():
    verdicts = [
        {"status": "ok", "used_quant_signal": True, "engaged_divergence": True, "score": 0.9},
        {"status": "ok", "used_quant_signal": False, "engaged_divergence": True, "score": 0.3},
        {"status": "parse_error"},
    ]
    agg = aggregate_judge(verdicts)
    assert agg["n"] == 2
    assert agg["used_rate"] == 0.5
    assert agg["divergence_rate"] == 1.0
    assert agg["mean_score"] == 0.6


# ---------- orchestration with a mocked debate runner ----------

def test_run_ablation_with_mock_runner():
    from scripts.eval_debate_grounding import run_ablation

    class _Rec:
        def __init__(self, sym, ret, direction):
            self.symbol = sym
            self.as_of_date = "2025-03-01"
            self.forward_returns = {"ret_60d": ret}
            self.direction = direction
            self.composite_score = 0.3

    records = [_Rec("AAA", 0.10, "bullish"), _Rec("BBB", -0.10, "bullish")]

    def fake_debate(symbol, date, grounded):
        # Grounded debate correctly turns bearish on the loser BBB; control
        # blindly follows bullish on both.
        if grounded and symbol == "BBB":
            return {"rating": "Sell", "transcript": "used the quant signal"}
        return {"rating": "Buy", "transcript": "ignored it"}

    judged = []

    def fake_judge(transcript):
        used = "used the quant signal" in transcript
        judged.append(used)
        return {"status": "ok", "used_quant_signal": used,
                "engaged_divergence": True, "score": 1.0 if used else 0.0}

    res = run_ablation(records, fake_debate, horizon="ret_60d", judge_fn=fake_judge)
    assert res["n_pairs"] == 2
    # grounded: Buy on winner (hit), Sell on loser (hit) -> 100%
    assert res["grounded"]["hit_rate"] == 1.0
    # control: Buy on both -> 50%
    assert res["control"]["hit_rate"] == 0.5
    # grounded override on BBB was correct.
    assert res["override_grounded"]["debate_win_rate"] == 1.0
    assert res["judge"]["n"] == 2
