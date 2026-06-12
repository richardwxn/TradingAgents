"""Ablation A/B: does grounding the debate in the quant signal improve it?

Runs the multi-agent graph twice per (ticker, date) — WITH the Quant Analyst
node (grounded) and WITHOUT (control) — over a corpus slice with known forward
returns, then scores both against the quant model alone:

  - decision hit-rate + mean return-capture (grounded vs control vs quant)
  - agreement with the quant signal
  - OVERRIDE-CORRECTNESS: on names where the debate disagrees with the model,
    who was right? (the test of whether grounding adds value vs just echoing)
  - optional LLM-judge process score: did the grounded debate actually use the
    signal?

The scoring core (`tradingagents.analysis_only.debate_eval`) is pure and unit-
tested; this shell wires the real graph. Running the graph needs a working LLM
key and is expensive, so default to a small `--max-pairs` slice.

Example:
    python scripts/eval_debate_grounding.py \
        --reports-glob "reports/analysis_mvp/*.json" \
        --max-pairs 20 --horizon ret_60d --judge \
        --output backtest/results/debate_grounding_ablation.md
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import load_records  # noqa: E402
from tradingagents.agents.utils.rating import parse_rating  # noqa: E402
from tradingagents.analysis_only.debate_eval import (  # noqa: E402
    aggregate_judge,
    build_judge_prompt,
    parse_judge_response,
    render_debate_eval_markdown,
    score_pairs,
)


def run_ablation(records, run_debate, *, horizon="ret_60d", judge_fn=None):
    """Core orchestration (testable).

    ``run_debate(symbol, date, grounded) -> {"rating": str, "transcript": str}``
    is injected so tests can avoid running a real debate. ``judge_fn(transcript)
    -> verdict dict`` is optional. Returns the ``score_pairs`` result, with a
    ``judge`` block when ``judge_fn`` is provided.
    """
    pairs = []
    judge_verdicts = []
    for r in records:
        ret = r.forward_returns.get(horizon)
        symbol = getattr(r, "symbol", None)
        date = getattr(r, "as_of_date", None)
        if not symbol or not date:
            continue
        grounded = run_debate(symbol, date, True)
        control = run_debate(symbol, date, False)
        pairs.append({
            "symbol": symbol,
            "as_of_date": date,
            "realized_return": ret,
            "quant_direction": str(getattr(r, "direction", "") or "neutral").lower(),
            "quant_composite": getattr(r, "composite_score", None),
            "grounded_rating": grounded.get("rating"),
            "control_rating": control.get("rating"),
        })
        if judge_fn is not None:
            judge_verdicts.append(judge_fn(grounded.get("transcript", "")))

    result = score_pairs(pairs)
    if judge_fn is not None:
        result["judge"] = aggregate_judge(judge_verdicts)
    return result


def _make_graph_runner(base_config):
    """Build grounded + control graphs once and reuse across pairs."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graphs = {}
    for grounded in (True, False):
        cfg = dict(base_config)
        cfg["enable_quant_analyst"] = grounded
        graphs[grounded] = TradingAgentsGraph(config=cfg)

    def run(symbol, date, grounded):
        graph = graphs[bool(grounded)]
        final_state, _decision = graph.propagate(symbol, str(date))
        rating = parse_rating(final_state.get("final_trade_decision", "") or "")
        debate = final_state.get("investment_debate_state") or {}
        transcript = (final_state.get("quant_report", "") + "\n\n"
                      + (debate.get("history", "") or ""))
        return {"rating": rating, "transcript": transcript}

    return run


def _make_judge(provider, model, base_url=None):
    from tradingagents.llm_clients.factory import create_llm_client

    client = create_llm_client(
        provider=provider, model=model, base_url=base_url, temperature=0.0,
    )
    llm = client.get_llm()

    def judge(transcript):
        try:
            resp = llm.invoke(build_judge_prompt(transcript))
            return parse_judge_response(str(getattr(resp, "content", resp)))
        except Exception as exc:
            return {"status": "judge_error", "error": str(exc)}

    return judge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-glob", default="reports/analysis_mvp/*.json")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--horizon", default="ret_60d")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--judge", action="store_true",
                        help="Run the LLM process-score judge on grounded transcripts.")
    parser.add_argument("--output", default="backtest/results/debate_grounding_ablation.md")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f"[fail] no reports matched {args.reports_glob}", file=sys.stderr)
        return 2
    print(f"Loading {len(paths)} reports + forward returns...")
    records = load_records(
        paths, horizons=args.horizons, capture_factor_scores=False,
        capture_market_context=False, benchmark_symbol=None,
    )
    # Evenly sample to the requested cap so the slice spans the corpus.
    if args.max_pairs and len(records) > args.max_pairs:
        step = len(records) / args.max_pairs
        records = [records[int(i * step)] for i in range(args.max_pairs)]
    print(f"Running ablation over {len(records)} (ticker, date) pairs "
          f"(2 debates each) at horizon {args.horizon}...")

    from tradingagents.default_config import DEFAULT_CONFIG

    runner = _make_graph_runner(DEFAULT_CONFIG.copy())
    judge_fn = None
    if args.judge:
        judge_fn = _make_judge(
            DEFAULT_CONFIG.get("llm_provider", "openai"),
            DEFAULT_CONFIG.get("quick_think_llm", "gpt-5.4-mini"),
            DEFAULT_CONFIG.get("backend_url"),
        )

    result = run_ablation(
        records, runner, horizon=args.horizon, judge_fn=judge_fn,
    )
    md = render_debate_eval_markdown(result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print("\n" + md)
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
