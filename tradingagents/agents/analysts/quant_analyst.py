"""Quant Analyst node — grounds the multi-agent debate in the quant model.

Unlike the LLM tool-calling analysts, this node does not call an LLM. It loads
the analysis-only pipeline's report for (ticker, trade_date) — the validated
factor composite, per-horizon (5d/20d/60d) composites, IC-weighted factors,
options flow, calibrated confidence, and (when present) the SEC filing digest —
and renders it into a `quant_report` the bull/bear researchers and managers read.

This turns the debate from argumentation over free-text analyst reports into a
debate that must engage with the quantitative signal and, in particular, with
the 20d-vs-60d horizon divergence the model surfaces.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

_DEFAULT_REPORTS_DIR = "reports/analysis_mvp"
_FILENAME_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})$")


def _load_latest_report(
    reports_dir: str, ticker: str, trade_date: str
) -> dict[str, Any] | None:
    """Load the latest analysis-only report for ``ticker`` on/before ``trade_date``.

    Reports are named ``TICKER_YYYY-MM-DD.json``. Point-in-time: filings dated
    after ``trade_date`` are ignored so a back-dated debate never sees a future
    signal.
    """
    pattern = str(Path(reports_dir) / f"{ticker.upper()}_*.json")
    best_path: str | None = None
    best_date: str | None = None
    for path in glob.glob(pattern):
        m = _FILENAME_RE.search(Path(path).stem)
        if not m:
            continue
        d = m.group(1)
        if d <= trade_date and (best_date is None or d > best_date):
            best_date, best_path = d, path
    if best_path is None:
        return None
    try:
        return json.loads(Path(best_path).read_text())
    except Exception:
        return None


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _direction_of(composite: Any, threshold: float = 0.15) -> str:
    try:
        c = float(composite)
    except (TypeError, ValueError):
        return "neutral"
    if c >= threshold:
        return "bullish"
    if c <= -threshold:
        return "bearish"
    return "neutral"


def format_quant_report(payload: dict[str, Any]) -> str:
    """Render the debate-relevant slice of an analysis-only report as text."""
    if not payload:
        return "Quant model signal unavailable for this ticker/date."

    kf = payload.get("key_features") or {}
    ms = kf.get("model_scoring") or {}
    symbol = payload.get("symbol", "?")
    as_of = payload.get("as_of_date", "?")
    direction = (payload.get("direction") or "neutral").lower()
    confidence = payload.get("confidence")
    composite = ms.get("composite_score")

    lines = [
        f"QUANTITATIVE MODEL SIGNAL — {symbol} (as of {as_of})",
        "",
        "This is the output of a backtested, walk-forward-validated factor "
        "model. Treat it as evidence to engage with, not gospel.",
        "",
        f"- Primary direction: **{direction}**",
        f"- Composite score (60d-anchored, [-1,1]): {_fmt(composite)}",
        f"- Calibrated confidence: {_fmt(confidence, 2)}",
    ]

    # Per-horizon composites + the 20d/60d divergence callout.
    ph = ms.get("per_horizon_composites") or {}
    if ph:
        c20 = (ph.get("ret_20d") or {}).get("composite_score")
        c60 = (ph.get("ret_60d") or {}).get("composite_score")
        c5 = (ph.get("ret_5d") or {}).get("composite_score")
        lines += [
            "",
            "Per-horizon composites:",
            f"- 5d: {_fmt(c5)}  ·  20d: {_fmt(c20)}  ·  60d: {_fmt(c60)}",
        ]
        d20, d60 = _direction_of(c20), _direction_of(c60)
        if c20 is not None and c60 is not None and d20 != d60:
            lines.append(
                f"- ⚠ HORIZON DIVERGENCE: the 20d view is **{d20}** while the "
                f"60d view is **{d60}**. The debate should explicitly resolve "
                f"this — is the near-term setup a buying opportunity against a "
                f"strong longer-term trend, or an early warning the 60d call is "
                f"about to break?"
            )

    # Top weighted factors (the model's actual drivers).
    factors = [
        f for f in (ms.get("factor_scores") or [])
        if f.get("data_available") and f.get("weighted_score") is not None
    ]
    factors.sort(key=lambda f: abs(f.get("weighted_score") or 0.0), reverse=True)
    if factors:
        lines += ["", "Top model factors (weighted score · rationale):"]
        for f in factors[:8]:
            lines.append(
                f"- {f.get('factor')} [{f.get('pillar')}]: "
                f"{_fmt(f.get('weighted_score'), 4)} — "
                f"{str(f.get('rationale') or '').strip()}"
            )

    pillars = ms.get("pillar_scores") or {}
    if pillars:
        lines += [
            "",
            "Pillar scores: "
            + " · ".join(f"{k}: {_fmt(v)}" for k, v in pillars.items()),
        ]

    # Options flow.
    of = kf.get("options_flow") or {}
    if of and of.get("scan_status") not in (None, "unavailable"):
        lines += [
            "",
            "Options flow: "
            f"unusual={of.get('unusual_count', 0)}, "
            f"net call/put notional={_fmt(of.get('net_call_put_notional'), 0)}, "
            f"IV rank={_fmt(of.get('iv_rank'), 2)}, "
            f"IV skew={_fmt(of.get('iv_skew'), 3)}",
        ]

    # SEC filing digest (if the report carried filing_analysis).
    fa = ((kf.get("filings_context") or {}).get("filing_analysis") or {})
    out = (fa.get("analysis") or {}).get("output") or {}
    if out:
        filing = fa.get("filing") or {}
        lines += [
            "",
            f"Latest SEC filing ({filing.get('form', '?')} "
            f"{filing.get('filing_date', '')}) — tone: {out.get('tone', '?')}",
        ]
        if out.get("summary"):
            lines.append(f"- {out['summary']}")
        for risk in (out.get("key_risks") or [])[:3]:
            lines.append(f"- Risk: {risk}")

    risk_flags = payload.get("risk_flags") or []
    if risk_flags:
        lines += ["", "Model risk flags: " + "; ".join(map(str, risk_flags[:6]))]

    dq = payload.get("data_quality") or {}
    pit = dq.get("pit_warnings") or []
    if pit:
        lines.append("")
        lines.append(
            f"Data caveats: {len(pit)} point-in-time warning(s) — weight the "
            "signal accordingly."
        )

    return "\n".join(lines)


def create_quant_analyst(config: dict[str, Any] | None = None):
    """Build the Quant Analyst node.

    ``config['analysis_only_reports_dir']`` (default ``reports/analysis_mvp``)
    is the directory of analysis-only report JSONs to ground the debate on.
    """
    reports_dir = (config or {}).get(
        "analysis_only_reports_dir", _DEFAULT_REPORTS_DIR
    )

    def quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = str(state["trade_date"])
        payload = _load_latest_report(reports_dir, ticker, trade_date)
        report = format_quant_report(payload) if payload else (
            "Quant model signal unavailable for this ticker/date "
            f"(no analysis-only report found under {reports_dir})."
        )
        return {"quant_report": report}

    return quant_analyst_node
