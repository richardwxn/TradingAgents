"""LangGraph tool: SEC filing document analysis.

Exposes the FinRobot-style document agent to the fundamentals analyst. The
tool resolves the company's latest point-in-time 10-K/10-Q/8-K, downloads it,
extracts the risk-factor and MD&A sections, and asks an LLM for a structured
digest. The LLM provider/model are read from the active config so the tool
follows the same backbone the rest of the run uses.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config


@tool
def get_sec_filing_analysis(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Analyze the company's most recent SEC filing (10-K, 10-Q, or 8-K) as of
    ``curr_date``. Downloads the actual filing document from SEC EDGAR,
    extracts the Risk Factors and Management's Discussion & Analysis sections,
    and returns a structured digest: a summary, key risks, MD&A highlights,
    management tone, and notable changes from prior periods.

    Use this for qualitative, filing-grounded context that raw financial
    statements do not capture (disclosed risks, guidance, litigation, etc.).

    Args:
        ticker (str): Ticker symbol of the company.
        curr_date (str): Current date you are trading at, yyyy-mm-dd. Only
            filings made public on or before this date are considered.

    Returns:
        str: A JSON string with the filing metadata and the analysis digest.
    """
    from tradingagents.analysis_only.sec_analysis import analyze_filing

    config = get_config()
    block = analyze_filing(
        ticker,
        curr_date,
        provider=config.get("llm_provider", "openai"),
        model=config.get("quick_think_llm", "gpt-5.4-mini"),
        base_url=config.get("backend_url"),
    )
    return json.dumps(block, indent=2, default=str)
