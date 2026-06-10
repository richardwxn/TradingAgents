"""Golden-file regression tests for SEC section extraction.

These run against the stripped text of real filings (committed, gzipped under
tests/fixtures/sec/) so the heuristic section parser cannot silently regress
on the formatting quirks of actual EDGAR documents — table-of-contents rows
with dot leaders and page numbers, Part I/Part II "Item N" collisions in
10-Qs, and in-prose "Risk Factors" boilerplate mentions.

To refresh a fixture (e.g. after a filing format change), re-run the snippet
in the module docstring of conftest-adjacent tooling; the assertions below key
on stable content phrases, not dates.
"""

import gzip
from pathlib import Path

import pytest

from tradingagents.analysis_only.providers import extract_filing_sections

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "sec"


def _load(name: str) -> str:
    with gzip.open(_FIXTURES / name, "rt", encoding="utf-8") as fh:
        return fh.read()


def _looks_like_toc_fragment(section: str) -> bool:
    """A real section is prose; a TOC capture is a short page-numbered stub."""
    head = section[:80].lower()
    return len(section) < 400 or head.rstrip().endswith(tuple("0123456789"))


# --------------------------------------------------------------------------
# Apple 10-Q (filed 2026-05-01)
# --------------------------------------------------------------------------

def test_aapl_10q_extracts_both_sections():
    sections = extract_filing_sections(_load("AAPL_10Q.txt.gz"), "10-Q")
    assert set(sections) == {"mdna", "risk_factors"}


def test_aapl_10q_mdna_is_real_section():
    sections = extract_filing_sections(_load("AAPL_10Q.txt.gz"), "10-Q")
    mdna = sections["mdna"]
    assert mdna.startswith("Management's Discussion and Analysis")
    assert len(mdna) > 5000
    assert not _looks_like_toc_fragment(mdna)


def test_aapl_10q_risk_factors_is_real_not_boilerplate():
    # The real Item 1A — not the forward-looking-statements mention of
    # "Risk Factors" inside MD&A boilerplate (which has no Item 1A prefix).
    sections = extract_filing_sections(_load("AAPL_10Q.txt.gz"), "10-Q")
    rf = sections["risk_factors"]
    assert rf.startswith("Risk Factors")
    assert "materially and adversely affected" in rf
    assert not _looks_like_toc_fragment(rf)


# --------------------------------------------------------------------------
# Costco 10-K (filed 2025-10-08)
# --------------------------------------------------------------------------

def test_cost_10k_extracts_both_sections():
    sections = extract_filing_sections(_load("COST_10K.txt.gz"), "10-K")
    assert set(sections) == {"mdna", "risk_factors"}


def test_cost_10k_risk_factors_is_real_section():
    sections = extract_filing_sections(_load("COST_10K.txt.gz"), "10-K")
    rf = sections["risk_factors"]
    assert rf.startswith("Risk Factors")
    assert "The risks described below" in rf
    assert len(rf) > 5000
    assert not _looks_like_toc_fragment(rf)


def test_cost_10k_mdna_is_real_section():
    sections = extract_filing_sections(_load("COST_10K.txt.gz"), "10-K")
    mdna = sections["mdna"]
    assert mdna.startswith("Management's Discussion and Analysis")
    assert not _looks_like_toc_fragment(mdna)


# --------------------------------------------------------------------------
# Truncation contract holds on real text
# --------------------------------------------------------------------------

def test_sections_respect_max_section_chars_on_real_text():
    sections = extract_filing_sections(
        _load("COST_10K.txt.gz"), "10-K", max_section_chars=3000
    )
    assert all(len(v) <= 3000 for v in sections.values())
