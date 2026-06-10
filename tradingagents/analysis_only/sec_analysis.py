"""SEC filing document analysis (FinRobot-style document agent).

Given the extracted sections of a company's latest 10-K / 10-Q / 8-K, send a
frozen prompt to an LLM at temperature=0 and ask for a structured digest:
a plain-language summary, key risks, MD&A highlights, an overall tone, and
notable changes. Output is validated against ``FilingAnalysisOutput`` and
returned as a JSON-serialisable block with the model id and a prompt hash, so
the call is reproducible — exactly the contract used by ``llm_critic``.

``analyze_filing`` is the end-to-end orchestrator: it resolves the latest
point-in-time filing, downloads and section-splits the document, then runs the
LLM. It is reused by both the analysis pipeline and the LangGraph agent tool.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEC_ANALYSIS_PROMPT_VERSION = "v1.0"

# Frozen prompt. Edit only by bumping SEC_ANALYSIS_PROMPT_VERSION; any change
# invalidates historical cached analyses.
SEC_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a buy-side equity analyst reading a company's SEC filing. You are "
    "given the extracted text of the most relevant sections (risk factors, "
    "management's discussion & analysis, and/or the filing body). Produce a "
    "terse, factual digest for a portfolio manager. Do not speculate beyond "
    "the text; do not give investment advice.\n\n"
    "Output STRICT JSON only. No prose. No markdown. No code fences. "
    "Keys (all required):\n"
    "- summary: string, 2-4 sentences describing what this filing says and "
    "why it matters.\n"
    "- key_risks: list[str], 0-6 short phrases naming the most material risks "
    "disclosed. Prefer specific, filing-grounded risks over boilerplate.\n"
    "- mdna_highlights: list[str], 0-6 short phrases capturing the most "
    "important points from management's discussion (revenue/margin drivers, "
    "guidance, liquidity, segment performance).\n"
    "- tone: one of \"positive\", \"neutral\", \"cautious\", \"negative\" — the "
    "overall tone of management's narrative.\n"
    "- notable_changes: list[str], 0-6 short phrases on anything that reads as "
    "a change from prior periods (new risk language, accounting changes, "
    "restructuring, litigation, leadership). Empty list if none stand out.\n\n"
    "FILING_METADATA:\n{meta_json}\n\n"
    "FILING_SECTIONS:\n{sections_text}"
)

# Bound the per-section text fed to the model to keep token cost predictable.
_MAX_SECTION_CHARS = 6_000


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def build_analysis_prompt(
    sections: dict[str, str],
    meta: dict[str, Any],
    *,
    max_section_chars: int = _MAX_SECTION_CHARS,
) -> str:
    """Render the frozen prompt from extracted sections and filing metadata."""
    blocks = []
    for name, text in sections.items():
        label = name.replace("_", " ").upper()
        blocks.append(f"--- {label} ---\n{(text or '')[:max_section_chars]}")
    sections_text = "\n\n".join(blocks) if blocks else "(no sections extracted)"
    return SEC_ANALYSIS_PROMPT_TEMPLATE.format(
        meta_json=json.dumps(meta, sort_keys=True, default=str),
        sections_text=sections_text,
    )


def _build_analysis_schema():
    """Local import so the module loads without pydantic available."""
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    class FilingAnalysisOutput(BaseModel):
        model_config = ConfigDict(extra="ignore")

        summary: str = Field(default="")
        key_risks: list[str] = Field(default_factory=list, max_length=6)
        mdna_highlights: list[str] = Field(default_factory=list, max_length=6)
        tone: str = Field(default="neutral")
        notable_changes: list[str] = Field(default_factory=list, max_length=6)

        @field_validator("tone", mode="before")
        @classmethod
        def _normalize_tone(cls, value):
            allowed = {"positive", "neutral", "cautious", "negative"}
            v = str(value or "neutral").strip().lower()
            return v if v in allowed else "neutral"

    return FilingAnalysisOutput


def _strip_code_fences(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return s


def run_filing_analysis(
    sections: dict[str, str],
    meta: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run the filing-analysis prompt and return a status block.

    The block mirrors ``llm_critic.run_critic``:
      - status: "ok" | "no_sections" | "llm_init_error" | "llm_call_error"
        | "llm_json_parse_error" | "llm_schema_error"
      - provider, model, prompt_version, prompt_hash
      - output: validated FilingAnalysisOutput dict (only on status="ok")
      - raw_response: trimmed raw text (only on schema/parse-error paths)
    """
    prompt = build_analysis_prompt(sections, meta)
    block: dict[str, Any] = {
        "status": "unknown",
        "provider": provider,
        "model": model,
        "prompt_version": SEC_ANALYSIS_PROMPT_VERSION,
        "prompt_hash": _prompt_hash(prompt),
        "temperature": temperature,
    }
    if not sections:
        block["status"] = "no_sections"
        return block

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
        schema = _build_analysis_schema()
        validated = schema(**parsed)
        block["status"] = "ok"
        block["output"] = validated.model_dump()
    except Exception as exc:
        block["status"] = "llm_schema_error"
        block["error"] = str(exc)
        block["raw_response"] = cleaned[:2000]
    return block


def analyze_filing(
    symbol: str,
    as_of_date: str | None,
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    sec_provider: Any | None = None,
    max_section_chars: int = 20_000,
) -> dict[str, Any]:
    """End-to-end: locate the latest PIT filing, fetch + section it, analyze.

    Returns a JSON-serialisable block:
      - status: "ok" | "unavailable" | "fetch_error" | "no_document" | <llm status>
      - filing: {form, accession, filing_date, cik, url} when a filing was found
      - sections_found: list[str]
      - analysis: the ``run_filing_analysis`` block (when sections were found)

    Network and LLM failures are returned as statuses rather than raised, so a
    caller can attach the block to a report without try/except.
    """
    from tradingagents.analysis_only.providers import (
        SECFetchError,
        SECFilingsProvider,
        extract_filing_sections,
    )

    sec = sec_provider or SECFilingsProvider()
    block: dict[str, Any] = {"status": "unknown", "symbol": symbol.upper()}

    try:
        latest = sec.get_latest_filing(symbol, as_of_date=as_of_date)
    except SECFetchError as exc:
        block["status"] = "fetch_error"
        block["error"] = str(exc)
        return block

    if not latest:
        block["status"] = "unavailable"
        return block

    block["filing"] = {
        "form": latest.get("form"),
        "accession": latest.get("accession"),
        "filing_date": latest.get("filing_date"),
        "cik": latest.get("cik"),
        "url": sec.filing_document_url(latest),
    }

    try:
        document = sec.fetch_filing_document(latest)
    except SECFetchError as exc:
        block["status"] = "fetch_error"
        block["error"] = str(exc)
        return block

    if not document:
        block["status"] = "no_document"
        return block

    sections = extract_filing_sections(
        document, latest.get("form"), max_section_chars=max_section_chars
    )
    block["sections_found"] = list(sections.keys())

    meta = {
        "symbol": symbol.upper(),
        "form": latest.get("form"),
        "filing_date": latest.get("filing_date"),
        "as_of_date": as_of_date,
    }
    analysis = run_filing_analysis(
        sections, meta, provider=provider, model=model, base_url=base_url
    )
    block["analysis"] = analysis
    block["status"] = analysis.get("status", "unknown")
    return block
