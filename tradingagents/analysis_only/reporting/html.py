"""Render an analysis-only report as a self-contained HTML equity-research doc.

Reuses the existing Markdown renderers for the prose (so the HTML never drifts
from the Markdown), converts that to HTML via the ``markdown`` library, and
prepends a chart gallery built from the same payload. Output is a single file
with inline CSS and base64-embedded PNGs — no external assets — so it renders
offline, emails cleanly, and prints to PDF from any browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .charts import build_all_charts
from .markdown import render_equity_research_markdown, render_markdown

# Human-friendly captions for the chart registry keys.
_CHART_TITLES = {
    "factor_scorecard": "Factor scorecard",
    "pillar_scores": "Pillar scores",
    "price_target": "Price-target scenarios",
    "forecast_fan": "Price range forecast",
}

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.55; color: #1f2328; background: #ffffff;
  max-width: 960px; margin: 0 auto; padding: 2.5rem 1.5rem;
}
h1 { font-size: 1.9rem; border-bottom: 2px solid #d0d7de; padding-bottom: .4rem; }
h2 { font-size: 1.35rem; margin-top: 2rem; border-bottom: 1px solid #eaeef2; padding-bottom: .3rem; }
h3 { font-size: 1.08rem; margin-top: 1.4rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }
th, td { border: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left; }
th { background: #f6f8fa; }
code { background: #f6f8fa; padding: .1rem .3rem; border-radius: 4px; font-size: .9em; }
blockquote { border-left: 3px solid #d0d7de; margin: 1rem 0; padding: .2rem 1rem; color: #57606a; }
a { color: #0969da; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1.2rem; margin: 1.5rem 0; }
.chart { border: 1px solid #eaeef2; border-radius: 8px; padding: .8rem; background: #fff; }
.chart h3 { margin: 0 0 .5rem; font-size: .95rem; color: #57606a; }
.chart img { width: 100%; height: auto; display: block; }
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eaeef2; color: #8c959f; font-size: .82rem; }
@media print { body { padding: 0; } .chart { break-inside: avoid; } }
"""


def _charts_html(charts: dict[str, str]) -> str:
    if not charts:
        return ""
    cards = []
    for name, uri in charts.items():
        title = _CHART_TITLES.get(name, name.replace("_", " ").title())
        cards.append(
            f'<figure class="chart"><h3>{title}</h3>'
            f'<img alt="{title}" src="{uri}"></figure>'
        )
    return '<section class="charts">' + "".join(cards) + "</section>"


def _markdown_to_html(md_text: str) -> str:
    import markdown as _md

    return _md.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


def render_html(
    payload: dict[str, Any],
    *,
    equity_research: bool = True,
    embed_charts: bool = True,
) -> str:
    """Render ``payload`` to a standalone HTML document string.

    Args:
        payload: A parsed analysis-only report JSON.
        equity_research: Use the FinRobot-style equity-research narrative
            layout; when ``False`` use the full technical Markdown report.
        embed_charts: Build and embed the chart gallery.
    """
    symbol = payload.get("symbol", "Report")
    as_of = payload.get("as_of_date", "")
    md_text = (
        render_equity_research_markdown(payload)
        if equity_research
        else render_markdown(payload)
    )
    body_html = _markdown_to_html(md_text)
    charts_html = _charts_html(build_all_charts(payload)) if embed_charts else ""
    generated = payload.get("generated_at_utc", "")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{symbol} — Equity Research ({as_of})</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"{charts_html}\n"
        f'<article class="report">{body_html}</article>\n'
        f'<div class="footer">Generated {generated} · TradingAgents analysis-only report. '
        "For research purposes only — not investment advice.</div>\n"
        "</body>\n</html>\n"
    )


def render_html_file(
    json_path: str | Path,
    output_path: str | Path | None = None,
    *,
    equity_research: bool = True,
    embed_charts: bool = True,
) -> Path:
    """Render ``json_path`` to a sibling ``.html`` (or ``output_path``)."""
    json_path = Path(json_path)
    payload = json.loads(json_path.read_text())
    html = render_html(
        payload, equity_research=equity_research, embed_charts=embed_charts
    )
    target = Path(output_path) if output_path else json_path.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    return target
