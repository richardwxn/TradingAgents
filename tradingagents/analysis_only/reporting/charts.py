"""Chart rendering for equity-research reports.

Pure functions that turn an analysis-report payload into base64-encoded PNGs
(``data:`` URI ready). matplotlib runs on the headless ``Agg`` backend so this
works offline and on a server with no display. Every function returns ``None``
when the payload lacks the data it needs, so the HTML report degrades cleanly
on partial reports rather than raising.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# Sign-based palette shared across charts.
_BULL = "#1a7f37"
_BEAR = "#cf222e"
_NEUTRAL = "#8c959f"
_ACCENT = "#0969da"


def _fig_to_data_uri(fig) -> str:
    """Serialise a matplotlib figure to a base64 ``data:`` PNG URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _color_for(value: float) -> str:
    if value > 0:
        return _BULL
    if value < 0:
        return _BEAR
    return _NEUTRAL


def factor_scorecard_chart(payload: dict[str, Any], *, top_n: int = 12) -> str | None:
    """Horizontal bar of the top weighted factor scores (by absolute weight)."""
    scoring = (payload.get("key_features") or {}).get("model_scoring") or {}
    factors = [
        f for f in (scoring.get("factor_scores") or [])
        if f.get("data_available") and f.get("weighted_score") is not None
    ]
    if not factors:
        return None
    factors.sort(key=lambda f: abs(f.get("weighted_score") or 0.0), reverse=True)
    factors = factors[:top_n][::-1]  # reverse so the largest sits on top

    labels = [str(f.get("factor", "?")).replace("_", " ") for f in factors]
    values = [float(f.get("weighted_score") or 0.0) for f in factors]
    colors = [_color_for(v) for v in values]

    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.34 * len(factors))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#d0d7de", linewidth=0.8)
    ax.set_title("Top weighted factor scores")
    ax.set_xlabel("Weighted score")
    ax.tick_params(axis="y", labelsize=8)
    return _fig_to_data_uri(fig)


def pillar_scores_chart(payload: dict[str, Any]) -> str | None:
    """Bar chart of the per-pillar composite contributions."""
    scoring = (payload.get("key_features") or {}).get("model_scoring") or {}
    pillars = scoring.get("pillar_scores") or {}
    items = [(k, v) for k, v in pillars.items() if isinstance(v, (int, float))]
    if not items:
        return None
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [float(v) for _, v in items]
    colors = [_color_for(v) for v in values]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#d0d7de", linewidth=0.8)
    ax.set_title("Pillar scores")
    ax.set_ylabel("Score")
    return _fig_to_data_uri(fig)


def price_target_chart(payload: dict[str, Any]) -> str | None:
    """Bear / base / bull price-target scenarios versus spot."""
    target = (payload.get("key_features") or {}).get("price_target") or {}
    if target.get("status") != "ok":
        return None
    spot = target.get("spot")
    scenarios = [
        ("Bear", target.get("bear"), _BEAR),
        ("Base", target.get("base"), _NEUTRAL),
        ("Bull", target.get("bull"), _BULL),
    ]
    scenarios = [(n, float(v), c) for n, v, c in scenarios if isinstance(v, (int, float))]
    if not scenarios:
        return None

    labels = [n for n, _, _ in scenarios]
    values = [v for _, v, _ in scenarios]
    colors = [c for _, _, c in scenarios]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:,.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    if isinstance(spot, (int, float)):
        ax.axhline(float(spot), color=_ACCENT, linestyle="--", linewidth=1.2,
                   label=f"Spot {float(spot):,.2f}")
        ax.legend(fontsize=8)
    horizon = target.get("time_horizon", "")
    ax.set_title(f"Price-target scenarios{f' ({horizon})' if horizon else ''}")
    ax.set_ylabel("Price")
    return _fig_to_data_uri(fig)


# Order horizons sensibly when they are present in the forecast block.
_HORIZON_ORDER = ["1d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"]


def forecast_fan_chart(payload: dict[str, Any]) -> str | None:
    """Center price with 80% confidence band across forecast horizons."""
    forecast = (payload.get("key_features") or {}).get("price_range_forecast") or {}
    horizons = [h for h in _HORIZON_ORDER if h in forecast]
    horizons += [h for h in forecast if h not in horizons]
    points = []
    for h in horizons:
        block = forecast.get(h) or {}
        center = block.get("center_price")
        lo = block.get("lower_80")
        hi = block.get("upper_80")
        if all(isinstance(v, (int, float)) for v in (center, lo, hi)):
            points.append((h, float(center), float(lo), float(hi)))
    if not points:
        return None

    xs = list(range(len(points)))
    labels = [p[0] for p in points]
    centers = [p[1] for p in points]
    lowers = [p[2] for p in points]
    uppers = [p[3] for p in points]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(xs, lowers, uppers, color=_ACCENT, alpha=0.18,
                    label="80% band")
    ax.plot(xs, centers, color=_ACCENT, marker="o", label="Center")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_title("Price range forecast")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    return _fig_to_data_uri(fig)


# Registry: (title, function). render_html iterates this so adding a chart is
# a one-line change and tests can introspect the full set.
CHART_BUILDERS = [
    ("factor_scorecard", factor_scorecard_chart),
    ("pillar_scores", pillar_scores_chart),
    ("price_target", price_target_chart),
    ("forecast_fan", forecast_fan_chart),
]


def build_all_charts(payload: dict[str, Any]) -> dict[str, str]:
    """Build every chart that has data; skip the rest. Returns name -> data URI."""
    out: dict[str, str] = {}
    for name, builder in CHART_BUILDERS:
        try:
            uri = builder(payload)
        except Exception:
            uri = None
        if uri:
            out[name] = uri
    return out
