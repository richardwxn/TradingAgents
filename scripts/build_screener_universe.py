"""Build the Nasdaq Composite filtered screener universe (Unit 4 / Phase B2).

Hits Polygon's ticker-reference + daily-aggs endpoints to derive a structured
yaml of NASDAQ-listed common stocks above market-cap + ADV thresholds, with
a coarse sector label per ticker derived from the SIC code.

Usage:
    POLYGON_API_KEY=... python scripts/build_screener_universe.py \\
        --output configs/screener_universe_nasdaq.yaml \\
        --pace-seconds 0.2

Output schema (Unit 4 / Phase B2):

    generated_at: 2026-05-31
    source: polygon-v3-reference-tickers
    filters:
      market_cap_min_usd: 500000000
      adv_min_usd: 5000000
      adv_lookback_days: 20
    tickers:
      - symbol: AAPL
        sector: Tech-MegaCap
        market_cap_usd: 3500000000000
        adv_usd: 12000000000
        sic_code: "3674"
      ...

The script is **idempotent**: re-running on the same date overwrites the
output yaml. Intent is monthly refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ----------------------------------------------------------------------
# SIC -> coarse sector mapping.
#
# Authoritative bucket vocabulary is `configs/universe.yaml::sectors`. The
# 26-name core universe uses: Semiconductors, Tech-MegaCap, Software,
# Networking, Photonics, Specialty-Materials, Energy-Nuclear,
# Financial-Data, Aerospace, Financials, Energy, Healthcare,
# Consumer-Staples, Utilities, Consumer-Discretionary. Anything not
# matched falls back to "Other".
#
# SIC codes are 4-digit standard industrial classification codes. We map
# the most common Nasdaq SIC families to the coarse buckets. Where the SIC
# range straddles multiple buckets (e.g. 7370-7379 is "Computer Services"
# = Software in our mapping but also includes data processing), we err
# toward the most-common-Nasdaq-stock interpretation. This is a
# coarse-grained heuristic, not a perfect mapping — the screener's job is
# to surface candidates for human review, not to make taxonomic claims.
#
# Reference for SIC ranges:
#   https://www.sec.gov/info/edgar/siccodes.htm
#
# The map uses 4-digit string keys for exact match; the helper
# `_sector_for_sic` also tries 4-digit prefix matches (e.g. "367*").
# ----------------------------------------------------------------------

# Exact 4-digit SIC matches.
_SIC_EXACT_MAP: dict[str, str] = {
    # Semiconductors
    "3674": "Semiconductors",  # Semiconductors & Related Devices
    "3559": "Semiconductors",  # Special industry machinery (semi equipment)
    "3825": "Semiconductors",  # Lab/measurement instruments (KLAC, etc)
    "3812": "Aerospace",       # Search/Detection/Nav (incl defense optics)
    # Networking
    "3669": "Networking",      # Communications equipment
    "3576": "Networking",      # Computer communications equipment
    "3661": "Networking",      # Telephone & telegraph apparatus
    # Software / tech services
    "7372": "Software",        # Prepackaged software
    "7370": "Software",        # Services-computer services
    "7371": "Software",        # Services-computer programming
    "7389": "Software",        # Services-business services NEC
    # Tech mega-cap (mfg of computers / phones)
    "3571": "Tech-MegaCap",    # Electronic computers
    "3572": "Tech-MegaCap",    # Computer storage devices
    "3575": "Tech-MegaCap",    # Computer terminals
    # Photonics / optical
    "3669": "Networking",      # (dup above) — fiber/comms equipment
    # Specialty Materials
    "3211": "Specialty-Materials",  # Glass
    "3357": "Specialty-Materials",  # Drawing/insulating non-ferrous wire
    # Energy / nuclear
    "1311": "Energy",          # Crude petroleum & natural gas
    "1381": "Energy",          # Drilling oil & gas wells
    "2911": "Energy",          # Petroleum refining
    "4924": "Utilities",       # Natural gas distribution
    "4931": "Utilities",       # Electric/gas services
    "4911": "Utilities",       # Electric services
    "4922": "Utilities",       # Natural gas transmission
    "4923": "Utilities",       # Natural gas transmission & distribution
    "4932": "Utilities",       # Gas/other services combined
    "4941": "Utilities",       # Water supply
    "1389": "Energy",          # Services-oil & gas field services
    # Aerospace
    "3721": "Aerospace",       # Aircraft
    "3724": "Aerospace",       # Aircraft engines & engine parts
    "3728": "Aerospace",       # Aircraft parts & auxiliary equipment NEC
    "3761": "Aerospace",       # Guided missiles & space vehicles
    "3812": "Aerospace",       # (dup above)
    # Financial data
    "7374": "Financial-Data",  # Services-computer processing & data prep
    "6199": "Financial-Data",  # Finance services
    "6770": "Financial-Data",  # Blank checks / holding offices
    # Financials
    "6020": "Financials",      # State commercial banks
    "6021": "Financials",      # National commercial banks
    "6022": "Financials",      # State commercial banks
    "6029": "Financials",      # Commercial banks NEC
    "6035": "Financials",      # Savings institution federal
    "6036": "Financials",      # Savings institution state-chartered
    "6099": "Financials",      # Functions related to deposit banking
    "6141": "Financials",      # Personal credit institutions
    "6199": "Financial-Data",  # (dup above)
    "6200": "Financials",      # Security & commodity brokers
    "6211": "Financials",      # Security brokers/dealers
    "6311": "Financials",      # Life insurance
    "6331": "Financials",      # Fire/marine/casualty insurance
    "6411": "Financials",      # Insurance agents/brokers/service
    "6726": "Financials",      # Investment offices NEC
    "6798": "Financials",      # Real estate investment trusts
    # Healthcare
    "2834": "Healthcare",      # Pharmaceutical preparations
    "2836": "Healthcare",      # Pharma — biological products
    "3841": "Healthcare",      # Surgical & medical instruments
    "3842": "Healthcare",      # Orthopedic/prosthetic/surgical appliances
    "3845": "Healthcare",      # Electromedical & electrotherapeutic apparatus
    "8731": "Healthcare",      # Services-commercial physical/biological R&D
    "8000": "Healthcare",      # Services-health
    "8090": "Healthcare",      # Services-health NEC
    "5912": "Healthcare",      # Retail drug stores
    # Consumer staples
    "2080": "Consumer-Staples",  # Beverages
    "2086": "Consumer-Staples",  # Bottled/canned soft drinks
    "2090": "Consumer-Staples",  # Food and kindred products
    "2099": "Consumer-Staples",  # Food preparations NEC
    "2111": "Consumer-Staples",  # Tobacco
    "5411": "Consumer-Staples",  # Grocery stores
    "5912": "Healthcare",        # (dup above)
    "2840": "Consumer-Staples",  # Soap, detergents
    "2844": "Consumer-Staples",  # Perfumes/cosmetics
    # Consumer discretionary
    "5651": "Consumer-Discretionary",   # Retail family clothing stores
    "5712": "Consumer-Discretionary",   # Retail furniture stores
    "5731": "Consumer-Discretionary",   # Retail radio/TV/electronics
    "5812": "Consumer-Discretionary",   # Eating places
    "5961": "Consumer-Discretionary",   # Catalog/mail-order houses
    "5990": "Consumer-Discretionary",   # Retail stores NEC
    "5999": "Consumer-Discretionary",   # Retail stores NEC
    "7011": "Consumer-Discretionary",   # Hotels & motels
    "7372": "Software",                 # (dup above)
    "7812": "Consumer-Discretionary",   # Services-motion picture production
    "7990": "Consumer-Discretionary",   # Services-amusement & recreation
    # Nuclear energy
    "1094": "Energy-Nuclear",          # Uranium-radium-vanadium ores
    # Transport — Consumer-Discretionary (airlines / hotels / cruise)
    "4512": "Consumer-Discretionary",  # Air transportation, scheduled
    "4513": "Consumer-Discretionary",  # Air courier services
    "4522": "Consumer-Discretionary",  # Air transportation, nonscheduled
    "4011": "Consumer-Discretionary",  # Railroads, line-haul operating
    "4731": "Consumer-Discretionary",  # Arrangement of transport of freight
    "4412": "Consumer-Discretionary",  # Deep sea foreign transport of freight
    "4731": "Consumer-Discretionary",
    # Online services / platforms / classified as Software
    "7340": "Software",                # Services-personal services (incl ABNB)
    "7311": "Software",                # Services-advertising
    "7320": "Software",                # Services-consumer credit reporting
    "7359": "Software",                # Services-equipment rental & leasing
    "7385": "Software",                # Services-info retrieval (web search)
    # Industrials / equipment — Other (no industrial bucket in vocab)
    "3585": "Other",                   # Refrigeration & service industry mach
    # Biotech / R&D
    "8731": "Healthcare",              # (already above; here for clarity)
}

# Prefix ranges as fallbacks. Order matters: more specific (4-digit prefix)
# tried before broader (3-digit, 2-digit).
_SIC_PREFIX_RULES: list[tuple[str, str]] = [
    # 4-digit start
    ("367", "Semiconductors"),   # Electronic components & accessories
    ("737", "Software"),         # Computer services
    ("738", "Software"),         # Services - business services NEC
    ("357", "Tech-MegaCap"),     # Computer & office equipment
    ("366", "Networking"),       # Communications equipment
    ("283", "Healthcare"),       # Drugs / pharma
    ("384", "Healthcare"),       # Medical instruments
    ("873", "Healthcare"),       # R&D services
    ("800", "Healthcare"),       # 8000-8099 health services
    ("809", "Healthcare"),
    # 3-digit (broader)
    ("131", "Energy"),           # Oil/gas extraction
    ("138", "Energy"),           # Oil/gas services
    ("291", "Energy"),           # Petroleum refining
    ("492", "Utilities"),
    ("493", "Utilities"),
    ("491", "Utilities"),
    ("494", "Utilities"),
    ("602", "Financials"),
    ("603", "Financials"),
    ("621", "Financials"),
    ("631", "Financials"),
    ("633", "Financials"),
    ("641", "Financials"),
    ("672", "Financials"),
    ("679", "Financials"),
    ("615", "Financials"),
    ("616", "Financials"),
    ("617", "Financials"),
    ("372", "Aerospace"),
    ("376", "Aerospace"),
    ("208", "Consumer-Staples"),
    ("209", "Consumer-Staples"),
    ("211", "Consumer-Staples"),
    ("541", "Consumer-Staples"),
    ("284", "Consumer-Staples"),
    ("561", "Consumer-Discretionary"),
    ("562", "Consumer-Discretionary"),
    ("563", "Consumer-Discretionary"),
    ("565", "Consumer-Discretionary"),
    ("566", "Consumer-Discretionary"),
    ("571", "Consumer-Discretionary"),
    ("572", "Consumer-Discretionary"),
    ("573", "Consumer-Discretionary"),
    ("581", "Consumer-Discretionary"),
    ("594", "Consumer-Discretionary"),
    ("596", "Consumer-Discretionary"),
    ("599", "Consumer-Discretionary"),
    ("701", "Consumer-Discretionary"),
    ("731", "Consumer-Discretionary"),
    ("781", "Consumer-Discretionary"),
    ("799", "Consumer-Discretionary"),
    # 2-digit catch-alls
    ("60", "Financials"),
    ("61", "Financials"),
    ("62", "Financials"),
    ("63", "Financials"),
    ("64", "Financials"),
    ("65", "Financials"),
    ("67", "Financials"),
]


def _sector_for_sic(sic_code: str | None) -> str:
    """Map a 4-digit SIC code → coarse sector bucket.

    Falls back through (exact 4-digit) → (prefix rule) → "Other".
    """
    if not sic_code:
        return "Other"
    code = str(sic_code).strip()
    if not code:
        return "Other"
    if code in _SIC_EXACT_MAP:
        return _SIC_EXACT_MAP[code]
    for prefix, sector in _SIC_PREFIX_RULES:
        if code.startswith(prefix):
            return sector
    return "Other"


# ----------------------------------------------------------------------
# Polygon HTTP helpers.
# ----------------------------------------------------------------------


@dataclass
class PolygonClient:
    api_key: str
    pace_seconds: float = 0.2
    session: requests.Session | None = None
    timeout: float = 30.0
    _last_request_at: float = 0.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _pace(self) -> None:
        if self.pace_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.pace_seconds:
            time.sleep(self.pace_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        max_retries: int = 4,
    ) -> dict[str, Any] | None:
        """GET with paced retries. Backs off on 429/5xx."""
        merged = dict(params or {})
        merged.setdefault("apiKey", self.api_key)
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._pace()
                resp = self.session.get(url, params=merged, timeout=self.timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    # Back off exponentially: 2s, 4s, 8s, 16s.
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        if last_exc is not None:
            print(
                f"WARN: Polygon GET failed after {max_retries} attempts: "
                f"{url} :: {type(last_exc).__name__}: {last_exc}",
                file=sys.stderr,
            )
        return None


# ----------------------------------------------------------------------
# Universe building.
# ----------------------------------------------------------------------


def fetch_nasdaq_tickers(client: PolygonClient) -> list[dict[str, Any]]:
    """Paginated fetch of all active NASDAQ-listed common stocks.

    Polygon's reference endpoint returns at most 1000 per page with a
    `next_url` for pagination. We keep only `type == "CS"`.
    """
    out: list[dict[str, Any]] = []
    url: str | None = "https://api.polygon.io/v3/reference/tickers"
    params: dict[str, Any] | None = {
        "market": "stocks",
        "exchange": "XNAS",
        "active": "true",
        "type": "CS",
        "limit": 1000,
    }
    page = 0
    while url:
        page += 1
        payload = client.get(url, params=params)
        if not payload:
            break
        results = payload.get("results") or []
        out.extend(results)
        print(
            f"  page {page}: +{len(results)} tickers (running total: {len(out)})",
            file=sys.stderr,
        )
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        # Polygon's next_url already includes the cursor; only apiKey is
        # injected by the client.
        params = None
    return out


def fetch_ticker_details(
    client: PolygonClient, symbol: str
) -> dict[str, Any] | None:
    """Fetch market cap + SIC code via the ticker-details endpoint.

    Returns the `results` dict (possibly empty) on HTTP success, ``None``
    only on HTTP / parse failure so the caller can distinguish
    "no metadata available" from "request failed".
    """
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}"
    payload = client.get(url)
    if payload is None:
        return None
    results = payload.get("results")
    if results is None:
        return {}
    return results if isinstance(results, dict) else None


def fetch_adv_usd(
    client: PolygonClient,
    symbol: str,
    *,
    end_date: date_cls,
    lookback_days: int = 20,
) -> float | None:
    """Median daily dollar volume (close * volume) over trailing N sessions.

    Uses the per-symbol daily aggs endpoint. We pull ~lookback_days * 1.6
    calendar days of bars and take the trailing N by date to handle
    weekends/holidays.

    **Note (Future-work #6):** for universe-scale builds (3000+ tickers)
    `precompute_adv_via_grouped` is dramatically faster — one HTTP call
    per date returns ALL tickers' aggs vs one call per ticker here.
    This function stays for fallback / single-ticker callers.
    """
    start = (end_date - timedelta(days=int(lookback_days * 1.8))).isoformat()
    end = end_date.isoformat()
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}"
        f"/range/1/day/{start}/{end}"
    )
    payload = client.get(url, params={"adjusted": "true", "sort": "asc", "limit": 5000})
    if not payload:
        return None
    rows = payload.get("results") or []
    if not rows:
        return None
    return _median_dollar_volume(rows[-lookback_days:])


def _median_dollar_volume(rows: list[dict[str, Any]]) -> float | None:
    """Median `close × volume` across the given Polygon agg rows.

    Common helper between the per-symbol fetch and the grouped batched
    path. Returns None when no row has positive close × volume.
    """
    dollar_vols: list[float] = []
    for row in rows:
        try:
            close = float(row.get("c") or 0.0)
            vol = float(row.get("v") or 0.0)
        except (TypeError, ValueError):
            continue
        if close > 0 and vol > 0:
            dollar_vols.append(close * vol)
    if not dollar_vols:
        return None
    dollar_vols.sort()
    n = len(dollar_vols)
    mid = n // 2
    if n % 2 == 1:
        return dollar_vols[mid]
    return (dollar_vols[mid - 1] + dollar_vols[mid]) / 2.0


def precompute_adv_via_grouped(
    client: PolygonClient,
    *,
    end_date: date_cls,
    lookback_days: int = 20,
) -> dict[str, float]:
    """Compute ADV (median dollar volume) for ALL tickers via grouped aggs.

    Future-work #6 batched path: replaces N per-ticker HTTP calls with
    ~lookback_days * 1.6 per-date calls returning every ticker's row in
    one shot. For a 3000-ticker universe scan that's ~3000 → ~32 HTTP
    calls (~100x reduction). For ticker symbols not seen on any of the
    fetched dates (e.g. delisted before window), they simply won't
    appear in the returned dict.

    Returns dict[symbol_upper, median_dollar_volume].
    """
    # Walk back lookback_days*1.8 calendar days to absorb weekends and
    # holidays, same as `fetch_adv_usd`'s per-symbol path.
    cur = end_date
    horizon = end_date - timedelta(days=int(lookback_days * 1.8))
    dates_iso: list[str] = []
    while cur >= horizon:
        # Skip weekends — Polygon returns empty for Sat/Sun but no point
        # in burning HTTP calls on them.
        if cur.weekday() < 5:
            dates_iso.append(cur.isoformat())
        cur = cur - timedelta(days=1)

    print(
        f"  precompute_adv_via_grouped: fetching {len(dates_iso)} trading "
        f"dates' grouped aggs ({horizon.isoformat()}..{end_date.isoformat()})",
        file=sys.stderr,
    )
    per_ticker_rows: dict[str, list[dict[str, Any]]] = {}
    for date_iso in dates_iso:
        url = (
            "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{date_iso}"
        )
        payload = client.get(url, params={"adjusted": "true"})
        if not payload:
            continue
        rows = payload.get("results") or []
        for row in rows:
            sym = (row.get("T") or "").upper()
            if sym:
                per_ticker_rows.setdefault(sym, []).append(row)

    # Trailing N dates per ticker (some symbols won't have a full window
    # — pre-IPO portions, delisted before window). `_median_dollar_volume`
    # gracefully returns None on insufficient data.
    out: dict[str, float] = {}
    for sym, rows in per_ticker_rows.items():
        # Already date-major sorted desc by our outer loop; reverse to asc
        # then take trailing N, matching `fetch_adv_usd[-lookback_days:]`.
        rows_asc = list(reversed(rows))
        adv = _median_dollar_volume(rows_asc[-lookback_days:])
        if adv is not None:
            out[sym] = adv
    print(
        f"  precompute_adv_via_grouped: computed ADV for {len(out)} symbols",
        file=sys.stderr,
    )
    return out


def build_universe(
    *,
    client: PolygonClient,
    as_of: date_cls,
    market_cap_min_usd: float,
    adv_min_usd: float,
    adv_lookback_days: int,
    max_tickers: int | None = None,
    progress_every: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch + filter the Nasdaq Composite universe.

    Returns `(kept_rows, stats)` where `stats` counts how many tickers
    were dropped at each stage. `kept_rows` is the per-ticker dict
    that gets serialized to the output yaml's `tickers` list.
    """
    print("Fetching NASDAQ common-stock tickers from Polygon...", file=sys.stderr)
    tickers = fetch_nasdaq_tickers(client)
    print(f"  fetched {len(tickers)} total CS tickers", file=sys.stderr)
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
        print(
            f"  --max-tickers cap: limiting to {len(tickers)} for this run",
            file=sys.stderr,
        )

    stats: dict[str, int] = {
        "fetched": len(tickers),
        "details_failed": 0,
        "dropped_market_cap": 0,
        "dropped_adv": 0,
        "dropped_missing_market_cap": 0,
        "kept": 0,
    }
    kept: list[dict[str, Any]] = []

    # Future-work #6: precompute ADV for every ticker via grouped-aggs
    # endpoint in a single ~20-30 HTTP call burst (one per trading date)
    # instead of one per-ticker call inside the loop. ~100x reduction on
    # HTTP requests for a 3000-name universe.
    adv_map = precompute_adv_via_grouped(
        client, end_date=as_of, lookback_days=adv_lookback_days,
    )

    for i, t in enumerate(tickers, start=1):
        symbol = str(t.get("ticker") or "").strip().upper()
        if not symbol:
            continue
        if i % progress_every == 0:
            print(
                f"  [{i}/{len(tickers)}] kept={len(kept)} "
                f"dropped_mc={stats['dropped_market_cap']} "
                f"dropped_adv={stats['dropped_adv']}",
                file=sys.stderr,
            )
        details = fetch_ticker_details(client, symbol)
        if details is None:
            stats["details_failed"] += 1
            continue
        market_cap = details.get("market_cap")
        if market_cap is None:
            stats["dropped_missing_market_cap"] += 1
            continue
        try:
            mc = float(market_cap)
        except (TypeError, ValueError):
            stats["dropped_missing_market_cap"] += 1
            continue
        if mc < market_cap_min_usd:
            stats["dropped_market_cap"] += 1
            continue
        # Look up ADV from the precomputed map (Future-work #6 batched
        # path). Fall back to the legacy per-symbol fetch if the symbol
        # wasn't in the grouped-aggs window (rare: pre-IPO mid-window,
        # delisted, etc.).
        adv_usd = adv_map.get(symbol)
        if adv_usd is None:
            adv_usd = fetch_adv_usd(
                client,
                symbol,
                end_date=as_of,
                lookback_days=adv_lookback_days,
            )
        if adv_usd is None or adv_usd < adv_min_usd:
            stats["dropped_adv"] += 1
            continue
        sic_code = str(details.get("sic_code") or "").strip() or None
        sector = _sector_for_sic(sic_code)
        kept.append(
            {
                "symbol": symbol,
                "sector": sector,
                "market_cap_usd": int(round(mc)),
                "adv_usd": int(round(adv_usd)),
                "sic_code": sic_code,
            }
        )
        stats["kept"] += 1

    return kept, stats


# ----------------------------------------------------------------------
# YAML serialization (minimal, no third-party formatting opinions).
# ----------------------------------------------------------------------


def _serialize_yaml(
    rows: list[dict[str, Any]],
    *,
    generated_at: str,
    market_cap_min_usd: float,
    adv_min_usd: float,
    adv_lookback_days: int,
    stats: dict[str, int],
) -> str:
    """Hand-rolled YAML emit. Keeps the diff minimal vs PyYAML's defaults."""
    lines: list[str] = []
    lines.append(
        "# Nasdaq Composite filtered screener universe, generated by "
        "scripts/build_screener_universe.py."
    )
    lines.append(
        "# Schema: structured rows per ticker with sector + market_cap + ADV "
        "+ SIC code. The screener (`scripts/screener.py`) reads either this "
        "shape or the Phase B1 flat ticker list."
    )
    lines.append(
        f"# Filters: NASDAQ common stocks (type=CS), active=true, "
        f"market_cap >= ${market_cap_min_usd:,.0f}, "
        f"trailing-{adv_lookback_days}d median ADV >= ${adv_min_usd:,.0f}."
    )
    lines.append(f"generated_at: \"{generated_at}\"")
    lines.append("source: polygon-v3-reference-tickers")
    lines.append("filters:")
    lines.append(f"  market_cap_min_usd: {int(market_cap_min_usd)}")
    lines.append(f"  adv_min_usd: {int(adv_min_usd)}")
    lines.append(f"  adv_lookback_days: {int(adv_lookback_days)}")
    lines.append("stats:")
    for k, v in stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("tickers:")
    for r in sorted(rows, key=lambda x: x["symbol"]):
        sic = r.get("sic_code")
        sic_repr = f"\"{sic}\"" if sic else "null"
        lines.append(f"  - symbol: {r['symbol']}")
        lines.append(f"    sector: {r['sector']}")
        lines.append(f"    market_cap_usd: {r['market_cap_usd']}")
        lines.append(f"    adv_usd: {r['adv_usd']}")
        lines.append(f"    sic_code: {sic_repr}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="configs/screener_universe_nasdaq.yaml",
        help="Output yaml path (overwritten).",
    )
    parser.add_argument(
        "--market-cap-min",
        type=float,
        default=500_000_000,
        help="Drop tickers with market cap below this USD threshold "
        "(default 500M).",
    )
    parser.add_argument(
        "--adv-min",
        type=float,
        default=5_000_000,
        help="Drop tickers with trailing-20d median dollar volume below "
        "this USD threshold (default 5M).",
    )
    parser.add_argument(
        "--adv-lookback-days",
        type=int,
        default=20,
        help="Trailing trading-day window for the ADV median (default 20).",
    )
    parser.add_argument(
        "--as-of",
        default=date_cls.today().isoformat(),
        help="ADV window end date (ISO). Default: today.",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.2,
        help="Minimum seconds between Polygon HTTP requests. Default 0.2 "
        "(paid-tier-ish). Use 12.0 for the free tier (5 req/min).",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="For development/debugging: limit the fetched ticker list to "
        "this many before filtering. Default: no limit.",
    )
    parser.add_argument(
        "--stats-json",
        default="",
        help="Optional path to write a JSON summary of the run "
        "(counts at each filter stage). Default: skipped.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key:
        print("ERROR: POLYGON_API_KEY env var is not set", file=sys.stderr)
        return 2

    try:
        as_of = date_cls.fromisoformat(args.as_of)
    except ValueError:
        print(f"ERROR: invalid --as-of: {args.as_of}", file=sys.stderr)
        return 2

    client = PolygonClient(api_key=api_key, pace_seconds=args.pace_seconds)

    started = time.time()
    kept, stats = build_universe(
        client=client,
        as_of=as_of,
        market_cap_min_usd=args.market_cap_min,
        adv_min_usd=args.adv_min,
        adv_lookback_days=args.adv_lookback_days,
        max_tickers=args.max_tickers,
    )
    elapsed = time.time() - started

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = _serialize_yaml(
        kept,
        generated_at=date_cls.today().isoformat(),
        market_cap_min_usd=args.market_cap_min,
        adv_min_usd=args.adv_min,
        adv_lookback_days=args.adv_lookback_days,
        stats=stats,
    )
    out_path.write_text(yaml_text)

    print(
        f"\nDone in {int(elapsed // 60)}m{int(elapsed % 60):02d}s. "
        f"Kept {stats['kept']} of {stats['fetched']} fetched tickers.",
        file=sys.stderr,
    )
    print(f"Output: {out_path}", file=sys.stderr)
    print(f"Stats: {json.dumps(stats, indent=2)}", file=sys.stderr)
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(
            json.dumps({
                "kept": stats["kept"],
                "fetched": stats["fetched"],
                "stats": stats,
                "elapsed_s": round(elapsed, 1),
                "filters": {
                    "market_cap_min_usd": args.market_cap_min,
                    "adv_min_usd": args.adv_min,
                    "adv_lookback_days": args.adv_lookback_days,
                },
                "as_of": args.as_of,
            }, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
