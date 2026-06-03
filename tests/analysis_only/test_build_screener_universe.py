"""Unit tests for `scripts/build_screener_universe.py`.

These tests mock Polygon HTTP via a stub `PolygonClient.get` to verify:
- Paginated NASDAQ ticker fetch follows `next_url`.
- Market-cap + ADV filters drop the right rows.
- SIC → sector mapping resolves to coarse buckets matching
  configs/universe.yaml::sectors vocabulary.
- The hand-rolled yaml serializer round-trips through PyYAML.
"""
from __future__ import annotations

import sys
from datetime import date as date_cls
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing a script file by path — `scripts/build_screener_universe.py` is
# not a package module, so import via importlib to keep things robust.
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "build_screener_universe",
    _REPO_ROOT / "scripts" / "build_screener_universe.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
# Register before exec so @dataclass can resolve cls.__module__.
sys.modules["build_screener_universe"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


_sector_for_sic = _MODULE._sector_for_sic
PolygonClient = _MODULE.PolygonClient
fetch_nasdaq_tickers = _MODULE.fetch_nasdaq_tickers
fetch_ticker_details = _MODULE.fetch_ticker_details
fetch_adv_usd = _MODULE.fetch_adv_usd
build_universe = _MODULE.build_universe
_serialize_yaml = _MODULE._serialize_yaml


# ---------- _sector_for_sic ----------


@pytest.mark.parametrize(
    "sic,expected",
    [
        ("3674", "Semiconductors"),
        ("7372", "Software"),
        ("3571", "Tech-MegaCap"),
        ("3576", "Networking"),
        ("3674", "Semiconductors"),
        ("2834", "Healthcare"),
        ("4911", "Utilities"),
        ("1311", "Energy"),
        ("6021", "Financials"),
        ("3721", "Aerospace"),
        ("1094", "Energy-Nuclear"),
        # Prefix fallback
        ("3679", "Semiconductors"),  # 367x → Semiconductors
        ("7373", "Software"),        # 737x → Software
        # Unknown SIC → Other
        ("9999", "Other"),
        ("0100", "Other"),
        # Missing / empty
        (None, "Other"),
        ("", "Other"),
    ],
)
def test_sector_for_sic(sic, expected):
    assert _sector_for_sic(sic) == expected


# ---------- mock PolygonClient ----------


class _StubClient:
    """Minimal stand-in for PolygonClient that returns canned responses."""

    def __init__(self, responses: dict[str, list[dict]]):
        # `responses[url]` is a list of payloads — each `get` call pops one.
        self._responses = {k: list(v) for k, v in responses.items()}
        self.requests: list[tuple[str, dict | None]] = []

    def get(self, url, *, params=None, max_retries=4):
        self.requests.append((url, dict(params) if params else None))
        queue = self._responses.get(url)
        if queue is None:
            # Allow url-prefix match for the next_url pagination case.
            for key, q in self._responses.items():
                if url.startswith(key) and q:
                    queue = q
                    break
        if not queue:
            return None
        return queue.pop(0)


# ---------- fetch_nasdaq_tickers ----------


def test_fetch_nasdaq_tickers_follows_pagination():
    base = "https://api.polygon.io/v3/reference/tickers"
    page1 = {
        "results": [{"ticker": "AAA"}, {"ticker": "BBB"}],
        "next_url": f"{base}?cursor=p2",
    }
    page2 = {
        "results": [{"ticker": "CCC"}],
        # No next_url — last page.
    }
    client = _StubClient({base: [page1], f"{base}?cursor=p2": [page2]})
    tickers = fetch_nasdaq_tickers(client)
    assert [t["ticker"] for t in tickers] == ["AAA", "BBB", "CCC"]


def test_fetch_nasdaq_tickers_handles_empty_first_page():
    base = "https://api.polygon.io/v3/reference/tickers"
    client = _StubClient({base: [{"results": []}]})
    tickers = fetch_nasdaq_tickers(client)
    assert tickers == []


def test_fetch_nasdaq_tickers_swallows_failed_request():
    base = "https://api.polygon.io/v3/reference/tickers"
    client = _StubClient({base: []})  # get returns None
    tickers = fetch_nasdaq_tickers(client)
    assert tickers == []


# ---------- fetch_ticker_details / fetch_adv_usd ----------


def test_fetch_ticker_details_returns_results_dict():
    url = "https://api.polygon.io/v3/reference/tickers/NVDA"
    client = _StubClient({url: [{"results": {"market_cap": 3.5e12, "sic_code": "3674"}}]})
    details = fetch_ticker_details(client, "NVDA")
    assert details["market_cap"] == pytest.approx(3.5e12)
    assert details["sic_code"] == "3674"


def test_fetch_ticker_details_returns_none_on_missing_payload():
    url = "https://api.polygon.io/v3/reference/tickers/MISSING"
    client = _StubClient({url: []})
    assert fetch_ticker_details(client, "MISSING") is None


def test_fetch_adv_usd_returns_median_of_dollar_volumes():
    base = "https://api.polygon.io/v2/aggs/ticker/NVDA"
    payload = {
        "results": [
            {"c": 100.0, "v": 1_000_000},  # $100M
            {"c": 110.0, "v": 1_000_000},  # $110M
            {"c": 120.0, "v": 1_000_000},  # $120M
        ]
    }
    client = _StubClient({base: [payload]})
    adv = fetch_adv_usd(
        client, "NVDA", end_date=date_cls(2026, 5, 22), lookback_days=3
    )
    assert adv == pytest.approx(110_000_000)


def test_fetch_adv_usd_returns_none_when_no_results():
    base = "https://api.polygon.io/v2/aggs/ticker/EMPTY"
    client = _StubClient({base: [{"results": []}]})
    adv = fetch_adv_usd(
        client, "EMPTY", end_date=date_cls(2026, 5, 22), lookback_days=3
    )
    assert adv is None


# ---------- build_universe end-to-end (mocked) ----------


def _build_e2e_client():
    base_tickers = "https://api.polygon.io/v3/reference/tickers"
    tickers_payload = {
        "results": [
            {"ticker": "BIG"},        # Above thresholds, semi sector
            {"ticker": "SMALL"},      # Below market cap
            {"ticker": "ILLIQUID"},   # Above market cap, below ADV
            {"ticker": "NOMETA"},     # details endpoint returns no market_cap
        ]
    }
    details_big = {"market_cap": 1.0e12, "sic_code": "3674"}
    details_small = {"market_cap": 1.0e8, "sic_code": "3674"}  # $100M < $500M
    details_illiquid = {"market_cap": 1.0e9, "sic_code": "7372"}
    details_nometa = {}  # missing market_cap

    aggs_big = {"results": [{"c": 100.0, "v": 1_000_000}] * 20}  # $100M ADV
    aggs_illiquid = {"results": [{"c": 1.0, "v": 100_000}] * 20}  # $100K ADV

    return _StubClient({
        base_tickers: [tickers_payload],
        "https://api.polygon.io/v3/reference/tickers/BIG": [
            {"results": details_big}
        ],
        "https://api.polygon.io/v3/reference/tickers/SMALL": [
            {"results": details_small}
        ],
        "https://api.polygon.io/v3/reference/tickers/ILLIQUID": [
            {"results": details_illiquid}
        ],
        "https://api.polygon.io/v3/reference/tickers/NOMETA": [
            {"results": details_nometa}
        ],
        "https://api.polygon.io/v2/aggs/ticker/BIG": [aggs_big],
        "https://api.polygon.io/v2/aggs/ticker/ILLIQUID": [aggs_illiquid],
    })


def test_build_universe_applies_market_cap_and_adv_filters():
    client = _build_e2e_client()
    kept, stats = build_universe(
        client=client,
        as_of=date_cls(2026, 5, 22),
        market_cap_min_usd=500_000_000,
        adv_min_usd=5_000_000,
        adv_lookback_days=20,
    )
    kept_symbols = [r["symbol"] for r in kept]
    assert kept_symbols == ["BIG"]
    assert stats["dropped_market_cap"] == 1   # SMALL
    assert stats["dropped_adv"] == 1          # ILLIQUID
    assert stats["dropped_missing_market_cap"] == 1  # NOMETA
    assert stats["kept"] == 1
    big = kept[0]
    assert big["sector"] == "Semiconductors"
    assert big["sic_code"] == "3674"
    assert big["market_cap_usd"] == 1_000_000_000_000
    assert big["adv_usd"] == 100_000_000


# ---------- _serialize_yaml ----------


def test_serialize_yaml_roundtrips_through_pyyaml():
    import yaml

    rows = [
        {
            "symbol": "ZZZ",
            "sector": "Software",
            "market_cap_usd": 750_000_000,
            "adv_usd": 12_000_000,
            "sic_code": "7372",
        },
        {
            "symbol": "AAA",
            "sector": "Semiconductors",
            "market_cap_usd": 3_500_000_000_000,
            "adv_usd": 25_000_000_000,
            "sic_code": "3674",
        },
    ]
    stats = {"fetched": 100, "kept": 2}
    text = _serialize_yaml(
        rows,
        generated_at="2026-05-31",
        market_cap_min_usd=500_000_000,
        adv_min_usd=5_000_000,
        adv_lookback_days=20,
        stats=stats,
    )
    payload = yaml.safe_load(text)
    assert payload["generated_at"] == "2026-05-31"
    assert payload["source"] == "polygon-v3-reference-tickers"
    assert payload["filters"]["market_cap_min_usd"] == 500_000_000
    assert payload["filters"]["adv_min_usd"] == 5_000_000
    assert payload["filters"]["adv_lookback_days"] == 20
    assert payload["stats"] == stats
    # Tickers sorted alphabetically by symbol
    symbols = [t["symbol"] for t in payload["tickers"]]
    assert symbols == ["AAA", "ZZZ"]
    # SIC stays a string (yaml quoting); market caps integer.
    assert payload["tickers"][0]["sic_code"] == "3674"
    assert payload["tickers"][0]["market_cap_usd"] == 3_500_000_000_000


def test_serialize_yaml_handles_null_sic():
    import yaml

    rows = [{
        "symbol": "NOSIC",
        "sector": "Other",
        "market_cap_usd": 600_000_000,
        "adv_usd": 6_000_000,
        "sic_code": None,
    }]
    text = _serialize_yaml(
        rows,
        generated_at="2026-05-31",
        market_cap_min_usd=500_000_000,
        adv_min_usd=5_000_000,
        adv_lookback_days=20,
        stats={"kept": 1},
    )
    payload = yaml.safe_load(text)
    assert payload["tickers"][0]["sic_code"] is None
    assert payload["tickers"][0]["sector"] == "Other"
