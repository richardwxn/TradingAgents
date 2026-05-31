"""Historical options chain reconstruction (Polygon Options Starter plan).

Polygon's snapshot endpoints (`/v3/snapshot/options/*`) are current-only:
they ignore `as_of` and return today's chain with today's IV/Greeks. For
PIT-correct historical IV we have to assemble the chain ourselves:

1. List the contract universe that existed on `as_of_date` via
   `/v3/reference/options/contracts?as_of=...` (paginated).
2. For each in-window contract, fetch the day-close (and VWAP) via
   `/v2/aggs/ticker/{contractSymbol}/range/1/day/{date}/{date}`.
3. Invert Black-Scholes to recover IV from the close price using a
   contemporaneous risk-free rate.

Returns a chain shaped identically to `_normalize_polygon_option_row` so
downstream consumers (`compute_iv_surface`, `build_option_strategies`,
etc.) work unchanged.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import requests
import yfinance as yf

from tradingagents.analysis_only.bsm import bs_vega, implied_vol
from tradingagents.analysis_only.cache import CACHE_SCHEMA_VERSION, DiskCache, stable_hash


# Process-shared cache of loaded ^IRX series, keyed by (start, end). Threads
# spawned from the same process (e.g. ThreadPoolExecutor in scripts/
# generate_corpus.py) share this dict, so the ^IRX yfinance fetch happens
# once per (start, end) combination per process — not once per report.
# Each report job previously paid ~0.5-1s of redundant fetch latency.
_RATE_SERIES_CACHE: dict[tuple[str, str | None], dict[str, float]] = {}
_RATE_SERIES_LOCK = threading.Lock()


def _load_irx_series(
    start: str,
    end: str | None,
    logger: logging.Logger,
) -> dict[str, float]:
    """Load ^IRX into a date→rate dict. Returns an empty dict on failure."""
    series: dict[str, float] = {}
    try:
        ticker = yf.Ticker("^IRX")
        kwargs: dict[str, Any] = {"start": start}
        if end:
            kwargs["end"] = end
        hist = ticker.history(**kwargs)
        if hist is not None and not hist.empty:
            closes = hist["Close"].dropna()
            for ts, value in closes.items():
                d_str = ts.date().isoformat()
                try:
                    series[d_str] = float(value) / 100.0
                except (TypeError, ValueError):
                    continue
        dates = sorted(series.keys())
        logger.info(
            "RiskFreeRateProvider loaded %d ^IRX observations from %s to %s",
            len(series), dates[0] if dates else "?", dates[-1] if dates else "?",
        )
    except Exception as exc:
        logger.warning(
            "RiskFreeRateProvider failed to load ^IRX (%s); rate lookups "
            "will fall back to constant.",
            exc,
        )
    return series


def reset_rate_cache() -> None:
    """Clear the module-level ^IRX cache. For tests."""
    with _RATE_SERIES_LOCK:
        _RATE_SERIES_CACHE.clear()


class RiskFreeRateProvider:
    """Daily 13-week T-bill yield from yfinance `^IRX`, in absolute terms.

    `^IRX` is quoted in percent (e.g. 4.50 = 4.50%). We divide by 100 and
    cache the full series **at module scope** (see `_RATE_SERIES_CACHE`) so
    repeated provider construction across reports / threads doesn't refetch.

    For dates with no daily observation (weekends, holidays, holes in the
    Yahoo series), we walk back up to 10 trading days to find a valid one.
    """

    def __init__(
        self,
        start: str | None = None,
        end: str | None = None,
        fallback_rate: float = 0.045,
        logger: logging.Logger | None = None,
    ) -> None:
        self._fallback_rate = fallback_rate
        self._logger = logger or logging.getLogger(__name__)
        self._start = start or "2020-01-01"
        self._end = end
        self._dates_sorted: list[str] = []
        self._series: dict[str, float] = {}

    def _ensure_loaded(self) -> None:
        if self._series:
            return
        cache_key = (self._start, self._end)
        with _RATE_SERIES_LOCK:
            cached = _RATE_SERIES_CACHE.get(cache_key)
            if cached is None:
                cached = _load_irx_series(
                    self._start, self._end, self._logger,
                )
                _RATE_SERIES_CACHE[cache_key] = cached
        self._series = cached
        self._dates_sorted = sorted(self._series.keys())

    def rate_for(self, target_date: str) -> float:
        """Return r for `target_date` (ISO). Walks back ≤10 trading days."""
        self._ensure_loaded()
        if not self._series:
            return self._fallback_rate
        if target_date in self._series:
            return self._series[target_date]
        try:
            d = date.fromisoformat(target_date)
        except ValueError:
            return self._fallback_rate
        for _ in range(10):
            d -= timedelta(days=1)
            key = d.isoformat()
            if key in self._series:
                return self._series[key]
        # Fallback: use the closest available date in the series.
        closest = min(
            self._dates_sorted,
            key=lambda dd: abs(
                date.fromisoformat(dd).toordinal()
                - date.fromisoformat(target_date).toordinal()
            ),
        )
        return self._series[closest]


class PolygonOptionsHistoricalProvider:
    """Assemble a PIT-correct historical option chain from Polygon Starter.

    Construction:
        rfr = RiskFreeRateProvider()
        provider = PolygonOptionsHistoricalProvider(
            api_key=POLYGON_API_KEY, risk_free_rates=rfr,
        )
        chain = provider.build_chain(
            symbol="NVDA", as_of_date="2024-06-21", spot=130.5,
        )
    """

    REFERENCE_URL = "https://api.polygon.io/v3/reference/options/contracts"
    AGGS_URL_TMPL = (
        "https://api.polygon.io/v2/aggs/ticker/{contract}/range/1/day/{d}/{d}"
    )

    def __init__(
        self,
        api_key: str,
        risk_free_rates: RiskFreeRateProvider,
        *,
        max_workers: int = 8,
        request_timeout: float = 20.0,
        logger: logging.Logger | None = None,
        cache_dir: str | None = None,
        enable_cache: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY required for historical options.")
        self._api_key = api_key
        self._rfr = risk_free_rates
        self._max_workers = max_workers
        self._timeout = request_timeout
        self._logger = logger or logging.getLogger(__name__)
        self._cache = DiskCache(cache_dir) if enable_cache else None

    # ---------- public ----------

    def build_chain(
        self,
        *,
        symbol: str,
        as_of_date: str,
        spot: float | None,
        dte_min: int = 7,
        dte_max: int = 120,
        strike_band: float = 0.20,
    ) -> dict[str, Any]:
        """Return a chain dict shaped like `_load_option_chain_polygon`.

        Filters contracts to:
        - DTE in [dte_min, dte_max] relative to `as_of_date`
        - |strike - spot| / spot ≤ strike_band

        Each entry has the same shape `_normalize_polygon_option_row`
        produces (type, expiry, dte, strike, bid/ask/mid, last, volume, OI,
        implied_volatility, delta, spot_distance_pct). `delta`/Greeks other
        than IV stay `None` on the historical path — they could be
        BSM-computed but downstream code doesn't require them today.
        """
        if spot is None or spot <= 0:
            return {
                "status": "unavailable",
                "source": "polygon_historical_reconstructed",
                "pit_status": "pit",
                "contracts": [],
                "reason": "Spot price unavailable.",
            }

        as_of = _parse_iso(as_of_date)
        if as_of is None:
            return {
                "status": "unavailable",
                "source": "polygon_historical_reconstructed",
                "pit_status": "pit",
                "contracts": [],
                "reason": f"Invalid as_of_date: {as_of_date!r}",
            }

        exp_lo = (as_of + timedelta(days=dte_min)).isoformat()
        exp_hi = (as_of + timedelta(days=dte_max)).isoformat()
        strike_lo = float(spot) * (1.0 - strike_band)
        strike_hi = float(spot) * (1.0 + strike_band)

        contract_meta = self._list_contracts(
            symbol=symbol,
            as_of_date=as_of_date,
            expiration_gte=exp_lo,
            expiration_lte=exp_hi,
            strike_gte=strike_lo,
            strike_lte=strike_hi,
        )
        if not contract_meta:
            return {
                "status": "unavailable",
                "source": "polygon_historical_reconstructed",
                "pit_status": "pit",
                "contracts": [],
                "reason": "No contracts in window.",
            }

        rate = self._rfr.rate_for(as_of_date)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(
                    self._fetch_contract_agg, c["ticker"], as_of_date,
                ): c
                for c in contract_meta
            }
            results: list[dict[str, Any]] = []
            for fut in as_completed(futures):
                meta = futures[fut]
                try:
                    agg = fut.result()
                except Exception as exc:
                    self._logger.debug(
                        "agg fetch failed for %s on %s: %s",
                        meta.get("ticker"), as_of_date, exc,
                    )
                    continue
                if not agg:
                    continue
                row = self._build_row(
                    meta=meta, agg=agg, as_of=as_of,
                    spot=float(spot), risk_free_rate=rate,
                )
                if row is not None:
                    results.append(row)

        if not results:
            return {
                "status": "unavailable",
                "source": "polygon_historical_reconstructed",
                "pit_status": "pit",
                "contracts": [],
                "reason": "No tradeable contract aggs in window.",
            }

        return {
            "status": "ok",
            "source": "polygon_historical_reconstructed",
            "pit_status": "pit",
            "contracts": results,
            "contracts_scanned": len(contract_meta),
            "contracts_usable": len(results),
            "risk_free_rate": rate,
        }

    # ---------- internals ----------

    def _list_contracts(
        self, *, symbol: str, as_of_date: str,
        expiration_gte: str, expiration_lte: str,
        strike_gte: float, strike_lte: float,
    ) -> list[dict[str, Any]]:
        cache_payload = {
            "symbol": symbol.upper(),
            "as_of_date": as_of_date,
            "expiration_gte": expiration_gte,
            "expiration_lte": expiration_lte,
            "strike_gte": round(float(strike_gte), 6),
            "strike_lte": round(float(strike_lte), 6),
        }
        cached = self._get_cached_json("contracts", cache_payload)
        if isinstance(cached, list):
            return cached
        params: dict[str, Any] = {
            "underlying_ticker": symbol.upper(),
            "as_of": as_of_date,
            "expiration_date.gte": expiration_gte,
            "expiration_date.lte": expiration_lte,
            "strike_price.gte": strike_gte,
            "strike_price.lte": strike_lte,
            "limit": 250,
            "apiKey": self._api_key,
        }
        next_url: str | None = self.REFERENCE_URL
        out: list[dict[str, Any]] = []
        page = 0
        while next_url and page < 10:
            page += 1
            try:
                resp = requests.get(
                    next_url, params=params, timeout=self._timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                self._logger.warning(
                    "list_contracts failed page=%d: %s", page, exc,
                )
                break
            for item in payload.get("results", []) or []:
                ct = item.get("contract_type")
                strike = item.get("strike_price")
                expiry = item.get("expiration_date")
                ticker = item.get("ticker")
                if ct not in {"call", "put"} or not (ticker and expiry and strike):
                    continue
                out.append({
                    "ticker": ticker,
                    "contract_type": ct,
                    "expiration_date": expiry,
                    "strike_price": float(strike),
                })
            next_url = payload.get("next_url")
            params = None
            if next_url and "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={self._api_key}"
        self._set_cached_json("contracts", cache_payload, out)
        return out

    def _fetch_contract_agg(
        self, contract_ticker: str, as_of_date: str,
    ) -> dict[str, Any] | None:
        cache_payload = {
            "contract_ticker": contract_ticker,
            "as_of_date": as_of_date,
        }
        cached = self._get_cached_json("contract_aggs", cache_payload)
        if isinstance(cached, dict):
            return cached
        url = self.AGGS_URL_TMPL.format(contract=contract_ticker, d=as_of_date)
        try:
            resp = requests.get(
                url,
                params={"adjusted": "true", "apiKey": self._api_key},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return None
        results = payload.get("results") or []
        if not results:
            return None
        bar = results[0]
        close = _safe_float(bar.get("c"))
        vwap = _safe_float(bar.get("vw"))
        volume = int(_safe_float(bar.get("v")) or 0)
        if close is None and vwap is None:
            return None
        out = {
            "close": close,
            "vwap": vwap,
            "volume": volume,
            "open": _safe_float(bar.get("o")),
            "high": _safe_float(bar.get("h")),
            "low": _safe_float(bar.get("l")),
        }
        self._set_cached_json("contract_aggs", cache_payload, out)
        return out

    def _cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        return stable_hash(
            {
                "schema": CACHE_SCHEMA_VERSION,
                "kind": f"historical_options_{namespace}",
                "payload": payload,
            }
        )

    def _get_cached_json(
        self,
        namespace: str,
        payload: dict[str, Any],
    ) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get_json(
            f"options_historical_{namespace}",
            self._cache_key(namespace, payload),
        )

    def _set_cached_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
    ) -> None:
        if self._cache is None:
            return
        self._cache.set_json(
            f"options_historical_{namespace}",
            self._cache_key(namespace, payload),
            value,
        )

    def _build_row(
        self, *, meta: dict[str, Any], agg: dict[str, Any],
        as_of: date, spot: float, risk_free_rate: float,
    ) -> dict[str, Any] | None:
        expiry = meta["expiration_date"]
        try:
            exp_date = date.fromisoformat(expiry)
        except ValueError:
            return None
        dte = (exp_date - as_of).days
        if dte <= 0:
            return None
        # Prefer VWAP (more stable for moderately-traded contracts);
        # fall back to close.
        price = agg.get("vwap")
        if price is None or price <= 0:
            price = agg.get("close")
        if price is None or price <= 0:
            return None
        strike = float(meta["strike_price"])
        kind = meta["contract_type"]
        iv = implied_vol(
            price=float(price),
            spot=spot, strike=strike,
            time_to_expiry=dte / 365.0,
            risk_free_rate=risk_free_rate,
            kind=kind,
        )
        spot_distance = (
            (strike - spot) / spot if spot > 0 else None
        )
        # Approximate Greeks for downstream code that wants them. Delta and
        # vega from BS at the recovered IV; gamma/theta left None to keep
        # the cost-to-quality tradeoff explicit.
        delta = None
        if iv is not None:
            try:
                delta = _bs_delta(
                    spot=spot, strike=strike,
                    T=dte / 365.0, r=risk_free_rate,
                    sigma=iv, kind=kind,
                )
            except (ValueError, ZeroDivisionError):
                delta = None
        return {
            "contract_symbol": meta["ticker"],
            "type": kind,
            "expiry": expiry,
            "dte": dte,
            "strike": strike,
            "bid": None,
            "ask": None,
            "mid": float(price),
            "last": _safe_float(agg.get("close")),
            "volume": int(agg.get("volume") or 0),
            "open_interest": 0,  # not exposed on aggs endpoint
            "implied_volatility": iv,
            "delta": delta,
            "gamma": None,
            "theta": None,
            "vega": None,
            "spot_distance_pct": spot_distance,
        }


def _bs_delta(
    *, spot: float, strike: float, T: float, r: float, sigma: float, kind: str,
    q: float = 0.0,
) -> float:
    """Plain BS delta. Used for downstream code that wants to filter by delta."""
    if not (spot > 0 and strike > 0 and T > 0 and sigma > 0):
        return 0.0
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * T) / (
        sigma * math.sqrt(T)
    )
    n = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    if kind == "call":
        return math.exp(-q * T) * n
    return math.exp(-q * T) * (n - 1.0)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _parse_iso(d: str) -> date | None:
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None
