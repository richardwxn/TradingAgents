"""Option-position tracking for the portfolio ledger.

`portfolio/positions.json` is extended with an optional `options` array
per symbol. Backward-compatible: positions with only `shares` + `avg_cost`
keep working; adding options doesn't break any existing code path.

Schema example:

```json
{
  "cash": 100000.0,
  "positions": {
    "NVDA": {
      "shares": 100,
      "avg_cost": 150.00,
      "options": [
        {
          "right": "call",
          "strike": 200.0,
          "expiry": "2026-06-19",
          "quantity": 1,
          "avg_cost": 5.50
        },
        {
          "right": "put",
          "strike": 180.0,
          "expiry": "2026-06-19",
          "quantity": -1,
          "avg_cost": 3.20
        }
      ]
    }
  }
}
```

Conventions:
- `right`: "call" or "put"
- `strike`: float (per-share strike price)
- `expiry`: ISO date string
- `quantity`: int. Positive = long, negative = short. 1 contract = 100 shares.
- `avg_cost`: float (per-share premium; multiply by 100 for $ basis per contract).

This module contains:
- `OptionPosition` dataclass
- `load_option_positions(positions_payload)` parser with validation
- Greeks/exposure aggregation helpers (Section 28)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import requests


# ---------- data shapes ----------


@dataclass(frozen=True)
class OptionPosition:
    """A single option contract holding.

    `quantity` is signed: positive = long, negative = short.
    `avg_cost` is per-share premium (i.e. multiply by 100 × |quantity| for
    total $ basis).
    """

    symbol: str
    right: str  # "call" or "put"
    strike: float
    expiry: str  # ISO date
    quantity: int
    avg_cost: float

    def __post_init__(self) -> None:
        if self.right not in {"call", "put"}:
            raise ValueError(
                f"OptionPosition.right must be 'call' or 'put', got {self.right!r}"
            )
        if not isinstance(self.quantity, int) or self.quantity == 0:
            raise ValueError(
                f"OptionPosition.quantity must be a non-zero int, got "
                f"{self.quantity!r}"
            )
        if self.strike <= 0:
            raise ValueError(
                f"OptionPosition.strike must be positive, got {self.strike}"
            )
        if self.avg_cost < 0:
            raise ValueError(
                f"OptionPosition.avg_cost must be non-negative, got "
                f"{self.avg_cost}"
            )
        # Expiry must be parseable ISO date.
        try:
            date.fromisoformat(self.expiry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OptionPosition.expiry must be ISO date, got {self.expiry!r}"
            ) from exc

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def contract_basis(self) -> float:
        """Total dollar basis: avg_cost × |quantity| × 100 (shares/contract)."""
        return float(self.avg_cost) * abs(int(self.quantity)) * 100.0

    def dte(self, as_of: str) -> int | None:
        """Calendar days from `as_of` (ISO) to expiry. Negative if expired."""
        try:
            expiry_d = date.fromisoformat(self.expiry)
            as_of_d = date.fromisoformat(as_of)
        except (TypeError, ValueError):
            return None
        return (expiry_d - as_of_d).days


# ---------- loader ----------


def load_option_positions(
    positions_payload: dict[str, Any],
) -> dict[str, list[OptionPosition]]:
    """Extract option contracts from the positions ledger payload.

    Returns a dict mapping symbol → list[OptionPosition]. Symbols with no
    options (legacy shares-only entries) simply don't appear in the result.
    Raises `ValueError` if any entry has malformed required fields.
    Unknown / extra keys on contract entries are ignored (forward-compatible).
    """
    out: dict[str, list[OptionPosition]] = {}
    positions = (positions_payload or {}).get("positions") or {}
    for sym, body in positions.items():
        if not isinstance(body, dict):
            continue
        raw_options = body.get("options") or []
        if not isinstance(raw_options, list):
            raise ValueError(
                f"positions[{sym!r}].options must be a list, got "
                f"{type(raw_options).__name__}"
            )
        contracts: list[OptionPosition] = []
        for i, entry in enumerate(raw_options):
            if not isinstance(entry, dict):
                continue
            try:
                contracts.append(
                    OptionPosition(
                        symbol=str(sym).upper(),
                        right=str(entry["right"]).strip().lower(),
                        strike=float(entry["strike"]),
                        expiry=str(entry["expiry"]),
                        quantity=int(entry["quantity"]),
                        avg_cost=float(entry["avg_cost"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"positions[{sym!r}].options[{i}] missing field {exc.args[0]!r}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"positions[{sym!r}].options[{i}] invalid: {exc}"
                ) from exc
        if contracts:
            out[str(sym).upper()] = contracts
    return out


# ---------- summary aggregation ----------


@dataclass(frozen=True)
class OptionBookSummary:
    """High-level view of all option holdings for one symbol."""

    symbol: str
    long_call_contracts: int
    short_call_contracts: int
    long_put_contracts: int
    short_put_contracts: int
    total_premium_paid: float  # gross spend on long positions
    total_premium_collected: float  # gross received on short positions
    net_premium_basis: float  # paid − collected

    @property
    def total_contracts(self) -> int:
        return (
            self.long_call_contracts + self.short_call_contracts
            + self.long_put_contracts + self.short_put_contracts
        )


# ---------- Greeks enrichment + book-level aggregation ----------


@dataclass(frozen=True)
class EnrichedOption:
    """An OptionPosition joined to current chain data (mark + Greeks)."""

    position: OptionPosition
    mark: float | None  # current per-share mid/last
    implied_volatility: float | None
    delta: float | None  # per-contract delta as quoted (positive for calls, negative for puts)
    gamma: float | None
    theta: float | None  # per-day
    vega: float | None  # per 1% IV move

    @property
    def signed_delta_shares(self) -> float | None:
        """Delta in share equivalents: qty × delta × 100.

        Short positions correctly negate (a short call with delta 0.5 and
        quantity=-1 contributes -50 share-equivalent delta to the book).
        Returns None when delta isn't available.
        """
        if self.delta is None:
            return None
        return float(self.position.quantity) * float(self.delta) * 100.0

    @property
    def signed_vega(self) -> float | None:
        if self.vega is None:
            return None
        return float(self.position.quantity) * float(self.vega) * 100.0

    @property
    def signed_theta(self) -> float | None:
        if self.theta is None:
            return None
        return float(self.position.quantity) * float(self.theta) * 100.0

    @property
    def signed_gamma(self) -> float | None:
        if self.gamma is None:
            return None
        return float(self.position.quantity) * float(self.gamma) * 100.0

    @property
    def market_basis(self) -> float | None:
        """Current $ value of this leg: mark × |qty| × 100 (signed by long/short)."""
        if self.mark is None:
            return None
        # For long positions, the basis is positive (the value of the calls
        # we own). For short positions, the basis is negative (the liability
        # — what we'd pay to close).
        sign = 1 if self.position.is_long else -1
        return sign * float(self.mark) * abs(int(self.position.quantity)) * 100.0

    @property
    def unrealized_pnl(self) -> float | None:
        """Mark-to-market P&L vs avg_cost. Positive = profit."""
        if self.mark is None:
            return None
        per_share_pnl = float(self.mark) - float(self.position.avg_cost)
        # Long: profit when mark > cost. Short: profit when mark < cost (negate).
        sign = 1 if self.position.is_long else -1
        return (
            sign * per_share_pnl
            * abs(int(self.position.quantity)) * 100.0
        )


def enrich_with_chain(
    positions: Iterable[OptionPosition],
    chain_contracts: Iterable[dict[str, Any]],
) -> list[EnrichedOption]:
    """Join each `OptionPosition` to the matching contract in `chain_contracts`.

    `chain_contracts` is the normalized list produced by
    `_load_option_chain_polygon` / `_load_option_chain_yfinance` — each
    item has `type` ("call"|"put"), `strike`, `expiry`, `mid`/`last`,
    `implied_volatility`, `delta`, `gamma`, `theta`, `vega`.

    Match key: (right, strike, expiry). If no match, the EnrichedOption
    is returned with all market-data fields = None so the caller can
    surface "no current quote" without losing the position record.
    """
    contracts = list(chain_contracts)
    out: list[EnrichedOption] = []
    for p in positions:
        match = _find_chain_match(p, contracts)
        if match is None:
            out.append(EnrichedOption(
                position=p, mark=None, implied_volatility=None,
                delta=None, gamma=None, theta=None, vega=None,
            ))
            continue
        out.append(EnrichedOption(
            position=p,
            mark=_safe_float(match.get("mid")) or _safe_float(match.get("last")),
            implied_volatility=_safe_float(match.get("implied_volatility")),
            delta=_safe_float(match.get("delta")),
            gamma=_safe_float(match.get("gamma")),
            theta=_safe_float(match.get("theta")),
            vega=_safe_float(match.get("vega")),
        ))
    return out


def _find_chain_match(
    position: OptionPosition,
    contracts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for c in contracts:
        if str(c.get("type", "")).lower() != position.right:
            continue
        if str(c.get("expiry", "")) != position.expiry:
            continue
        strike = _safe_float(c.get("strike"))
        if strike is None or abs(strike - position.strike) > 0.005:
            continue
        return c
    return None


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


@dataclass(frozen=True)
class BookGreeks:
    """Aggregated Greek exposure across all option positions for a symbol
    (and optionally the underlying share position)."""

    symbol: str
    shares: int
    # Net share-equivalent delta: shares + Σ(qty × delta × 100) across legs.
    # None when no positions had delta data.
    net_share_equivalent_delta: float | None
    # Net vega across option positions in $ per 1% IV change.
    net_vega_dollars_per_vol_pt: float | None
    # Net theta across option positions in $ per day (negative = time decay drag).
    net_theta_dollars_per_day: float | None
    # Net gamma across option positions.
    net_gamma_dollars_per_share: float | None
    # Mark-to-market total of all option legs.
    net_option_market_value: float | None
    # Mark-to-market unrealized P&L on options.
    net_option_unrealized_pnl: float | None
    # Count of positions used in the aggregation.
    options_count: int


def book_greeks(
    enriched: Iterable[EnrichedOption],
    shares: int = 0,
) -> BookGreeks | None:
    """Aggregate Greeks across a list of `EnrichedOption` entries.

    Returns None if `enriched` is empty AND `shares == 0` — caller can
    treat that as "nothing to summarize."
    """
    enriched = list(enriched)
    if not enriched and shares == 0:
        return None

    symbol = enriched[0].position.symbol if enriched else ""

    def _accumulate(getter) -> float | None:
        values = [v for v in (getter(e) for e in enriched) if v is not None]
        if not values:
            return None
        return float(sum(values))

    sum_delta = _accumulate(lambda e: e.signed_delta_shares)
    net_share_eq_delta: float | None
    if sum_delta is None and shares == 0:
        net_share_eq_delta = None
    else:
        net_share_eq_delta = float(shares) + (sum_delta or 0.0)

    return BookGreeks(
        symbol=symbol,
        shares=int(shares),
        net_share_equivalent_delta=(
            round(net_share_eq_delta, 4)
            if net_share_eq_delta is not None else None
        ),
        net_vega_dollars_per_vol_pt=_round_or_none(
            _accumulate(lambda e: e.signed_vega)
        ),
        net_theta_dollars_per_day=_round_or_none(
            _accumulate(lambda e: e.signed_theta)
        ),
        net_gamma_dollars_per_share=_round_or_none(
            _accumulate(lambda e: e.signed_gamma)
        ),
        net_option_market_value=_round_or_none(
            _accumulate(lambda e: e.market_basis)
        ),
        net_option_unrealized_pnl=_round_or_none(
            _accumulate(lambda e: e.unrealized_pnl)
        ),
        options_count=len(enriched),
    )


def _round_or_none(v: float | None, ndigits: int = 2) -> float | None:
    return round(v, ndigits) if v is not None else None


# ---------- live chain fetch for daily signals ----------

_POLYGON_SNAPSHOT_URL = "https://api.polygon.io/v3/snapshot/options/{ticker}"


def fetch_current_chain(
    symbol: str,
    api_key: str | None = None,
    *,
    max_pages: int = 24,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch the current Polygon options snapshot for `symbol` as a list of
    normalized contract dicts (same shape `enrich_with_chain` expects).

    Returns `[]` on any failure (no API key, HTTP error, empty payload). The
    daily-signals CLI calls this once per symbol that has option positions;
    failures degrade gracefully — the position still renders, the
    Greeks/mark columns just show as "—".
    """
    key = api_key or os.environ.get("POLYGON_API_KEY")
    if not key:
        return []
    url = _POLYGON_SNAPSHOT_URL.format(ticker=symbol.upper())
    params: dict[str, Any] | None = {"limit": 250, "apiKey": key}
    next_url: str | None = url
    results: list[dict[str, Any]] = []
    pages = 0
    while next_url and pages < max_pages:
        pages += 1
        try:
            resp = requests.get(next_url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []
        for row in payload.get("results") or []:
            details = row.get("details") or {}
            day = row.get("day") or {}
            quote = row.get("last_quote") or {}
            greeks = row.get("greeks") or {}
            ct = details.get("contract_type")
            expiry = details.get("expiration_date")
            strike = _safe_float(details.get("strike_price"))
            if ct not in {"call", "put"} or not expiry or strike is None:
                continue
            bid = _safe_float(quote.get("bid") or quote.get("bid_price"))
            ask = _safe_float(quote.get("ask") or quote.get("ask_price"))
            last = _safe_float(day.get("close") or day.get("vwap"))
            mid: float | None
            if bid is not None and ask is not None and ask > 0:
                mid = (bid + ask) / 2.0
            else:
                mid = last
            results.append({
                "contract_symbol": details.get("ticker"),
                "type": ct,
                "expiry": expiry,
                "strike": strike,
                "bid": bid, "ask": ask, "mid": mid, "last": last,
                "volume": int(_safe_float(day.get("volume")) or 0),
                "open_interest": int(
                    _safe_float(row.get("open_interest")) or 0
                ),
                "implied_volatility": _safe_float(row.get("implied_volatility")),
                "delta": _safe_float(greeks.get("delta")),
                "gamma": _safe_float(greeks.get("gamma")),
                "theta": _safe_float(greeks.get("theta")),
                "vega": _safe_float(greeks.get("vega")),
            })
        next_url = payload.get("next_url")
        params = None
        if next_url and "apiKey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}apiKey={key}"
    return results


def summarize_option_book(
    positions: Iterable[OptionPosition],
) -> OptionBookSummary | None:
    """Aggregate per-symbol option holdings into a high-level summary.

    Returns None when `positions` is empty (caller decides how to render
    "no options for this name"). All entries are assumed to share the same
    symbol — call once per symbol after `load_option_positions`.
    """
    positions = list(positions)
    if not positions:
        return None
    symbol = positions[0].symbol
    long_call = sum(p.quantity for p in positions
                    if p.right == "call" and p.is_long)
    short_call = sum(-p.quantity for p in positions
                     if p.right == "call" and p.is_short)
    long_put = sum(p.quantity for p in positions
                   if p.right == "put" and p.is_long)
    short_put = sum(-p.quantity for p in positions
                    if p.right == "put" and p.is_short)
    paid = sum(p.contract_basis for p in positions if p.is_long)
    collected = sum(p.contract_basis for p in positions if p.is_short)
    return OptionBookSummary(
        symbol=symbol,
        long_call_contracts=int(long_call),
        short_call_contracts=int(short_call),
        long_put_contracts=int(long_put),
        short_put_contracts=int(short_put),
        total_premium_paid=round(float(paid), 2),
        total_premium_collected=round(float(collected), 2),
        net_premium_basis=round(float(paid - collected), 2),
    )
