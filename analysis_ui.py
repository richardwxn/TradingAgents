from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import traceback
from urllib.parse import parse_qs, urlparse

import yaml

from portfolio.signals import (
    PriceContext,
    Position,
    compute_actions,
    load_latest_signals,
)
from portfolio.sizing import SizingConfig, compute_target_weights, sizing_config_from_dict
from tradingagents.analysis_only import AnalysisOnlyMVP, render_markdown
from tradingagents.analysis_only.cache import load_report_if_cache_hit, report_file
from tradingagents.analysis_only.pipeline import AnalysisReport
from tradingagents.analysis_only.scoring import DEFAULT_FACTOR_WEIGHTS
from tradingagents.llm_clients.validators import VALID_MODELS


DEFAULT_OUTPUT_DIR = "reports/analysis_ui"
DEFAULT_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_DATA_PROVIDER = "polygon"
DEFAULT_PORTFOLIO_PATH = "configs/portfolio_snapshot.json"
DEFAULT_REPORTS_GLOB = "reports/analysis_mvp/*.json"
DEFAULT_POSITIONS_PATH = "portfolio/positions.json"
DEFAULT_SIZING_CONFIG = "configs/sizing.yaml"
FORECAST_PRESETS: dict[str, dict[str, int]] = {
    "short_term_1_2_weeks": {"2d": 2, "1w": 5, "2w": 10, "1m": 21},
    "swing_1_4_weeks": {"1w": 5, "2w": 10, "1m": 21, "3m": 63},
    "position_1_3_months": {"1w": 5, "1m": 21, "2m": 42, "3m": 63},
    "long_3_6_months": {"1w": 5, "1m": 21, "3m": 63, "6m": 126},
}
LLM_MODEL_OPTIONS: dict[str, list[str]] = {
    **VALID_MODELS,
    "openrouter": [
        "openai/gpt-5.5",
        "openai/gpt-5.4-mini",
        "openai/gpt-5-mini",
        "openai/gpt-5",
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.5-pro",
    ],
    "ollama": ["llama3.1", "qwen2.5", "mistral"],
}
LLM_PROVIDER_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}
DATA_PROVIDER_KEY_ENV: dict[str, str | None] = {
    "polygon": "POLYGON_API_KEY",
    "auto": None,
    "yfinance": None,
    "openbb": None,
}
_BEST_BUY_STATUS_LOCK = threading.Lock()
_BEST_BUY_STATUS: dict[str, dict[str, Any]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local browser UI for analysis-only reports."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def make_handler(output_dir: str):
    class AnalysisUIHandler(BaseHTTPRequestHandler):
        server_version = "TradingAgentsAnalysisUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/" or self.path.startswith("/?"):
                self._send_html(_html_page())
                return
            if parsed.path == "/api/defaults":
                self._send_json(
                    {
                        "factor_weights": DEFAULT_FACTOR_WEIGHTS,
                        "default_date": datetime.now().strftime("%Y-%m-%d"),
                        "default_data_provider": DEFAULT_DATA_PROVIDER,
                        "data_provider_env": _data_provider_env_status(),
                        "llm_models": LLM_MODEL_OPTIONS,
                        "llm_env": _llm_env_status(),
                        "default_watchlist": _default_watchlist(),
                    }
                )
                return
            if parsed.path == "/api/best-buy-status":
                query = parse_qs(parsed.query)
                job_id = (query.get("job_id") or [""])[0]
                self._send_json(_best_buy_status(job_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/analyze":
                try:
                    payload = self._read_json()
                    response = _run_analysis(payload, output_dir=output_dir)
                    self._send_json(response)
                except Exception as exc:
                    self._send_json(
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if self.path == "/api/best-buy":
                try:
                    payload = self._read_json()
                    response = _run_best_buy(payload, output_dir=output_dir)
                    self._send_json(response)
                except Exception as exc:
                    self._send_json(
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[analysis-ui] {self.address_string()} - {fmt % args}")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            loaded = json.loads(raw or "{}")
            if not isinstance(loaded, dict):
                raise ValueError("Request body must be a JSON object.")
            return loaded

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return AnalysisUIHandler


def _run_analysis(payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    ticker = str(payload.get("ticker") or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")
    as_of_date = str(payload.get("date") or datetime.now().strftime("%Y-%m-%d"))
    datetime.strptime(as_of_date, "%Y-%m-%d")

    factor_weights = _clean_factor_weights(payload.get("factor_weights") or {})
    horizon = str(payload.get("horizon") or "swing_1_4_weeks")
    competitors = [
        item.strip().upper()
        for item in str(payload.get("competitors") or "").split(",")
        if item.strip()
    ]
    no_state = bool(payload.get("no_state", False))
    state_path = None if no_state else str(
        payload.get("state_store") or "state/analysis_state.sqlite"
    )

    mvp = AnalysisOnlyMVP(
        horizon=horizon,
        data_provider=str(payload.get("data_provider") or DEFAULT_DATA_PROVIDER),
        options_enabled=not bool(payload.get("disable_options_scan", False)),
        min_unusual_option_notional=float(
            payload.get("min_unusual_option_notional") or 500_000.0
        ),
        min_option_volume_oi_ratio=float(
            payload.get("min_option_volume_oi_ratio") or 3.0
        ),
        factor_weights=factor_weights,
        forecast_horizons=FORECAST_PRESETS.get(horizon),
        competitors=competitors,
        enable_llm_insights=bool(payload.get("enable_llm_insights", False)),
        enable_narrative=bool(payload.get("enable_narrative", False)),
        enable_tradingagents_review=bool(
            payload.get("enable_tradingagents_review", False)
        ),
        llm_provider=str(payload.get("llm_provider") or "openai"),
        llm_model=str(payload.get("llm_model") or DEFAULT_LLM_MODEL),
        llm_base_url=payload.get("llm_base_url") or None,
        portfolio_path=payload.get("portfolio_path") or DEFAULT_PORTFOLIO_PATH,
        state_store_path=state_path,
        verbose=False,
    )
    out_dir = Path(output_dir)
    cache_key = mvp.report_cache_key(ticker, as_of_date)
    force_refresh = bool(payload.get("force_refresh", False))
    report = None if force_refresh else load_report_if_cache_hit(
        AnalysisReport,
        out_dir,
        ticker,
        as_of_date,
        cache_key,
    )
    cache_hit = report is not None
    if report is None:
        report = mvp.run(symbol=ticker, as_of_date=as_of_date)
        json_path = mvp.save_report(report, out_dir)
    else:
        json_path = report_file(out_dir, ticker, as_of_date)
    markdown = render_markdown(report.to_json_dict())
    md_path = json_path.with_suffix(".md")
    md_path.write_text(markdown)
    data = report.to_json_dict()
    return {
        "ok": True,
        "json_path": str(json_path.resolve()),
        "markdown_path": str(md_path.resolve()),
        "cache_hit": cache_hit,
        "report": data,
        "markdown": markdown,
    }


def _best_buy_status(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {"ok": False, "error": "job_id is required"}
    with _BEST_BUY_STATUS_LOCK:
        status = dict(_BEST_BUY_STATUS.get(job_id) or {})
    if not status:
        return {"ok": False, "job_id": job_id, "state": "unknown"}
    if status.get("state") == "running" and status.get("started_at_epoch"):
        status["elapsed_seconds"] = round(time.time() - float(status["started_at_epoch"]), 1)
    status["ok"] = True
    return status


def _update_best_buy_status(job_id: str | None, **updates: Any) -> None:
    if not job_id:
        return
    with _BEST_BUY_STATUS_LOCK:
        current = dict(_BEST_BUY_STATUS.get(job_id) or {})
        current.update(updates)
        current["job_id"] = job_id
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _BEST_BUY_STATUS[job_id] = current


def _run_best_buy(payload: dict[str, Any], output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or "")
    started_at = time.time()
    _update_best_buy_status(
        job_id,
        state="running",
        phase="loading reports",
        started_at_epoch=started_at,
        current_symbol=None,
        current_reason=None,
        total=0,
        completed=0,
        queued=[],
        generated=[],
    )
    raw_symbols = payload.get("tickers") or payload.get("symbols") or ""
    symbols = _parse_symbols(raw_symbols) or _default_watchlist()
    if not symbols:
        raise ValueError("Provide at least one ticker to rank.")

    as_of_raw = str(payload.get("date") or datetime.now().strftime("%Y-%m-%d"))
    as_of = datetime.strptime(as_of_raw, "%Y-%m-%d").date()
    reports_glob = str(payload.get("reports_glob") or DEFAULT_REPORTS_GLOB)
    report_paths = [Path(p) for p in glob.glob(reports_glob)]
    signals = load_latest_signals(report_paths, universe=symbols, as_of=as_of)
    config = _load_sizing_config(Path(str(payload.get("sizing_config") or DEFAULT_SIZING_CONFIG)), symbols)
    generated_reports: list[dict[str, Any]] = []
    if bool(payload.get("run_missing_reports", False)) or bool(payload.get("refresh_stale_reports", False)):
        symbols_to_generate: list[tuple[str, str]] = []
        if bool(payload.get("run_missing_reports", False)):
            symbols_to_generate.extend(
                (sym, "missing") for sym in symbols if signals.get(sym) is None
            )
        if bool(payload.get("refresh_stale_reports", False)):
            missing = {sym for sym, reason in symbols_to_generate if reason == "missing"}
            symbols_to_generate.extend(
                (sym, "stale_promising")
                for sym in _stale_promising_symbols(signals, config, as_of)
                if sym not in missing
            )
        _update_best_buy_status(
            job_id,
            phase="generating reports" if symbols_to_generate else "ranking",
            total=len(symbols_to_generate),
            completed=0,
            queued=[{"symbol": sym, "reason": reason} for sym, reason in symbols_to_generate],
        )
        report_output_dir = str(
            payload.get("report_output_dir")
            or _output_dir_from_reports_glob(reports_glob)
            or output_dir
        )
        for idx, (sym, reason) in enumerate(symbols_to_generate, start=1):
            _update_best_buy_status(
                job_id,
                phase="generating reports",
                current_symbol=sym,
                current_reason=reason,
                completed=idx - 1,
            )
            analysis_payload = _best_buy_analysis_payload(payload, sym, as_of_raw)
            try:
                result = _run_analysis(analysis_payload, output_dir=report_output_dir)
            except Exception as exc:
                _update_best_buy_status(
                    job_id,
                    state="error",
                    phase="error",
                    current_symbol=sym,
                    current_reason=reason,
                    error=str(exc),
                    elapsed_seconds=round(time.time() - started_at, 1),
                )
                raise
            generated_reports.append(
                {
                    "symbol": sym,
                    "reason": reason,
                    "json_path": result.get("json_path"),
                    "cache_hit": result.get("cache_hit", False),
                }
            )
            _update_best_buy_status(
                job_id,
                completed=idx,
                generated=generated_reports,
            )
        if symbols_to_generate:
            report_paths = [Path(p) for p in glob.glob(reports_glob)]
            signals = load_latest_signals(report_paths, universe=symbols, as_of=as_of)
    else:
        _update_best_buy_status(job_id, phase="ranking", total=0, completed=0)
    _update_best_buy_status(
        job_id,
        phase="ranking",
        current_symbol=None,
        current_reason=None,
    )
    cash, positions = _load_positions(Path(str(payload.get("positions_path") or DEFAULT_POSITIONS_PATH)))
    reports_by_symbol = _load_signal_reports(signals)
    fetch_current_prices = bool(payload.get("fetch_current_prices", False))
    prices = _price_contexts_for_rank(signals, reports_by_symbol, fetch_current_prices)

    actions, summary = compute_actions(
        signals=signals,
        positions=positions,
        prices=prices,
        config=config,
        cash=cash,
        as_of=as_of,
    )
    by_symbol = {a.symbol: a for a in actions}
    candidates = [
        _rank_candidate(sym, signals.get(sym), by_symbol.get(sym), reports_by_symbol.get(sym), config)
        for sym in symbols
    ]
    candidates = [c for c in candidates if c is not None]
    candidates.sort(key=lambda c: c["rank_score"], reverse=True)
    best = candidates[0] if candidates else None
    _update_best_buy_status(
        job_id,
        state="complete",
        phase="complete",
        best_symbol=best.get("symbol") if best else None,
        best_score=best.get("rank_score") if best else None,
        elapsed_seconds=round(time.time() - started_at, 1),
    )
    return {
        "ok": True,
        "as_of": as_of.isoformat(),
        "reports_glob": reports_glob,
        "symbols": symbols,
        "best": best,
        "candidates": candidates,
        "generated_reports": generated_reports,
        "summary": summary,
        "notes": [
            "Ranking uses the latest cached report on or before the as-of date.",
            "Risk caps that need live beta/correlation are not applied in this quick picker.",
        ],
    }


def _output_dir_from_reports_glob(reports_glob: str) -> str | None:
    if not reports_glob:
        return None
    marker = "*"
    has_wildcard = marker in reports_glob
    prefix = reports_glob.split(marker, 1)[0] if marker in reports_glob else reports_glob
    if prefix.endswith(("/", "\\")):
        return prefix.rstrip("/\\")
    path = Path(prefix)
    if path.suffix:
        return str(path.parent)
    if not has_wildcard:
        return str(path)
    return str(path.parent if path.name else path)


def _stale_promising_symbols(
    signals: dict[str, Any],
    config: SizingConfig,
    as_of: Any,
) -> list[str]:
    """Return stale bullish symbols that would clear min size without decay."""
    sizing_input = _sizing_input_from_signals(signals, config=config, as_of=as_of)
    no_decay_config = replace(config, stale_signal_decay=1.0)
    no_decay_weights = compute_target_weights(sizing_input, config=no_decay_config)
    stale: list[str] = []
    for sym, sig in signals.items():
        if sig is None or sig.direction != "bullish":
            continue
        age = sizing_input.get(sym, {}).get("age_days")
        if age is None or age <= config.stale_composite_days:
            continue
        if no_decay_weights.get(sym, 0.0) >= config.min_position_weight:
            stale.append(sym)
    return stale


def _sizing_input_from_signals(
    signals: dict[str, Any],
    *,
    config: SizingConfig,
    as_of: Any,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym, sig in signals.items():
        age_days: int | None = None
        if sig is not None:
            try:
                sig_date = datetime.strptime(sig.as_of_date, "%Y-%m-%d").date()
                age_days = (as_of - sig_date).days
            except Exception:
                age_days = None
        if sig is None:
            out[sym] = {
                "direction": "neutral",
                "composite": None,
                "confidence": None,
                "age_days": age_days,
            }
        elif not config.enable_bearish and sig.direction == "bearish":
            out[sym] = {
                "direction": "neutral",
                "composite": None,
                "confidence": None,
                "age_days": age_days,
            }
        else:
            payload = sig.to_sizing_input()
            payload["age_days"] = age_days
            out[sym] = payload
    return out


def _best_buy_analysis_payload(
    payload: dict[str, Any],
    symbol: str,
    as_of_date: str,
) -> dict[str, Any]:
    passthrough_keys = {
        "horizon",
        "data_provider",
        "disable_options_scan",
        "min_unusual_option_notional",
        "min_option_volume_oi_ratio",
        "enable_narrative",
        "enable_llm_insights",
        "enable_tradingagents_review",
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "no_state",
        "state_store",
        "portfolio_path",
        "factor_weights",
    }
    out = {key: payload[key] for key in passthrough_keys if key in payload}
    out.update(
        {
            "ticker": symbol,
            "date": as_of_date,
            "force_refresh": False,
        }
    )
    return out


def _parse_symbols(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        pieces = [str(x) for x in raw]
    else:
        pieces = str(raw or "").replace(",", " ").split()
    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        sym = piece.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _default_watchlist() -> list[str]:
    universe_path = Path("configs/universe.yaml")
    try:
        data = yaml.safe_load(universe_path.read_text()) or {}
        core = data.get("core") or []
        parsed = _parse_symbols(core)
        if parsed:
            return parsed
    except Exception:
        pass
    try:
        data = yaml.safe_load(Path(DEFAULT_SIZING_CONFIG).read_text()) or {}
        parsed = _parse_symbols(data.get("universe") or [])
        if parsed:
            return parsed
    except Exception:
        pass
    return []


def _load_sizing_config(path: Path, symbols: list[str]) -> SizingConfig:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        data = {}
    if not data.get("universe"):
        data["universe"] = symbols
    return sizing_config_from_dict(data)


def _load_positions(path: Path) -> tuple[float, dict[str, Position]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return 0.0, {}
    cash = float(payload.get("cash") or 0.0)
    positions: dict[str, Position] = {}
    for sym, raw in (payload.get("positions") or {}).items():
        positions[str(sym).upper()] = Position(
            shares=float(raw.get("shares") or 0.0),
            avg_cost=float(raw.get("avg_cost") or 0.0),
        )
    return cash, positions


def _load_signal_reports(signals: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym, sig in signals.items():
        if sig is None:
            continue
        try:
            loaded = json.loads(Path(sig.source_path).read_text())
        except Exception:
            continue
        if isinstance(loaded, dict):
            out[sym] = loaded
    return out


def _price_contexts_for_rank(
    signals: dict[str, Any],
    reports_by_symbol: dict[str, dict[str, Any]],
    fetch_current_prices: bool,
) -> dict[str, PriceContext]:
    out: dict[str, PriceContext] = {}
    for sym in signals:
        if fetch_current_prices:
            try:
                from daily_signals import fetch_price_context
                out[sym] = fetch_price_context(sym)
                continue
            except Exception:
                pass
        report = reports_by_symbol.get(sym) or {}
        tech = ((report.get("key_features") or {}).get("technical") or {})
        decision = ((report.get("key_features") or {}).get("decision_summary") or {})
        out[sym] = PriceContext(
            last_close=_safe_float(decision.get("current_price") or tech.get("close")),
            sma20=_safe_float(tech.get("sma_20")),
            atr14=_safe_float(tech.get("atr_14")),
        )
    return out


def _rank_candidate(
    sym: str,
    sig: Any,
    action: Any,
    report: dict[str, Any] | None,
    config: SizingConfig,
) -> dict[str, Any] | None:
    if sig is None or action is None or not report:
        return {
            "symbol": sym,
            "rank_score": -100.0,
            "action": "MISSING",
            "direction": None,
            "composite": None,
            "confidence": None,
            "as_of_date": None,
            "signal_age_days": None,
            "reason": "No cached report found for this ticker.",
            "notes": ["Run analysis for this ticker before it can rank."],
        }
    kf = report.get("key_features") or {}
    decision = kf.get("decision_summary") or {}
    target = kf.get("price_target") or {}
    entry = decision.get("entry") or {}
    buy_zone = entry.get("preferred_buy_zone") or {}
    current_price = _safe_float(decision.get("current_price") or target.get("spot"))
    zone_high = _safe_float(buy_zone.get("high") or entry.get("starter_buy_at_or_below"))
    base_upside = _safe_float(target.get("base_upside_pct") or decision.get("base_upside_pct")) or 0.0
    win_probability = _safe_float(decision.get("estimated_win_probability"))
    composite = _safe_float(sig.composite) or 0.0
    confidence = _safe_float(sig.confidence) or 0.0
    conviction = max(0.0, composite) * max(0.0, confidence)
    max_name = max(config.max_per_name, 0.01)
    target_weight_component = min(1.0, max(0.0, action.target_weight / max_name))
    composite_component = min(1.0, max(0.0, (composite + 0.15) / 0.65))
    upside_component = min(1.0, max(0.0, base_upside / 0.35))
    entry_component, entry_label = _entry_component(current_price, zone_high)
    action_bonus = {"BUY": 0.08, "ADD": 0.04, "HOLD": -0.02, "SKIP": -0.16, "EXIT": -0.22}.get(action.action, -0.08)
    stale_penalty = 0.0
    if action.signal_age_days is not None and action.signal_age_days > config.stale_composite_days:
        stale_penalty = min(0.25, (action.signal_age_days - config.stale_composite_days) * 0.015)
    score = (
        0.34 * target_weight_component
        + 0.28 * composite_component
        + 0.16 * confidence
        + 0.12 * upside_component
        + 0.10 * entry_component
        + action_bonus
        - stale_penalty
    )
    rank_score = round(max(0.0, min(100.0, score * 100.0)), 1)
    reason_bits = [
        f"{action.action} by sizing policy",
        f"composite {composite:+.3f}",
        f"target {action.target_weight * 100:.1f}%",
    ]
    if base_upside:
        reason_bits.append(f"base upside {base_upside * 100:.1f}%")
    reason_bits.append(entry_label)
    return {
        "symbol": sym,
        "rank_score": rank_score,
        "action": action.action,
        "direction": sig.direction,
        "composite": composite,
        "confidence": confidence,
        "conviction": conviction,
        "as_of_date": sig.as_of_date,
        "signal_age_days": action.signal_age_days,
        "target_weight": action.target_weight,
        "current_weight": action.current_weight,
        "delta_pp": action.delta_pp,
        "target_shares": action.target_shares,
        "delta_shares": action.delta_shares,
        "limit_price": action.limit_price,
        "stop_loss": action.stop_loss,
        "last_close": action.last_close or current_price,
        "base_upside_pct": base_upside,
        "estimated_win_probability": win_probability,
        "entry_status": entry_label,
        "reason": "; ".join(reason_bits),
        "notes": action.notes,
        "source_path": sig.source_path,
    }


def _entry_component(current_price: float | None, zone_high: float | None) -> tuple[float, str]:
    if current_price is None or zone_high is None or zone_high <= 0:
        return 0.35, "entry zone unavailable"
    gap = (current_price / zone_high) - 1.0
    if gap <= 0:
        return 1.0, "inside buy zone"
    if gap <= 0.03:
        return 0.75, "near buy zone"
    if gap <= 0.08:
        return 0.45, "modestly extended"
    return 0.15, "extended above buy zone"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_factor_weights(raw: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        if key not in DEFAULT_FACTOR_WEIGHTS:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f >= 0:
            cleaned[key] = f
    return cleaned


def _llm_env_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for provider, env_var in LLM_PROVIDER_KEY_ENV.items():
        status[provider] = {
            "requires_key": env_var is not None,
            "env_var": env_var,
            "configured": True if env_var is None else bool(os.getenv(env_var)),
        }
    return status


def _data_provider_env_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for provider, env_var in DATA_PROVIDER_KEY_ENV.items():
        status[provider] = {
            "requires_key": env_var is not None,
            "env_var": env_var,
            "configured": True if env_var is None else bool(os.getenv(env_var)),
        }
    return status


def _html_page() -> str:
    defaults_json = json.dumps(DEFAULT_FACTOR_WEIGHTS, sort_keys=True)
    data_provider_env_json = json.dumps(
        _data_provider_env_status(), sort_keys=True
    )
    llm_models_json = json.dumps(LLM_MODEL_OPTIONS, sort_keys=True)
    llm_env_json = json.dumps(_llm_env_status(), sort_keys=True)
    watchlist_text = ", ".join(_default_watchlist())
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TradingAgents Analysis</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5c6870;
      --line: #d8dee3;
      --panel: #f7f9fa;
      --accent: #0c6b58;
      --warn: #9d4b00;
      --bad: #a33131;
      --good: #0b6f45;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #ffffff; color: var(--ink); }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; }}
    main {{
      display: grid;
      grid-template-columns: minmax(300px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 66px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
      background: var(--panel);
    }}
    section {{ padding: 20px 24px; overflow: auto; }}
    fieldset {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0 0 14px;
      padding: 14px;
      background: #fff;
    }}
    legend {{ padding: 0 6px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    label {{ display: block; font-size: 12px; font-weight: 650; color: #2c373d; margin: 10px 0 5px; }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      background: #fff;
    }}
    textarea {{ min-height: 92px; resize: vertical; line-height: 1.35; }}
    input[type="checkbox"] {{ width: auto; margin-right: 8px; }}
    input[type="range"] {{ padding: 0; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .checkrow {{ display: flex; align-items: center; margin-top: 10px; color: #2c373d; font-size: 13px; }}
    .hint {{ color: var(--muted); font-size: 12px; line-height: 1.35; margin: 8px 0 0; }}
    .hint.warn {{ color: var(--warn); }}
    .factor-row {{ display: grid; grid-template-columns: minmax(130px, 1fr) 110px 44px; gap: 8px; align-items: center; margin: 8px 0; }}
    .factor-row span {{ font-size: 12px; overflow-wrap: anywhere; }}
    .factor-row output {{ font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; text-align: right; }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 10px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{ background: var(--ink); margin-top: 8px; }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; font-size: 12px; color: var(--muted); background: #fff; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .metric {{ border-bottom: 2px solid var(--line); padding: 8px 0; min-width: 0; }}
    .metric b {{ display: block; font-size: 20px; overflow-wrap: anywhere; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .bullish {{ color: var(--good); }}
    .bearish {{ color: var(--bad); }}
    .neutral {{ color: var(--warn); }}
    .dashboard-hero {{
      border-top: 4px solid var(--accent);
      border-bottom: 1px solid var(--line);
      padding: 16px 0 18px;
      margin-bottom: 18px;
    }}
    .hero-line {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
    .action-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      border-radius: 6px;
      padding: 7px 10px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
      color: #fff;
      background: var(--accent);
    }}
    .action-badge.buy {{ background: var(--good); }}
    .action-badge.sell {{ background: var(--bad); }}
    .action-badge.hold, .action-badge.watch {{ background: var(--warn); }}
    .hero-copy {{ max-width: 760px; }}
    .hero-copy h2 {{ margin: 0 0 6px; font-size: 24px; }}
    .hero-copy p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .stat-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 16px; }}
    .stat-item {{ border-left: 3px solid var(--line); padding: 4px 0 4px 10px; min-width: 0; }}
    .stat-item b {{ display: block; font-size: 18px; overflow-wrap: anywhere; }}
    .stat-item span {{ color: var(--muted); font-size: 12px; }}
    .summary-columns {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(280px, .85fr); gap: 18px; align-items: start; }}
    .dashboard-panel {{ border-top: 1px solid var(--line); padding-top: 14px; margin-bottom: 18px; }}
    .dashboard-panel h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .dashboard-panel p {{ line-height: 1.45; }}
    .levels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
    .level {{ border-left: 3px solid var(--line); padding: 6px 0 6px 10px; }}
    .level strong {{ display: block; font-size: 16px; }}
    .level span {{ display: block; color: var(--muted); font-size: 12px; }}
    .level.buy-level {{ border-left-color: var(--good); }}
    .level.sell-level {{ border-left-color: var(--accent); }}
    .level.stop-level {{ border-left-color: var(--bad); }}
    .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
    .strategy-card {{ border-top: 3px solid var(--line); padding: 8px 0 10px; min-width: 0; }}
    .strategy-card.consider {{ border-top-color: var(--good); }}
    .strategy-card.conditional {{ border-top-color: var(--warn); }}
    .strategy-card.wait {{ border-top-color: var(--warn); }}
    .strategy-card.avoid_now {{ border-top-color: var(--bad); }}
    .strategy-card header {{ display: flex; justify-content: space-between; gap: 8px; align-items: start; margin: 0 0 6px; }}
    .strategy-card h3 {{ margin: 0; font-size: 14px; }}
    .strategy-card small {{ color: var(--muted); line-height: 1.35; }}
    .strategy-card dl {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; margin: 8px 0; }}
    .strategy-card dt {{ color: var(--muted); font-size: 11px; }}
    .strategy-card dd {{ margin: 0; font-size: 13px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
    .verdict {{ font-size: 11px; font-weight: 800; text-transform: uppercase; color: var(--muted); }}
    .factor-list {{ display: grid; gap: 8px; }}
    .factor-item {{ display: grid; grid-template-columns: minmax(150px, 1fr) 72px; gap: 10px; align-items: center; }}
    .factor-item small {{ color: var(--muted); display: block; margin-top: 2px; line-height: 1.3; }}
    .factor-score {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .bar {{ height: 5px; background: #edf1f3; border-radius: 999px; overflow: hidden; margin-top: 5px; }}
    .bar span {{ display: block; height: 100%; width: 0; }}
    .bar .positive {{ background: var(--good); }}
    .bar .negative {{ background: var(--bad); }}
    .muted-list {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .muted-list li {{ margin: 4px 0; }}
    .rank-list {{ display: grid; gap: 10px; }}
    .rank-row {{ border-top: 1px solid var(--line); padding: 12px 0; display: grid; grid-template-columns: 92px minmax(0, 1fr) 120px; gap: 12px; align-items: start; }}
    .rank-symbol {{ font-size: 20px; font-weight: 800; }}
    .rank-score {{ text-align: right; font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; }}
    .rank-meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 7px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .data-pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; font-size: 12px; color: var(--muted); background: #fff; }}
    .agent-room {{ display: grid; gap: 12px; }}
    .agent-room details {{ border-top: 1px solid var(--line); padding-top: 10px; }}
    .agent-room summary {{ cursor: pointer; font-weight: 750; margin-bottom: 8px; }}
    .agent-text {{ white-space: pre-wrap; color: #2c373d; line-height: 1.42; font-size: 13px; max-height: 360px; overflow: auto; background: #fbfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .agent-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .agreement-map {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 10px 0 12px; }}
    .agreement-item {{ border-left: 3px solid var(--line); padding: 6px 0 6px 10px; min-width: 0; }}
    .agreement-item b {{ display: block; overflow-wrap: anywhere; }}
    .agreement-item span {{ color: var(--muted); font-size: 12px; }}
    .disagreement-list {{ margin: 8px 0 14px; padding-left: 18px; color: var(--muted); }}
    .disagreement-list li {{ margin: 5px 0; }}
    details.details-block {{ border-top: 1px solid var(--line); padding-top: 12px; margin-top: 8px; }}
    details.details-block summary {{ cursor: pointer; font-weight: 700; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; }}
    ul {{ padding-left: 20px; }}
    pre {{ white-space: pre-wrap; border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfd; max-height: 520px; overflow: auto; }}
    .tabs {{ display: flex; gap: 6px; margin: 4px 0 14px; }}
    .tab {{ width: auto; background: #fff; color: var(--ink); border: 1px solid var(--line); padding: 7px 10px; }}
    .tab.active {{ background: var(--ink); color: white; border-color: var(--ink); }}
    .hidden {{ display: none; }}
    .error {{ color: var(--bad); white-space: pre-wrap; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .summary-columns {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TradingAgents Analysis</h1>
    <div class="toolbar">
      <span class="pill" id="status">Idle</span>
    </div>
  </header>
  <main>
    <aside>
      <form id="analysis-form">
        <fieldset>
          <legend>Run</legend>
          <div class="grid2">
            <div>
              <label for="ticker">Ticker</label>
              <input id="ticker" name="ticker" value="NVDA" required>
            </div>
            <div>
              <label for="date">Date</label>
              <input id="date" name="date" type="date" value="{today}">
            </div>
          </div>
          <label for="horizon">Horizon</label>
          <select id="horizon" name="horizon">
            <option value="short_term_1_2_weeks">short term 1-2 weeks</option>
            <option value="swing_1_4_weeks" selected>swing 1-4 weeks</option>
            <option value="position_1_3_months">position 1-3 months</option>
            <option value="long_3_6_months">long 3-6 months</option>
          </select>
          <label for="data_provider">Data Provider</label>
          <select id="data_provider" name="data_provider">
            <option value="auto">auto</option>
            <option value="yfinance">yfinance</option>
            <option value="polygon" selected>polygon</option>
            <option value="openbb">openbb</option>
          </select>
          <p id="data_provider_status" class="hint"></p>
          <label for="competitors">Competitors</label>
          <input id="competitors" name="competitors" placeholder="AMD,INTC,AVGO">
        </fieldset>

        <fieldset>
          <legend>Options</legend>
          <label class="checkrow"><input id="disable_options_scan" type="checkbox"> Disable options scan</label>
          <label for="min_unusual_option_notional">Min Unusual Notional</label>
          <input id="min_unusual_option_notional" type="number" min="0" step="50000" value="500000">
          <label for="min_option_volume_oi_ratio">Min Volume/OI Ratio</label>
          <input id="min_option_volume_oi_ratio" type="number" min="0" step="0.1" value="3.0">
        </fieldset>

        <fieldset>
          <legend>LLM</legend>
          <label class="checkrow"><input id="enable_narrative" type="checkbox"> Narrative</label>
          <label class="checkrow"><input id="enable_llm_insights" type="checkbox"> Insight Block</label>
          <label class="checkrow"><input id="enable_tradingagents_review" type="checkbox"> TradingAgents Review</label>
          <div class="grid2">
            <div>
              <label for="llm_provider">Provider</label>
              <select id="llm_provider">
                <option value="openai">openai</option>
                <option value="google">google</option>
                <option value="anthropic">anthropic</option>
                <option value="xai">xai</option>
                <option value="openrouter">openrouter</option>
                <option value="ollama">ollama</option>
              </select>
            </div>
            <div>
              <label for="llm_model_select">Model</label>
              <select id="llm_model_select"></select>
            </div>
          </div>
          <input id="llm_model_custom" class="hidden" placeholder="Custom model id">
          <p id="llm_status" class="hint"></p>
        </fieldset>

        <fieldset>
          <legend>Weights</legend>
          <div id="factors"></div>
        </fieldset>

        <fieldset>
          <legend>State</legend>
          <label class="checkrow"><input id="no_state" type="checkbox"> Disable state delta</label>
          <label class="checkrow"><input id="force_refresh" type="checkbox"> Refresh cached report</label>
          <label for="state_store">State Store</label>
          <input id="state_store" value="state/analysis_state.sqlite">
          <label for="portfolio_path">Portfolio Snapshot</label>
          <input id="portfolio_path" value="{DEFAULT_PORTFOLIO_PATH}">
        </fieldset>
        <button id="run" type="submit">Run Analysis</button>
      </form>

      <form id="best-buy-form">
        <fieldset>
          <legend>Best Buy</legend>
          <label for="watchlist">Care Tickers</label>
          <textarea id="watchlist">{watchlist_text}</textarea>
          <div class="grid2">
            <div>
              <label for="best_buy_date">As Of</label>
              <input id="best_buy_date" type="date" value="{today}">
            </div>
            <div>
              <label for="best_buy_reports_glob">Reports Glob</label>
              <input id="best_buy_reports_glob" value="{DEFAULT_REPORTS_GLOB}">
            </div>
          </div>
          <label for="best_buy_sizing_config">Sizing Config</label>
          <input id="best_buy_sizing_config" value="{DEFAULT_SIZING_CONFIG}">
          <label for="best_buy_positions_path">Positions Ledger</label>
          <input id="best_buy_positions_path" value="{DEFAULT_POSITIONS_PATH}">
          <label class="checkrow"><input id="best_buy_fetch_prices" type="checkbox"> Fetch current yfinance prices</label>
          <label class="checkrow"><input id="best_buy_run_missing" type="checkbox"> Run analysis for missing reports</label>
          <label class="checkrow"><input id="best_buy_refresh_stale" type="checkbox"> Refresh stale promising reports</label>
          <p class="hint">Ranks latest reports for the watchlist. Missing and stale promising reports can be generated first using the current analysis settings.</p>
          <button id="best-buy-run" class="secondary" type="submit">Pick Best Buy</button>
        </fieldset>
      </form>
    </aside>
    <section>
      <div class="tabs">
        <button class="tab active" data-tab="summary" type="button">Summary</button>
        <button class="tab" data-tab="best-buy" type="button">Best Buy</button>
        <button class="tab" data-tab="markdown" type="button">Markdown</button>
        <button class="tab" data-tab="json" type="button">JSON</button>
      </div>
      <div id="summary"></div>
      <div id="best-buy" class="hidden"></div>
      <pre id="markdown" class="hidden"></pre>
      <pre id="json" class="hidden"></pre>
    </section>
  </main>
  <script>
    const DEFAULT_WEIGHTS = {defaults_json};
    const DATA_PROVIDER_ENV = {data_provider_env_json};
    const LLM_MODELS = {llm_models_json};
    const LLM_ENV = {llm_env_json};
    let bestBuyPoll = null;
    const factors = document.getElementById('factors');
    for (const [name, value] of Object.entries(DEFAULT_WEIGHTS)) {{
      const row = document.createElement('div');
      row.className = 'factor-row';
      row.innerHTML = `<span>${{name}}</span><input type="range" min="0" max="0.3" step="0.005" value="${{value}}" data-factor="${{name}}"><output>${{Number(value).toFixed(3)}}</output>`;
      const slider = row.querySelector('input');
      const out = row.querySelector('output');
      slider.addEventListener('input', () => out.value = Number(slider.value).toFixed(3));
      factors.appendChild(row);
    }}

    const dataProvider = document.getElementById('data_provider');
    dataProvider.addEventListener('change', updateDataProviderStatus);
    updateDataProviderStatus();

    const llmProvider = document.getElementById('llm_provider');
    const llmModelSelect = document.getElementById('llm_model_select');
    const llmModelCustom = document.getElementById('llm_model_custom');
    llmProvider.addEventListener('change', () => {{
      populateModelSelect();
      updateLlmStatus();
    }});
    llmModelSelect.addEventListener('change', () => {{
      llmModelCustom.classList.toggle('hidden', llmModelSelect.value !== '__custom__');
    }});
    populateModelSelect('{DEFAULT_LLM_MODEL}');
    updateLlmStatus();

    document.querySelectorAll('.tab').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        for (const id of ['summary', 'best-buy', 'markdown', 'json']) {{
          document.getElementById(id).classList.toggle('hidden', id !== btn.dataset.tab);
        }}
      }});
    }});

    document.getElementById('analysis-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const status = document.getElementById('status');
      const run = document.getElementById('run');
      status.textContent = 'Running';
      run.disabled = true;
      document.getElementById('summary').innerHTML = '';
      document.getElementById('markdown').textContent = '';
      document.getElementById('json').textContent = '';
      const factorWeights = {{}};
      document.querySelectorAll('[data-factor]').forEach(input => factorWeights[input.dataset.factor] = Number(input.value));
      const payload = {{
        ticker: document.getElementById('ticker').value,
        date: document.getElementById('date').value,
        horizon: document.getElementById('horizon').value,
        data_provider: dataProvider.value,
        competitors: document.getElementById('competitors').value,
        disable_options_scan: document.getElementById('disable_options_scan').checked,
        min_unusual_option_notional: Number(document.getElementById('min_unusual_option_notional').value),
        min_option_volume_oi_ratio: Number(document.getElementById('min_option_volume_oi_ratio').value),
        enable_narrative: document.getElementById('enable_narrative').checked,
        enable_llm_insights: document.getElementById('enable_llm_insights').checked,
        enable_tradingagents_review: document.getElementById('enable_tradingagents_review').checked,
        llm_provider: llmProvider.value,
        llm_model: getSelectedModel(),
        no_state: document.getElementById('no_state').checked,
        force_refresh: document.getElementById('force_refresh').checked,
        state_store: document.getElementById('state_store').value,
        portfolio_path: document.getElementById('portfolio_path').value,
        factor_weights: factorWeights
      }};
      try {{
        const res = await fetch('/api/analyze', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Analysis failed');
        renderSummary(data);
        document.getElementById('markdown').textContent = data.markdown;
        document.getElementById('json').textContent = JSON.stringify(data.report, null, 2);
        status.textContent = data.cache_hit ? 'Complete (cached)' : 'Complete';
      }} catch (err) {{
        document.getElementById('summary').innerHTML = `<div class="error">${{err.message}}</div>`;
        status.textContent = 'Error';
      }} finally {{
        run.disabled = false;
      }}
    }});

    document.getElementById('best-buy-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const status = document.getElementById('status');
      const run = document.getElementById('best-buy-run');
      const runMissing = document.getElementById('best_buy_run_missing').checked;
      const refreshStale = document.getElementById('best_buy_refresh_stale').checked;
      const jobId = `best-buy-${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
      status.textContent = runMissing || refreshStale ? 'Analyzing + ranking' : 'Ranking';
      run.disabled = true;
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelector('[data-tab="best-buy"]').classList.add('active');
      for (const id of ['summary', 'best-buy', 'markdown', 'json']) {{
        document.getElementById(id).classList.toggle('hidden', id !== 'best-buy');
      }}
      document.getElementById('best-buy').innerHTML = `
        <div class="dashboard-panel">
          <h2>Best Buy progress</h2>
          <div id="best-buy-progress">${{runMissing || refreshStale ? 'Starting needed analyses...' : 'Ranking cached reports...'}}</div>
        </div>
      `;
      startBestBuyPolling(jobId);
      const factorWeights = {{}};
      document.querySelectorAll('[data-factor]').forEach(input => factorWeights[input.dataset.factor] = Number(input.value));
      const payload = {{
        job_id: jobId,
        tickers: document.getElementById('watchlist').value,
        date: document.getElementById('best_buy_date').value,
        reports_glob: document.getElementById('best_buy_reports_glob').value,
        sizing_config: document.getElementById('best_buy_sizing_config').value,
        positions_path: document.getElementById('best_buy_positions_path').value,
        fetch_current_prices: document.getElementById('best_buy_fetch_prices').checked,
        run_missing_reports: runMissing,
        refresh_stale_reports: refreshStale,
        horizon: document.getElementById('horizon').value,
        data_provider: dataProvider.value,
        disable_options_scan: document.getElementById('disable_options_scan').checked,
        min_unusual_option_notional: Number(document.getElementById('min_unusual_option_notional').value),
        min_option_volume_oi_ratio: Number(document.getElementById('min_option_volume_oi_ratio').value),
        enable_narrative: document.getElementById('enable_narrative').checked,
        enable_llm_insights: document.getElementById('enable_llm_insights').checked,
        enable_tradingagents_review: document.getElementById('enable_tradingagents_review').checked,
        llm_provider: llmProvider.value,
        llm_model: getSelectedModel(),
        no_state: document.getElementById('no_state').checked,
        state_store: document.getElementById('state_store').value,
        portfolio_path: document.getElementById('portfolio_path').value,
        factor_weights: factorWeights
      }};
      try {{
        const res = await fetch('/api/best-buy', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Ranking failed');
        renderBestBuy(data);
        status.textContent = 'Ranked';
      }} catch (err) {{
        document.getElementById('best-buy').innerHTML = `<div class="error">${{err.message}}</div>`;
        status.textContent = 'Error';
      }} finally {{
        stopBestBuyPolling();
        run.disabled = false;
      }}
    }});

    function startBestBuyPolling(jobId) {{
      stopBestBuyPolling();
      const poll = async () => {{
        try {{
          const res = await fetch(`/api/best-buy-status?job_id=${{encodeURIComponent(jobId)}}`);
          const data = await res.json();
          renderBestBuyProgress(data);
          if (data.state === 'complete' || data.state === 'error') stopBestBuyPolling();
        }} catch (_err) {{
          // Keep the main request in charge; polling is best-effort.
        }}
      }};
      poll();
      bestBuyPoll = setInterval(poll, 1000);
    }}
    function stopBestBuyPolling() {{
      if (bestBuyPoll) clearInterval(bestBuyPoll);
      bestBuyPoll = null;
    }}
    function renderBestBuyProgress(data) {{
      const el = document.getElementById('best-buy-progress');
      if (!el || !data || data.ok === false) return;
      const total = Number(data.total || 0);
      const completed = Number(data.completed || 0);
      const pct = total > 0 ? Math.round((completed / total) * 100) : (data.phase === 'complete' ? 100 : 0);
      const current = data.current_symbol
        ? `<span class="data-pill">Current: ${{escapeHtml(data.current_symbol)}} (${{escapeHtml(data.current_reason || '')}})</span>`
        : '';
      const generated = data.generated || [];
      const generatedList = generated.length
        ? `<ul class="muted-list">${{generated.map(x => `<li>${{escapeHtml(x.symbol)}} (${{escapeHtml(x.reason || 'generated')}}): ${{escapeHtml(x.json_path || '')}}</li>`).join('')}}</ul>`
        : '';
      const error = data.error ? `<p class="error">${{escapeHtml(data.error)}}</p>` : '';
      el.innerHTML = `
        <div class="pill-row">
          <span class="data-pill">State: ${{escapeHtml(data.state || 'starting')}}</span>
          <span class="data-pill">Phase: ${{escapeHtml(data.phase || 'starting')}}</span>
          <span class="data-pill">Progress: ${{completed}}/${{total}}</span>
          <span class="data-pill">Elapsed: ${{data.elapsed_seconds === undefined ? '—' : fmtNum(data.elapsed_seconds)}}s</span>
          ${{current}}
        </div>
        <div class="bar"><span class="positive" style="width:${{pct}}%"></span></div>
        ${{generatedList}}
        ${{error}}
      `;
    }}

    function renderSummary(data) {{
      const r = data.report;
      const kf = r.key_features || {{}};
      const scoring = kf.model_scoring || {{}};
      const tech = kf.technical || {{}};
      const market = kf.market_context || {{}};
      const decision = kf.decision_summary || {{}};
      const portfolio = kf.portfolio_context || {{}};
      const holding = portfolio.holding || {{}};
      const account = portfolio.account || {{}};
      const entry = decision.entry || {{}};
      const exit = decision.exit || {{}};
      const buyZone = entry.preferred_buy_zone || {{}};
      const target = kf.price_target || {{}};
      const optionStrategies = kf.option_strategies || {{}};
      const narrative = kf.llm_narrative || {{}};
      const insights = kf.llm_insights || {{}};
      const tradingReview = kf.tradingagents_review || {{}};
      const allFactors = [...(scoring.factor_scores || [])].sort((a, b) => Math.abs(b.weighted_score || 0) - Math.abs(a.weighted_score || 0));
      const topPositive = allFactors.filter(f => Number(f.weighted_score || 0) > 0).slice(0, 5);
      const topNegative = allFactors.filter(f => Number(f.weighted_score || 0) < 0).slice(0, 5);
      const targetSources = target.source_weights || [];
      const risks = r.risk_flags || [];
      const reviewContext = {{
        direction: r.direction,
        action: decision.action || decision.label || r.direction,
        composite: scoring.composite_score,
        confidence: r.confidence,
        topPositive,
        topNegative,
        riskFlags: risks
      }};
      const action = String(decision.action || r.direction || 'watch').toLowerCase();
      const actionClass = actionBucket(action);
      document.getElementById('summary').innerHTML = `
        <div class="dashboard-hero">
          <div class="hero-line">
            <div class="hero-copy">
              <h2>${{escapeHtml(decision.label || decision.action || r.direction || 'Watch')}}</h2>
              <p>${{escapeHtml(decision.summary || r.thesis || '')}}</p>
            </div>
            <span class="action-badge ${{actionClass}}">${{escapeHtml(action)}}</span>
          </div>
          <div class="stat-strip">
            ${{statItem('Current price', fmtMoney(decision.current_price || tech.close))}}
            ${{statItem('Win probability', fmtPct(decision.estimated_win_probability))}}
            ${{statItem('Decision confidence', fmtPct(decision.confidence))}}
            ${{statItem('Base target', fmtMoney(target.base || decision.base_target))}}
            ${{statItem('Base upside', fmtPct(target.base_upside_pct || decision.base_upside_pct))}}
            ${{statItem('Composite', fmtSigned(scoring.composite_score))}}
          </div>
        </div>

        <div class="summary-columns">
          <div>
            <div class="dashboard-panel">
              <h2>Price plan</h2>
              <div class="levels">
                ${{levelItem('Starter buy at/below', fmtMoney(entry.starter_buy_at_or_below), 'buy-level')}}
                ${{levelItem('Preferred buy zone', formatRange(buyZone.low, buyZone.high), 'buy-level')}}
                ${{levelItem('Add below', fmtMoney(entry.add_below), 'buy-level')}}
                ${{levelItem('Take profit 1', fmtMoney(exit.take_profit_1), 'sell-level')}}
                ${{levelItem('Take profit 2', fmtMoney(exit.take_profit_2), 'sell-level')}}
                ${{levelItem('Stop / invalidate', fmtMoney(exit.stop_loss), 'stop-level')}}
              </div>
            </div>

            <div class="dashboard-panel">
              <h2>Option strategy candidates</h2>
              ${{renderOptionStrategies(optionStrategies)}}
            </div>

            <div class="dashboard-panel">
              <h2>Top drivers</h2>
              <div class="factor-list">
                ${{topPositive.length ? topPositive.map(factorItem).join('') : '<p class="hint">No positive factor drivers.</p>'}}
              </div>
            </div>

            <div class="dashboard-panel">
              <h2>Top risks</h2>
              <div class="factor-list">
                ${{topNegative.length ? topNegative.map(factorItem).join('') : '<p class="hint">No negative factor drivers.</p>'}}
              </div>
              ${{risks.length ? `<ul class="muted-list">${{risks.map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul>` : ''}}
            </div>
          </div>

          <div>
            <div class="dashboard-panel">
              <h2>Scenario target</h2>
              <div class="levels">
                ${{levelItem('Bear', fmtMoney(target.bear), 'stop-level')}}
                ${{levelItem('Base', `${{fmtMoney(target.base)}} (${{fmtPct(target.base_upside_pct)}})`, 'sell-level')}}
                ${{levelItem('Bull', fmtMoney(target.bull), 'sell-level')}}
                ${{levelItem('Target confidence', fmtPct(target.confidence), '')}}
              </div>
              <table>
                <thead><tr><th>Source</th><th>Target</th><th>Weight</th></tr></thead>
                <tbody>${{targetSources.slice(0, 5).map(s => `<tr><td>${{escapeHtml(s.name || '')}}</td><td>${{fmtMoney(s.target)}}</td><td>${{fmtNum(s.weight)}}</td></tr>`).join('')}}</tbody>
              </table>
            </div>

            <div class="dashboard-panel">
              <h2>Market context</h2>
              <div class="pill-row">
                <span class="data-pill">SPY 20d: ${{fmtPct(market.spy_return_20d)}}</span>
                <span class="data-pill">VIX: ${{fmtNum(market.vix_level)}}</span>
                <span class="data-pill">Fear & Greed: ${{fmtNum(market.fear_greed_score)}} ${{escapeHtml(market.fear_greed_rating || '')}}</span>
                <span class="data-pill">Provider: ${{escapeHtml((r.data_quality || {{}}).data_provider_resolved || '—')}}</span>
              </div>
            </div>

            <div class="dashboard-panel">
              <h2>Portfolio context</h2>
              <div class="pill-row">
                <span class="data-pill">Position: ${{holding.has_position ? `${{fmtNum(holding.shares)}} sh / ${{fmtPct(holding.portfolio_weight)}}` : 'none'}}</span>
                <span class="data-pill">Cash: ${{fmtMoney(account.cash)}} (${{fmtPct(account.cash_pct)}})</span>
                <span class="data-pill">Short-put margin: ${{fmtPct(account.short_put_margin_utilization)}}</span>
                <span class="data-pill">Margin left: ${{fmtMoney(account.margin_remaining)}}</span>
              </div>
              ${{(decision.rationale || []).filter(x => String(x).includes('position') || String(x).includes('margin') || String(x).includes('Short puts')).length ? `<ul class="muted-list">${{(decision.rationale || []).filter(x => String(x).includes('position') || String(x).includes('margin') || String(x).includes('Short puts')).map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul>` : ''}}
            </div>

            <div class="dashboard-panel">
              <h2>LLM status</h2>
              <p class="hint">Narrative: ${{escapeHtml(kf.narrative_source || '—')}}${{narrative.provider ? ` / ${{escapeHtml(narrative.provider)}}:${{escapeHtml(narrative.model || '')}}` : ''}}</p>
              <p class="hint">Insight Block: ${{escapeHtml(insights.status || '—')}}${{insights.provider ? ` / ${{escapeHtml(insights.provider)}}:${{escapeHtml(insights.model || '')}}` : ''}}</p>
              <p class="hint">TradingAgents Review: ${{escapeHtml(tradingReview.status || '—')}}${{tradingReview.provider ? ` / ${{escapeHtml(tradingReview.provider)}}:${{escapeHtml(tradingReview.model || '')}}` : ''}}</p>
            </div>
          </div>
        </div>

        ${{renderTradingAgentsReview(tradingReview, reviewContext)}}

        <div class="dashboard-panel">
          <h2>Thesis</h2>
          <p>${{escapeHtml(r.thesis || '')}}</p>
          <div class="summary-columns">
            <div><h2>Bull case</h2><ul class="muted-list">${{(r.bull_case || []).map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul></div>
            <div><h2>Bear case</h2><ul class="muted-list">${{(r.bear_case || []).map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul></div>
          </div>
        </div>

        <details class="details-block">
          <summary>Detailed factor scorecard</summary>
          <table>
            <thead><tr><th>Factor</th><th>Pillar</th><th>Score</th><th>Weight</th><th>Weighted</th><th>Rationale</th></tr></thead>
            <tbody>${{allFactors.map(f => `<tr><td>${{escapeHtml(f.factor)}}</td><td>${{escapeHtml(f.pillar)}}</td><td>${{fmtSigned(f.score)}}</td><td>${{fmtNum(f.weight)}}</td><td>${{fmtSigned(f.weighted_score)}}</td><td>${{escapeHtml(f.rationale || '')}}</td></tr>`).join('')}}</tbody>
          </table>
        </details>

        <p class="hint">Saved: ${{escapeHtml(data.json_path)}}<br>${{escapeHtml(data.markdown_path)}}</p>
      `;
    }}
    function renderBestBuy(data) {{
      const best = data.best || null;
      const candidates = data.candidates || [];
      const notes = data.notes || [];
      const generated = data.generated_reports || [];
      document.getElementById('best-buy').innerHTML = `
        <div class="dashboard-hero">
          <div class="hero-line">
            <div class="hero-copy">
              <h2>${{best ? `${{escapeHtml(best.symbol)}} is the top buy candidate` : 'No buy candidate found'}}</h2>
              <p>${{best ? escapeHtml(best.reason || '') : 'No cached reports resolved for this watchlist.'}}</p>
            </div>
            <span class="action-badge ${{best ? actionBucket(best.action) : 'watch'}}">${{best ? escapeHtml(best.action) : 'watch'}}</span>
          </div>
          <div class="stat-strip">
            ${{statItem('Buy score', best ? fmtNum(best.rank_score) : '—')}}
            ${{statItem('Composite', best ? fmtSigned(best.composite) : '—')}}
            ${{statItem('Composite × confidence', best ? fmtSigned(best.conviction) : '—')}}
            ${{statItem('Base upside', best ? fmtPct(best.base_upside_pct) : '—')}}
            ${{statItem('Target weight', best ? fmtPct(best.target_weight) : '—')}}
            ${{statItem('Limit', best ? fmtMoney(best.limit_price) : '—')}}
            ${{statItem('Signal age', best && best.signal_age_days !== null ? `${{best.signal_age_days}}d` : '—')}}
          </div>
        </div>
        <div class="dashboard-panel">
          <h2>Ranked watchlist</h2>
          <div class="rank-list">
            ${{candidates.map(rankRow).join('') || '<p class="hint">No candidates available.</p>'}}
          </div>
        </div>
        <div class="dashboard-panel">
          <h2>Run context</h2>
          <div class="pill-row">
            <span class="data-pill">As of: ${{escapeHtml(data.as_of || '')}}</span>
            <span class="data-pill">Watchlist: ${{fmtNum((data.symbols || []).length)}} names</span>
            <span class="data-pill">Generated: ${{fmtNum(generated.length)}} reports</span>
            <span class="data-pill">Reports: ${{escapeHtml(data.reports_glob || '')}}</span>
          </div>
          ${{generated.length ? `<ul class="muted-list">${{generated.map(x => `<li>${{escapeHtml(x.symbol)}} (${{escapeHtml(x.reason || 'generated')}}): ${{escapeHtml(x.json_path || '')}}</li>`).join('')}}</ul>` : ''}}
          ${{notes.length ? `<ul class="muted-list">${{notes.map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul>` : ''}}
        </div>
      `;
    }}
    function rankRow(c) {{
      const stale = c.signal_age_days !== null && c.signal_age_days !== undefined && c.signal_age_days > 7 ? 'Stale' : '';
      const notes = (c.notes || []).slice(0, 2);
      return `
        <div class="rank-row">
          <div>
            <div class="rank-symbol">${{escapeHtml(c.symbol)}}</div>
            <span class="verdict">${{escapeHtml(c.action || '')}}</span>
          </div>
          <div>
            <strong>${{escapeHtml(c.reason || '')}}</strong>
            <div class="rank-meta">
              <span class="data-pill">Target ${{fmtPct(c.target_weight)}}</span>
              <span class="data-pill">C×C ${{fmtSigned(c.conviction)}}</span>
              <span class="data-pill">Δ ${{fmtPct(c.delta_pp)}}</span>
              <span class="data-pill">Limit ${{fmtMoney(c.limit_price)}}</span>
              <span class="data-pill">Stop ${{fmtMoney(c.stop_loss)}}</span>
              <span class="data-pill">Age ${{c.signal_age_days === null || c.signal_age_days === undefined ? '—' : `${{c.signal_age_days}}d`}}</span>
              ${{stale ? `<span class="data-pill">${{stale}}</span>` : ''}}
            </div>
            ${{notes.length ? `<ul class="muted-list">${{notes.map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul>` : ''}}
          </div>
          <div class="rank-score">${{fmtNum(c.rank_score)}}</div>
        </div>
      `;
    }}
    function statItem(label, value) {{
      return `<div class="stat-item"><b>${{value}}</b><span>${{escapeHtml(label)}}</span></div>`;
    }}
    function levelItem(label, value, cls) {{
      return `<div class="level ${{cls || ''}}"><strong>${{value}}</strong><span>${{escapeHtml(label)}}</span></div>`;
    }}
    function factorItem(f) {{
      const weighted = Number(f.weighted_score || 0);
      const width = Math.min(100, Math.max(6, Math.abs(weighted) * 1200));
      const polarity = weighted >= 0 ? 'positive' : 'negative';
      return `
        <div class="factor-item">
          <div>
            <strong>${{escapeHtml(f.factor || '')}}</strong>
            <small>${{escapeHtml(f.rationale || '')}}</small>
            <div class="bar"><span class="${{polarity}}" style="width:${{width}}%"></span></div>
          </div>
          <div class="factor-score ${{weighted >= 0 ? 'bullish' : 'bearish'}}">${{fmtSigned(weighted)}}</div>
        </div>
      `;
    }}
    function renderOptionStrategies(block) {{
      const strategies = block.strategies || [];
      const status = block.status || block.chain_status || 'unknown';
      const warning = block.capital_warning
        ? '<span class="data-pill">Buying power tight</span>'
        : '';
      const earnings = block.earnings_warning
        ? '<span class="data-pill">Earnings near</span>'
        : '';
      const header = `
        <div class="pill-row">
          <span class="data-pill">Status: ${{escapeHtml(status)}}</span>
          <span class="data-pill">Best: ${{escapeHtml(block.recommended || 'none')}}</span>
          <span class="data-pill">Contracts: ${{fmtNum(block.contracts_considered || 0)}}</span>
          ${{warning}}${{earnings}}
        </div>
      `;
      if (!strategies.length) {{
        return header + `<p class="hint">${{escapeHtml(block.reason || 'No option chain strategy candidates available.')}}</p>`;
      }}
      return header + `<div class="strategy-grid">${{strategies.map(strategyCard).join('')}}</div>`;
    }}
    function renderTradingAgentsReview(block, reportContext) {{
      if (!block || !block.enabled || block.status === 'disabled') return '';
      if (block.status !== 'ok') {{
        const error = block.error
          ? `<pre class="mini-log">${{escapeHtml(block.error)}}</pre>`
          : '';
        return `
          <div class="dashboard-panel">
            <h2>TradingAgents review</h2>
            <p class="hint">Status: ${{escapeHtml(block.status || 'unknown')}}</p>
            ${{error}}
          </div>
        `;
      }}
      const analysis = block.analysis || {{}};
      const hypotheses = analysis.factor_hypotheses || [];
      const critiques = analysis.candidate_risk_critiques || [];
      const overfit = analysis.overfit_explanations || [];
      const features = analysis.feature_recommendations || [];
      const graphContext = (block.graph_contexts || [])[0] || {{}};
      const disagreementItems = agentDisagreements(reportContext, graphContext, analysis);
      return `
        <div class="dashboard-panel">
          <h2>TradingAgents review</h2>
          ${{renderAgentAgreement(reportContext, graphContext, analysis)}}
          ${{renderDisagreements(disagreementItems)}}
          <div class="summary-columns">
            <div>
              <h2>Factor hypotheses</h2>
              ${{hypotheses.length ? `<ul class="muted-list">${{hypotheses.slice(0, 5).map(x => `<li><strong>${{escapeHtml(x.name || '')}}</strong>: ${{escapeHtml(x.rationale || '')}}</li>`).join('')}}</ul>` : '<p class="hint">No factor hypotheses returned.</p>'}}
            </div>
            <div>
              <h2>Next data/features</h2>
              ${{features.length ? `<ul class="muted-list">${{features.slice(0, 5).map(x => `<li><strong>${{escapeHtml(x.feature_or_dataset || '')}}</strong>: ${{escapeHtml(x.reason || '')}}</li>`).join('')}}</ul>` : '<p class="hint">No feature recommendations returned.</p>'}}
            </div>
          </div>
          ${{critiques.length ? `<h2>Candidate risk critiques</h2><ul class="muted-list">${{critiques.slice(0, 5).map(x => `<li><strong>${{escapeHtml(x.candidate_id || '')}}</strong>: ${{escapeHtml(x.concern || '')}}</li>`).join('')}}</ul>` : ''}}
          ${{overfit.length ? `<h2>Overfit warnings</h2><ul class="muted-list">${{overfit.slice(0, 5).map(x => `<li><strong>${{escapeHtml(x.candidate_id || '')}}</strong>: ${{escapeHtml(x.evidence || '')}}</li>`).join('')}}</ul>` : ''}}
          ${{renderAgentRoom(graphContext)}}
        </div>
      `;
    }}
    function renderAgentAgreement(reportContext, graphContext, analysis) {{
      if (!graphContext || !Object.keys(graphContext).length) return '';
      const status = graphContext.status || 'unknown';
      const reasons = (graphContext.selection_reasons || []).join(', ');
      const processed = graphContext.processed_decision || '—';
      const conflict = conflictLevel(agentDisagreements(reportContext, graphContext, analysis).length);
      const bull = firstFactorName(reportContext.topPositive) || firstHypothesisName(analysis) || '—';
      const bear = firstFactorName(reportContext.topNegative) || firstRiskFlag(reportContext) || '—';
      return `
        <div class="pill-row" style="margin-bottom:12px">
          <span class="data-pill">Graph: ${{escapeHtml(status)}}</span>
          <span class="data-pill">Context: ${{escapeHtml(graphContext.symbol || '')}} ${{escapeHtml(graphContext.as_of_date || '')}}</span>
          <span class="data-pill">Reason: ${{escapeHtml(reasons || 'single report')}}</span>
          <span class="data-pill">Graph decision: ${{escapeHtml(processed)}}</span>
        </div>
        <div class="agreement-map">
          ${{agreementItem('Quant direction', reportContext.direction || '—')}}
          ${{agreementItem('Composite / confidence', `${{fmtSigned(reportContext.composite)}} / ${{fmtNum(reportContext.confidence)}}`)}}
          ${{agreementItem('Analysis action', reportContext.action || '—')}}
          ${{agreementItem('Risk judge', processed)}}
          ${{agreementItem('Main bull evidence', bull)}}
          ${{agreementItem('Main bear concern', bear)}}
          ${{agreementItem('Conflict level', conflict)}}
        </div>
      `;
    }}
    function agreementItem(label, value) {{
      return `<div class="agreement-item"><b>${{escapeHtml(value || '—')}}</b><span>${{escapeHtml(label)}}</span></div>`;
    }}
    function renderDisagreements(items) {{
      if (!items.length) return '<p class="hint">No major model/agent disagreements detected.</p>';
      return `
        <h2>Disagreement highlights</h2>
        <ul class="disagreement-list">
          ${{items.map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}
        </ul>
      `;
    }}
    function agentDisagreements(reportContext, graphContext, analysis) {{
      const items = [];
      const quant = normalizeStance(reportContext.direction);
      const action = normalizeStance(reportContext.action);
      const graph = normalizeStance(graphContext.processed_decision || graphContext.final_trade_decision);
      if (quant && graph && quant !== graph) items.push(`Quant direction is ${{quant}}, while the TradingAgents graph resolves to ${{graph}}.`);
      if (action && graph && action !== graph) items.push(`Analysis action is ${{action}}, but the risk-judged graph decision is ${{graph}}.`);
      if (Number(reportContext.composite || 0) >= 0.15 && graph === 'bearish') items.push('Composite is bullish, but the graph risk process is bearish.');
      if (Number(reportContext.composite || 0) <= -0.15 && graph === 'bullish') items.push('Composite is bearish, but the graph risk process is bullish.');
      if ((reportContext.topPositive || []).length && (reportContext.topNegative || []).length) {{
        items.push(`Top model support is ${{firstFactorName(reportContext.topPositive)}}, while the largest model drag is ${{firstFactorName(reportContext.topNegative)}}.`);
      }}
      const featureCount = ((analysis || {{}}).feature_recommendations || []).filter(x => String(x.priority || '').toLowerCase() === 'high').length;
      if (featureCount >= 2) items.push(`${{featureCount}} high-priority feature gaps were recommended before leaning harder on this setup.`);
      if (graphContext.status && graphContext.status !== 'ok') items.push(`TradingAgents graph context is ${{graphContext.status}}, so the room transcript is incomplete.`);
      return [...new Set(items)].slice(0, 6);
    }}
    function normalizeStance(value) {{
      const s = String(value || '').toLowerCase();
      if (!s || s === '—') return '';
      if (s.includes('buy') || s.includes('bull') || s.includes('overweight')) return 'bullish';
      if (s.includes('sell') || s.includes('bear') || s.includes('underweight')) return 'bearish';
      if (s.includes('hold') || s.includes('neutral') || s.includes('watch') || s.includes('wait')) return 'neutral';
      return '';
    }}
    function conflictLevel(count) {{
      if (count >= 4) return 'high';
      if (count >= 2) return 'medium';
      return count === 1 ? 'low' : 'low';
    }}
    function firstFactorName(items) {{
      const first = (items || [])[0] || null;
      return first ? String(first.factor || first.name || '') : '';
    }}
    function firstHypothesisName(analysis) {{
      const first = (((analysis || {{}}).factor_hypotheses || [])[0]) || null;
      return first ? String(first.name || '') : '';
    }}
    function firstRiskFlag(reportContext) {{
      return String(((reportContext.riskFlags || [])[0]) || '');
    }}
    function renderAgentRoom(graphContext) {{
      if (!graphContext || !Object.keys(graphContext).length) return '';
      if (graphContext.status !== 'ok') {{
        return `
          <details class="details-block">
            <summary>TradingAgents room</summary>
            <p class="hint">Graph context unavailable: ${{escapeHtml(graphContext.status || 'unknown')}}</p>
            ${{graphContext.error ? `<div class="agent-text">${{escapeHtml(graphContext.error)}}</div>` : ''}}
          </details>
        `;
      }}
      const invest = graphContext.investment_debate || {{}};
      const risk = graphContext.risk_debate || {{}};
      return `
        <details class="details-block">
          <summary>TradingAgents room</summary>
          <div class="agent-room">
            <details open>
              <summary>Analysts</summary>
              <div class="agent-grid">
                ${{agentTextBlock('Market analyst', graphContext.market_report)}}
                ${{agentTextBlock('Sentiment analyst', graphContext.sentiment_report)}}
                ${{agentTextBlock('News analyst', graphContext.news_report)}}
                ${{agentTextBlock('Fundamentals analyst', graphContext.fundamentals_report)}}
              </div>
            </details>
            <details>
              <summary>Research debate</summary>
              <div class="agent-grid">
                ${{agentTextBlock('Bull researcher', invest.bull_history)}}
                ${{agentTextBlock('Bear researcher', invest.bear_history)}}
                ${{agentTextBlock('Research manager', invest.judge_decision)}}
              </div>
            </details>
            <details>
              <summary>Trader plan</summary>
              ${{agentTextBlock('Trader', graphContext.trader_plan)}}
            </details>
            <details>
              <summary>Risk debate</summary>
              <div class="agent-grid">
                ${{agentTextBlock('Aggressive risk', risk.aggressive_history)}}
                ${{agentTextBlock('Neutral risk', risk.neutral_history)}}
                ${{agentTextBlock('Conservative risk', risk.conservative_history)}}
                ${{agentTextBlock('Risk judge', risk.judge_decision || graphContext.final_trade_decision)}}
              </div>
            </details>
          </div>
        </details>
      `;
    }}
    function agentTextBlock(title, text) {{
      const value = text === undefined || text === null || text === '' ? 'Not returned.' : String(text);
      return `<div><h2>${{escapeHtml(title)}}</h2><div class="agent-text">${{escapeHtml(value)}}</div></div>`;
    }}
    function strategyCard(s) {{
      const verdict = String(s.verdict || 'unavailable').toLowerCase();
      const contract = strategyContractLabel(s);
      const cost = s.debit !== undefined && s.debit !== null
        ? `Debit ${{fmtMoney(s.debit)}}`
        : `Premium ${{fmtMoney(s.premium)}}`;
      return `
        <div class="strategy-card ${{escapeHtml(verdict)}}">
          <header>
            <h3>${{escapeHtml(s.label || s.type || '')}}</h3>
            <span class="verdict">${{escapeHtml(verdict)}}</span>
          </header>
          <small>${{escapeHtml(contract)}}</small>
          <dl>
            <div><dt>Cost / credit</dt><dd>${{cost}}</dd></div>
            <div><dt>${{strategyBreakevenLabel(s)}}</dt><dd>${{strategyBreakevenValue(s)}}</dd></div>
            <div><dt>Max loss</dt><dd>${{fmtMoney(s.max_loss)}}</dd></div>
            <div><dt>Max profit</dt><dd>${{fmtMoney(s.max_profit)}}</dd></div>
            <div><dt>Est. POP</dt><dd>${{fmtPct(s.estimated_pop)}}</dd></div>
            <div><dt>Mid</dt><dd>${{fmtMoney(s.mid)}}</dd></div>
          </dl>
          <small>${{escapeHtml(s.reason || '')}}</small>
        </div>
      `;
    }}
    function strategyBreakevenLabel(s) {{
      if (s.type === 'sell_call') {{
        return s.cost_basis_breakeven !== null && s.cost_basis_breakeven !== undefined
          ? 'Basis breakeven'
          : 'Current - premium';
      }}
      if (s.type === 'sell_put') return 'Assignment breakeven';
      return 'Breakeven';
    }}
    function strategyBreakevenValue(s) {{
      if (s.type === 'sell_call') {{
        const primary = s.cost_basis_breakeven !== null && s.cost_basis_breakeven !== undefined
          ? fmtMoney(s.cost_basis_breakeven)
          : fmtMoney(s.premium_adjusted_reference_price);
        if (s.cost_basis_breakeven !== null && s.cost_basis_breakeven !== undefined) {{
          return `${{primary}} / current-premium ${{fmtMoney(s.premium_adjusted_reference_price)}}`;
        }}
        return primary;
      }}
      return fmtMoney(s.breakeven);
    }}
    function strategyContractLabel(s) {{
      const expiry = s.expiry || '—';
      const dte = s.dte === null || s.dte === undefined ? '—' : String(s.dte);
      if (s.long_strike !== undefined || s.short_strike !== undefined) {{
        return `${{expiry}} / ${{dte}} DTE / ${{fmtMoney(s.long_strike)}}-${{fmtMoney(s.short_strike)}}`;
      }}
      return `${{expiry}} / ${{dte}} DTE / ${{fmtMoney(s.strike)}} ${{s.option_type || ''}}`;
    }}
    function formatRange(low, high) {{
      if (low === null || low === undefined || high === null || high === undefined) return '—';
      return `${{fmtMoney(low)}} - ${{fmtMoney(high)}}`;
    }}
    function actionBucket(action) {{
      const a = String(action || '').toLowerCase();
      if (a.includes('buy') || a === 'bullish') return 'buy';
      if (a.includes('sell') || a === 'bearish') return 'sell';
      if (a.includes('hold')) return 'hold';
      return 'watch';
    }}
    function populateModelSelect(preferred) {{
      const provider = llmProvider.value;
      const models = LLM_MODELS[provider] || [];
      const target = preferred || models[0] || '';
      llmModelSelect.innerHTML = '';
      for (const model of models) {{
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model;
        llmModelSelect.appendChild(opt);
      }}
      const custom = document.createElement('option');
      custom.value = '__custom__';
      custom.textContent = 'Custom...';
      llmModelSelect.appendChild(custom);
      if (models.includes(target)) {{
        llmModelSelect.value = target;
        llmModelCustom.classList.add('hidden');
      }} else {{
        llmModelSelect.value = '__custom__';
        llmModelCustom.value = target;
        llmModelCustom.classList.remove('hidden');
      }}
    }}
    function getSelectedModel() {{
      return llmModelSelect.value === '__custom__'
        ? llmModelCustom.value.trim()
        : llmModelSelect.value;
    }}
    function updateLlmStatus() {{
      const status = LLM_ENV[llmProvider.value] || {{}};
      const el = document.getElementById('llm_status');
      if (status.requires_key === false) {{
        el.className = 'hint';
        el.textContent = 'No API key required by this provider.';
        return;
      }}
      const envVar = status.env_var || 'provider API key';
      if (status.configured) {{
        el.className = 'hint';
        el.textContent = `${{envVar}} is visible to this UI server.`;
      }} else {{
        el.className = 'hint warn';
        el.textContent = `${{envVar}} is not visible to this UI server; LLM mode will likely fall back or error.`;
      }}
    }}
    function updateDataProviderStatus() {{
      const status = DATA_PROVIDER_ENV[dataProvider.value] || {{}};
      const el = document.getElementById('data_provider_status');
      if (!el) return;
      if (status.requires_key === false || !status.requires_key) {{
        el.className = 'hint';
        el.textContent = dataProvider.value === 'auto'
          ? 'Auto tries Polygon first when POLYGON_API_KEY is visible, then falls back.'
          : 'No API key required by this data provider.';
        return;
      }}
      const envVar = status.env_var || 'provider API key';
      if (status.configured) {{
        el.className = 'hint';
        el.textContent = `${{envVar}} is visible to this UI server.`;
      }} else {{
        el.className = 'hint warn';
        el.textContent = `${{envVar}} is not visible to this UI server; Polygon will fall back unless restarted with the key.`;
      }}
    }}
    function fmtNum(v) {{ return v === null || v === undefined ? '—' : Number(v).toFixed(2); }}
    function fmtSigned(v) {{ return v === null || v === undefined ? '—' : (Number(v) >= 0 ? '+' : '') + Number(v).toFixed(4); }}
    function fmtPct(v) {{ return v === null || v === undefined ? '—' : (Number(v) * 100).toFixed(2) + '%'; }}
    function fmtMoney(v) {{ return v === null || v === undefined ? '—' : '$' + Number(v).toFixed(2); }}
    function escapeHtml(str) {{
      return String(str).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    handler = make_handler(output_dir=args.output_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Analysis UI listening at {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
