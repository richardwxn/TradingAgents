from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import time

from .config import AppConfig
from .pipeline import AnalysisOnlyMVP
from .providers import (
    PolygonMarketStatusProvider,
    PolygonNewsProvider,
    SECFetchError,
    SECFilingsProvider,
    YFinanceNewsProvider,
)
from .state_store import StateStore


class AnalysisRuntime:
    def __init__(self, config: AppConfig):
        self.config = config
        self.state = StateStore(config.state_db_path)
        self.market_status = PolygonMarketStatusProvider()
        self.news_provider = PolygonNewsProvider()
        self.news_fallback = YFinanceNewsProvider()
        self.filings_provider = SECFilingsProvider()

    def run_once(self) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        usage_date = now_utc.date().isoformat()
        if self.config.runtime.only_when_market_open:
            if not self.market_status.is_market_open():
                return {
                    "status": "skipped_market_closed",
                    "symbols_processed": [],
                }

        due_symbols = self._select_due_symbols(now_utc)
        if not due_symbols:
            return {"status": "no_symbols_due", "symbols_processed": []}

        gate_map = self._build_llm_gate_map(due_symbols)
        processed: list[dict[str, Any]] = []
        as_of_date = now_utc.date().strftime("%Y-%m-%d")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, dict[str, Any]] = {}

        # Pass 1: numeric/deterministic run for all due symbols.
        for symbol in due_symbols:
            prev_state = self.state.get_symbol_state(symbol)
            report = self._run_symbol(
                symbol=symbol,
                as_of_date=as_of_date,
                enable_llm=False if self.config.llm.mode != "always" else True,
            )
            save_path = self._save_symbol_report(report, output_dir, symbol)
            metrics = self._extract_report_metrics(report)
            reasons = list(gate_map.get(symbol, []))
            reasons.extend(
                self._delta_gate_reasons(
                    prev_state=prev_state,
                    new_metrics=metrics,
                    new_direction=report.direction,
                )
            )
            gate_map[symbol] = sorted(set(reasons))
            records[symbol] = {
                "report": report,
                "save_path": save_path,
                "metrics": metrics,
                "gate_reasons": gate_map[symbol],
            }

        llm_symbols = self._select_llm_symbols(
            symbols=due_symbols,
            gate_map=gate_map,
            usage_date=usage_date,
        )

        # Pass 2: selective LLM rerun only for gated symbols.
        if self.config.llm.mode == "selective":
            for symbol in llm_symbols:
                report = self._run_symbol(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    enable_llm=True,
                )
                save_path = self._save_symbol_report(report, output_dir, symbol)
                records[symbol]["report"] = report
                records[symbol]["save_path"] = save_path
                records[symbol]["metrics"] = self._extract_report_metrics(report)

        if llm_symbols:
            self.state.add_llm_usage(
                usage_date=usage_date,
                calls=len(llm_symbols),
                est_input_tokens=(
                    len(llm_symbols)
                    * self.config.runtime.llm_est_input_tokens_per_call
                ),
                est_output_tokens=(
                    len(llm_symbols)
                    * self.config.runtime.llm_est_output_tokens_per_call
                ),
            )

        # Persist state and build output.
        diagnostics: list[dict[str, Any]] = []
        for symbol in due_symbols:
            record = records[symbol]
            report = record["report"]
            metrics = record["metrics"]
            save_path = record["save_path"]
            self.state.upsert_symbol_state(
                symbol=symbol,
                last_price=metrics.get("close"),
                last_signal=report.direction,
                last_composite_score=metrics.get("composite_score"),
                last_options_unusual_count=metrics.get("options_unusual_count"),
            )
            filing_ctx = (
                report.to_json_dict()
                .get("key_features", {})
                .get("filings_context", {})
            )
            filing_accession = filing_ctx.get("latest_accession")
            if filing_accession:
                self.state.set_last_seen_filing_accession(
                    symbol=symbol,
                    accession=str(filing_accession),
                )
            news_items = self._get_news_items(symbol)
            news_ids = self._extract_news_ids(news_items)
            self.state.mark_news_seen(symbol, news_ids)
            processed.append(
                {
                    "symbol": symbol,
                    "direction": report.direction,
                    "confidence": report.confidence,
                    "path": str(save_path),
                    "llm_enabled": symbol in llm_symbols
                    if self.config.llm.mode != "off"
                    else False,
                    "llm_gate_reasons": record["gate_reasons"],
                }
            )
            diagnostics.append(
                {
                    "symbol": symbol,
                    "llm_selected": symbol in llm_symbols
                    if self.config.llm.mode != "off"
                    else False,
                    "gate_reasons": record["gate_reasons"],
                    "metrics": metrics,
                    "direction": report.direction,
                    "confidence": report.confidence,
                    "report_path": str(save_path),
                }
            )

        diagnostics_path = self._write_gate_diagnostics(
            output_dir=output_dir,
            diagnostics=diagnostics,
            run_time=now_utc,
        )
        return {
            "status": "ok",
            "symbols_processed": processed,
            "gate_diagnostics_path": str(diagnostics_path),
            "llm_quota": self._get_quota_snapshot(usage_date),
        }

    def run_loop(self) -> None:
        interval = max(5, int(self.config.runtime.interval_minutes)) * 60
        while True:
            self.run_once()
            time.sleep(interval)

    def _select_due_symbols(self, now_utc: datetime) -> list[str]:
        interval = timedelta(minutes=int(self.config.runtime.interval_minutes))
        due: list[str] = []
        for symbol in self.config.watchlist:
            state = self.state.get_symbol_state(symbol)
            if not state.last_run_time:
                due.append(symbol)
                continue
            try:
                last_run = datetime.fromisoformat(state.last_run_time)
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)
            except ValueError:
                due.append(symbol)
                continue
            if now_utc - last_run >= interval:
                due.append(symbol)
                continue
            news_items = self._get_news_items(symbol)
            new_ids = self._extract_new_news_ids(symbol, news_items)
            if new_ids:
                due.append(symbol)
                continue
            try:
                latest_filing = self.filings_provider.get_latest_filing(symbol)
            except SECFetchError:
                # Transient SEC fetch failure — skip the filings gate for
                # this symbol; news/price gates still applied above.
                continue
            latest_accession = (
                str(latest_filing.get("accession"))
                if latest_filing and latest_filing.get("accession")
                else None
            )
            if not latest_accession:
                continue
            seen_accession = self.state.get_last_seen_filing_accession(symbol)
            if seen_accession != latest_accession:
                due.append(symbol)
        return due

    def _select_llm_symbols(
        self,
        symbols: list[str],
        gate_map: dict[str, list[str]],
        usage_date: str,
    ) -> set[str]:
        if self.config.llm.mode == "off":
            return set()
        if self.config.llm.mode == "always":
            return self._apply_llm_quota(list(symbols), usage_date)
        # selective mode: call LLM only on gated symbols; prioritize more reasons.
        scored: list[tuple[int, str]] = []
        for symbol in symbols:
            reasons = gate_map.get(symbol, [])
            if not reasons:
                continue
            scored.append((len(reasons), symbol))
        scored.sort(reverse=True)
        capped = [symbol for _, symbol in scored]
        return self._apply_llm_quota(capped, usage_date)

    def _build_llm_gate_map(
        self,
        symbols: list[str],
    ) -> dict[str, list[str]]:
        gate_map: dict[str, list[str]] = {}
        for symbol in symbols:
            reasons: list[str] = []
            state = self.state.get_symbol_state(symbol)
            news_items = self._get_news_items(symbol)
            new_ids = self._extract_new_news_ids(symbol, news_items)
            if len(new_ids) >= self.config.runtime.llm_gate_min_new_headlines:
                reasons.append("new_headlines")

            try:
                latest_filing = self.filings_provider.get_latest_filing(symbol)
            except SECFetchError:
                latest_filing = None
            latest_accession = (
                str(latest_filing.get("accession"))
                if latest_filing and latest_filing.get("accession")
                else None
            )
            seen_accession = self.state.get_last_seen_filing_accession(symbol)
            if latest_accession and latest_accession != seen_accession:
                reasons.append("new_filing")

            latest_price = self._fetch_latest_price(symbol)
            if (
                latest_price is not None
                and state.last_price is not None
                and state.last_price > 0
            ):
                move = abs((latest_price / state.last_price) - 1.0)
                if move >= self.config.runtime.llm_gate_price_move_pct:
                    reasons.append("price_move")

            gate_map[symbol] = sorted(set(reasons))
        return gate_map

    def _delta_gate_reasons(
        self,
        prev_state: Any,
        new_metrics: dict[str, Any],
        new_direction: str,
    ) -> list[str]:
        reasons: list[str] = []
        prev_comp = prev_state.last_composite_score
        new_comp = new_metrics.get("composite_score")
        if prev_comp is not None and new_comp is not None:
            if (
                abs(float(new_comp) - float(prev_comp))
                >= self.config.runtime.llm_gate_composite_delta
            ):
                reasons.append("composite_shift")

        prev_opts = prev_state.last_options_unusual_count
        new_opts = new_metrics.get("options_unusual_count")
        if prev_opts is not None and new_opts is not None:
            if int(new_opts) - int(prev_opts) >= int(
                self.config.runtime.llm_gate_options_unusual_jump
            ):
                reasons.append("options_unusual_jump")

        if self.config.runtime.llm_gate_signal_change_only:
            if prev_state.last_signal and prev_state.last_signal != new_direction:
                reasons.append("signal_change")
        return reasons

    def _run_symbol(
        self,
        symbol: str,
        as_of_date: str,
        enable_llm: bool,
    ) -> Any:
        peers = self.config.peers.get(symbol, [])
        runner = AnalysisOnlyMVP(
            horizon="swing_1_4_weeks",
            data_provider=self.config.data_provider,
            competitors=peers,
            enable_llm_insights=enable_llm,
            llm_provider=self.config.llm.provider,
            llm_model=(
                self.config.llm.model_deep
                if enable_llm
                else self.config.llm.model_fast
            ),
            llm_base_url=self.config.llm.base_url,
            state_store_path=self.config.state_db_path,
        )
        return runner.run(symbol=symbol, as_of_date=as_of_date)

    def _save_symbol_report(
        self,
        report: Any,
        output_dir: Path,
        symbol: str,
    ) -> Path:
        runner = AnalysisOnlyMVP()
        return runner.save_report(report, output_dir=output_dir / symbol)

    def _extract_report_metrics(self, report: Any) -> dict[str, Any]:
        payload = report.to_json_dict()
        key_features = payload.get("key_features", {})
        return {
            "close": key_features.get("technical", {}).get("close"),
            "composite_score": key_features.get("model_scoring", {}).get(
                "composite_score"
            ),
            "options_unusual_count": int(
                key_features.get("options_flow", {}).get("unusual_count", 0)
            ),
        }

    def _apply_llm_quota(
        self,
        candidates: list[str],
        usage_date: str,
    ) -> set[str]:
        if not candidates:
            return set()
        usage = self.state.get_llm_usage(usage_date)
        remaining_calls_day = max(
            0,
            int(self.config.runtime.llm_max_calls_per_day) - int(usage.calls),
        )
        remaining_input_day = max(
            0,
            int(self.config.runtime.llm_max_est_input_tokens_per_day)
            - int(usage.est_input_tokens),
        )
        remaining_output_day = max(
            0,
            int(self.config.runtime.llm_max_est_output_tokens_per_day)
            - int(usage.est_output_tokens),
        )
        max_by_input = (
            remaining_input_day // max(1, self.config.runtime.llm_est_input_tokens_per_call)
        )
        max_by_output = (
            remaining_output_day // max(1, self.config.runtime.llm_est_output_tokens_per_call)
        )
        max_by_run = max(0, int(self.config.runtime.llm_max_calls_per_run))
        final_cap = min(
            max_by_run,
            remaining_calls_day,
            max_by_input,
            max_by_output,
        )
        return set(candidates[:final_cap])

    def _get_quota_snapshot(self, usage_date: str) -> dict[str, Any]:
        usage = self.state.get_llm_usage(usage_date)
        return {
            "usage_date": usage_date,
            "calls_used": usage.calls,
            "calls_limit": self.config.runtime.llm_max_calls_per_day,
            "est_input_tokens_used": usage.est_input_tokens,
            "est_input_tokens_limit": self.config.runtime.llm_max_est_input_tokens_per_day,
            "est_output_tokens_used": usage.est_output_tokens,
            "est_output_tokens_limit": self.config.runtime.llm_max_est_output_tokens_per_day,
        }

    def _write_gate_diagnostics(
        self,
        output_dir: Path,
        diagnostics: list[dict[str, Any]],
        run_time: datetime,
    ) -> Path:
        diag_dir = output_dir / "_runtime"
        diag_dir.mkdir(parents=True, exist_ok=True)
        ts = run_time.strftime("%Y%m%dT%H%M%SZ")
        file_path = diag_dir / f"gate_diagnostics_{ts}.json"
        payload = {
            "run_time_utc": run_time.isoformat(),
            "llm_mode": self.config.llm.mode,
            "diagnostics": diagnostics,
        }
        file_path.write_text(json.dumps(payload, indent=2))
        return file_path

    def _fetch_latest_price(self, symbol: str) -> float | None:
        try:
            items = self.news_fallback.get_news(symbol, limit=1)  # warm ticker cache
            _ = items
            # yfinance fast path
            import yfinance as yf

            hist = yf.download(
                symbol.upper(),
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            return None

    def _get_news_items(self, symbol: str) -> list[dict[str, Any]]:
        items = self.news_provider.get_news(symbol=symbol, limit=30)
        if items:
            return items
        return self.news_fallback.get_news(symbol=symbol, limit=30)

    def _extract_news_ids(self, news_items: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for item in news_items:
            nid = (
                item.get("id")
                or item.get("uuid")
                or item.get("article_url")
                or item.get("title")
            )
            if nid:
                ids.append(str(nid))
        return ids

    def _extract_new_news_ids(
        self,
        symbol: str,
        news_items: list[dict[str, Any]],
    ) -> list[str]:
        seen = self.state.get_seen_news_ids(symbol)
        all_ids = self._extract_news_ids(news_items)
        return [nid for nid in all_ids if nid not in seen]
