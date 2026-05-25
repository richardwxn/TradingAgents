from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import sqlite3


@dataclass
class SymbolState:
    symbol: str
    last_run_time: str | None
    last_price: float | None
    last_signal: str | None
    last_composite_score: float | None
    last_options_unusual_count: int | None
    daily_summary_date: str | None
    daily_summary_json: dict[str, Any] | None
    last_as_of_date: str | None = None
    last_confidence: float | None = None
    last_factor_scores: list[dict[str, Any]] | None = None
    last_thesis: str | None = None


@dataclass
class DailyLLMUsage:
    usage_date: str
    calls: int
    est_input_tokens: int
    est_output_tokens: int


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_state (
                    symbol TEXT PRIMARY KEY,
                    last_run_time TEXT,
                    last_price REAL,
                    last_signal TEXT,
                    last_composite_score REAL,
                    last_options_unusual_count INTEGER,
                    daily_summary_date TEXT,
                    daily_summary_json TEXT
                )
                """
            )
            self._ensure_column(
                conn,
                "symbol_state",
                "last_composite_score",
                "REAL",
            )
            self._ensure_column(
                conn,
                "symbol_state",
                "last_options_unusual_count",
                "INTEGER",
            )
            for col, ctype in (
                ("last_as_of_date", "TEXT"),
                ("last_confidence", "REAL"),
                ("last_factor_scores_json", "TEXT"),
                ("last_thesis", "TEXT"),
            ):
                self._ensure_column(conn, "symbol_state", col, ctype)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news_seen (
                    symbol TEXT NOT NULL,
                    news_id TEXT NOT NULL,
                    seen_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, news_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS filing_seen (
                    symbol TEXT PRIMARY KEY,
                    last_seen_filing_accession TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage_daily (
                    usage_date TEXT PRIMARY KEY,
                    calls INTEGER NOT NULL,
                    est_input_tokens INTEGER NOT NULL,
                    est_output_tokens INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS iv_history (
                    symbol TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    atm_iv_30d REAL,
                    atm_iv_60d REAL,
                    atm_iv_90d REAL,
                    skew_25d_30d REAL,
                    term_slope_30_to_60 REAL,
                    recorded_at_utc TEXT NOT NULL,
                    PRIMARY KEY (symbol, as_of_date)
                )
                """
            )

    def get_symbol_state(self, symbol: str) -> SymbolState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, last_run_time, last_price, last_signal,
                       last_composite_score, last_options_unusual_count,
                       daily_summary_date, daily_summary_json,
                       last_as_of_date, last_confidence,
                       last_factor_scores_json, last_thesis
                FROM symbol_state
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if not row:
            return SymbolState(
                symbol=symbol,
                last_run_time=None,
                last_price=None,
                last_signal=None,
                last_composite_score=None,
                last_options_unusual_count=None,
                daily_summary_date=None,
                daily_summary_json=None,
            )
        summary = json.loads(row[7]) if row[7] else None
        factor_scores = json.loads(row[10]) if row[10] else None
        return SymbolState(
            symbol=row[0],
            last_run_time=row[1],
            last_price=row[2],
            last_signal=row[3],
            last_composite_score=row[4],
            last_options_unusual_count=row[5],
            daily_summary_date=row[6],
            daily_summary_json=summary,
            last_as_of_date=row[8],
            last_confidence=row[9],
            last_factor_scores=factor_scores,
            last_thesis=row[11],
        )

    def upsert_symbol_state(
        self,
        symbol: str,
        last_price: float | None,
        last_signal: str | None,
        last_composite_score: float | None = None,
        last_options_unusual_count: int | None = None,
        daily_summary_date: str | None = None,
        daily_summary_json: dict[str, Any] | None = None,
        last_as_of_date: str | None = None,
        last_confidence: float | None = None,
        last_factor_scores: list[dict[str, Any]] | None = None,
        last_thesis: str | None = None,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        summary_str = (
            json.dumps(daily_summary_json) if daily_summary_json else None
        )
        factor_str = (
            json.dumps(last_factor_scores) if last_factor_scores else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_state (
                    symbol, last_run_time, last_price, last_signal,
                    last_composite_score, last_options_unusual_count,
                    daily_summary_date, daily_summary_json,
                    last_as_of_date, last_confidence,
                    last_factor_scores_json, last_thesis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_run_time = excluded.last_run_time,
                    last_price = excluded.last_price,
                    last_signal = excluded.last_signal,
                    last_composite_score = excluded.last_composite_score,
                    last_options_unusual_count = excluded.last_options_unusual_count,
                    daily_summary_date = COALESCE(excluded.daily_summary_date, symbol_state.daily_summary_date),
                    daily_summary_json = COALESCE(excluded.daily_summary_json, symbol_state.daily_summary_json),
                    last_as_of_date = COALESCE(excluded.last_as_of_date, symbol_state.last_as_of_date),
                    last_confidence = COALESCE(excluded.last_confidence, symbol_state.last_confidence),
                    last_factor_scores_json = COALESCE(excluded.last_factor_scores_json, symbol_state.last_factor_scores_json),
                    last_thesis = COALESCE(excluded.last_thesis, symbol_state.last_thesis)
                """,
                (
                    symbol,
                    now_iso,
                    last_price,
                    last_signal,
                    last_composite_score,
                    last_options_unusual_count,
                    daily_summary_date,
                    summary_str,
                    last_as_of_date,
                    last_confidence,
                    factor_str,
                    last_thesis,
                ),
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        col_type: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(r[1]) for r in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def get_seen_news_ids(self, symbol: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT news_id FROM news_seen WHERE symbol = ?",
                (symbol,),
            ).fetchall()
        return {str(r[0]) for r in rows}

    def mark_news_seen(self, symbol: str, news_ids: list[str]) -> None:
        if not news_ids:
            return
        seen_at = datetime.now(timezone.utc).isoformat()
        rows = [(symbol, str(nid), seen_at) for nid in news_ids]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO news_seen (symbol, news_id, seen_at)
                VALUES (?, ?, ?)
                """,
                rows,
            )

    def get_last_seen_filing_accession(self, symbol: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT last_seen_filing_accession
                FROM filing_seen
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if not row:
            return None
        return row[0]

    def set_last_seen_filing_accession(
        self,
        symbol: str,
        accession: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO filing_seen(symbol, last_seen_filing_accession)
                VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_seen_filing_accession = excluded.last_seen_filing_accession
                """,
                (symbol, accession),
            )

    def get_llm_usage(self, usage_date: str) -> DailyLLMUsage:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT usage_date, calls, est_input_tokens, est_output_tokens
                FROM llm_usage_daily
                WHERE usage_date = ?
                """,
                (usage_date,),
            ).fetchone()
        if not row:
            return DailyLLMUsage(
                usage_date=usage_date,
                calls=0,
                est_input_tokens=0,
                est_output_tokens=0,
            )
        return DailyLLMUsage(
            usage_date=row[0],
            calls=int(row[1] or 0),
            est_input_tokens=int(row[2] or 0),
            est_output_tokens=int(row[3] or 0),
        )

    def record_iv_snapshot(
        self,
        symbol: str,
        as_of_date: str,
        atm_iv_30d: float | None,
        atm_iv_60d: float | None,
        atm_iv_90d: float | None,
        skew_25d_30d: float | None,
        term_slope_30_to_60: float | None,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iv_history (
                    symbol, as_of_date, atm_iv_30d, atm_iv_60d, atm_iv_90d,
                    skew_25d_30d, term_slope_30_to_60, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, as_of_date) DO UPDATE SET
                    atm_iv_30d = excluded.atm_iv_30d,
                    atm_iv_60d = excluded.atm_iv_60d,
                    atm_iv_90d = excluded.atm_iv_90d,
                    skew_25d_30d = excluded.skew_25d_30d,
                    term_slope_30_to_60 = excluded.term_slope_30_to_60,
                    recorded_at_utc = excluded.recorded_at_utc
                """,
                (
                    symbol.upper(),
                    as_of_date,
                    atm_iv_30d,
                    atm_iv_60d,
                    atm_iv_90d,
                    skew_25d_30d,
                    term_slope_30_to_60,
                    now_iso,
                ),
            )

    def get_iv_history(
        self,
        symbol: str,
        before_date: str,
        window_days: int = 252,
    ) -> list[dict[str, Any]]:
        """Return IV snapshots with `as_of_date < before_date`, ordered ascending.

        `before_date` is exclusive so callers running a historical date never
        leak future IV observations. `window_days` is a calendar-day cap so
        any reasonable backtest cadence (weekly/daily/monthly) inherits a
        ~1Y trailing window without the caller doing the math.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT as_of_date, atm_iv_30d, atm_iv_60d, atm_iv_90d,
                       skew_25d_30d, term_slope_30_to_60
                FROM iv_history
                WHERE symbol = ?
                  AND as_of_date < ?
                  AND as_of_date >= date(?, ?)
                ORDER BY as_of_date ASC
                """,
                (
                    symbol.upper(),
                    before_date,
                    before_date,
                    f"-{int(window_days)} days",
                ),
            ).fetchall()
        return [
            {
                "as_of_date": r[0],
                "atm_iv_30d": r[1],
                "atm_iv_60d": r[2],
                "atm_iv_90d": r[3],
                "skew_25d_30d": r[4],
                "term_slope_30_to_60": r[5],
            }
            for r in rows
        ]

    def add_llm_usage(
        self,
        usage_date: str,
        calls: int,
        est_input_tokens: int,
        est_output_tokens: int,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_usage_daily(
                    usage_date, calls, est_input_tokens,
                    est_output_tokens, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(usage_date) DO UPDATE SET
                    calls = llm_usage_daily.calls + excluded.calls,
                    est_input_tokens = llm_usage_daily.est_input_tokens + excluded.est_input_tokens,
                    est_output_tokens = llm_usage_daily.est_output_tokens + excluded.est_output_tokens,
                    updated_at = excluded.updated_at
                """,
                (
                    usage_date,
                    calls,
                    est_input_tokens,
                    est_output_tokens,
                    now_iso,
                ),
            )

