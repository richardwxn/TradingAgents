from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class RuntimeConfig:
    schedule: str = "hourly"
    only_when_market_open: bool = True
    max_symbols_per_run_llm: int = 3
    interval_minutes: int = 60
    llm_gate_min_new_headlines: int = 2
    llm_gate_price_move_pct: float = 0.02
    llm_gate_composite_delta: float = 0.2
    llm_gate_options_unusual_jump: int = 2
    llm_gate_signal_change_only: bool = True
    llm_max_calls_per_day: int = 30
    llm_max_calls_per_run: int = 3
    llm_max_est_input_tokens_per_day: int = 120000
    llm_max_est_output_tokens_per_day: int = 60000
    llm_est_input_tokens_per_call: int = 2500
    llm_est_output_tokens_per_call: int = 800


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_fast: str = "gpt-5.4-mini"
    model_deep: str = "gpt-5.5"
    mode: str = "selective"  # selective|always|off
    base_url: str | None = None


@dataclass
class BenchmarksConfig:
    market: str = "SPY"
    sectors: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    watchlist: list[str] = field(default_factory=list)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    peers: dict[str, list[str]] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data_provider: str = "polygon"
    output_dir: str = "reports/analysis_runtime"
    state_db_path: str = "state/analysis_state.sqlite"


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_config(path)
    benchmarks_raw = raw.get("benchmarks", {})
    runtime_raw = raw.get("runtime", {})
    llm_raw = raw.get("llm", {})
    config = AppConfig(
        watchlist=[str(s).upper() for s in raw.get("watchlist", [])],
        benchmarks=BenchmarksConfig(
            market=str(benchmarks_raw.get("market", "SPY")).upper(),
            sectors={
                str(k): str(v).upper()
                for k, v in (benchmarks_raw.get("sectors", {}) or {}).items()
            },
        ),
        peers={
            str(k).upper(): [str(t).upper() for t in v]
            for k, v in (raw.get("peers", {}) or {}).items()
        },
        runtime=RuntimeConfig(
            schedule=str(runtime_raw.get("schedule", "hourly")),
            only_when_market_open=bool(
                runtime_raw.get("only_when_market_open", True)
            ),
            max_symbols_per_run_llm=int(
                runtime_raw.get("max_symbols_per_run_llm", 3)
            ),
            interval_minutes=int(runtime_raw.get("interval_minutes", 60)),
            llm_gate_min_new_headlines=int(
                runtime_raw.get("llm_gate_min_new_headlines", 2)
            ),
            llm_gate_price_move_pct=float(
                runtime_raw.get("llm_gate_price_move_pct", 0.02)
            ),
            llm_gate_composite_delta=float(
                runtime_raw.get("llm_gate_composite_delta", 0.2)
            ),
            llm_gate_options_unusual_jump=int(
                runtime_raw.get("llm_gate_options_unusual_jump", 2)
            ),
            llm_gate_signal_change_only=bool(
                runtime_raw.get("llm_gate_signal_change_only", True)
            ),
            llm_max_calls_per_day=int(
                runtime_raw.get("llm_max_calls_per_day", 30)
            ),
            llm_max_calls_per_run=int(
                runtime_raw.get("llm_max_calls_per_run", 3)
            ),
            llm_max_est_input_tokens_per_day=int(
                runtime_raw.get("llm_max_est_input_tokens_per_day", 120000)
            ),
            llm_max_est_output_tokens_per_day=int(
                runtime_raw.get("llm_max_est_output_tokens_per_day", 60000)
            ),
            llm_est_input_tokens_per_call=int(
                runtime_raw.get("llm_est_input_tokens_per_call", 2500)
            ),
            llm_est_output_tokens_per_call=int(
                runtime_raw.get("llm_est_output_tokens_per_call", 800)
            ),
        ),
        llm=LLMConfig(
            provider=str(llm_raw.get("provider", "openai")),
            model_fast=str(llm_raw.get("model_fast", "gpt-5.4-mini")),
            model_deep=str(llm_raw.get("model_deep", "gpt-5.5")),
            mode=str(llm_raw.get("mode", "selective")),
            base_url=llm_raw.get("base_url"),
        ),
        data_provider=str(raw.get("data_provider", "polygon")),
        output_dir=str(raw.get("output_dir", "reports/analysis_runtime")),
        state_db_path=str(raw.get("state_db_path", "state/analysis_state.sqlite")),
    )
    return config


def _load_raw_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with `pip install pyyaml` or use JSON config."
            ) from exc
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file root must be a mapping/object.")
        return loaded

    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Config file root must be an object.")
    return loaded
