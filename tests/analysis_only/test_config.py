from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.analysis_only.config import (
    AppConfig,
    BenchmarksConfig,
    LLMConfig,
    RuntimeConfig,
    load_config,
)


def test_appconfig_defaults():
    cfg = AppConfig()
    assert cfg.watchlist == []
    assert cfg.peers == {}
    assert cfg.data_provider == "polygon"
    assert cfg.output_dir == "reports/analysis_runtime"
    assert cfg.state_db_path == "state/analysis_state.sqlite"
    assert isinstance(cfg.benchmarks, BenchmarksConfig)
    assert isinstance(cfg.runtime, RuntimeConfig)
    assert isinstance(cfg.llm, LLMConfig)


def test_nested_dataclass_defaults():
    bench = BenchmarksConfig()
    assert bench.market == "SPY"
    assert bench.sectors == {}

    rt = RuntimeConfig()
    assert rt.schedule == "hourly"
    assert rt.only_when_market_open is True
    assert rt.max_symbols_per_run_llm == 3
    assert rt.interval_minutes == 60
    assert rt.llm_gate_min_new_headlines == 2
    assert rt.llm_gate_price_move_pct == pytest.approx(0.02)
    assert rt.llm_gate_composite_delta == pytest.approx(0.2)
    assert rt.llm_gate_options_unusual_jump == 2
    assert rt.llm_gate_signal_change_only is True
    assert rt.llm_max_calls_per_day == 30
    assert rt.llm_max_calls_per_run == 3
    assert rt.llm_max_est_input_tokens_per_day == 120000
    assert rt.llm_max_est_output_tokens_per_day == 60000
    assert rt.llm_est_input_tokens_per_call == 2500
    assert rt.llm_est_output_tokens_per_call == 800

    llm = LLMConfig()
    assert llm.provider == "openai"
    assert llm.model_fast == "gpt-5.4-mini"
    assert llm.model_deep == "gpt-5.5"
    assert llm.mode == "selective"
    assert llm.base_url is None


def test_default_factory_instances_are_independent():
    a = AppConfig()
    b = AppConfig()
    a.watchlist.append("AAPL")
    a.benchmarks.sectors["tech"] = "XLK"
    assert b.watchlist == []
    assert b.benchmarks.sectors == {}


def test_load_config_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_empty_json_uses_defaults(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    cfg = load_config(path)
    assert cfg.watchlist == []
    assert cfg.benchmarks.market == "SPY"
    assert cfg.runtime.schedule == "hourly"
    assert cfg.llm.provider == "openai"
    assert cfg.data_provider == "polygon"


def test_load_json_full_config(tmp_path: Path):
    payload = {
        "watchlist": ["aapl", "Msft"],
        "benchmarks": {
            "market": "spy",
            "sectors": {"tech": "xlk", "energy": "xle"},
        },
        "peers": {"aapl": ["msft", "googl"]},
        "runtime": {
            "schedule": "daily",
            "only_when_market_open": False,
            "max_symbols_per_run_llm": 5,
            "interval_minutes": 30,
            "llm_gate_min_new_headlines": 4,
            "llm_gate_price_move_pct": 0.05,
            "llm_gate_composite_delta": 0.4,
            "llm_gate_options_unusual_jump": 3,
            "llm_gate_signal_change_only": False,
            "llm_max_calls_per_day": 10,
            "llm_max_calls_per_run": 2,
            "llm_max_est_input_tokens_per_day": 99999,
            "llm_max_est_output_tokens_per_day": 88888,
            "llm_est_input_tokens_per_call": 1000,
            "llm_est_output_tokens_per_call": 500,
        },
        "llm": {
            "provider": "anthropic",
            "model_fast": "fast-x",
            "model_deep": "deep-x",
            "mode": "always",
            "base_url": "https://example.test",
        },
        "data_provider": "fmp",
        "output_dir": "out/dir",
        "state_db_path": "db/state.sqlite",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    cfg = load_config(path)

    # symbols are normalized to uppercase
    assert cfg.watchlist == ["AAPL", "MSFT"]
    assert cfg.benchmarks.market == "SPY"
    assert cfg.benchmarks.sectors == {"tech": "XLK", "energy": "XLE"}
    assert cfg.peers == {"AAPL": ["MSFT", "GOOGL"]}

    assert cfg.runtime.schedule == "daily"
    assert cfg.runtime.only_when_market_open is False
    assert cfg.runtime.max_symbols_per_run_llm == 5
    assert cfg.runtime.interval_minutes == 30
    assert cfg.runtime.llm_gate_min_new_headlines == 4
    assert cfg.runtime.llm_gate_price_move_pct == pytest.approx(0.05)
    assert cfg.runtime.llm_gate_composite_delta == pytest.approx(0.4)
    assert cfg.runtime.llm_gate_options_unusual_jump == 3
    assert cfg.runtime.llm_gate_signal_change_only is False
    assert cfg.runtime.llm_max_calls_per_day == 10
    assert cfg.runtime.llm_max_calls_per_run == 2
    assert cfg.runtime.llm_max_est_input_tokens_per_day == 99999
    assert cfg.runtime.llm_max_est_output_tokens_per_day == 88888
    assert cfg.runtime.llm_est_input_tokens_per_call == 1000
    assert cfg.runtime.llm_est_output_tokens_per_call == 500

    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model_fast == "fast-x"
    assert cfg.llm.model_deep == "deep-x"
    assert cfg.llm.mode == "always"
    assert cfg.llm.base_url == "https://example.test"

    assert cfg.data_provider == "fmp"
    assert cfg.output_dir == "out/dir"
    assert cfg.state_db_path == "db/state.sqlite"


def test_load_yaml_config(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    payload = {
        "watchlist": ["tsla"],
        "benchmarks": {"market": "qqq"},
        "runtime": {"interval_minutes": 15},
        "llm": {"mode": "off"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = load_config(path)
    assert cfg.watchlist == ["TSLA"]
    assert cfg.benchmarks.market == "QQQ"
    assert cfg.runtime.interval_minutes == 15
    assert cfg.llm.mode == "off"
    # untouched keys still default
    assert cfg.data_provider == "polygon"


def test_yaml_extension_loaded_as_yaml(tmp_path: Path):
    pytest.importorskip("yaml")
    # .yml suffix is also treated as YAML.
    path = tmp_path / "config.yml"
    path.write_text("watchlist: [spy]\n")
    cfg = load_config(path)
    assert cfg.watchlist == ["SPY"]


def test_malformed_json_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)


def test_json_root_not_object_raises(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        load_config(path)


def test_yaml_root_not_mapping_raises(tmp_path: Path):
    pytest.importorskip("yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_malformed_yaml_raises(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    # Unbalanced brackets -> YAML scanner/parser error.
    path.write_text("watchlist: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_none_sectors_and_peers_tolerated(tmp_path: Path):
    # Explicit nulls in YAML/JSON must not crash the comprehensions.
    payload = {"benchmarks": {"sectors": None}, "peers": None}
    path = tmp_path / "nulls.json"
    path.write_text(json.dumps(payload))
    cfg = load_config(path)
    assert cfg.benchmarks.sectors == {}
    assert cfg.peers == {}
