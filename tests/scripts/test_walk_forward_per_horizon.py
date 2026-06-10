"""Tests for the per-horizon weight extensions in walk_forward_eval."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_walk_forward_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_walk_forward_eval_for_tests",
        repo_root / "scripts" / "walk_forward_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_per_horizon_weights_round_trips(tmp_path):
    mod = _load_walk_forward_module()
    src = {
        "horizons": ["ret_5d", "ret_20d", "ret_60d"],
        "weights_by_horizon": {
            "ret_5d": {},
            "ret_20d": {"market_vix_regime": -0.08, "options_iv_skew": -0.06},
            "ret_60d": {"options_iv_term_structure": 0.13, "news_sentiment": -0.06},
        },
    }
    path = tmp_path / "per_horizon.json"
    path.write_text(json.dumps(src))
    loaded = mod._load_per_horizon_weights(str(path))
    assert loaded["ret_5d"] == {}
    assert loaded["ret_20d"]["market_vix_regime"] == pytest.approx(-0.08)
    assert loaded["ret_60d"]["options_iv_term_structure"] == pytest.approx(0.13)


def test_load_per_horizon_weights_rejects_malformed(tmp_path):
    mod = _load_walk_forward_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"horizons": ["ret_5d"]}))  # missing weights_by_horizon
    with pytest.raises(SystemExit):
        mod._load_per_horizon_weights(str(path))
