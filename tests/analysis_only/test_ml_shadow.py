"""Tests for paper-log shadow ML prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.analysis_only.ml_shadow import _latest_records_from_signals


class _Sig:
    def __init__(self, source_path: str):
        self.source_path = source_path


def test_latest_records_from_signals_loads_factor_scores(tmp_path: Path):
    report = {
        "symbol": "NVDA",
        "as_of_date": "2026-06-08",
        "direction": "bullish",
        "confidence": 0.7,
        "key_features": {
            "model_scoring": {
                "composite_score": 0.25,
                "factor_scores": [{"factor": "momentum_rsi", "score": 0.4}],
            }
        },
    }
    path = tmp_path / "NVDA.json"
    path.write_text(json.dumps(report))
    records = _latest_records_from_signals({"NVDA": _Sig(str(path))})
    assert len(records) == 1
    assert records[0].symbol == "NVDA"
    assert records[0].factor_scores[0]["factor"] == "momentum_rsi"
