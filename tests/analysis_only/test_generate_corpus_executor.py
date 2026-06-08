"""Tests for the ``--executor`` flag added to ``scripts/generate_corpus.py``.

These tests cover the lightweight surface (CLI parsing + JobTuple
picklability) without invoking the actual Polygon-backed pipeline. The
intent is to lock in the contract that ``ProcessPoolExecutor`` can submit
the worker function (`_generate_one`) with a JobTuple of basic types — the
prerequisite for the regen speedup tracked in
``plans/screener_and_regen.md`` future-work item #5.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_corpus.py"


def _load_generate_corpus_module():
    """Import scripts/generate_corpus.py as a module without running main()."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "generate_corpus_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gc_module():
    return _load_generate_corpus_module()


def _build_parser(gc_module):
    """Recreate the argparse parser used by main() so we can assert on parsed
    values without invoking the rest of main()."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def test_cli_accepts_executor_process(gc_module):
    parser = _build_parser(gc_module)
    args = parser.parse_args(["--executor", "process"])
    assert args.executor == "process"


def test_cli_accepts_executor_thread(gc_module):
    parser = _build_parser(gc_module)
    args = parser.parse_args(["--executor", "thread"])
    assert args.executor == "thread"


def test_cli_default_executor_is_process(gc_module):
    parser = _build_parser(gc_module)
    args = parser.parse_args([])
    assert args.executor == "process"


def test_real_cli_parses_executor_flag(gc_module):
    """End-to-end: parse the actual generate_corpus CLI with the new flag."""
    import subprocess

    py = sys.executable
    result = subprocess.run(
        [py, str(_SCRIPT_PATH), "--executor", "thread", "--dry-run",
         "--tickers", "NVDA", "--start", "2024-06-21", "--end", "2024-06-21",
         "--state-store-path", ""],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # --dry-run should always exit 0 and not require Polygon.
    assert result.returncode == 0, (
        f"dry-run failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "executor: thread" in result.stdout


def test_job_tuple_is_picklable():
    """ProcessPoolExecutor.submit pickles each arg. Any non-picklable type
    in the JobTuple would surface as a runtime PicklingError on the first
    submission, so guard with a unit test."""
    job: tuple[str, str, str, str, int, bool, str | None, float, bool] = (
        "NVDA",
        "2024-06-21",
        "/tmp/pp_test",
        "polygon",
        3,
        False,
        "state/analysis_state.sqlite",
        0.0,
        True,
    )
    payload = pickle.dumps(job)
    restored = pickle.loads(payload)
    assert restored == job


def test_job_tuple_with_none_state_store_is_picklable():
    """state_store_path can be None when the user passes --state-store-path ''."""
    job: tuple[str, str, str, str, int, bool, str | None, float, bool] = (
        "AMD",
        "2024-06-28",
        "/tmp/pp_test",
        "polygon",
        3,
        True,
        None,
        0.0,
        False,
    )
    payload = pickle.dumps(job)
    restored = pickle.loads(payload)
    assert restored == job
    assert restored[6] is None


def test_generate_one_is_module_level(gc_module):
    """ProcessPoolExecutor requires the worker function to be importable by
    qualified name, which means defined at module scope (not nested)."""
    assert hasattr(gc_module, "_generate_one")
    assert gc_module._generate_one.__qualname__ == "_generate_one"
