from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import time

import pandas as pd


CACHE_SCHEMA_VERSION = "analysis-cache-v1"


def default_cache_dir() -> Path:
    raw = os.getenv("TRADINGAGENTS_ANALYSIS_CACHE_DIR")
    if raw:
        return Path(raw).expanduser()
    base = os.getenv(
        "TRADINGAGENTS_CACHE_DIR",
        str(Path.home() / ".tradingagents" / "cache"),
    )
    return Path(base).expanduser() / "analysis_only"


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def json_path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def dataframe_path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.df.json"

    def get_json(
        self,
        namespace: str,
        key: str,
        *,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        path = self.json_path(namespace, key)
        if not self._fresh(path, ttl_seconds):
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def set_json(self, namespace: str, key: str, value: Any) -> Path:
        path = self.json_path(namespace, key)
        self._write_text_atomic(path, json.dumps(value, indent=2, default=str))
        return path

    def get_dataframe(
        self,
        namespace: str,
        key: str,
        *,
        ttl_seconds: int | None = None,
    ) -> pd.DataFrame | None:
        path = self.dataframe_path(namespace, key)
        if not self._fresh(path, ttl_seconds):
            return None
        try:
            return pd.read_json(StringIO(path.read_text()), orient="table")
        except Exception:
            return None

    def set_dataframe(
        self,
        namespace: str,
        key: str,
        value: pd.DataFrame,
    ) -> Path:
        path = self.dataframe_path(namespace, key)
        payload = value.to_json(orient="table", date_format="iso")
        self._write_text_atomic(path, payload)
        return path

    def _fresh(self, path: Path, ttl_seconds: int | None) -> bool:
        if not path.exists():
            return False
        if ttl_seconds is None:
            return True
        return (time.time() - path.stat().st_mtime) <= ttl_seconds

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_name, path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass


def report_cache_key(params: dict[str, Any]) -> str:
    return stable_hash(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "kind": "analysis_report",
            "params": params,
        }
    )


def report_file(output_dir: str | Path, symbol: str, as_of_date: str) -> Path:
    return Path(output_dir) / f"{symbol.upper()}_{as_of_date}.json"


def load_report_if_cache_hit(
    report_cls: type,
    output_dir: str | Path,
    symbol: str,
    as_of_date: str,
    cache_key: str,
) -> Any | None:
    path = report_file(output_dir, symbol, as_of_date)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    meta = (payload.get("data_quality") or {}).get("analysis_cache") or {}
    if meta.get("key") != cache_key:
        return None
    payload.setdefault("data_quality", {}).setdefault(
        "analysis_cache", {}
    )["source"] = "cache"
    if not is_dataclass(report_cls):
        return payload
    allowed = {f.name for f in fields(report_cls)}
    kwargs = {k: v for k, v in payload.items() if k in allowed}
    for f in fields(report_cls):
        if f.name not in kwargs and f.default is not MISSING:
            kwargs[f.name] = f.default
    return report_cls(**kwargs)
