"""Minimal in-house TOML writer.

Supports what trainer configs use: scalar values, homogeneous/heterogeneous
arrays (including nested arrays), and one level of tables. Keys are written
bare when possible, quoted otherwise. stdlib ``tomllib`` round-trips the output.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _fmt_key(key: str) -> str:
    key = str(key)
    return key if _BARE_KEY.match(key) else '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        text = repr(value)
        return text if ("." in text or "e" in text or "E" in text) else text + ".0"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_fmt_key(k)} = {_fmt_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def dumps(data: dict[str, Any]) -> str:
    scalars: list[str] = []
    tables: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            body = "\n".join(f"{_fmt_key(k)} = {_fmt_value(v)}" for k, v in value.items())
            tables.append(f"[{_fmt_key(key)}]\n{body}")
        else:
            scalars.append(f"{_fmt_key(key)} = {_fmt_value(value)}")
    parts = []
    if scalars:
        parts.append("\n".join(scalars))
    parts.extend(tables)
    return "\n\n".join(parts) + "\n"


def dump(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data), encoding="utf-8")
    return path
