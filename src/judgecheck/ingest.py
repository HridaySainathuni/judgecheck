"""Trace loaders for judgecheck.

Each loader reads one trace source and returns a list of
:class:`~judgecheck.records.TraceRecord` values.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from judgecheck.records import TraceRecord


class MalformedTraceError(ValueError):
    """A line in a trace file is not valid JSON or lacks required fields."""


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise MalformedTraceError(f"timestamp must be an ISO 8601 string, got {value!r}")


def _record_from_dict(data: dict, source: str) -> TraceRecord:
    try:
        trace_id = data["trace_id"]
        messages = data["messages"]
        output = data["output"]
    except KeyError as exc:
        raise MalformedTraceError(f"missing required field: {exc.args[0]}") from exc
    if not isinstance(messages, list):
        raise MalformedTraceError("messages must be a list of message dicts")
    return TraceRecord(
        trace_id=str(trace_id),
        timestamp=_parse_timestamp(data.get("timestamp")),
        messages=messages,
        output=str(output),
        model=data.get("model"),
        metadata=data.get("metadata") or {},
        source=source,
    )


def load_jsonl(path: str | Path) -> list[TraceRecord]:
    """Load trace records from a JSONL file, one trace dict per line.

    Blank lines are skipped. Any other line that is not valid JSON or
    that lacks required fields raises :class:`MalformedTraceError`
    naming the offending line number.
    """
    path = Path(path)
    records: list[TraceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedTraceError(
                    f"{path}: line {line_number}: not valid JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise MalformedTraceError(
                    f"{path}: line {line_number}: expected a JSON object"
                )
            try:
                records.append(_record_from_dict(data, source=str(path)))
            except MalformedTraceError as exc:
                raise MalformedTraceError(
                    f"{path}: line {line_number}: {exc}"
                ) from exc
    return records
