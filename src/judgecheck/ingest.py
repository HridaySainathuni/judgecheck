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


def _langsmith_timestamp(value: object) -> datetime | None:
    """Parse a LangSmith ``created_at`` value, defaulting to None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _langsmith_messages(inputs: object) -> list[dict]:
    """Turn a LangSmith example's ``inputs`` into chat messages.

    A ``messages`` list passes through unchanged. A single string
    input becomes one user message; anything else is serialized as
    JSON in one user message. Missing or non-dict inputs give no
    messages.
    """
    if not isinstance(inputs, dict):
        return []
    messages = inputs.get("messages")
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict)]
    if len(inputs) == 1:
        value = next(iter(inputs.values()))
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
    if not inputs:
        return []
    return [{"role": "user", "content": json.dumps(inputs, sort_keys=True)}]


def _langsmith_output(outputs: object) -> str:
    """Turn a LangSmith example's ``outputs`` into the record output.

    An ``output`` key wins; a single string value is used as-is;
    anything else is serialized as JSON. Missing or empty outputs
    give an empty string.
    """
    if not isinstance(outputs, dict) or not outputs:
        return ""
    value = outputs.get("output")
    if isinstance(value, str):
        return value
    if len(outputs) == 1:
        value = next(iter(outputs.values()))
        if isinstance(value, str):
            return value
    return json.dumps(outputs, sort_keys=True)


def load_langsmith(path: str | Path) -> list[TraceRecord]:
    """Load trace records from a LangSmith JSON dataset export.

    The export is either a JSON list of examples or an object with an
    ``examples`` key. Each example carries ``inputs`` and ``outputs``
    dicts; inputs map to the record messages, outputs to the record
    output, and the model name and metadata carry through when the
    export has them. Missing fields fall back to clear defaults
    instead of raising: no id becomes ``langsmith-<index>``, missing
    inputs give no messages, missing outputs give an empty output,
    and a missing or unparseable timestamp gives ``None``.

    A file that is not valid JSON, or an example that is not a JSON
    object, raises :class:`MalformedTraceError`.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise MalformedTraceError(f"{path}: not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        examples = data.get("examples")
        if examples is None:
            raise MalformedTraceError(
                f"{path}: expected a list of examples or an object "
                "with an 'examples' key"
            )
    elif isinstance(data, list):
        examples = data
    else:
        raise MalformedTraceError(
            f"{path}: expected a list of examples or an object "
            "with an 'examples' key"
        )
    if not isinstance(examples, list):
        raise MalformedTraceError(
            f"{path}: 'examples' must be a list of example objects"
        )
    records: list[TraceRecord] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise MalformedTraceError(
                f"{path}: example {index}: expected a JSON object"
            )
        metadata = example.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        model = example.get("model")
        if not isinstance(model, str):
            model = metadata.get("model")
        if not isinstance(model, str):
            model = None
        records.append(
            TraceRecord(
                trace_id=str(example.get("id") or f"langsmith-{index}"),
                timestamp=_langsmith_timestamp(example.get("created_at")),
                messages=_langsmith_messages(example.get("inputs")),
                output=_langsmith_output(example.get("outputs")),
                model=model,
                metadata=metadata,
                source=str(path),
            )
        )
    return records
