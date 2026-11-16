import json
from datetime import datetime, timezone

import pytest

from judgecheck import MalformedTraceError, TraceRecord, load_jsonl, load_langsmith


def _write_export(tmp_path, data):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _normal_examples():
    return [
        {
            "id": "ls-1",
            "created_at": "2026-11-12T10:00:00Z",
            "inputs": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
            "outputs": {"output": "4"},
            "metadata": {"model": "gpt-4o", "dataset": "math-qa"},
        },
        {
            "id": "ls-2",
            "inputs": {"question": "Say hello"},
            "outputs": {"answer": "Hello"},
            "metadata": {"model": "gpt-4o-mini"},
        },
    ]


def test_loads_normal_export(tmp_path):
    path = _write_export(tmp_path, _normal_examples())

    records = load_langsmith(path)

    assert len(records) == 2
    first, second = records
    assert isinstance(first, TraceRecord)
    assert first.trace_id == "ls-1"
    assert first.timestamp == datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc)
    assert first.messages == [{"role": "user", "content": "What is 2+2?"}]
    assert first.output == "4"
    assert first.model == "gpt-4o"
    assert first.metadata == {"model": "gpt-4o", "dataset": "math-qa"}
    assert first.source == str(path)
    assert second.trace_id == "ls-2"
    assert second.timestamp is None
    assert second.messages == [{"role": "user", "content": "Say hello"}]
    assert second.output == "Hello"
    assert second.model == "gpt-4o-mini"


def test_loads_examples_object_wrapper(tmp_path):
    path = _write_export(tmp_path, {"examples": _normal_examples()})

    records = load_langsmith(path)

    assert len(records) == 2
    assert records[0].trace_id == "ls-1"
    assert records[0].output == "4"


def test_missing_fields_use_defaults(tmp_path):
    path = _write_export(tmp_path, [{}, {"id": "ls-3", "inputs": {}, "outputs": {}}])

    records = load_langsmith(path)

    assert len(records) == 2
    first, second = records
    assert first.trace_id == "langsmith-0"
    assert first.timestamp is None
    assert first.messages == []
    assert first.output == ""
    assert first.model is None
    assert first.metadata == {}
    assert first.source == str(path)
    assert second.trace_id == "ls-3"
    assert second.messages == []
    assert second.output == ""


def test_unparseable_timestamp_defaults_to_none(tmp_path):
    path = _write_export(
        tmp_path,
        [
            {
                "id": "ls-4",
                "created_at": "not a date",
                "inputs": {"prompt": "Hi"},
                "outputs": {"output": "Hello"},
            }
        ],
    )

    records = load_langsmith(path)

    assert records[0].timestamp is None
    assert records[0].messages == [{"role": "user", "content": "Hi"}]


def test_matches_jsonl_records(tmp_path):
    jsonl_path = tmp_path / "traces.jsonl"
    jsonl_path.write_text(
        '{"trace_id": "t1", "timestamp": "2026-11-12T10:00:00Z", '
        '"messages": [{"role": "user", "content": "What is 2+2?"}], '
        '"output": "4", "model": "gpt-4o", '
        '"metadata": {"dataset": "math-qa"}}\n',
        encoding="utf-8",
    )
    export_path = _write_export(
        tmp_path,
        [
            {
                "id": "t1",
                "created_at": "2026-11-12T10:00:00Z",
                "inputs": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
                "outputs": {"output": "4"},
                "metadata": {"model": "gpt-4o", "dataset": "math-qa"},
            }
        ],
    )

    jsonl_record = load_jsonl(jsonl_path)[0]
    langsmith_record = load_langsmith(export_path)[0]

    assert langsmith_record.messages == jsonl_record.messages
    assert langsmith_record.output == jsonl_record.output
    assert langsmith_record.model == jsonl_record.model
    assert langsmith_record.trace_id == jsonl_record.trace_id
    assert langsmith_record.timestamp == jsonl_record.timestamp


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(MalformedTraceError, match="not valid JSON"):
        load_langsmith(path)


def test_non_object_example_raises(tmp_path):
    path = _write_export(tmp_path, ["not an example"])

    with pytest.raises(MalformedTraceError, match="example 0"):
        load_langsmith(path)
