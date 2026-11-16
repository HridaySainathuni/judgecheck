from datetime import datetime, timezone

import pytest

from judgecheck import MalformedTraceError, TraceRecord, load_jsonl


def _write_jsonl(tmp_path, text):
    path = tmp_path / "traces.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_normal_file(tmp_path):
    path = _write_jsonl(
        tmp_path,
        '{"trace_id": "t1", "timestamp": "2026-11-12T10:00:00Z", '
        '"messages": [{"role": "user", "content": "What is 2+2?"}], '
        '"output": "4", "model": "gpt-4o", '
        '"metadata": {"judge": "gpt-4o"}, "source": "langsmith"}\n'
        '{"trace_id": "t2", "messages": [{"role": "user", "content": "Hi"}], '
        '"output": "Hello"}\n'
        "\n",
    )

    records = load_jsonl(path)

    assert len(records) == 2
    first, second = records
    assert isinstance(first, TraceRecord)
    assert first.trace_id == "t1"
    assert first.timestamp == datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc)
    assert first.messages == [{"role": "user", "content": "What is 2+2?"}]
    assert first.output == "4"
    assert first.model == "gpt-4o"
    assert first.metadata == {"judge": "gpt-4o"}
    assert first.source == str(path)
    assert second.trace_id == "t2"
    assert second.timestamp is None
    assert second.model is None
    assert second.metadata == {}


def test_malformed_json_line_raises(tmp_path):
    path = _write_jsonl(tmp_path, '{"trace_id": "t1", "messages": [], "output": "ok"}\nnot json\n')

    with pytest.raises(MalformedTraceError, match="line 2"):
        load_jsonl(path)


def test_missing_required_field_raises(tmp_path):
    path = _write_jsonl(tmp_path, '{"trace_id": "t1", "messages": []}\n')

    with pytest.raises(MalformedTraceError, match="line 1"):
        load_jsonl(path)


def test_non_object_line_raises(tmp_path):
    path = _write_jsonl(tmp_path, '["not", "a", "trace"]\n')

    with pytest.raises(MalformedTraceError, match="line 1"):
        load_jsonl(path)
