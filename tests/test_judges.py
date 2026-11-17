"""Tests for the judge interface and the relevance judge."""

from __future__ import annotations

from types import SimpleNamespace

from judgecheck.judges import Judgement, RelevanceJudge
from judgecheck.records import TraceRecord


class _FakeMessages:
    """Stands in for ``client.messages`` and records create() calls."""

    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply_text)])


class _FakeClient:
    """Stands in for an Anthropic client."""

    def __init__(self, reply_text: str) -> None:
        self.messages = _FakeMessages(reply_text)


def _trace(output: str = "Paris is the capital of France.") -> TraceRecord:
    return TraceRecord(
        trace_id="trace-1",
        timestamp=None,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        output=output,
    )


def test_relevance_judge_returns_verdict_and_rationale() -> None:
    client = _FakeClient(
        "VERDICT: RELEVANT\nThe output directly answers the question."
    )
    judge = RelevanceJudge(client=client)

    result = judge.judge(_trace())

    assert isinstance(result, Judgement)
    assert result.verdict is True
    assert "directly answers" in result.rationale
    assert len(client.messages.calls) == 1


def test_relevance_judge_negative_verdict() -> None:
    client = _FakeClient(
        "VERDICT: NOT RELEVANT\nThe output talks about cooking, not capitals."
    )
    judge = RelevanceJudge(client=client)

    result = judge.judge(_trace(output="Here is a recipe for soup."))

    assert result.verdict is False
    assert "cooking" in result.rationale


def test_empty_output_returns_negative_verdict_without_calling_llm() -> None:
    client = _FakeClient("VERDICT: RELEVANT\nThis should never be used.")
    judge = RelevanceJudge(client=client)

    result = judge.judge(_trace(output="   "))

    assert result.verdict is False
    assert "empty" in result.rationale
    assert client.messages.calls == []


def test_prompt_and_model_are_configurable() -> None:
    client = _FakeClient("VERDICT: RELEVANT\nCustom prompt works.")
    judge = RelevanceJudge(
        model="claude-opus-4-1",
        prompt_template="CUSTOM MARKER\nQuestion: {input}\nAnswer: {output}",
        client=client,
    )

    result = judge.judge(_trace())

    assert result.verdict is True
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-4-1"
    prompt = call["messages"][0]["content"]
    assert "CUSTOM MARKER" in prompt
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
