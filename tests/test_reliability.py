"""Tests for perturbation stability checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from judgecheck.judges import RelevanceJudge
from judgecheck.records import TraceRecord
from judgecheck.reliability import (
    PERTURBATIONS,
    PerturbationReport,
    run_perturbation_check,
)


class _FakeMessages:
    """Stands in for ``client.messages`` and returns a fixed reply."""

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


class _PunctuationSensitiveMessages(_FakeMessages):
    """Replies NOT RELEVANT when the judged output ends with ``!``.

    The trailing punctuation perturbation turns a trailing period
    into an exclamation mark, so a judge backed by this client flips
    its verdict on that perturbation only.
    """

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        prompt = str(kwargs["messages"][0]["content"])
        if "!\n\nDoes the response" in prompt:
            reply = "VERDICT: NOT RELEVANT\nThe output reads as shouting."
        else:
            reply = "VERDICT: RELEVANT\nThe output directly answers the question."
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])


class _PunctuationSensitiveClient:
    """Stands in for an Anthropic client with a punctuation quirk."""

    def __init__(self) -> None:
        self.messages = _PunctuationSensitiveMessages("")


def _trace(output: str = "Paris is the capital of France.") -> TraceRecord:
    return TraceRecord(
        trace_id="trace-1",
        timestamp=None,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        output=output,
    )


def test_stable_judge_reports_stable_with_all_verdicts() -> None:
    client = _FakeClient(
        "VERDICT: RELEVANT\nThe output directly answers the question."
    )
    judge = RelevanceJudge(client=client)

    report = run_perturbation_check(judge, _trace())

    assert isinstance(report, PerturbationReport)
    assert report.original_verdict is True
    assert report.stable is True
    assert report.flipped_perturbations == []
    assert [result.name for result in report.results] == [
        "case_change",
        "extra_whitespace",
        "trailing_punctuation",
    ]
    assert all(result.verdict is True for result in report.results)
    assert len(client.messages.calls) == 4


def test_flipping_judge_reports_unstable_and_names_the_perturbation() -> None:
    client = _PunctuationSensitiveClient()
    judge = RelevanceJudge(client=client)

    report = run_perturbation_check(judge, _trace())

    assert report.original_verdict is True
    assert report.stable is False
    assert report.flipped_perturbations == ["trailing_punctuation"]
    flipped = [result for result in report.results if result.flipped]
    assert len(flipped) == 1
    assert flipped[0].name == "trailing_punctuation"
    assert flipped[0].perturbed_output == "Paris is the capital of France!"
    assert flipped[0].verdict is False


def test_perturbations_change_only_the_surface_of_the_output() -> None:
    output = "Paris is the capital of France."

    assert PERTURBATIONS["case_change"](output) == "pARIS IS THE CAPITAL OF fRANCE."
    assert (
        PERTURBATIONS["extra_whitespace"](output)
        == "Paris  is  the  capital  of  France.  "
    )
    assert PERTURBATIONS["trailing_punctuation"](output) == (
        "Paris is the capital of France!"
    )


def test_original_trace_is_not_modified() -> None:
    client = _FakeClient(
        "VERDICT: RELEVANT\nThe output directly answers the question."
    )
    judge = RelevanceJudge(client=client)
    trace = _trace()

    run_perturbation_check(judge, trace)

    assert trace.output == "Paris is the capital of France."


def test_unknown_perturbation_name_raises() -> None:
    client = _FakeClient(
        "VERDICT: RELEVANT\nThe output directly answers the question."
    )
    judge = RelevanceJudge(client=client)

    with pytest.raises(ValueError, match="Unknown perturbation"):
        run_perturbation_check(judge, _trace(), perturbations=["shuffle_words"])
