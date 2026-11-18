"""Configurable LLM judges for judgecheck.

A judge scores a :class:`~judgecheck.records.TraceRecord` and returns a
verdict plus a rationale. Judges call an LLM through the Anthropic SDK;
the API key comes from the ``ANTHROPIC_API_KEY`` environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anthropic

from judgecheck.records import TraceRecord

DEFAULT_MODEL = "claude-sonnet-4-5"

DEFAULT_RELEVANCE_PROMPT = """You are judging whether a response answers the question it was asked.

Question:
{input}

Response:
{output}

Does the response answer the question? Reply with exactly one line
starting with "VERDICT: RELEVANT" or "VERDICT: NOT RELEVANT", followed
by a short rationale explaining your verdict.
"""

DEFAULT_GROUNDING_PROMPT = """You are judging whether a response stays grounded in the
source text it was given.

Source text:
{input}

Response:
{output}

Is every claim in the response supported by the source text, with no
invented claims? Reply with exactly one line starting with
"VERDICT: GROUNDED" or "VERDICT: NOT GROUNDED", followed by a short
rationale explaining your verdict.
"""


@dataclass
class Judgement:
    """The result of running a judge over one trace."""

    verdict: bool
    rationale: str


class Judge(Protocol):
    """A judge that scores one trace."""

    def judge(self, trace: TraceRecord) -> Judgement:
        """Score ``trace`` and return a verdict plus a rationale."""
        ...


def _trace_input(trace: TraceRecord) -> str:
    """Return the text the trace was asked to answer.

    Uses the last user message when the trace has messages, and falls
    back to joining every message's content.
    """
    user_messages = [
        message.get("content", "")
        for message in trace.messages
        if message.get("role") == "user"
    ]
    if user_messages:
        return str(user_messages[-1])
    return "\n".join(str(message.get("content", "")) for message in trace.messages)


class RelevanceJudge:
    """Judge whether a trace output answers the trace input.

    The prompt and model are set at construction. Pass ``client`` only
    to inject a pre-built Anthropic client (used by tests); otherwise
    a client is built from the ``ANTHROPIC_API_KEY`` environment
    variable.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_template: str | None = None,
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_RELEVANCE_PROMPT
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens

    def judge(self, trace: TraceRecord) -> Judgement:
        """Return whether the trace output answers the trace input."""
        if not trace.output.strip():
            return Judgement(
                verdict=False,
                rationale="Trace output is empty, so it cannot answer the input.",
            )
        prompt = self.prompt_template.format(
            input=_trace_input(trace),
            output=trace.output,
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_verdict(
            response.content[0].text,
            positive="relevant",
            negative="not relevant",
        )


class GroundingJudge:
    """Judge whether a trace output stays grounded in the trace input.

    Every claim in the output must be supported by the input, with no
    invented claims. The prompt and model are set at construction. Pass
    ``client`` only to inject a pre-built Anthropic client (used by
    tests); otherwise a client is built from the
    ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_template: str | None = None,
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_GROUNDING_PROMPT
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens

    def judge(self, trace: TraceRecord) -> Judgement:
        """Return whether the trace output is grounded in the trace input."""
        if not trace.output.strip():
            return Judgement(
                verdict=False,
                rationale="Trace output is empty, so there is nothing to ground.",
            )
        prompt = self.prompt_template.format(
            input=_trace_input(trace),
            output=trace.output,
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_verdict(
            response.content[0].text,
            positive="grounded",
            negative="not grounded",
        )


def _parse_verdict(text: str, *, positive: str, negative: str) -> Judgement:
    """Parse a judge reply of the form ``VERDICT: ...`` plus rationale."""
    lowered = text.lower()
    if negative in lowered:
        verdict = False
    elif positive in lowered:
        verdict = True
    else:
        return Judgement(
            verdict=False,
            rationale=f"Judge reply did not contain a verdict: {text.strip()}",
        )
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rationale_lines = [
        line for line in lines if not line.lower().startswith("verdict:")
    ]
    rationale = " ".join(rationale_lines) if rationale_lines else text.strip()
    return Judgement(verdict=verdict, rationale=rationale)
