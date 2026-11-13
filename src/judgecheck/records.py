"""Shared record types for judgecheck.

Every loader (JSONL, LangSmith, ...) produces ``TraceRecord`` values,
so judges and reliability checks only ever see one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TraceRecord:
    """One evaluated trace: a single prompt/response pair from an LLM run."""

    trace_id: str
    timestamp: datetime | None
    messages: list[dict]
    output: str
    model: str | None = None
    metadata: dict = field(default_factory=dict)
    source: str = ""
