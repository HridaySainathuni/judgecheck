"""judgecheck: LLM eval reliability toolkit.

Catch misleading judge scores with perturbation tests, agreement
checks, and confidence intervals.
"""

__version__ = "0.1.0"

from judgecheck.ingest import MalformedTraceError, load_jsonl, load_langsmith
from judgecheck.judges import GroundingJudge, Judge, Judgement, RelevanceJudge
from judgecheck.records import TraceRecord

__all__ = [
    "GroundingJudge",
    "Judge",
    "Judgement",
    "MalformedTraceError",
    "RelevanceJudge",
    "TraceRecord",
    "__version__",
    "load_jsonl",
    "load_langsmith",
]
