"""Command line tool for judgecheck.

Loads traces from a JSONL file, runs a judge over them, computes the
pass rate with a Wilson interval, and writes a static HTML report.
The judge is injectable: pass ``judge`` to :func:`main` (used by
tests) or leave it out to build the default
:class:`~judgecheck.judges.RelevanceJudge`, which reads the
``ANTHROPIC_API_KEY`` environment variable.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from judgecheck.ingest import load_jsonl
from judgecheck.judges import Judge, RelevanceJudge
from judgecheck.records import TraceRecord
from judgecheck.stats import wilson_interval

PASS_LABEL = "PASS"
FAIL_LABEL = "FAIL"


@dataclass
class TraceResult:
    """One trace plus the judge's verdict and rationale for it."""

    trace_id: str
    verdict: bool
    rationale: str


def render_report(
    results: Sequence[TraceResult],
    *,
    passes: int,
    total: int,
    lower: float,
    upper: float,
) -> str:
    """Return a static HTML report for the per-trace results.

    The report has a summary table (total traces, passes, pass rate,
    and the Wilson interval) and one row per trace with its trace id,
    verdict, and rationale. Pure function: no I/O, no LLM calls.
    """
    if total:
        pass_rate = f"{100 * passes / total:.1f}%"
    else:
        pass_rate = "n/a"
    rows = "\n".join(
        "      <tr>"
        f"<td>{html.escape(result.trace_id)}</td>"
        f"<td>{PASS_LABEL if result.verdict else FAIL_LABEL}</td>"
        f"<td>{html.escape(result.rationale)}</td>"
        "</tr>"
        for result in results
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>judgecheck report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #999; padding: 0.4rem 0.8rem; text-align: left; }}
</style>
</head>
<body>
<h1>judgecheck report</h1>
<h2>Summary</h2>
<table>
<tr><th>Total traces</th><td>{total}</td></tr>
<tr><th>Passes</th><td>{passes}</td></tr>
<tr><th>Pass rate</th><td>{pass_rate}</td></tr>
<tr><th>Wilson 95% interval</th><td>{lower:.3f} &ndash; {upper:.3f}</td></tr>
</table>
<h2>Traces</h2>
<table>
<tr><th>Trace id</th><th>Verdict</th><th>Rationale</th></tr>
{rows}
</table>
</body>
</html>
"""


def run(traces: Sequence[TraceRecord], judge: Judge) -> list[TraceResult]:
    """Run ``judge`` over each trace and collect the results."""
    results: list[TraceResult] = []
    for trace in traces:
        judgement = judge.judge(trace)
        results.append(
            TraceResult(
                trace_id=trace.trace_id,
                verdict=judgement.verdict,
                rationale=judgement.rationale,
            )
        )
    return results


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="judgecheck",
        description=(
            "Judge traces from a JSONL file and write a static HTML report."
        ),
    )
    parser.add_argument(
        "trace_file",
        help="JSONL file of traces, one trace dict per line",
    )
    parser.add_argument(
        "output",
        help="Path of the HTML report to write",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, judge: Judge | None = None) -> int:
    """Run the CLI: judge the traces and write the HTML report.

    Returns the process exit code (0 on success). Pass ``judge`` to
    inject a judge (used by tests); otherwise a default
    :class:`~judgecheck.judges.RelevanceJudge` is built.
    """
    args = _parse_args(argv)
    judge = judge if judge is not None else RelevanceJudge()
    traces = load_jsonl(args.trace_file)
    results = run(traces, judge)
    passes = sum(1 for result in results if result.verdict)
    total = len(results)
    lower, upper = wilson_interval(passes, total)
    report = render_report(
        results, passes=passes, total=total, lower=lower, upper=upper
    )
    Path(args.output).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
