"""Tests for the CLI and the static HTML report generator."""

from __future__ import annotations

import json
from pathlib import Path

from judgecheck.cli import TraceResult, main, render_report
from judgecheck.judges import Judgement
from judgecheck.records import TraceRecord


class _MockJudge:
    """A judge that passes traces whose output contains 'Paris'."""

    def judge(self, trace: TraceRecord) -> Judgement:
        if "Paris" in trace.output:
            return Judgement(
                verdict=True,
                rationale="The output answers the question.",
            )
        return Judgement(
            verdict=False,
            rationale="The output does not answer the question.",
        )


def _results() -> list[TraceResult]:
    return [
        TraceResult(
            trace_id="trace-1",
            verdict=True,
            rationale="The output answers the question.",
        ),
        TraceResult(
            trace_id="trace-2",
            verdict=False,
            rationale="The output does not answer the question.",
        ),
    ]


def test_render_report_contains_trace_ids_and_verdicts() -> None:
    html = render_report(_results(), passes=1, total=2, lower=0.1, upper=0.9)

    assert "trace-1" in html
    assert "trace-2" in html
    assert "PASS" in html
    assert "FAIL" in html
    assert "The output answers the question." in html
    assert "The output does not answer the question." in html


def test_render_report_contains_summary_table() -> None:
    html = render_report(_results(), passes=1, total=2, lower=0.1, upper=0.9)

    assert "Total traces" in html
    assert "50.0%" in html
    assert "0.100 &ndash; 0.900" in html


def test_render_report_escapes_html_in_rationale() -> None:
    results = [
        TraceResult(trace_id="trace-1", verdict=True, rationale="<b>bold</b>")
    ]
    html = render_report(results, passes=1, total=1, lower=0.5, upper=1.0)

    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "<b>bold</b>" not in html


def test_main_writes_report_with_pass_rate(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "messages": [
                        {"role": "user", "content": "What is the capital?"},
                    ],
                    "output": output,
                }
            )
            for trace_id, output in [
                ("trace-1", "Paris is the capital of France."),
                ("trace-2", "Here is a recipe for soup."),
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.html"

    exit_code = main([str(traces_path), str(report_path)], judge=_MockJudge())

    assert exit_code == 0
    html = report_path.read_text(encoding="utf-8")
    assert "trace-1" in html
    assert "trace-2" in html
    assert "PASS" in html
    assert "FAIL" in html
    assert "50.0%" in html
