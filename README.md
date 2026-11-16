# judgecheck

LLM eval reliability toolkit. It catches misleading judge scores with perturbation tests, agreement checks, and confidence intervals.

An LLM judge can score a de-identified response as a bad answer and sound confident while it is wrong. judgecheck exists to catch that. It takes the traces your judges already produce, runs reliability checks on them, and reports where the scores hold up and where they do not.

## Status

Milestone 1 is done: trace ingestion works. `load_jsonl` loads traces from JSONL files and `load_langsmith` loads them from LangSmith dataset exports, both into one shared `TraceRecord` type. Next up are the judges and reliability checks.

## Install

```bash
pip install judgecheck
```

Requires Python 3.12 or newer.

## Quick example

Each line of a JSONL trace file is one JSON object with `trace_id`, `messages`, and `output`:

```json
{"trace_id": "t1", "messages": [{"role": "user", "content": "What is 2+2?"}], "output": "4", "model": "gpt-4o"}
```

Load the traces and work with them as `TraceRecord` values:

```python
from judgecheck.ingest import load_jsonl, load_langsmith

records = load_jsonl("traces.jsonl")
for record in records:
    print(record.trace_id, record.model, record.output)

langsmith_records = load_langsmith("langsmith-export.json")
print(f"loaded {len(langsmith_records)} traces from LangSmith")
```

A line that is not valid JSON or lacks a required field raises `MalformedTraceError` naming the file and line number.

## What it will do

- Ingest traces from OpenTelemetry and JSONL files and from LangSmith exports.
- Run configurable LLM-as-judge rubrics against those traces.
- Report Wilson confidence intervals on judge scores.
- Measure Cohen's kappa against human labels.
- Run perturbation tests to see if small input changes flip judge scores.
- Measure agreement across judges.

## Contributing

See DECISIONS.md for the design choices behind this project. Pull requests need tests and a passing CI run.
