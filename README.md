# judgecheck

LLM eval reliability toolkit. It catches misleading judge scores with perturbation tests, agreement checks, and confidence intervals.

An LLM judge can score a de-identified response as a bad answer and sound confident while it is wrong. judgecheck exists to catch that. It takes the traces your judges already produce, runs reliability checks on them, and reports where the scores hold up and where they do not.

## Status

This project is just starting. The first milestone is trace ingestion: loading traces from JSONL files and LangSmith exports into one common record format.

## Install

```bash
pip install judgecheck
```

Requires Python 3.12 or newer.

## Quick example

Coming in Milestone 1, once trace ingestion lands. The plan is a small CLI: point it at a traces file, and it prints a reliability summary and writes a static HTML report.

## What it will do

- Ingest traces from OpenTelemetry and JSONL files and from LangSmith exports.
- Run configurable LLM-as-judge rubrics against those traces.
- Report Wilson confidence intervals on judge scores.
- Measure Cohen's kappa against human labels.
- Run perturbation tests to see if small input changes flip judge scores.
- Measure agreement across judges.

## Contributing

See DECISIONS.md for the design choices behind this project. Pull requests need tests and a passing CI run.
