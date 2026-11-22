# judgecheck

LLM eval reliability toolkit. It catches misleading judge scores with perturbation tests, agreement checks, and confidence intervals.

An LLM judge can score a de-identified response as a bad answer and sound confident while it is wrong. judgecheck exists to catch that. It takes the traces your judges already produce, runs reliability checks on them, and reports where the scores hold up and where they do not.

## Status

Milestone 4 is in progress: perturbation stability checks and cross-judge agreement are built. `run_perturbation_check` tests whether a judge's verdict survives surface changes to a trace output, and `run_cross_judge_agreement` compares two judges over the same traces with a raw agreement rate and Cohen's kappa.

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

## Judges

A judge scores one trace and returns a verdict plus a rationale. Every judge follows the `Judge` protocol: a `judge` method that takes a `TraceRecord` and returns a `Judgement` with a boolean `verdict` and a string `rationale`.

Two judges are built in:

- `RelevanceJudge`: does the trace output answer the trace input?
- `GroundingJudge`: is every claim in the trace output supported by the trace input, with no invented claims?

Both call an LLM through the Anthropic SDK, with the API key read from the `ANTHROPIC_API_KEY` environment variable. The prompt template and model are set at construction, so a judge can be tuned without editing library code.

```python
from judgecheck.ingest import load_jsonl
from judgecheck.judges import GroundingJudge, RelevanceJudge

records = load_jsonl("traces.jsonl")
relevance = RelevanceJudge()
grounding = GroundingJudge()

for record in records:
    relevance_result = relevance.judge(record)
    grounding_result = grounding.judge(record)
    print(record.trace_id, relevance_result.verdict, grounding_result.verdict)
    print(grounding_result.rationale)
```

An empty trace output returns a negative verdict without calling the LLM.

## Statistics

Two pure functions in `judgecheck.stats` quantify how much a judge score can be trusted. Neither makes LLM calls or reads files.

- `wilson_interval(successes, trials, confidence=0.95)`: returns the (lower, upper) bounds of the Wilson score interval for a rate, as a pair of floats. Use it for any judge rate, for example the share of traces a judge marked relevant. With zero trials the interval is (0.0, 1.0).
- `cohens_kappa(labels_a, labels_b)`: returns Cohen's kappa for two lists of labels, as a float. 1.0 is perfect agreement, 0.0 is agreement equal to chance, and negative values mean agreement worse than chance. Use it to compare a judge's labels against human labels.

```python
from judgecheck.stats import cohens_kappa, wilson_interval

lower, upper = wilson_interval(38, 40)
print(f"relevant: 95% Wilson interval [{lower:.2f}, {upper:.2f}]")

judge_labels = ["relevant", "relevant", "not relevant"]
human_labels = ["relevant", "not relevant", "not relevant"]
print(f"kappa: {cohens_kappa(judge_labels, human_labels):.2f}")
```

## Reliability

Two checks in `judgecheck.reliability` test whether judge scores can be trusted. Neither needs human labels.

- `run_perturbation_check(judge, trace, perturbations=None)`: runs the judge on the trace and on deterministic surface perturbations of its output (case change, extra whitespace, trailing punctuation change). Returns a `PerturbationReport` with the original verdict, whether the verdict stayed the same across every perturbation (`stable`), and the per-perturbation verdicts with a `flipped_perturbations` list naming the ones that changed the verdict. A flip means the judge is reacting to surface detail instead of content.
- `run_cross_judge_agreement(judge_a, judge_b, traces)`: runs both judges over the same traces. Returns a `CrossJudgeAgreement` with both verdict lists, the raw agreement rate (the share of traces where the verdicts match), and Cohen's kappa between the verdict lists. Kappa is 1.0 for perfect agreement and 0.0 for agreement at chance level.

```python
from judgecheck.ingest import load_jsonl
from judgecheck.judges import GroundingJudge, RelevanceJudge
from judgecheck.reliability import run_cross_judge_agreement, run_perturbation_check

records = load_jsonl("traces.jsonl")

report = run_perturbation_check(RelevanceJudge(), records[0])
print(f"stable: {report.stable}, flipped: {report.flipped_perturbations}")

agreement = run_cross_judge_agreement(RelevanceJudge(), GroundingJudge(), records)
print(f"agreement: {agreement.agreement:.2f}, kappa: {agreement.kappa:.2f}")
```

## What it will do

- Ingest traces from OpenTelemetry and JSONL files and from LangSmith exports.
- Run configurable LLM-as-judge rubrics against those traces.
- Report Wilson confidence intervals on judge scores.
- Measure Cohen's kappa against human labels.

## Contributing

See DECISIONS.md for the design choices behind this project. Pull requests need tests and a passing CI run.
