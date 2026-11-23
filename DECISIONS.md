# Decisions

Each entry records a choice, the alternatives considered, and why the choice won.

## Python package with src layout

Date: 2026-11-12

Choice: judgecheck is a standalone Python package, laid out with the source under `src/judgecheck`, built with hatchling, tested with pytest, linted with ruff.

Alternatives considered:

- Flat layout with the package at the repo root. Simpler to start, but tests can import the local package by accident instead of the installed one, which hides packaging mistakes.
- A module inside the resume-tailor-saas repo. The eval toolkit is a general tool, not part of the resume app, so it deserves its own repo and release history.
- Poetry or PDM instead of plain pip and pyproject. Both are fine tools, but they add a lockfile and a workflow the project does not need yet. Plain pip with an editable install keeps the setup light.

Why: the src layout forces tests to run against the installed package, which is how users will use it. Hatchling is the smallest build backend that supports the layout, and pytest plus ruff match the setup used in the other portfolio projects.

## One shared TraceRecord for every loader

Date: 2026-11-13

Choice: every loader returns `TraceRecord` values from `src/judgecheck/records.py`, with fields `trace_id`, `timestamp`, `messages`, `output`, `model`, `metadata`, and `source`. `timestamp` and `model` are optional and default to `None`, `metadata` defaults to an empty dict, and `source` records where the trace came from.

Alternatives considered:

- A separate record type per source, one for JSONL and one for LangSmith. That would push format handling into every judge and reliability check instead of keeping it in the loaders.
- Plain dicts with no shared type. Simpler to start, but nothing enforces which fields exist, and typos in field names would fail silently downstream.

Why: judges and reliability checks should only ever see one shape. The loaders own the messy mapping from each source format, and everything after ingestion works with the same fields.

## JSONL and LangSmith loaders

Date: 2026-11-14

Choice: two loaders in `src/judgecheck/ingest.py`, both returning a list of `TraceRecord`. `load_jsonl` reads one trace dict per line from a JSONL file, skipping blank lines. `load_langsmith` reads a LangSmith JSON dataset export, either a list of examples or an object with an `examples` key. LangSmith inputs map to messages: a `messages` list passes through, a single string input becomes one user message, and anything else is serialized as JSON in one user message. LangSmith outputs map to the record output: an `output` key wins, a single string value is used as-is, and anything else is serialized as JSON.

Alternatives considered:

- One loader with a format flag. That would mix two very different parsing rules in one function and make each harder to test on its own.
- Requiring LangSmith exports to be normalized into JSONL first. That puts work on the user for a mapping the toolkit can do itself.

Why: two small loaders are easier to test than one loader with a branch, and each loader keeps its source format knowledge in one place. JSONL is strict about required fields because a hand-built trace file should fail loudly. LangSmith is lenient about missing fields because exports vary, so missing ids become `langsmith-<index>`, missing inputs give no messages, missing outputs give an empty output, and a missing or unparseable timestamp gives `None`.

## MalformedTraceError for bad traces

Date: 2026-11-14

Choice: one error type, `MalformedTraceError`, a subclass of `ValueError`, raised whenever a trace file cannot be loaded. JSONL errors name the file and the offending line number. A file that is not valid JSON, or an example that is not a JSON object, raises too. A bad line fails the whole load instead of being skipped.

Alternatives considered:

- Skipping bad lines and warning. That would silently drop traces, so a judge score could be computed over fewer traces than the user expected with no clear sign.
- Returning a result object that carries errors alongside records. That forces every caller to check for errors, and most callers would forget.
- Raising plain `ValueError`. That gives callers no way to tell a bad trace file apart from any other value error.

Why: one named error type lets callers catch exactly this failure, and failing the whole load with the file and line number makes a bad trace easy to find and fix. Raising instead of skipping keeps the trace count honest.

## One Judge protocol with prompt and model set at construction

Date: 2026-11-18

Choice: every judge implements the `Judge` protocol in `src/judgecheck/judges.py`: one `judge` method that takes a `TraceRecord` and returns a `Judgement` with a boolean `verdict` and a string `rationale`. Each judge calls an LLM through the Anthropic SDK, with the prompt template and model set at construction and the API key read from the `ANTHROPIC_API_KEY` environment variable. A pre-built client can be injected instead, which the tests use to avoid network calls.

Alternatives considered:

- A single judge class with a mode flag for each check. That would mix the prompts and verdict parsing for different checks in one class, and adding a check would mean editing the class instead of adding a new one.
- A plain function per judge. Functions cannot hold a configured prompt, model, and client together, so every call site would repeat that setup.
- Reading the API key from a config file. An environment variable is the standard way to pass secrets to a CLI tool, and it keeps keys out of the repo.

Why: one protocol means the reliability checks in later milestones can run any judge the same way, without knowing which check it performs. Prompt and model at construction let a user tune a judge for their own traces without editing library code. The env API key keeps the secret out of the code, and injecting a client keeps the tests fast and offline.

## Relevance and grounding as the first two judges

Date: 2026-11-18

Choice: the first two judges are `RelevanceJudge` and `GroundingJudge`. `RelevanceJudge` checks whether the trace output answers the trace input. `GroundingJudge` checks whether every claim in the trace output is supported by the trace input, with no invented claims. Both follow the `Judge` protocol and share the same construction pattern: prompt template and model configurable, client injected or built from the environment.

Alternatives considered:

- Starting with a style or tone judge. Style is subjective and varies by use case, so a style verdict is hard to trust as a reliability signal.
- Starting with a judge that scores on a scale instead of a yes or no verdict. A scale adds calibration work before the toolkit can even report a pass rate, and the reliability checks need binary outcomes first.
- Building more judges before shipping these two. Each new judge is cheap to add under the protocol, so the two most common failure modes are worth shipping first and the rest can follow.

Why: relevance and grounding are the two most common ways an LLM answer goes wrong in a way a judge should catch: an answer that misses the question, and an answer that invents facts. A judge that cannot catch those two is not useful, so they come first. Both are binary verdicts, which the later reliability checks can count directly.

## Wilson score interval for judge rates

Date: 2026-11-20

Choice: `wilson_interval` in `src/judgecheck/stats.py` reports a judge rate, for example the share of traces a judge marked relevant, as a Wilson score interval at a chosen confidence level, defaulting to 95 percent. With zero trials there is no information, so the interval is the full [0, 1] range. The function is pure: it takes counts and returns a pair of floats.

Alternatives considered:

- The normal approximation, also called the Wald interval: `p_hat` plus or minus `z` times the standard error. It is the textbook default, but it breaks down exactly where judge evals live. With small samples it can produce bounds outside [0, 1], and with a rate near 0 or 1 it collapses to a zero-width interval, which claims certainty the data does not support.
- The exact Clopper-Pearson interval. It guarantees coverage, but it is wider than Wilson and needs a beta distribution, which adds a dependency or hand-rolled math for no practical gain at these sample sizes.
- Bootstrapping the interval. Resampling works, but it is slower, nondeterministic unless seeded, and harder to explain in a report than a closed-form interval.

Why: judge evals usually run on tens of traces, not thousands, and a rate of 0 or 1 is common when a judge is consistent. Wilson stays inside [0, 1], stays honest at the edges, and needs only the counts, so a report can print it next to any rate without extra machinery.

## Cohen's kappa for judge vs human agreement

Date: 2026-11-20

Choice: `cohens_kappa` in `src/judgecheck/stats.py` measures agreement between a judge's labels and human labels as Cohen's kappa. 1.0 is perfect agreement, 0.0 is agreement equal to chance, and negative values mean agreement worse than chance. The function is pure: it takes two label lists and returns a float.

Alternatives considered:

- Raw agreement, the share of labels where judge and human match. It is easy to read, but it rewards a judge that always picks the majority label. If 90 percent of traces are relevant, a judge that says relevant every time scores 90 percent agreement while telling the user nothing.
- Percent agreement with a fixed threshold, such as requiring 80 percent. A threshold does not fix the problem above, it just moves it: the same always-relevant judge passes the threshold on an unbalanced set.
- Krippendorff's alpha. It handles more raters and missing labels, but the toolkit compares one judge against one human label set, so the extra generality adds complexity the milestone does not need.

Why: kappa subtracts the agreement expected by chance from the observed agreement, so a judge only scores well when it agrees with humans beyond what guessing the majority label would give. That is the number a user needs before trusting a judge score, and it takes nothing more than two label lists to compute.

## Perturbation stability checks

Date: 2026-11-22

Choice: `run_perturbation_check` in `src/judgecheck/reliability.py` runs a judge on a trace and on deterministic surface perturbations of its output: case change, extra whitespace, and trailing punctuation change. It returns a `PerturbationReport` with the original verdict, whether the verdict stayed the same across every perturbation, and the per-perturbation verdicts. The original trace is never modified; each perturbed run gets a copy with only `output` changed.

Alternatives considered:

- Semantic paraphrases of the output. A paraphrase changes meaning, so a verdict flip would be expected rather than a sign of unreliability, and generating paraphrases needs an LLM call, which makes the check nondeterministic and slow.
- Random fuzzing of the output. Random changes are not reproducible, so two runs of the same check could disagree and a failure could not be replayed.
- Adversarial prompts designed to trick the judge. That tests a different property, whether a judge can be manipulated, rather than whether it is stable on ordinary surface variation.
- Checking stability by running the judge twice on the same trace. That only catches nondeterminism in the judge itself, not sensitivity to how the output happens to be written.

Why: a verdict should depend on what the output says, not on its casing, spacing, or trailing punctuation. Surface perturbations are deterministic, free, and fast, so the check runs offline in CI and any flip names exactly which surface detail the judge reacted to. A stable verdict is not proof the judge is right, but an unstable one is a clear sign its scores cannot be trusted.

## Cross-judge agreement with kappa

Date: 2026-11-22

Choice: `run_cross_judge_agreement` in `src/judgecheck/reliability.py` runs two judges over the same traces and returns a `CrossJudgeAgreement` with both verdict lists, the raw agreement rate (the share of traces where the verdicts match), and Cohen's kappa between the verdict lists, computed by `cohens_kappa` in `src/judgecheck/stats.py`.

Alternatives considered:

- Raw agreement alone. It is easy to read, but it rewards two judges that both pick the majority label. If 90 percent of traces are relevant, two always-relevant judges score 90 percent agreement while telling the user nothing about whether they actually agree on the hard cases.
- Kappa alone. Kappa corrects for chance, but a single number hides where the judges diverge. Keeping the verdict lists lets the caller look at the specific traces the judges disagreed on.
- Comparing each judge against human labels instead. That is the right check when human labels exist, and `cohens_kappa` already supports it, but it needs labelled data. Cross-judge agreement works with no human labels at all, so it can run on any trace set.
- Requiring the rationales to match as well as the verdicts. Two judges can reach the same verdict for different reasons, and rationale text is free-form, so matching it would be stricter than the verdicts the toolkit reports.

Why: two independent judges that agree beyond chance are more trustworthy than one judge alone, and disagreement between them flags traces where the verdict is genuinely uncertain. Kappa is the number that says whether the agreement beats chance, and the raw rate is the number a user reads first, so the check reports both and keeps the verdict lists for inspection.
