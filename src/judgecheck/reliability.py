"""Reliability checks for judgecheck.

Two checks, both of which run judges through the ``Judge`` protocol
and need no human labels:

- ``run_perturbation_check``: a judge's verdict should not depend on
  surface details of the trace output, so it runs a judge on the
  original trace and on a small set of deterministic surface
  perturbations of the trace output and reports whether the verdict
  stayed the same across all of them.
- ``run_cross_judge_agreement``: two judges should agree on the same
  traces more than chance would predict, so it runs both judges over
  the same traces and reports their verdict lists, the raw agreement
  rate, and Cohen's kappa between them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from judgecheck.judges import Judge
from judgecheck.records import TraceRecord
from judgecheck.stats import cohens_kappa


def _perturb_case(output: str) -> str:
    """Swap the case of every letter in the output."""
    return output.swapcase()


def _perturb_whitespace(output: str) -> str:
    """Double the spaces between words and add trailing whitespace."""
    return output.replace(" ", "  ") + "  "


def _perturb_punctuation(output: str) -> str:
    """Change the trailing punctuation of the output.

    A trailing period becomes an exclamation mark, a trailing
    exclamation mark or question mark becomes a period, and an output
    with no trailing punctuation gains a period.
    """
    if output.endswith("."):
        return output[:-1] + "!"
    if output.endswith(("!", "?")):
        return output[:-1] + "."
    return output + "."


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "case_change": _perturb_case,
    "extra_whitespace": _perturb_whitespace,
    "trailing_punctuation": _perturb_punctuation,
}

DEFAULT_PERTURBATIONS: tuple[str, ...] = tuple(PERTURBATIONS)


@dataclass
class PerturbationResult:
    """One perturbation and the verdict the judge gave it."""

    name: str
    perturbed_output: str
    verdict: bool
    flipped: bool


@dataclass
class PerturbationReport:
    """Whether a judge's verdict survived every perturbation."""

    original_verdict: bool
    stable: bool
    results: list[PerturbationResult] = field(default_factory=list)

    @property
    def flipped_perturbations(self) -> list[str]:
        """Names of the perturbations whose verdict differed."""
        return [result.name for result in self.results if result.flipped]


def run_perturbation_check(
    judge: Judge,
    trace: TraceRecord,
    perturbations: Sequence[str] | None = None,
) -> PerturbationReport:
    """Run ``judge`` on ``trace`` and on each perturbed copy of it.

    Returns whether the verdict was stable across the original and
    every perturbation, plus the per-perturbation verdicts. The
    original trace is never modified: each perturbed run gets a copy
    of the trace with only ``output`` changed.
    """
    names = list(perturbations) if perturbations is not None else list(DEFAULT_PERTURBATIONS)
    unknown = [name for name in names if name not in PERTURBATIONS]
    if unknown:
        raise ValueError(f"Unknown perturbation(s): {', '.join(unknown)}")

    original = judge.judge(trace)
    results = []
    for name in names:
        perturbed_output = PERTURBATIONS[name](trace.output)
        perturbed_trace = replace(trace, output=perturbed_output)
        judgement = judge.judge(perturbed_trace)
        results.append(
            PerturbationResult(
                name=name,
                perturbed_output=perturbed_output,
                verdict=judgement.verdict,
                flipped=judgement.verdict != original.verdict,
            )
        )
    return PerturbationReport(
        original_verdict=original.verdict,
        stable=all(not result.flipped for result in results),
        results=results,
    )


@dataclass
class CrossJudgeAgreement:
    """How two judges' verdicts over the same traces compare."""

    verdicts_a: list[bool]
    verdicts_b: list[bool]
    kappa: float
    agreement: float


def run_cross_judge_agreement(
    judge_a: Judge,
    judge_b: Judge,
    traces: Sequence[TraceRecord],
) -> CrossJudgeAgreement:
    """Run two judges over the same traces and compare their verdicts.

    Returns both verdict lists, the raw agreement rate (the share of
    traces where the verdicts match), and Cohen's kappa between the
    verdict lists. Kappa is 1.0 for perfect agreement and 0.0 for
    agreement at chance level. At least one trace is required.
    """
    if not traces:
        raise ValueError("at least one trace is required")

    verdicts_a = [judge_a.judge(trace).verdict for trace in traces]
    verdicts_b = [judge_b.judge(trace).verdict for trace in traces]
    agreement = sum(
        1 for a, b in zip(verdicts_a, verdicts_b) if a == b
    ) / len(traces)
    return CrossJudgeAgreement(
        verdicts_a=verdicts_a,
        verdicts_b=verdicts_b,
        kappa=cohens_kappa(verdicts_a, verdicts_b),
        agreement=agreement,
    )
