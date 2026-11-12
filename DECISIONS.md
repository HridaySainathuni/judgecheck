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
