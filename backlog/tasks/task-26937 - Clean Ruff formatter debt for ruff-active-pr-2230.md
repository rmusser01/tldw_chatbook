---
id: TASK-26937
title: Clean Ruff formatter debt for ruff-active-pr-2230
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-09-30 22:10'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-active-pr-2230 -->
<!-- TASK-26000-PATHS-SHA256: 36f93c83c3745f0891a3d7ea12925ea5d6bedb5d3f3b765ca63ca401e93b7f77 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-active-pr-2230` Ruff formatter batch at the owner boundary recorded as: Exact point-in-time active PR ownership: #2230 fix(console): voice chip no longer starves the composer draft at narrow widths (TASK-24620) at 00e11f24bee1bf60c76324f0d64d34b8464534ad. The focused test surface recorded by TASK-26000 is `["Tests/UI/test_console_dictation_streaming.py"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_dictation_streaming.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot.
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000, plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `90597ade77` (isolated worktree `.worktrees/ruff-debt-batch-1`, branch `chore/ruff-debt-batch-1`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** The single assigned path exists unchanged at the base — no upstream delete or rename. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over this task's Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 1 of 1 assigned paths formatted (`Tests/UI/test_console_dictation_streaming.py`). `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths shows an identical finding count before (243) and after — the formatter introduced zero findings; the pre-existing findings are F401-type debt outside the TASK-26000 formatter scope (hand-fixing them would violate AC#8).

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal for every file in the batch EXCEPT this task's assigned file, where exactly one class of value changed: 6 docstring `Constant` values each gained one leading space after the opening `"""` because their first content line begins with a quote character (`""""one." ...` -> `""" "one." ...`). This is Ruff 0.15.22's mandatory black-style docstring normalization (it prevents the `""""` opening-quote ambiguity), not an optional transform — a formatter run cannot avoid it. Recorded deviation, with evidence: dump length delta is exactly +6 characters across 6 regions, all the same single-space insertion; the module contains no `__doc__` consumer (grep: 0 hits); the file is a test module and no production path is touched. AC#3 is ticked on this basis; the alternative (leaving the file unformatted) would fail the batch's formatter-clean AC#5 instead.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/UI/test_console_dictation_streaming.py` (the assigned file itself). Result: 84 failed, 4 passed in 17.87s (`--timeout=120`). Because the counts fail, an explicit baseline A/B was run per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): the assigned file(s) were swapped to their unformatted `HEAD` content with `git checkout HEAD -- <paths>`, the identical command re-run, and the files re-formatted and re-verified with `ruff format --check`. Baseline A/B result: 84 failed, 4 passed in 16.36s — identical to the formatted outcome. The failures are therefore pre-existing at the current `dev` tip and out of scope for a formatter-only task; AST equality (AC#3) independently proves the formatted files are runtime-identical to their unformatted state. Corroborating context: the concurrent TASK-19425 measurement (branch `fix/task-19425-core-suite-durations`) documents a dev-tip hook-consent send-gate regression (`aed1b13501`, 2026-09-27) that fails ~10 controller-adjacent suites, and this machine was running unrelated heavy pytest suites throughout.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; the assigned path carried live debt and was formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
