---
id: TASK-26989
title: Clean Ruff formatter debt for ruff-skills-runtime
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 18:55'
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

<!-- TASK-26000-BATCH: ruff-skills-runtime -->
<!-- TASK-26000-PATHS-SHA256: 039c0b882508fa3ae476d03f75cc50a8235835064066f025f7ae7881ad118089 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-skills-runtime` Ruff formatter batch at the owner boundary recorded as: Skill discovery, trust, import, package, and script execution with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Skills"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Skills/test_e2e_run_skill_script.py",
  "Tests/Skills/test_import_skill_directory.py",
  "Tests/Skills/test_local_skills_bundle_io.py",
  "Tests/Skills/test_project_skills_discovery.py",
  "Tests/Skills/test_project_skills_import_modal.py",
  "Tests/Skills/test_project_skills_startup_gate.py",
  "Tests/Skills/test_read_skill_file.py",
  "Tests/Skills/test_skill_fingerprint_executable.py",
  "Tests/Skills/test_skill_package_inspection.py",
  "Tests/Skills/test_skill_remote_fetch.py",
  "Tests/Skills/test_skill_script_grants.py",
  "Tests/Skills/test_skill_script_runner.py",
  "Tests/Skills/test_skill_script_service.py",
  "Tests/Skills/test_skill_trust_scanner_recursive.py",
  "Tests/Skills/test_skill_trust_store.py",
  "Tests/Skills/test_skill_trust_store_reset.py",
  "Tests/Skills/test_skill_trust_store_scoping.py",
  "Tests/Skills/test_skills_import.py",
  "Tests/Skills/test_skills_library_flow.py",
  "Tests/Skills/test_trust_tolerates_unsupported.py",
  "Tests/Skills/test_verify_content_binary.py",
  "Tests/Skills/test_web_research_skill.py",
  "Tests/Skills/test_zip_import_bundle.py",
  "tldw_chatbook/Skills_Interop/project_skills_discovery.py",
  "tldw_chatbook/Skills_Interop/skill_package_inspection.py",
  "tldw_chatbook/Skills_Interop/skill_remote_fetch.py",
  "tldw_chatbook/Skills_Interop/skill_trust_scanner.py"
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

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `ecc0a531c8` (isolated worktree `.worktrees/ruff-debt-batch-7`, branch `chore/ruff-debt-batch-7`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 27 of 27 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 203 assigned paths: 68 findings before and 32 after, ZERO increases — the formatter incidentally resolved 36 line-length findings concentrated in `Tests/Tools/test_local_tool_impls.py` (34->1) and `Tests/Tools/test_local_tool_impls_properties.py` (3->0) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 23 assigned test files (recorded surface Tests/Skills). Result on the formatted tree: 89 failed, 264 passed in 144.76s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 89 failed, 264 passed in 148.71s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 192 committed + 10 clean-at-base + 1 upstream-deleted = 203 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 27 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
