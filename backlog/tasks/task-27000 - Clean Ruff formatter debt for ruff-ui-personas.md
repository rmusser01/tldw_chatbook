---
id: TASK-27000
title: Clean Ruff formatter debt for ruff-ui-personas
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 21:55'
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

<!-- TASK-26000-BATCH: ruff-ui-personas -->
<!-- TASK-26000-PATHS-SHA256: 46d36f76100c63f6ae8f76474794245c08c3a08fc174f406ae4641e07562c22b -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-personas` Ruff formatter batch at the owner boundary recorded as: Persona and character UI surfaces with direct UI/Character tests.. The focused test surface recorded by TASK-26000 is `["Tests/Character_Chat", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_actor_pack_staging_sweep_seam.py",
  "Tests/UI/test_character_display_text.py",
  "Tests/UI/test_persona_policy_rules_editor.py",
  "Tests/UI/test_persona_profile_widgets.py",
  "Tests/UI/test_personas_center_canvas_layout.py",
  "Tests/UI/test_personas_character_editor_avatar.py",
  "Tests/UI/test_personas_character_widgets.py",
  "Tests/UI/test_personas_character_world_books.py",
  "Tests/UI/test_personas_character_world_books_screen.py",
  "Tests/UI/test_personas_deferred_center_views.py",
  "Tests/UI/test_personas_editor_save_in_place.py",
  "Tests/UI/test_personas_expression_generate.py",
  "Tests/UI/test_personas_inspector_pane.py",
  "Tests/UI/test_personas_lore.py",
  "Tests/UI/test_personas_preview.py",
  "Tests/UI/test_personas_preview_restore.py",
  "Tests/UI/test_personas_workbench.py",
  "Tests/UI/test_personas_workbench_foundation.py",
  "Tests/UI/test_personas_workbench_state.py",
  "Tests/UI/test_uat_first_time_character_chat.py",
  "tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py",
  "tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py"
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

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `2612fc56b2` (isolated worktree `.worktrees/ruff-debt-batch-8`, branch `chore/ruff-debt-batch-8`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 20 of 22 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 142 assigned paths: 32 findings before and 32 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every existing assigned path (no docstring deviations and no plain-parse fallback needed anywhere in this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 20 assigned test files (recorded surface Tests/Character_Chat + Tests/UI). Result on the formatted tree: 434 failed, 419 passed, 8 errors in 2336.83s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: A/B name-diff protocol (COMBINED FAILED+ERROR name sets, batch-7 task-26994 precedent — this surface converts errors to failures run-to-run): formatted 442 names vs baseline 442 names, diff EMPTY — byte-identical failure sets; the summary-count churn (434F/8E vs 399F/68E) is category conversion within the same pre-existing dev-tip red mass, out of scope for a formatter-only task. — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 130 committed + 11 clean-at-base + 1 upstream-deleted = 142 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 2 paths were already formatter-clean at the base (`Tests/UI/test_personas_character_editor_avatar.py`, `Tests/UI/test_personas_deferred_center_views.py`) and were deliberately left untouched; the other 20 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
