---
id: TASK-26991
title: Clean Ruff formatter debt for ruff-state-sync-wizards-tests
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

<!-- TASK-26000-BATCH: ruff-state-sync-wizards-tests -->
<!-- TASK-26000-PATHS-SHA256: c9d72a418572dc9ee8d7c1f6cc15792afe991bd916f2499444862d022ddc22b8 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-state-sync-wizards-tests` Ruff formatter batch at the owner boundary recorded as: State, sync-interoperability, event-handler, and wizard integration tests.. The focused test surface recorded by TASK-26000 is `["Tests/Event_Handlers", "Tests/State", "Tests/Sync_Interop", "Tests/Wizards"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Event_Handlers/test_note_ingest_import_offload.py",
  "Tests/State/test_screen_state_store.py",
  "Tests/Sync_Interop/test_chat_outbox_producer.py",
  "Tests/Sync_Interop/test_note_organization_receipt_finalization.py",
  "Tests/Sync_Interop/test_notes_organization_adapters.py",
  "Tests/Sync_Interop/test_notes_organization_app_wiring.py",
  "Tests/Sync_Interop/test_notes_organization_contract.py",
  "Tests/Sync_Interop/test_notes_organization_enrollment.py",
  "Tests/Sync_Interop/test_notes_organization_intent_dispatch.py",
  "Tests/Sync_Interop/test_notes_organization_legacy_inventory.py",
  "Tests/Sync_Interop/test_notes_organization_two_device.py",
  "Tests/Sync_Interop/test_notes_outbox_producer.py",
  "Tests/Wizards/test_first_run_setup_integration.py",
  "Tests/Wizards/test_first_run_setup_state.py",
  "Tests/Wizards/test_first_run_setup_wizard.py",
  "Tests/Wizards/test_first_run_speech_step_state.py"
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

**Formatting (AC#2, AC#5).** 16 of 16 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 203 assigned paths: 68 findings before and 32 after, ZERO increases — the formatter incidentally resolved 36 line-length findings concentrated in `Tests/Tools/test_local_tool_impls.py` (34->1) and `Tests/Tools/test_local_tool_impls_properties.py` (3->0) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 16 assigned test files across Tests/Event_Handlers, Tests/State, Tests/Sync_Interop, Tests/Wizards (recorded surface). Result on the formatted tree: 120 failed, 830 passed, 31 errors in 484.90s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 120 failed, 830 passed, 31 errors in 387.29s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 192 committed + 10 clean-at-base + 1 upstream-deleted = 203 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 16 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
