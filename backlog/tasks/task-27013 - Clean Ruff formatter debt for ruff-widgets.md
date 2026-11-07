---
id: TASK-27013
title: Clean Ruff formatter debt for ruff-widgets
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-03 14:30'
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

<!-- TASK-26000-BATCH: ruff-widgets -->
<!-- TASK-26000-PATHS-SHA256: 363308e7911957a39dc222b8a801bc80733f5824dc6701522c0ec88d5946139c -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-widgets` Ruff formatter batch at the owner boundary recorded as: Shared non-Console widgets and direct widget tests.. The focused test surface recorded by TASK-26000 is `["Tests/Widgets"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Widgets/test_chapter_editor_widget_inplace_edit_refresh.py",
  "Tests/Widgets/test_console_video_card.py",
  "Tests/Widgets/test_console_video_card_rows.py",
  "Tests/Widgets/test_inline_loader_timer.py",
  "Tests/Widgets/test_library_collections_panel.py",
  "Tests/Widgets/test_model_search_picker.py",
  "Tests/Widgets/test_password_dialog_encryption_warning.py",
  "Tests/Widgets/test_pausable_progress.py",
  "Tests/Widgets/test_prune_safe_select.py",
  "Tests/Widgets/test_reactive_default_aliasing.py",
  "Tests/Widgets/test_tool_diff_widgets.py",
  "Tests/Widgets/test_watchlists_operation_card.py",
  "tldw_chatbook/Widgets/AppFooterStatus.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py",
  "tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py",
  "tldw_chatbook/Widgets/Persona_Widgets/character_tts_portability_dialogs.py",
  "tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py",
  "tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_character_dictionaries.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py",
  "tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py",
  "tldw_chatbook/Widgets/detailed_progress.py",
  "tldw_chatbook/Widgets/enhanced_file_picker.py",
  "tldw_chatbook/Widgets/model_search_picker.py",
  "tldw_chatbook/Widgets/project_skills_import_modal.py",
  "tldw_chatbook/Widgets/settings_agents_panel.py",
  "tldw_chatbook/Widgets/settings_image_gen_panel.py",
  "tldw_chatbook/Widgets/settings_internal_prompts_editor_modal.py",
  "tldw_chatbook/Widgets/settings_splash_screen_viewer.py",
  "tldw_chatbook/Widgets/settings_theme_editor.py",
  "tldw_chatbook/Widgets/settings_video_gen_panel.py",
  "tldw_chatbook/Widgets/splash_screen.py",
  "tldw_chatbook/Widgets/workspace_create_modal.py"
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

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `3c439d606e` (isolated worktree `.worktrees/ruff-debt-batch-9`, branch `chore/ruff-debt-batch-9` — the series' FINAL batch). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 39 assigned paths reconcile mechanically (hash matches marker and evidence JSON); 36 exist, 3 deleted upstream — see Lineage.

**Formatting (AC#2, AC#5).** 34 of the 36 existing assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting. Lint: `ruff check --output-format concise` over the batch's 399 assigned paths: 119 findings before and 117 after, ZERO increases (the formatter incidentally resolved 2 in `Docs/superpowers/qa/skills-script-execution-2026-07-25/seed3.py` by reflowing); the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 9 existing assigned test files plus Tests/Widgets (recorded surface). Result on the formatted tree: 1 failed, 125 passed in 30.18s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 1 failed, 125 passed in 29.21s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 369 committed + 25 clean-at-base + 5 upstream-deleted = 399 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 3 paths deleted upstream: `Tests/Widgets/test_inline_loader_timer.py` and `tldw_chatbook/Widgets/detailed_progress.py` by `5f3adeca33` (ADR-161 task-2 dead-module cleanup), and `Tests/Widgets/test_library_collections_panel.py` by `5dd1077df6` (collections retirement — its THIRD census-path deletion); 2 paths already formatter-clean at the base (`tldw_chatbook/Widgets/model_search_picker.py`, `workspace_create_modal.py`), deliberately left untouched; the remaining 34 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
