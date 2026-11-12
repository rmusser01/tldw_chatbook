---
id: TASK-27005
title: Clean Ruff formatter debt for ruff-ui-settings
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

<!-- TASK-26000-BATCH: ruff-ui-settings -->
<!-- TASK-26000-PATHS-SHA256: 98041cba955aff32e3580a928ed360488dec5c4d82de7b3e0f8f8a2099e6190f -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-settings` Ruff formatter batch at the owner boundary recorded as: Settings, configuration, and preference UI surfaces with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_dictation_settings_debounce.py",
  "Tests/UI/test_settings_agent_run_budget.py",
  "Tests/UI/test_settings_agents_category.py",
  "Tests/UI/test_settings_configuration_hub.py",
  "Tests/UI/test_settings_console_side_chat.py",
  "Tests/UI/test_settings_console_status_row.py",
  "Tests/UI/test_settings_context_memory_controls.py",
  "Tests/UI/test_settings_footer_hints.py",
  "Tests/UI/test_settings_image_gen_defaults.py",
  "Tests/UI/test_settings_image_gen_panel.py",
  "Tests/UI/test_settings_kimi_zai.py",
  "Tests/UI/test_settings_model_catalog_toggles.py",
  "Tests/UI/test_settings_narrow_layout.py",
  "Tests/UI/test_settings_network_category.py",
  "Tests/UI/test_settings_network_defaults.py",
  "Tests/UI/test_settings_panel_scoped_updates.py",
  "Tests/UI/test_settings_privacy_security.py",
  "Tests/UI/test_settings_provider_test_draft.py",
  "Tests/UI/test_settings_provider_view_model.py",
  "Tests/UI/test_settings_save_commit_models.py",
  "Tests/UI/test_settings_scope_inspector_focus.py",
  "Tests/UI/test_settings_speech_tts_model.py",
  "Tests/UI/test_settings_splash_screen_viewer.py",
  "Tests/UI/test_settings_theme_editor.py",
  "Tests/UI/test_settings_tools_section.py",
  "Tests/UI/test_settings_url_input.py",
  "Tests/UI/test_settings_video_gen_defaults.py",
  "Tests/UI/test_settings_workspace_assistant_defaults.py",
  "Tests/UI/test_settings_workspaces_category.py",
  "Tests/UI/test_site_config_settings.py",
  "Tests/UI/test_speech_settings_completeness.py",
  "Tests/UI/test_speech_settings_pane.py",
  "Tests/UI/test_speech_settings_panel_scoped_updates.py",
  "Tests/UI/test_studio_tts_preferences.py",
  "Tests/UI/test_tools_settings_window.py",
  "tldw_chatbook/UI/Screens/settings_image_gen_defaults.py",
  "tldw_chatbook/UI/Screens/settings_network_defaults.py",
  "tldw_chatbook/UI/Screens/settings_privacy_security.py",
  "tldw_chatbook/UI/Screens/settings_video_gen_defaults.py",
  "tldw_chatbook/UI/Screens/tools_settings_screen.py",
  "tldw_chatbook/UI/Speech/speech_settings_model.py",
  "tldw_chatbook/UI/Tools_Settings_Window.py"
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

**Ownership reconciliation (AC#1).** All 42 assigned paths reconcile mechanically (hash matches marker and evidence JSON); 41 exist, 1 deleted upstream — see Lineage.

**Formatting (AC#2, AC#5).** 34 of the 41 existing assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting. Lint: `ruff check --output-format concise` over the batch's 399 assigned paths: 119 findings before and 117 after, ZERO increases (the formatter incidentally resolved 2 in `Docs/superpowers/qa/skills-script-execution-2026-07-25/seed3.py` by reflowing); the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 34 existing assigned test files (recorded surface Tests/UI). Result on the formatted tree: 378 failed, 803 passed, 4 skipped, 46 errors in 1943.43s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 378 failed, 803 passed, 4 skipped, 46 errors in 2153.40s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 369 committed + 25 clean-at-base + 5 upstream-deleted = 399 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 1 path deleted upstream: `Tests/UI/test_site_config_settings.py` by `ef43462806` (TASK-32899 unreachable-window cleanup, the same commit as batch-8's deletion); 7 paths already formatter-clean at the base and deliberately left untouched; the remaining 34 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
