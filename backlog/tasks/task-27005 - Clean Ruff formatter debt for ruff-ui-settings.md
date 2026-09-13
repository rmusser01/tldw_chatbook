---
id: TASK-27005
title: Clean Ruff formatter debt for ruff-ui-settings
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
