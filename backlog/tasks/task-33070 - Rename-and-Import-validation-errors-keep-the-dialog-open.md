---
id: TASK-33070
title: Rename and Import validation errors keep the dialog open
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. 'Name taken' on Rename and Import validation errors appear as toasts after the dialog closes, forcing the user to reopen and retype. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rename and Import show validation errors inside the dialog and keep the typed value
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Give RagProfileNameModal an optional validator callback that returns an inline error (shown in the dialog, input kept) or None to dismiss.
2. Split the editor's Rename and Import refusal checks into rename_refusal/import_refusal (same checks, same editor instance) and reuse them in rename_user_theme/import_theme.
3. Wire the validators in settings_screen's Rename/Import prompts; tests drive the real modal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RagProfileNameModal (settings_screen.py) gained an optional validate callback: a returned reason shows in a markup-free #settings-rag-profile-name-error line inside the dialog and the input keeps its value; None dismisses as before. The file checks stay in SettingsThemeEditor: rename_user_theme's checks moved into _rename_check (public rename_refusal), Import gained import_refusal (_parse_import + _write_target), and _refuse_link/_resolve_write_target now wrap text-returning _link_refusal/_write_target. The operations re-run every check after dismiss (time passes). Screen handlers pass validators through _with_theme_editor. CSS: .settings-rag-profile-name-error in features/_settings.tcss (+ regenerated screen_agentic_settings.tcss). Tests: two picker-screen journeys (rename taken/invalid then success; import bad colour then fixed).
<!-- SECTION:NOTES:END -->
