---
id: TASK-90
title: Console Settings keeps provider model when switching to configured provider
status: Done
assignee: []
created_date: '2026-06-10 01:15'
updated_date: '2026-06-10 01:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure Console Settings provider switching preserves or discovers the configured model for the selected provider so users can return to a local/provider-backed runtime without manually retyping the model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching from a custom provider to llama_cpp pre-fills the configured llama_cpp model.
- [x] #2 Switching providers preserves the configured base URL.
- [x] #3 Users can save the switched provider without an empty model when config defines one.
- [x] #4 Regression coverage proves the provider-switch path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: UI/settings bugfix aligning provider-switch behavior with existing config ownership; no new storage, sync, or service boundary decision.

1. Add a mounted regression for switching Console Settings from custom to llama_cpp when app_config defines api_settings.llama_cpp.model and endpoint settings.
2. Verify the regression fails because the switched model is blank.
3. Update ConsoleSettingsModal model resolution to fallback to provider config model when no catalog/runtime options exist.
4. Run the focused Console Settings tests and diff checks.
5. Relaunch the textual-web UAT app from this branch, capture CDP screenshots for the switched model and local llama send path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a mounted regression covering custom-to-llama_cpp provider switching with a configured model and local endpoint.
- Updated ConsoleSettingsModal model resolution so provider-specific draft values still win, then provider config model is used before falling back to current settings or catalog options.
- Verified the regression failed before the fix, then passed after the fix.
- Verified broader Console Settings behavior with focused UI and pure helper tests.
- Verified in CDP/textual-web that switching to llama_cpp prefilled `uat-model` and `http://127.0.0.1:9099`, saving updated the Console rail, and sending `say ok` rendered the local response.
<!-- SECTION:NOTES:END -->
