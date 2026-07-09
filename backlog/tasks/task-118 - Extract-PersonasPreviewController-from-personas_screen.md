---
id: TASK-118
title: Extract PersonasPreviewController from personas_screen
status: Done
assignee: []
created_date: '2026-06-11 13:25'
updated_date: '2026-07-03 09:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The preview block (~180-210 lines: seeding rule, system-prompt builder, worker, gateway lifecycle) is self-contained; extract to UI/Persona_Modules mirroring personas_conversations_controller before the screen grows past legacy size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Screen delegates preview logic to a controller,Tests repointed,Behavior preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: bounded UI/controller extraction that preserves existing Personas preview behavior and does not change storage, schema, sync policy, service contracts, or security boundaries.

1. Run the existing Personas preview integration baseline from current dev.
2. Add a failing structural regression requiring PersonasScreen to own a PersonasPreviewController and delegate preview reply/reset/open-console handling to it.
3. Extract the preview state, gateway lifecycle, system-prompt builder, seeding, worker, and open-console logic into UI/Persona_Modules/personas_preview_controller.py.
4. Keep PersonasScreen event handlers and selection paths thin, delegating preview operations to the controller.
5. Re-run focused preview tests, the full Personas workbench file, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted the ephemeral Personas preview conversation state, gateway lifecycle, system prompt construction, reply worker, reset handling, and Console handoff into `PersonasPreviewController`. `PersonasScreen` now owns a `preview` controller and keeps only selection/lifecycle/event delegation, matching the existing conversations-controller pattern.

Updated preview integration tests to assert through the controller seam (`screen.preview.history`, `screen.preview.ensure_gateway`, and `screen.preview.system_prompt`) and added a structural regression that verifies `PersonasScreen.preview` is a `PersonasPreviewController`.

Review follow-up replaced Loguru `exc_info=True` calls in `PersonasPreviewController` with native `logger.opt(exception=True)` exception capture and bound safe provider/model/selection/generation context on preview provider failures. Added a regression that captures real Loguru records for streaming and non-streaming provider errors.

ADR check: no new ADR required; this remains a bounded UI/controller extraction under existing Personas ADR-004 and ADR-007, with no storage, schema, sync, service-contract, provider-boundary, or security-boundary change.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py::TestPreviewIntegration --tb=short` -> 23 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short` -> 161 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_conversation_browser_search_ignores_stale_results 'Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts[settings-#settings-category-strip-#settings-workbench-panes1-actions1]' Tests/UI/test_settings_configuration_hub.py::test_settings_console_background_effects_save_nested_config Tests/UI/test_settings_configuration_hub.py::test_settings_console_background_workbench_loaded_scope_unrelated_save_falls_back Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_revert_restores_global_defaults --tb=short` -> 5 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py Tests/UI/test_personas_workbench.py` -> passed.
- `git diff --check` -> passed.
<!-- SECTION:NOTES:END -->
