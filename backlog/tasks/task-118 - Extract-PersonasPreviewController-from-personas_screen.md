---
id: TASK-118
title: Extract PersonasPreviewController from personas_screen
status: Done
assignee: []
created_date: '2026-06-11 13:25'
updated_date: '2026-06-29 05:27'
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

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py::TestPreviewIntegration --tb=short` -> 22 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short` -> 160 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py tldw_chatbook/UI/Screens/personas_screen.py` -> passed.
- `git diff --check` -> passed.
<!-- SECTION:NOTES:END -->
