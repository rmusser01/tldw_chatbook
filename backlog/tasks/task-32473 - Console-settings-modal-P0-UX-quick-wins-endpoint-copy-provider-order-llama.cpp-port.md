---
id: TASK-32473
title: >-
  Console settings modal P0 UX quick wins (endpoint copy, provider order,
  llama.cpp port)
status: Done
assignee: []
created_date: '2026-09-11 03:40'
updated_date: '2026-09-11 03:40'
labels: []
dependencies: []
---

## Renumbering provenance

Renumbered from TASK-32473 on 2026-09-11: the id collided with a task that
arrived on dev while this branch was in review (owner rule TASK-19601 — the
older arrival keeps the id). No dependencies referenced the old id.


## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stopgap fixes from the 2026-09-10 conversation-settings UX review, scoped to compose with the TASK-30012 connection-first modal redesign rather than duplicate it: correct the endpoint-not-saved recovery copy to name the in-modal action, order Console provider options by the shared Settings/Wizard group taxonomy, and single-source the llama.cpp default base URL while local discovery scans both common ports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Endpoint-not-saved blocker copy names Save model defaults (Console Settings) or F9 Settings
- [x] Provider options ordered Cloud, then Local, then Custom-and-legacy, by display name
- [x] chat_screen blocked-reason matcher keys off the new copy signal
- [x] DEFAULT_LLAMACPP_BASE_URL defined once and shared by gateway/session settings
- [x] Local discovery candidates include 8080 and 9099
- [x] Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update UNSAVED_ENDPOINT_COPY + modal scope/tooltip copy and the chat_screen matcher
2. Group-rank and display-name-sort build_console_provider_options
3. Move DEFAULT_LLAMACPP_BASE_URL/INVALID_LLAMACPP_BASE_URL_COPY to console_provider_endpoints; add 9099 discovery candidate
4. Update affected tests; run targeted suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented 2026-09-10. Approach: stopgaps that compose with the TASK-30012 connection-first modal redesign instead of duplicating it. (1) UNSAVED_ENDPOINT_COPY now names Save model defaults in Console Settings or F9 Settings; modal save-default tooltip and model-scope copy state endpoint persistence; chat_screen _console_setup_blocked_reason matcher keys on 'endpoint is not saved'. (2) build_console_provider_options orders Cloud > Local > Custom-and-legacy by display name (same taxonomy as F9 Settings / First-Run Wizard; the function TASK-30012's picker will consume). (3) DEFAULT_LLAMACPP_BASE_URL + INVALID_LLAMACPP_BASE_URL_COPY single-sourced in console_provider_endpoints (gateway/session-settings import); local discovery candidates now scan stock llama-server 8080 AND the documented 9099 convention. Files: Chat/console_provider_endpoints.py, Chat/console_provider_gateway.py, Chat/console_session_settings.py, Chat/local_server_discovery.py, UI/Screens/chat_screen.py, Widgets/Console/console_settings_modal.py + tests in Tests/Chat and Tests/UI. ADR: not required (copy/ordering/port-candidate fixes; no schema, boundary, or security decision). Verification: Tests/Chat test_local_server_discovery+test_console_provider_gateway 294 passed; Tests/Chat test_console_session_settings 121 passed; Tests/UI test_console_session_settings 194 passed with 2 failures verified pre-existing on the stashed pristine tree (inspector staged-context ordering, unmount-timeout repair). Ruff: no new findings vs pristine baseline.
<!-- SECTION:NOTES:END -->
