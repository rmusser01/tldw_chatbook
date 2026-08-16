---
id: TASK-16473
title: Console warn when a saved provider endpoint will not survive restart
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 15:10'
labels: []
dependencies: []
---
## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report (2026-08-15): a llama.cpp user re-enters their custom IP:Port on every boot. Root cause: the Console settings modal's "Save" (and the Alt+M popover's apply) write session settings only, and Console session settings are per-process in-memory state (`ConsoleChatStore`, `ScreenStateStore`), so on the next boot the endpoint re-derives from env / `[console] llama_cpp_base_url_override` / `[api_settings.<provider>]` and falls back to the default (`http://127.0.0.1:9099`). For llama.cpp the trap is invisible: `build_console_settings_readiness` skips the "Endpoint not saved" comparison for the direct llama path, so a session carrying an unsaved custom endpoint still reports "Ready" and nothing nudges the user toward "Save as default" (`Chat/console_session_settings.py`, the `uses_direct_llama_path` guard in `build_console_settings_readiness`; apply path `UI/Screens/chat_screen.py::_apply_console_settings_result`; session-scoped save `Widgets/Console/console_settings_modal.py` `_save`).

ADR required: no — bug fix adding a warning at an existing UI seam; no storage, sync, or boundary decision changes.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan (the how)

1. Pure helper `llamacpp_endpoint_unsaved_notice`-style decision in `Chat/console_session_settings.py` reusing `_endpoint_differs_for_provider`/`first_configured_endpoint` vocabulary: returns warning copy when the session endpoint differs from everything persisted, None otherwise
2. Red test exists (`test_console_settings_apply_warns_when_llamacpp_endpoint_not_persisted`); add pure-helper unit tests for the four AC cases
3. Fire the warning from `_apply_console_settings_result` after a successful session-scoped replace (one warning per apply)
4. Run the Console settings modal + session settings suites

ADR required: no
ADR path: N/A
Reason: warning at an existing UI seam; no storage/sync/boundary change

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Applying Console settings whose provider endpoint differs from everything persisted for that provider (no matching `[api_settings]` endpoint key, no `[console] llama_cpp_base_url_override`, no env override) surfaces a warning that names the consequence (endpoint is session-only and will not survive a restart) and the persist action ("Save as default")
- [x] #2 No warning when the session endpoint matches persisted config, and no warning for providers that use no endpoint
- [x] #3 The differs-from-persisted decision is a pure, unit-tested helper (reusing the `_endpoint_differs_for_provider` / `first_configured_endpoint` vocabulary), covering: custom endpoint with no config, custom endpoint differing from configured, endpoint matching config, llama.cpp default-endpoint-no-config
- [x] #4 Regression test `test_console_settings_apply_warns_when_llamacpp_endpoint_not_persisted` (added red in `Tests/UI/test_console_provider_persistence_regressions.py`) passes, and the existing Console settings-modal suite stays green
- [x] #5 Warning fires on the modal "Save" apply path and does not spam: one warning per apply that actually carries an unsaved differing endpoint
<!-- AC:END -->
## Implementation Notes

- Added pure helpers in `Chat/console_session_settings.py`: `console_session_endpoint_survives_restart` (compares the session endpoint against the real restart fallback chain: env -> `[console] llama_cpp_base_url_override` -> configured endpoint -> default) and `unsaved_console_endpoint_warning` (copy naming the restart consequence and "Save as default").
- `_apply_console_settings_result` (`UI/Screens/chat_screen.py`) now fires that warning after the success toast when the applied endpoint has no persisted backing; one warning per apply, none when backed or provider uses no endpoint.
- Modified files: `Chat/console_session_settings.py`, `UI/Screens/chat_screen.py`, `Tests/UI/test_console_provider_persistence_regressions.py` (apply-path regression test + 7-case helper unit test, both verified red on dev baseline).
