---
id: TASK-32272
title: Console boot crash when a keyless provider has an api_key configured
status: In Progress
assignee:
  - '@contact@rmusser.net'
created_date: '2026-09-10 19:09'
updated_date: '2026-09-10 19:37'
labels:
  - console
  - provider-readiness
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console readiness validator raises 'Console credential source conflicts with its facet' when the credential facet is not_required but a stored key exists, and the app exits before rendering anything. Reproduced with [api_settings.custom] api_key on the keyless custom provider; introduced with the Console conversation-settings redesign (939dee8dc2, 2026-09-04). Keyless OpenAI-compatible servers commonly still accept a bearer token, so this configuration is legitimate. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The app starts and Console reaches Ready when a keyless provider has an api_key configured.
- [x] #2 Readiness reports the credential facet and its source consistently for keyless providers with and without a stored key, without raising.
- [x] #3 A regression test covers a keyless provider with a stored key.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression test for a keyless provider (custom) with a stored api_key.
2. Confirm it fails with the reported ValueError.
3. Fix credential_source derivation in build_console_settings_readiness so it is forced to none whenever readiness.requires_api_key is False.
4. Run the full session-settings and provider-evidence test files.
5. Live-check app boot with a scratch config matching the repro.
6. Close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
build_console_settings_readiness() computed credential_source from readiness.api_key_source before checking readiness.requires_api_key, so a keyless provider (e.g. custom) with a stored api_key produced credential='not_required' alongside credential_source='stored', which the structural validator (ConsoleSettingsReadiness._validate_structured_state, L566-568) rejects. Fix: in the 'if not readiness.requires_api_key:' branch (console_session_settings.py, now L1392-1394) also force credential_source = 'none' -- the key is irrelevant to readiness for a keyless provider, matching the validator's existing rule. One-line fix. Added test_keyless_provider_with_stored_key_reports_not_required_without_source in Tests/Chat/test_console_session_settings.py (next to the existing keyless-provider readiness test), asserting credential == 'not_required', credential_source == 'none', operability == 'ready_to_send' for provider=custom with app_config[api_settings.custom].api_key='dummy'. TDD: test failed red with the exact reported ValueError before the fix, passed green after. Tests/Chat/test_console_session_settings.py + Tests/Chat/test_provider_test_evidence.py: 237 passed, 1 pre-existing unrelated failure (test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh, a SimpleNamespace mock missing capture_console_settings_origin, present before this change too). Live-checked: booted the app with HOME pointed at a scratch config ([chat_defaults] provider=custom/model=x, [api_settings.custom] api_key=dummy api_url=http://127.0.0.1:1) inside tmux; app reached the Console screen with its nav bar, no traceback, no first-run wizard.
<!-- SECTION:NOTES:END -->
