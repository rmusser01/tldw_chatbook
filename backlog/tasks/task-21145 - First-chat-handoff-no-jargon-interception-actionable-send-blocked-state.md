---
id: TASK-21145
title: 'First-chat handoff: no jargon interception, actionable send-blocked state'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:14'
updated_date: '2026-08-25 16:14'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings H-2, H-3 (findings.md section H): on a fresh profile the first message triggers a 'Project instructions need a folder' dialog exposing raw no_eligible_binding; with a broken provider the composer says 'Send blocked - finish provider setup to continue' with no way to reach that setup, and validation can sit 30s+ with no error or cancel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing a first message on a fresh profile is never intercepted by the project-instructions folder dialog
- [x] #2 The send-blocked state offers a working affordance that opens provider setup
- [x] #3 Provider validation surfaces a terminal result (success or actionable error) within a bounded time, with the run cancellable
- [x] #4 No raw internal error codes are shown to the user in this flow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace no_eligible_binding folder dialog trigger on first send; stop interception on fresh profiles\n2. Trace 'Send blocked — finish provider setup' + 'Validating provider.' lifecycle; add open-provider-setup affordance; bound the validation with terminal result + cancel\n3. Replace raw error codes with human copy\n4. Tests + live tmux first-chat run
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
H-2 root cause: ProjectInstructionControlState.new_session() defaults enabled=True with no binding, so a fresh profile's first send raised no_eligible_binding into the modal recovery dialog (raw code on screen). Fix at the single send-path interception site (_run_agent_reply): project_recovery_should_skip_send_interception() — a session that NEVER had a binding with no eligible folders has nothing to recover; instructions simply don't apply to that send. Broken-existing-binding and choose-among-folders dialogs remain (the resolver's raise semantics and its security tests are untouched). Verified live end-to-end: fresh profile, bad-key wizard, first 'Hello' sent with zero interception.

H-3: all seven send-path resolve_for_send awaits now route through _resolve_for_send_bounded (30s hard deadline -> not-ready with actionable timeout copy; cancellation propagates). Live: the 401 send reached a terminal 'Assistant Failed … HTTP 401' in seconds. The setup-blocked composer reason is now itself an action link (whole-text + ' ›' chevron -> app.run_setup_wizard, new idempotent app action; an appended 'Open setup' label blew the strip's width budget — caught by the workbench contract test). Docs/User_Guide/console.md updated.

Verification honesty: full Tests/Chat sweep showed ~68 failures + 9 errors that are PRE-EXISTING on dev — baselined by stashing my edits and diffing failure sets (byte-identical). Wizard trio: 873 passed. New tests: 6 handoff units + app-action idempotence + 3 strip-pin updates.

Files: console_chat_controller.py, console_composer_bar.py, app.py, Tests/Chat/test_console_first_chat_handoff.py (new), 4 test files updated, Docs/User_Guide/console.md.
<!-- SECTION:NOTES:END -->
