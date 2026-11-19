---
id: TASK-35560
title: Applying a session-only endpoint in Chat settings warns that it will not survive a restart
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: high
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_session_settings.py
- Tests/UI/test_console_provider_persistence_regressions.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's live capture phase (2026-10-18) and confirmed in triage to predate Phase 9: it reproduces at the phase base 111f8e48fc.

TASK-16473 made Console warn when an endpoint applied to a chat is saved for this session only, so a restart will lose it. The running app has not shown that warning since 7cf89de6c0 (2026-08-29, "feat: consolidate long-lived branch changes"). That commit routed the modal's result to `ChatScreen._dispatch_console_settings_submission`, the live apply path for Chat settings and Switch model, which only says "This chat updated". The only call to `unsaved_console_endpoint_warning` stayed in `ChatScreen._apply_console_settings_result`, which nothing in tldw_chatbook/ calls any more.

Its tests stay green because they call the dead method directly. Those tests are `test_console_settings_apply_warns_when_llamacpp_endpoint_not_persisted` and `test_console_settings_result_applies_name_override_without_losing_prompt_source`.

The warning's copy also names "Save as default", which no current control is called. The action that saves an endpoint is "Default for new chats (Ctrl+N)".

Live evidence: Phase 9 capture 01k, and the triage's base capture 01k-base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Applying a chat's settings with an endpoint that is saved for this session only shows exactly one warning that it will not survive a restart, on the live Chat settings and Switch model apply paths
- [ ] #2 No warning shows when the endpoint is already saved, when the provider has no endpoint, or when the action itself saves the endpoint as the default for new chats
- [ ] #3 The warning names a control that exists and saves the endpoint, and renders runtime text literally (markup-safe)
- [ ] #4 Tests drive the live apply path, not a method the product never calls, and the dead path is removed or no longer pinned
- [ ] #5 Docs/User_Guide/console.md says when the warning appears
<!-- AC:END -->
