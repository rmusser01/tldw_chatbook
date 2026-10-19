---
id: TASK-35310
title: >-
  Guard Console sends while a conversation resume activation is still in flight
status: Done
assignee:
  - Robert
created_date: '2026-10-18 00:00'
labels:
  - console
  - send-admission
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console UAT 2026-10-17 (`qa/console-uat-2026-10-17/`, origin/dev `6c0bde2437`) round 11: a
Send pressed shortly after clicking a workspace-rail conversation row was **silently dropped** —
no provider call, no DB write, no transcript turn, no user feedback. With a ~1.5 s settle the
send works.

Cause: `open_console_workspace_conversation`'s slow path occupies the screen pump across
multiple awaits during which `active_session_id` *and* the composer binding still name the OLD
session, so every switch-progress guard sees a consistent OLD state and waves the Send through,
pinned to OLD. The decoupled send task then dies in branches that return without any notice
(`chat_screen.py` pending-token mismatches, `send_acknowledgement.py` Enter-guard drop) or in
received-intent custody whose failure only logs. `refuse_send_from_unbound_composer` cannot
fire because bound == active == OLD throughout the window.

Governance: ADR-098 ("boundary-racing drafts never stranded or duplicated"), ADR-230, ADR-094;
the refusal UX follows the shipped TASK-33622.7 tab-switch guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:AC:BEGIN -->
- [x] Enter/Send during the resume-activation window shows the existing switching-chats refusal and dispatches nothing
- [x] No silent drop remains on this path: the pending-token and Enter-guard branches notify the user
- [x] After the resume settles, Enter sends into the newly activated conversation
- [x] A `console_send_stage` breadcrumb records the switch-in-flight refusal
- [x] Existing tab-switch / send-guard / acknowledgement tests stay green
<!-- SECTION:AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test in `Tests/UI/test_console_resume_send_guard.py` cloning the Gate pattern from
   `test_console_tab_switch_binds_composer.py`: hold `load_console_conversation_tree`
   mid-resume, deliver Enter with a draft, assert refusal + no dispatch.
2. Set a conversation-activation-pending flag over the slow path of
   `open_console_workspace_conversation` (try/finally covering all rollbacks).
3. `refuse_send_from_unbound_composer` refuses while the flag is set (existing copy).
4. Louden the silent branches; add the send-stage breadcrumb.
5. GREEN + adjacent suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: a reentrant `conversation_activation_window` on the Console session controller marks the
  rail-row resume's slow path (hydration) as in flight; `refuse_send_from_unbound_composer` refuses
  with the existing TASK-33622.7 "Still switching chats…" copy while the window is open, so Enter and
  the Send button both get a visible refusal instead of pinning to the OLD chat and dying silently.
- Belt: the two silent `return False` branches in `_send_console_message_from_visible_action_observed`
  (pending-token mismatch, tokenless send while the slot is busy) now notify, and the Enter-guard
  release in `send_acknowledgement._schedule` notifies ("Send is not available in this chat yet…").
- The refusal also logs a WARNING (the press happens outside any `send_diagnostic_scope`, so a
  `record_send_stage` breadcrumb there would be a silent no-op; the loudened in-scope branches do
  emit `ui_submit/not_dispatched` with reasons).
- Tests: `Tests/UI/test_console_resume_send_guard.py` (RED first: mid-window Enter produced no
  notice and dispatched to the OLD chat) — mid-window refusal + post-settle delivery to the resumed
  chat; adjacency suites (`tab_switch_binds_composer`, `send_resend_guard`, `send_acknowledgement`)
  stay green.
- Harness notes captured for the next person: seeded conversations need a FILE-backed DB at the
  sandbox path plus `app._wire_chat_conversation_services()` after attach (the conversation browser
  wires from `app.chachanotes_db` at __init__), and resumed sessions base their settings on
  `chat_defaults` (seed llama defaults via `_configure_native_ready_console`).
- Modified files: `UI/Console_Modules/session.py`, `UI/Console_Modules/workspace.py`,
  `UI/Screens/chat_screen.py`, `UI/Console_Modules/send_acknowledgement.py`, plus the new test file.
- ADR check: no new ADR — direct implementation of ADR-098's "boundary-racing drafts never stranded
  or duplicated" promise using the refusal UX TASK-33622.7 already shipped.
