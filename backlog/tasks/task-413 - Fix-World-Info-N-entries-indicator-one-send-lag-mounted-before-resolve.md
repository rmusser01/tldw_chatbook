---
id: TASK-413
title: 'Fix [World Info: N entries] indicator one-send lag (mounted before resolve)'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 16:21'
updated_date: '2026-07-24 01:10'
labels:
  - bug
  - chat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In handle_chat_send_button_pressed (chat_events.py), the [World Info: N entries activated] indicator ChatMessage is mounted (~line 1180, reads app.current_world_info_active/count) BEFORE the send path resolves world-info and updates those reactives (~line 1395). So the indicator on send K reflects send K-1's world-info, not the current send. Pre-existing (the mount-before-resolve ordering predates P2g-3); surfaced by the opus whole-branch review + Qodo on PR #740. Fix by reordering — resolve world-info before mounting the indicator, or move the indicator mount after the resolution — so the indicator reflects the CURRENT send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The [World Info: N entries] indicator reflects the CURRENT send's matched-entry count/active state (not the previous send's),No regression to the world-info injection itself (byte-parity preserved),import tldw_chatbook.app OK; a test covers the corrected ordering
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read handle_chat_send_button_pressed end-to-end (~1080-1450) to map the mount order: user message (step 6, ~1174) -> [stale indicator mount, ~1180-1191] -> DB save/UI updates -> API key check -> AI placeholder mount (step 10, ~1326) -> RAG (10.5) -> chat dictionaries (10.6) -> world-info resolution updating app.current_world_info_active/count (10.7, ~1394).
2. Confirm the AI placeholder mounts before world-info resolves (Textual Widget.mount supports before=/after= params), so transcript position CAN be preserved without restructuring by mounting the indicator later with before=ai_placeholder_widget.
3. Delete the indicator-mount block from step 6 (leave a pointer comment); re-add an equivalent block immediately after the world-info resolution (end of step 10.7), guarded by the same not reuse_last_user_bubble / not resend_conversation conditions plus user_msg_widget_instance is not None, reading the now-current app.current_world_info_active/count, and mounting with before=ai_placeholder_widget.
4. Do not touch resolve_world_info_injection or message_text_with_world_info construction (byte-parity requirement).
5. Add a regression test in Tests/Event_Handlers/Chat_Events/test_chat_events.py driving two sends via handle_chat_send_button_pressed with resolve_world_info_injection mocked to different counts (3 then 5), asserting each send's mounted indicator carries that send's own count and is mounted before that send's AI placeholder.
6. Verify RED by stashing only the chat_events.py fix (keeping the test) and confirming the new test fails against the pre-fix code, then restore the fix and confirm GREEN plus no regressions in related world-info/chat_events test modules.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the [World Info: N entries activated] indicator mount in handle_chat_send_button_pressed (tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py) from step 6 (right after mounting the user message, ~old line 1180 — read app.current_world_info_active/count BEFORE resolve_world_info_injection had run for this send, so it showed the PREVIOUS send's world-info) to immediately after step 10.7's world-info resolution (~new line 1394, right after app.current_world_info_active/count are set for the CURRENT send). The AI reply placeholder is mounted earlier (step 10) than world-info resolution, so restructuring to keep strict mount-order wasn't possible without moving the placeholder too; instead the indicator is mounted with chat_container.mount(world_info_widget, before=ai_placeholder_widget), using Textual's Widget.mount(before=) to preserve its transcript position (after the user message, before the AI reply) regardless of construction order. The world-info resolution call (resolve_world_info_injection) and message_text_with_world_info construction were not touched — byte-parity of the injected prompt preserved.

Added a regression test, test_world_info_indicator_reflects_current_send_not_previous, in Tests/Event_Handlers/Chat_Events/test_chat_events.py: drives two sends through handle_chat_send_button_pressed with resolve_world_info_injection mocked to return (text, 3) then (text, 5), and asserts each send's mounted -world-info-indicator widget shows that send's own count (not the other send's) and is mounted with before=<that send's AI placeholder>. Verified RED by git-stashing only the chat_events.py fix (keeping the test) — pre-fix, send 1 mounts 0 indicators (stale False/0 from before resolution) instead of 1, failing the test; restored the fix (git stash pop) and reran to confirm GREEN.

Verified: Tests/Event_Handlers/Chat_Events/test_chat_events.py (16 passed, incl. new test), plus Tests/Character_Chat/test_resolve_world_info_injection.py, Tests/Character_Chat/test_character_world_book_send_path.py, Tests/UI/test_console_world_info_send_integration.py, Tests/Chat/test_console_world_info_application.py (36 passed total across the 5 files); also Tests/UI/test_ux_audit_smoke.py, Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py, Tests/unit/test_chat_image_unit.py, Tests/integration/test_chat_tabs_integration.py, Tests/Widgets/test_chat_session.py, Tests/UI/test_send_stop_button.py (83 passed, 21 skipped, no regressions). `python -c "import tldw_chatbook.app"` succeeds.

Modified files: tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py, Tests/Event_Handlers/Chat_Events/test_chat_events.py, backlog/tasks/task-413 - Fix-World-Info-N-entries-indicator-one-send-lag-mounted-before-resolve.md.
<!-- SECTION:NOTES:END -->
