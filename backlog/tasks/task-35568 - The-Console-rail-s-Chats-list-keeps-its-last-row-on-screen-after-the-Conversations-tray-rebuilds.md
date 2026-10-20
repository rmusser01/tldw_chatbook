---
id: TASK-35568
title: The Console rail's Chats list keeps its last row on screen after the Conversations tray rebuilds
status: To Do
created_date: 2026-10-18 22:40
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_workspace_context.py
- tldw_chatbook/Widgets/Console/console_bounded_section.py
- tldw_chatbook/UI/Console_Modules/left_rail.py
- tldw_chatbook/UI/Screens/chat_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's capture check run (2026-10-18). Triage reproduced it at the phase base 111f8e48fc, so it predates Phase 9.

After the Conversations tray rebuilds, the Conversations section can end up one line shorter than its content. A rebuild follows a width relabel on resize, or a state change. The last Chats row ('Chat 1 💬' under 'Today') is then clipped, and the sections below move up one line. The user's open chat disappears from the rail's Chats list, along with its actions button. It stays hidden until a later rebuild happens to come out right.

No data is lost. The browser state still holds the row, and its 'Today' divider still shows. The chat stays open, and its summary under 'Search all chats…' still shows.

A rebuild drops the 'New conversation' alias, and the screen mounts it again afterwards (since TASK-33008). Both the tray fit and the bounded viewport's height are snapshot measurements, and their requests are coalesced. The clipped amount is exactly the alias's one line. The likeliest cause is a final measurement taken before the re-mounted alias is laid out. Which owner keeps the stale height is not yet proved: a pilot harness came out healthy every time.

Live, from a fresh scratch profile: resizing from 235x52 to 211x44 at least about 40 s after launch lost the row every time. The same cycle at about 30 s kept it, and Alt+M and Esc never removed it on their own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In a full Console at 211x44 and at 235x52, the last Chats row stays on screen whole after any number of resize cycles and Alt+M/Esc opens, and nothing below it shifts
- [ ] #2 A deterministic test rebuilds the tray with the alias mounted again late, and it fails before the fix: the last row's region lies outside the Conversations viewport
- [ ] #3 The fix measures again at most once per rebuild, and the tray's recompose-guard and alias pins still hold, or are updated on purpose
<!-- AC:END -->
