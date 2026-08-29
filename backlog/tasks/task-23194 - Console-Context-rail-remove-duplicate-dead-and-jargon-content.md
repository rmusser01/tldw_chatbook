---
id: TASK-23194
title: 'Console Context rail: remove duplicate, dead and jargon content'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-29 23:20'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four content defects in the Context rail: 'No character in this chat' renders twice from two different widgets; the Agent section mounts three zero-size focusable widgets including a text Input; four controls (Switch, Star, star glyph, Clear) ship disabled with no explanation; and 'Local stars unavailable' exposes developer language to users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The no-character empty state renders exactly once
- [x] #2 No KEYBOARD-REACHABLE Context rail control paints nothing (revised - see Implementation Notes; the original wording rested on a false audit finding)
- [ ] #3 Controls that cannot act are hidden rather than disabled, or explain their precondition - MOVED to the conversation action menu task
- [ ] #4 'Local stars unavailable' is removed or replaced with user-facing copy - MOVED to the conversation action menu task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered AC #1 and a revised AC #2. AC #3 and #4 moved to the conversation action-menu task, because the reviewer asked for the dead star column to be REPLACED by an actionable menu rather than hidden - the opposite remedy to the one this task assumed.

AC #1 - duplicate empty state. Two different widgets rendered the identical sentence on consecutive rows: the avatar placeholder (_build_character_avatar_widget) and the name row (#console-character-name). The placeholder now paints nothing when there is no character; the name row keeps the copy, which is where it belongs semantically. Confirmed in the rendered capture: the Character section went from two identical lines to one.

AC #2 was rewritten because the audit finding behind it was WRONG. The audit reported three zero-size focusable widgets (console-workspace-tree-star, console-inspector-section-agent-fleet-toggle, console-agent-steering-input) and concluded keyboard focus could land on a text Input painting nothing. It cannot. The audit queried 'can_focus and display', but a widget's own display stays True while an ANCESTOR is hidden; Textual's real focus chain excludes all three (probed: in_focus_chain=False for each). No fix was needed. The test now pins the invariant the audit was reaching for and which is genuinely user-facing: nothing in screen.focus_chain inside the rail may have a zero-size region. It passes without production changes and guards against a real future regression.

Three existing tests asserted the placeholder renders 'No character in this chat'. Rather than delete that coverage, test_avatar_placeholder_paints_nonzero_region_in_auto_holder now exercises the case where the placeholder IS shown (character set, no image) so the task-3793 0x0-collapse guard still runs on a visible widget; the property it protects belongs to the auto/auto holder, not to the string inside it.

Files: tldw_chatbook/UI/Screens/chat_screen.py; Tests/UI/test_console_context_rail_content.py (new); Tests/UI/test_console_character_avatar.py.
<!-- SECTION:NOTES:END -->
