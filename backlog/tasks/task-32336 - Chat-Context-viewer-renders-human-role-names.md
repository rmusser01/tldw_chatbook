---
id: TASK-32336
title: >-
  Chat Context viewer renders human role names
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C7. The Ctrl+Shift+P viewer displays internal enum forms like '[ConsoleMessageRole.USER] complete' (task-2704). Map roles to display names in the viewer; this is the power user's audit surface.

Filed from the 2026-09-10 Console rail UX review (review item C7).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a RED test pinning rendered titles '[User] complete' / '[Assistant] streaming' with no enum repr. 2. Add _display_role_name helper and use it in _build_current_context_widgets. 3. Audit other role render sites in the viewer. 4. Re-run the inspector suite.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The viewer renders 'User'/'Assistant'/etc. instead of enum reprs for message roles
- [x] #2 Any other internal-form leaks in the viewer's rendered rows are mapped or filed
- [x] #3 Viewer tests assert the display mapping
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The Current Context tab built every message Collapsible's
title from `f"[{msg.role}] {msg.status}"`; `msg.role` is a
`ConsoleMessageRole` str-mixin Enum whose f-string form on the supported
runtime (>=3.11) is the qualified "ConsoleMessageRole.USER". Added a
module-level `_display_role_name()` helper (value-passthrough + title-case)
used at that one title site. Audited the viewer's other role renderings
(`_exchange_turn_title`, exchange message rows, cost rows): all receive
plain strings already and render fine; the payload-based exchange rows
(provider dicts) are also plain strings. This retires the task-2704
symptom the user guide still footnotes.

**ADR check.** Not required — display copy inside one widget.

**Modified.**
`tldw_chatbook/Widgets/Console/console_conversation_inspector.py`,
`Tests/UI/test_console_conversation_inspector.py` (+1 test asserting the
RENDERED title shows "[User] complete" and no enum repr; RED before the
fix on exactly that string). Verified: full inspector suite — 43 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): console_conversation_inspector.py:1810-1822 renders f'[{msg.role}] {msg.status}' with enum-derived values; on py>=3.11 that prints '[ConsoleMessageRole.USER] complete'.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
