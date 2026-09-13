---
id: TASK-32324
title: >-
  Stacked collapsed rail labels discoverable plus badge legend
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
UX review A5. Collapsed handles read 'Context->' / '<-Inspect' with abbreviated badges ('1 appr', 'art') in 13/11 columns. The vertical stacked-label mode exists (console.stack_collapsed_rail_labels, default false) but is buried; and no legend explains the badge abbreviations.

Filed from the 2026-09-10 Console rail UX review (review item A5).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the badges legend line to F1's Panes group. 2. Pin it via the existing F1 test. 3. Document legend + toggles beside the stacked-labels setting and the handle row. 4. Run keyboard-route suite.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The F1 help panel documents what the handle badges mean (approvals, artifact)
- [x] #2 The stacked-labels setting is documented in the user guide next to the rail presentation settings
- [x] #3 No behavior change to default handle rendering
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The F1 help's Panes group now carries a "Handle badges"
legend line decoding "N appr" (N approvals pending) and "art" (artifact
ready) — pinned by test in the keyboard-route suite. User guide: the
handle badge row in console.md's rails table points at the F1 legend, and
the Collapsed-rail-labels section in chat-basics.md now also names the
keyboard toggles (Alt+C / Alt+I — the width-independent alternative) and
the badge legend. Default handle rendering unchanged (AC#3).

**ADR check.** Not required — help copy + docs only.

**Modified.** `UI/Screens/chat_screen.py` (F1 group legend),
`Tests/UI/test_console_inspector_keyboard_route.py` (legend pin),
`Docs/User_Guide/console.md`, `Docs/User_Guide/console/chat-basics.md`.
Verified: keyboard-route suite 13 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): labels 'Context >'/'<-Inspect', badges '1 appr'/'art' (console_rail_handle.py:120-135); F1 built from CONSOLE_WORKBENCH_SHORTCUT_GROUPS (chat_screen.py:1199-1293).
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
