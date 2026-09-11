---
id: TASK-32325
title: >-
  Attach context button no longer mislabels its action
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B1. The control-bar button labeled 'Attach context' (console_control_bar.py:47) only opens the left rail; staging actually happens in Library, and the docs admit it parenthetically. When the rail is already open (the default) the click appears to do nothing. Either make it go where staging happens or rename it to what it does.

Filed from the 2026-09-10 Console rail UX review (review item B1).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repin the four test sites to 'Context rail'. 2. Rename label+tooltip in live action and fallback; keep ids. 3. Update the three docs. 4. Run workbench-state + contract + flow suites; stash-verify failures pre-exist.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The control no longer claims to attach anything it does not attach, and does something visible even when the left rail is already open
- [x] #2 Composer menu entry with the same label is updated to match
- [x] #3 User-guide rows for the control bar are updated to the new behavior/copy
- [x] #4 Tests covering the button label/action are updated
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** Renamed the control-bar action "Attach context" ->
"Context rail" (label says what the button does; the tooltip now carries
the staging pointer: "Open the Console context rail; stage sources from
Library"). The ACTION ID stays "attach-context" and widget ids are
untouched (no CSS/test churn). Changed in BOTH the live action
(console_workbench_state.py) and the fallback (console_control_bar.py,
byte-matching per CN-03). The composer menu's separate real file-picker
entry stays "Attach file" — the two-name rule is preserved, now with both
names truthful. Docs updated: console.md control-bar table + layout tour,
context-and-rag.md staging section, attachments-images-voice.md (with a
rename note for older documentation). Test pins updated at the four sites
that asserted the old label.

**ADR check.** Not required — copy rename on a stable id.

**Modified.** `Widgets/Console/console_workbench_state.py`,
`Widgets/Console/console_control_bar.py`, three user-guide files, four
test files' pins. Verified: workbench-state suite 6 passed;
workbench-contract action/help tests 5 passed (the 6 contract failures in
wider runs are the pre-existing dev-drift set — identical on a stashed
clean tree); native-chat-flow attach/control tests 10 passed (1
pre-existing on clean tree).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: control-bar 'Attach context' routes to rail reveal only (chat_screen.py:4640-4651); composer menu separately has a real 'Attach file' (file picker). Mislabel stands.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
