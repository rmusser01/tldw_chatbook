---
id: TASK-16800
title: 'Turn file card: annotate/feedback loop and Review affordance'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-15'
updated_date: '2026-08-17 03:28'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-1972
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console turn file card (`ConsoleTurnFileCard`) lets a user expand any
changed file's diff inline, but it is read-only: there is no way to leave
feedback on a specific hunk, and the only path into the full Review screen
(turns, retention, guarded per-path revert) is the keyboard-only `v`
binding, undiscoverable from the card itself. This is V1.5 of the turn
file card design (`Docs/superpowers/specs/2026-08-15-console-turn-file-review-design.md`,
"Out of scope" section): the review screen and guarded revert already
exist via TASK-1972, so this task is scoped to two additions — an
annotate/feedback loop on expanded hunks, and a `Review` button on the
card that opens that same screen at the turn.

Feedback recorded here should be usable as context for the agent's next
reply, closing the loop between "the agent shows me a diff" and "I tell it
what to change" without leaving the transcript to type a follow-up
message by hand.

Two other V1.5 polish items were also trimmed from V1 and belong in this
same follow-up bucket: a header collapse/expand-all chevron (today each
row toggles independently, with no all-at-once control) and middle-elided
per-row paths (today a long path is shown in full rather than elided to
fit the row).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An expanded diff row exposes an action to attach a note to a specific hunk, without leaving the transcript
- [ ] #2 A note attached to a hunk is durably recorded (survives session resume, like the rest of the card's source data) and is available to the agent as context on its next reply
- [ ] #3 The card exposes a `Review` affordance that opens the existing Review screen scoped to that turn — equivalent to pressing `v`, reachable without the keyboard shortcut
- [ ] #4 No control added to the card performs a destructive action; revert remains exclusively on the Review screen behind its existing confirm (the TASK-1845/TASK-1972 precedent)
- [ ] #5 With `[console] turn_file_cards` set to `false`, the plain-text marker row and its `v` binding are unaffected by this feature (no regression to the kill-switch fallback)
<!-- AC:END -->
