---
id: TASK-37840
title: >-
  Console: the always-visible Temporary-chat surfaces no longer say the chat is
  not saved
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-30 01:30'
updated_date: '2026-10-31 00:09'
labels:
  - console
  - copy
  - needs-owner
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.17 changed the status-strip segment to 'Temporary · traces off' and the tab-strip ◌ tooltip decode with it, both owner-specified copy. The chip's own tooltip (which still says 'not saved locally') has not been rendered since TASK-35160 turned the chips into one status line, so the only on-screen 'not saved' a Temporary chat now shows is the composer Menu's Save chat hint. A user glancing at the strip learns that traces are off but not that the whole chat is lost when the tab closes. Raised by the pre-merge review of TASK-37811; needs a UX look and an owner call on whether the always-visible copy should say both.

A second question for the same call, from the second review of TASK-37811: the segment keeps saying 'Temporary · traces off' after someone explicitly turns Capture On for the Temporary chat (This conversation or Next send). That is literally true (nothing is written while the chat stays temporary), but those sends now pause on the Save chat to capture this send card, so the strip no longer predicts what a send will do.

**Owner decision (2026-10-30).** The always-visible Temporary status segment says both: 'Temporary · not saved · traces off', and 'Temporary · not saved · capture on' once Capture is explicitly turned On for that Temporary chat (This conversation or Next send). The tab-strip ◌ tooltip decode stays consistent with it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on what the Temporary segment and tab marker say is recorded on this task
- [ ] #2 The status-strip segment and the tab-strip marker's decode say what the decision requires, and the User Guide matches
- [ ] #3 A test pins the visible copy
- [ ] #4 The decision also says what the segment reads when Capture On is chosen for a Temporary chat (its sends pause), and a test pins that case
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on this head (done 2026-10-30, mounted Console, real dialog path): the strip reads 'Temporary · traces off' and the tab decode 'Temporary · traces off.' in all three states (default, This conversation On, Next send On), though the last two pause each/the next send. Next send On survives the pause card's Cancel and is spent by Send without capture.
2. Tests first: unit pins for the segment render and the tab decode (both cases), a pure helper that says whether a Temporary chat's next send is captured, checked against the controller snapshot; one mounted file (stock ChatScreen/controller/store, the real Inspector > Trace Privacy dialog) that reads the painted strip and tab tooltip for default, This conversation On, back to Inherit, Next send On, and after Send without capture spends it. Red on this head.
3. Implement: labels in console_ephemeral; helper in console_exchange_capture; the status segment and tab decode take the active chat's resolved capture; the screen re-syncs the segment on every full sync pass and on resume (return from the dialog).
4. User Guide: sessions-tabs-workspaces.md, context-and-rag.md, semantic-trace-capture.md.
5. Rebase onto newest dev, gates, live round in the real app.
<!-- SECTION:PLAN:END -->
