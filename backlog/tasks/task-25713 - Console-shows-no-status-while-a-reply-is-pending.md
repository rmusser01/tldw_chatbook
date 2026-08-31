---
id: TASK-25713
title: Console shows no status while a reply is pending
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 13:29'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An assistant row mounts as an empty bordered block with no spinner, elapsed timer, or state label. Against a provider that answers in 0.8s, Console showed this blank block for over 30 seconds before any card appeared. A pending reply, a stalled run, and a silently failed run are visually identical, which is the highest-frequency moment in the product.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pending assistant row always shows a live state label distinguishing generating, waiting on an action, and failed
- [ ] #2 Elapsed time is visible while a reply is pending
- [ ] #3 A run that ends without content renders an explicit outcome instead of an empty block
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
BLOCKED ON TASK-25712 -- evidence is entangled; do not design a fix from my original observation.

I reported 30+ seconds of an empty assistant row against a provider answering in
0.8s. That is accurate but the run was BLOCKED, never streaming: the
trace-capture guard (TASK-25712) refused dispatch, so no tokens were ever
requested. An empty row may well be the CORRECT render for 'accepted but not
started'.

The app already has the indicators my finding said were missing:
  - console_transcript.py carries a '[streaming]'/'[stopped]'/'[failed]' status
    token rendered as a status line (task-2154.16, FB-01)
  - console_display_state.py resolves 'Generating…' for a running generation
    (TASK-347)
Neither could fire, because nothing was running.

So the real question is narrower and currently unanswerable: when a turn is
ACCEPTED but BLOCKED before dispatch, the row shows neither 'Generating…' (not
running) nor a block reason (that lives in a card above the transcript), so it
reads as a hung response. Whether that gap survives once dispatch works can only
be judged after 25712 is fixed and a genuine streaming turn can be observed.

NEXT STEP: re-observe a real streaming turn once the trace wiring lands, then
re-scope this to the accepted-but-blocked state only, if it still reproduces.
<!-- SECTION:NOTES:END -->
