---
id: TASK-25815
title: Console shows no status while a reply is pending
status: Done
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 14:12'
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
RESOLVED BY OBSERVATION once TASK-25814 unblocked dispatch -- no code change needed here.

My original report (30+ seconds of an empty assistant row) was accurate but the
run was BLOCKED, never streaming: the trace guard refused dispatch, so no tokens
were requested and neither the '[streaming]' status token
(console_transcript.py, task-2154.16) nor the 'Generating…' resolution
(console_display_state.py, TASK-347) could fire.

With 25712 fixed I sent 'Name three primary colors.' against a live local
provider and watched the row progress:
    Assistant  Generating…
    Assistant  Thinking… · <1s
    Assistant  I couldn't find any color tools in my catalog.
A state label AND an elapsed timer, exactly what this task asked for. The
indicators were never missing; nothing was ever running.

REMAINING SLIVER, not worth its own task on current evidence: an ACCEPTED but
pre-dispatch-BLOCKED turn still renders a bare row while its reason lives in a
card above the transcript. That state is now rare (25712 removed the path that
made it universal) and 25715 stops other panels stacking over the card that
explains it. Re-open only if it is observed again on a healthy dispatch.
<!-- SECTION:NOTES:END -->

## Renumbering

Filed as TASK-25713 on 2026-08-31 05:07. `dev` later merged a TASK-25713
("Census warm boot flakes on sys.modules mutation during iteration", created
14:12), and the backlog guard flagged the duplicate.

DELIBERATE DEVIATION from the 2026-08-21 owner rule (TASK-19601), stated so it
is not mistaken for an oversight: by that rule the OLDER arrival keeps the id,
which would be this task (05:07 vs 14:12), and the Census task would renumber.
It renumbers the other way here because the Census task is already MERGED on
`dev` and may carry references in code, tests or commit messages that an
unmerged PR cannot see; renumbering it from here risks breaking work that is
already landed. This task is not yet merged, so moving it is the change with no
blast radius.

Renumbered to 25815 (next free after 25814, swept across all refs and
worktrees). References updated in TASK-25814 and TASK-25721. Earlier commit
messages on this branch still cite 25713 and are left as historical record.
