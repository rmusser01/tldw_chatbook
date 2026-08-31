---
id: TASK-25721
title: Provider continuation card shows remote-handoff copy during local sends
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 13:46'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A plain local send surfaced a card reading that response delivery status is unknown on the source device and warning that retrying may send a duplicate request. There is no source device in a local single-machine send. The card also omits the Owner, Problem and Impact structure the sibling interrupt card uses, so the product presents two different grammars for the same class of blocking decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Continuation-recovery copy appears only for sends that genuinely crossed a device boundary
- [ ] #2 All blocking interrupt cards use one consistent Owner, Problem, Impact and action structure
- [ ] #3 Card copy names a concrete cause rather than an internal subsystem
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RE-FRAMED -- the copy is not the defect, and rewriting it would be wrong.

'Response delivery status is unknown on the source device' lives in
_UNRESOLVED_IMPORTED_STATE_COPY (assistant_generation_state.py) and is emitted
for ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED. For its intended state
-- a conversation IMPORTED from another device whose dispatch outcome cannot be
resolved locally -- 'the source device' is exactly right, and three test sites
pin it.

The real defect is that a plain local single-machine send reached that state at
all. I hit it by choosing 'Send without capture' on a trace-blocked turn, after
which the turn was reconciled through the dispatch-recovery path and classified
REMOTE_DISPATCH_STARTED. So the question is why a local unresolved dispatch is
treated as remote -- a classification bug in the recovery path, not wording.

BLOCKED ON TASK-25814, like TASK-25713: the only route I have to this state runs
through the trace block, so I cannot tell whether it reproduces on a healthy
dispatch. Re-test once the trace wiring lands; if a local send can still land in
REMOTE_DISPATCH_STARTED, fix the classification and leave the remote copy alone.

The second half of this task -- that the card omits the Owner/Problem/Impact
structure its sibling uses -- remains valid and unaddressed. It is a small,
self-contained follow-up once the classification question is settled.
<!-- SECTION:NOTES:END -->
