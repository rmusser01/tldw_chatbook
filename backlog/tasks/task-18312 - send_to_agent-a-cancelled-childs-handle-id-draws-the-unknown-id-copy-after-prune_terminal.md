---
id: TASK-18312
title: >-
  send_to_agent: a cancelled child's handle id draws the unknown-id copy after
  prune_terminal, not the honest not-retained copy
status: To Do
assignee: []
created_date: '2026-08-18 15:40'
labels:
  - agents
  - console
priority: low
dependencies: []
---

## Description (the why)

Flagged by PR 3b Task 5's landing report ("the pruned-cancelled-handle-id
refusal gap") and re-verified reproducing at dev `cf5db6f50` by Task 6 before
filing (per the plan: file, do not patch — changing the resolution ladder is
Task 4's shipped design and deserves its own review).

Within the turn it was cancelled in, a cancelled child's `send_to_agent`
refusal is honest: "has finished (cancelled) and no retained transcript is
available … cannot be resumed" (pinned by
`test_a_cancelled_child_draws_the_honest_not_retained_refusal_not_unknown`).
But after the NEXT turn starts, `prune_terminal` drops the terminal handle,
and cancelled children are never retained — so the handle id falls through
every ladder tier (live handles → retention store → un-pruned terminal
handles → DB run-id tier) to the unknown-id copy. Task 6's probe, verbatim at
`cf5db6f50`:

    ERROR: send_to_agent: no sub-agent matches id '292a9e3c…' (checked
    handle ids and run ids). Live sub-agent ids: none.

A supervisor that spawned, cancelled, and later re-addresses a real child by
the id its own spawn result gave it is told the child never existed. (The
same child's RUN id still resolves — to the DB tier's post-restart copy,
which is itself slightly off for a same-session cancel: "finished in an
earlier session".)

Candidate shapes from Task 5's report: a DB tier for handle ids, or teaching
the unknown-id refusal to mention that run ids survive where handle ids do
not. Either changes Task 4's resolution ladder, so it needs its own review
against the six ladder-order pins in `Tests/Agents/test_fleet_continuation.py`.

## Acceptance Criteria (the what)

- [ ] After the next turn's prune, steering a real (but cancelled/pruned) child's handle id draws a copy acknowledging the child existed and cannot be resumed — never the "no sub-agent matches" unknown-id copy
- [ ] The same-session cancelled child's run id no longer draws the "finished in an earlier session" wording, or that wording is generalized honestly
- [ ] The existing resolution-order pins (live-before-retained, retained-before-terminal, handle-before-run-id) stay green
