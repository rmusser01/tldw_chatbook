---
id: TASK-18312
title: >-
  send_to_agent: a cancelled child's handle id draws the unknown-id copy after
  prune_terminal, not the honest not-retained copy
status: Done
assignee:
  - '@codex'
created_date: '2026-08-18 15:40'
updated_date: '2026-09-08 03:48'
labels:
  - agents
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After the next turn's prune, steering a real (but cancelled/pruned) child's handle id draws a copy acknowledging the child existed and cannot be resumed — never the "no sub-agent matches" unknown-id copy
- [x] #2 The same-session cancelled child's run id no longer draws the "finished in an earlier session" wording, or that wording is generalized honestly
- [x] #3 The existing resolution-order pins (live-before-retained, retained-before-terminal, handle-before-run-id) stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the TASK-18312 section of Docs/superpowers/plans/2026-09-07-fleet-lifecycle-and-identity-repairs.md. Reproduce first, implement the bounded repair, and run its targeted regressions before marking Done.

ADR required: no additional ADR
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: Direct implementation of the accepted resource/identity contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Terminal pruning now retains up to 256 immutable identity records per coordinator, containing only handle ID, optional run ID, and status. send_to_agent consults these after live/retained/unpruned-terminal resolution and before the conversation-scoped DB fallback. A real pruned child receives the existing not-retained/cannot-resume response; DB-only rows no longer imply an earlier session, and expired handle-ID copy suggests the durable run ID. No new transcript retention, continuation authority, schema, or dependency.

Regression tests first reproduced incorrect refusals through the real service for cancelled, superseded, and error children addressed by either ID. The final tests cover bounded oldest-first identity expiry, payload absence, handle/run collision ordering, live precedence, and foreign-conversation isolation. Existing retained-continuation and resolution-order tests pass. Normalized the legacy task section markers for CLI acceptance-criteria tracking while preserving its original description.

Files: tldw_chatbook/Agents/fleet_coordinator.py, tldw_chatbook/Agents/agent_service.py, Tests/Agents/test_fleet_pruned_identity.py.

ADR required: no additional ADR; implemented the pruned-identity amendment in backlog/decisions/129-fleet-mailbox-and-wake-reliability.md.

Verification: 710 targeted tests passed across disjoint groups: 471 gateway/bridge/lifeline tests (469 passed initially; two localhost-server fixtures needed sandbox permission and both passed on the permitted rerun), 195 identity/continuation/coordinator/runtime tests, and 44 teardown/close-session/runtime-lifetime/fanout/stop tests. New test modules pass Ruff lint/format. The four edited production modules add no Ruff diagnostics compared with this pass's starting working tree. Scoped git diff whitespace checks pass. No full suite or live-provider run. Self-review completed.

Docs: backlog/docs/agent-orchestration-review-2026-09-07.md, Docs/superpowers/plans/2026-09-07-fleet-lifecycle-and-identity-repairs.md, and Docs/User_Guide/console/agent-runs-and-tools.md. Existing unrelated working-tree changes and the earlier mailbox/wake repairs were preserved. Changes are uncommitted.
<!-- SECTION:NOTES:END -->
