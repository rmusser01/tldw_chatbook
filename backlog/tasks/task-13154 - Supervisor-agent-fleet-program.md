---
id: TASK-13154
title: Supervisor agent fleet program
status: In Progress
assignee: []
created_date: '2026-08-09 13:57'
updated_date: '2026-09-12 16:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Named sub-agent definitions, background/parallel execution, steering, Console fleet panel. Spec: Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All approved fleet outcomes are delivered and verified: definitions, concurrency runtime, fleet panel, cross-turn lifetime, wake and notification, steering and continuation, and phase-four polish; every remaining TASK-13154 child is Done.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md plus current accounting/admission and implementation decisions. Reason: records-only reconciliation after all implementation children complete. Execute Docs/superpowers/plans/2026-09-12-agent-fleet-program-closeout.md last: verify six historical core deliveries and the seventh polish slice, checked child criteria/reviews, prior and fresh evidence with limits, current authority/remaining optional extensions, independent records and final branch review, then Backlog Done. Do not invent retroactive children or erase unfinished outcomes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Historical PR-1 review triage is preserved below; it is not a claim that the remaining work is complete. Current child tasks and the closeout plan reconcile each item.

### Deferred to later phases (final-review triage, 2026-08-09)

The final whole-branch review of PR-1 (named agent definitions) accepted
these as real but out of scope for the fix wave that landed the review's
findings. Recorded here because the SDD ledger that carried the full triage
(`.superpowers/sdd/2026-08-08-supervisor-fleet-pr1-agent-definitions/`) is
gitignored and does not survive merge.

- **PR-2a** — convert the spawn-closure disjoint-path `assert` to a `raise`
  (an `assert` is stripped under `python -O`, silently turning a real
  invariant violation into undefined behavior); add a load-once-per-turn
  call-count guard where `run_turn` changes.
- **PR-2b** — memoize/close the Settings ▸ Agents panel's `AgentRunsDB`
  handle. It is currently opened fresh on every category visit and relies
  on garbage collection to close the underlying connection rather than an
  explicit lifecycle.
- **Phase-4 polish** — give per-save feedback when `RUNTIME_TOOL_NAMES`
  entries are silently dropped from a typed tool list, so a user who lists
  `spawn_subagent` sees why it didn't stick.
- **Owner taste call** — where the Agents category belongs in Settings
  navigation (Troubleshooting vs. Expert) is a placement judgment call, not
  a defect; left for the owner to decide.
<!-- SECTION:NOTES:END -->
