---
id: TASK-32195
title: Design local chat schedules and agent self timers
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 01:56'
updated_date: '2026-10-03 03:34'
labels:
  - console
  - scheduling
  - agents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define one local-first scheduling capability for repeated responses in an existing Console chat and for agent-created timers, with a concrete design ready for implementation review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design specifies the approved composer flow, same-chat scheduled responses, enabled tools, local execution, and agent-created one-time and recurring timers.
- [x] #2 A canonical ADR records storage and runtime ownership, durable scheduling authority, permission enforcement, accounting, and restart behavior.
- [x] #3 The reviewed specification defines testable behavior for occurrence deduplication, busy chats, missed runs, cancellation, tool retries, and agent scheduling limits, with scope assumptions explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/143-local-chat-schedules-and-agent-timers.md
Reason: This adds durable scheduling authority, an automatic submission origin, agent tool contracts, and a cross-module runtime boundary; ADR-018/019 and ADR-131/134/135/141 remain applicable.

1. Record the approved local, same-chat composer flow and inspect Scheduling, Console runtime, tool registry, permissions and automatic accounting.
2. Compare storage and execution approaches, specify the agent timer contract, and write a proposed canonical ADR plus a concrete design under Docs/superpowers/specs/.
3. Self-review authority, deduplication, concurrency, recovery and verification requirements; explicitly identify any scope assumption still awaiting user input.
4. Commit the design artifacts for review. Treat this as a design task; create the implementation plan and execution tasks after the written design review. No runtime implementation or test-suite run is part of this task.
5. Under the user's authorization to correct the review, revise the spec/ADR, recheck authority and runtime transitions, and prepare the concrete implementation plan with integration prerequisites and targeted qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the scheduling design and review corrections under accepted ADR-143.
The revised spec binds reviewed root tasks to grants, defines one-time and
completed-root controls, skips slots covered by active work, re-arms the queue,
stores receipts for every mutation and bounds retained state without losing
uncertain accounting or cancellation capacity. Strict automatic ownership,
atomic maintenance exclusion and per-conversation locks preserve unrelated
manual chats across app processes. Current scratch, live policy, hook, history
projection and design-token requirements are explicit.

Updated the spec, ADR, review resolution map and this task; added the six-slice
implementation plan linked above. Two independent revision reviews checked the
eight findings, with their follow-up lifecycle/control clarifications included.
Documentation verification checked local links, YAML/status, whitespace/fences,
Python example syntax, all 22 qualification cases and preservation of the original
user-authored goal-plan bytes. No runtime implementation or application test run
is claimed by this design-only task. The later task-ID claimant and current dev
prerequisites must be reconciled during isolated implementation integration.
<!-- SECTION:NOTES:END -->

## Design Progress

- Written specification: [Local chat schedules and agent timers](../../Docs/superpowers/specs/2026-09-09-chat-schedules-and-agent-timers-design.md).
- Accepted design decision: [ADR-143](../decisions/143-local-chat-schedules-and-agent-timers.md).
- The user approved the local same-chat composer flow with existing tools and added agent-created timers. The written draft assumes main native agents; spawned-agent resumption was raised separately and has not been selected.
- Self-review made future-work authority explicit: automatic timer descendants share a finite grant; fixed human cadence requires explicit follow-up authorization; cancellation of the root covers its descendants. It also specifies durable logical tool-call identity, viewless startup, exact Save payloads, and separate approval-timeout and uncertain-effect recovery.
- Documentation validation covered the three artifacts and their local links, YAML/IDs/status, Markdown fences and whitespace. The original goal-plan line bytes were preserved. No application code or runtime tests changed.
- Written design review: [2026-10-02 scheduling review](../../Docs/superpowers/reviews/2026-10-02-chat-schedules-and-agent-timers-review.md). Independent authority and runtime reviews found eight contracts to revise before implementation planning: live-owner exclusion, reviewed-task authority, queue re-arming, one-time transitions, due slots during execution, mutation replay, retained-state bounds and scheduled history projection. Current scratch/profile/hook/design-token integration requirements and timeout qualification are recorded there too.
- The review also records a TASK-32195 collision with a later web-search task now on dev. Add-commit provenance makes this design the earlier claimant; reconcile the later record and its references before integration rather than silently renumbering this design.
- The user authorized correcting the review. All eight design contracts are revised and independently checked; additional completed-root, full-quota control, active retime and maintenance/recovery cases are explicit. The review's resolution map links each correction to qualification requirements.
- Implementation plan: [2026-10-02 chat schedules and agent timers](../../Docs/superpowers/plans/2026-10-02-chat-schedules-and-agent-timers.md), with six independently reviewable slices and all 22 spec cases mapped. It starts by integrating current dev and reconciling the later task-ID claimant. Runtime implementation and qualification remain execution work, not results of this design task.
