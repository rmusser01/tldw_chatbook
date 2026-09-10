---
id: TASK-32195
title: Design local chat schedules and agent self timers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-10 01:56'
updated_date: '2026-09-10 02:00'
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
- [ ] #1 The design specifies the approved composer flow, same-chat scheduled responses, enabled tools, local execution, and agent-created one-time and recurring timers.
- [ ] #2 A canonical ADR records storage and runtime ownership, durable scheduling authority, permission enforcement, accounting, and restart behavior.
- [ ] #3 The reviewed specification defines testable behavior for occurrence deduplication, busy chats, missed runs, cancellation, tool retries, and agent scheduling limits, with scope assumptions explicit.
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
<!-- SECTION:PLAN:END -->

## Design Progress

- Written specification: [Local chat schedules and agent timers](../../Docs/superpowers/specs/2026-09-09-chat-schedules-and-agent-timers-design.md).
- Proposed decision: [ADR-143](../decisions/143-local-chat-schedules-and-agent-timers.md).
- The user approved the local same-chat composer flow with existing tools and added agent-created timers. The written draft assumes main native agents; spawned-agent resumption was raised separately and has not been selected.
- Self-review made future-work authority explicit: automatic timer descendants share a finite grant; fixed human cadence requires explicit follow-up authorization; cancellation of the root covers its descendants. It also specifies durable logical tool-call identity, viewless startup, exact Save payloads, and separate approval-timeout and uncertain-effect recovery.
- Documentation validation covered the three artifacts and their local links, YAML/IDs/status, Markdown fences and whitespace. The original goal-plan line bytes were preserved. No application code or runtime tests changed.
- Written design review remains outstanding. Keep this task In Progress and ADR-143 Proposed until that review; implementation planning follows the approved written design.
