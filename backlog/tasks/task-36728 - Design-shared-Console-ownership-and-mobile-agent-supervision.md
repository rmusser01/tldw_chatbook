---
id: TASK-36728
title: Design shared Console ownership and mobile agent supervision
status: In Progress
created_date: 2026-10-25 16:01
assignee:
- '@codex'
labels:
- console
- web-server
- mobile
- design
documentation:
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 16:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a mobile interface to existing served Console chats and agents, remove the separate mobile data and execution island, and preserve running work across browser and VPN disconnects. Include direct subagent steering and safe consolidation of existing mobile history; native Console attachment follows in a later implementation phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source-anchored spec defines canonical storage, one live served owner per chat, reconnectable process lifetime, and explicit-target remote commands without introducing another execution engine.
- [x] #2 The spec defines the approved Chats-first phone workflow, live transcript and decisions, main-agent controls, and direct subagent steering and Stop.
- [x] #3 The spec defines restartable legacy-mobile consolidation, identity/reference conflicts, preserved permissions, uncertain-command reconciliation, and targeted automated/live verification.
- [x] #4 A linked ADR records the replacement ownership contract and identifies the superseded parts of ADR-234 and ADR-236.
- [ ] #5 The written spec has passed self-review for scope, contradictions, placeholders and local links, and the user has reviewed it before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Replaces frontend-specific storage and execution ownership, changes served process lifetime and authenticated control contracts, and requires a recoverable data consolidation. Existing ADR-094, ADR-126, ADR-150, ADR-161 and ADR-233 remain governing boundaries.

1. Trace the existing Console runtime, served child lifetime, mobile engine/databases, control broker, decision host, and subagent commands.
2. Write the user-approved ownership, migration, mobile workflow, reconnect and verification contracts in the design spec and replacement ADR.
3. Self-review scope and failure semantics, validate source/local links and task metadata, and commit only this design's artifacts.
4. Present the committed spec for user review; keep product code and implementation planning pending that review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the conversation-approved design spec and ADR-244, linked the ADR index, and added metadata-only supersession/amendment pointers to ADR-234/236. The design replaces frontend-specific storage and engine construction with canonical Console ownership, retained served hosts, exact-target mobile controls and recoverable legacy-history consolidation.

Source review identified two integration requirements: detached served output needs a full terminal repaint on reattachment, and legacy mobile stores need explicit recovery inventory coverage. Self-review also found AgentRunsDB's constructor reconciliation assumes one app per data directory. The revised design requires one retained served runtime per profile, an execution-owner fence honored by native/served entry points, and sweep-free non-owning reads. Native phone attachment is deferred; exclusive native-or-served execution for the same canonical profile is an explicit first-release limit for written review.

The spec defines Chats-first navigation, direct subagent guidance/Stop, native decision ownership, independent phone/desktop selection, command receipt reconciliation, bounded projections, migration interruption/conflict handling, and targeted automated/live qualification. No product implementation plan has been written.

Design self-review and local-link/source-anchor checks are complete. Final task metadata, whitespace and commit-scope verification are performed before the design commit. Written-spec user review remains pending: AC 5 stays unchecked and the task remains In Progress. No product code, user database, provider call, runtime test suite, push or merge is part of this design-writing task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
