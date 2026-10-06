---
id: TASK-34563
title: Design Console Send preparation and I/O ownership
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 21:51'
updated_date: '2026-10-06 23:03'
labels:
  - console
  - architecture
  - performance
dependencies: []
documentation:
  - Docs/Development/2026-10-06-console-send-architecture-review.md
  - >-
    Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md
  - backlog/decisions/222-console-send-preparation-and-io-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a simpler and extensible Console Send process that gives immediate terminal feedback and removes repeated preparation I/O while preserving durable acceptance, permissions, recovery and cancellation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reviewed design defines one owner for each lifecycle stage, shared input and storage effect, with explicit data and authority freshness rules.
- [x] #2 The design covers approval, refusal, errors, retry, navigation, concurrent sessions, Stop and shutdown without duplicating existing state authorities.
- [x] #3 Extension contracts and phased migration preserve supported entry points and specify targeted correctness and real input/render/performance evidence.
- [ ] #4 The written specification and architectural decision are reviewed with the user before an implementation plan or product changes begin.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/222-console-send-preparation-and-io-ownership.md.
Reason: received-request custody, shared preparation/result contracts, native outcome ownership and unknown-authority handling cross existing subsystem interfaces. Extends ADR-094/098/126/148/163/197/220.
1. Read current runtime, preparation, request and storage owners and relevant ADRs (completed).
2. Compare practical approaches and review the recommended ownership/data flow with the user (completed).
3. Review shared-result freshness, lifecycle/error behavior, extension contracts and migration/verification sections; incorporate the requested reviews (completed in chat).
4. Consolidate the approved sections into Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md and ADR-222, self-review and validate document links/scope, and obtain written-spec review (written review pending).
Product implementation and implementation-plan execution are outside this design task. No runtime performance or regression acceptance is claimed.
<!-- SECTION:PLAN:END -->
