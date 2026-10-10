---
id: TASK-34563
title: Design Console Send preparation and I/O ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-10-06 21:51'
updated_date: '2026-10-06 23:55'
labels:
  - console
  - architecture
  - performance
dependencies: []
documentation:
  - Docs/Development/2026-10-06-console-send-architecture-review.md
  - >-
    Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md
  - backlog/decisions/225-console-send-preparation-and-io-ownership.md
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
- [x] #4 The written specification and architectural decision are reviewed with the user before an implementation plan or product changes begin.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: received-request custody, shared domain results, native outcome ownership and unknown-authority handling cross subsystem interfaces. Extends ADR-094/098/126/148/163/197/220.
1. Read current runtime, preparation, storage and relevant ADRs (completed).
2. Compare approaches and review the ownership/data flow with the user (completed).
3. Review freshness, lifecycle/errors, extension and migration/verification contracts; incorporate requested reviews (completed).
4. Write/commit the consolidated spec and ADR; complete the user's written-spec review and requested issue audit, correct the concrete gaps, and validate document scope/links (completed).
Product implementation and implementation-plan execution are outside this design task; implementation-plan review remains required before product changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Designed the Console Send preparation and I/O ownership architecture around existing runtime/controller/store authorities. The specification defines lightweight received-intent custody, domain-owned immutable results, live authority checks, cancellation/commit outcome settlement, recovery, extension contracts and phased qualification. ADR-225 records the architectural decision and intentional unknown-authority hardening.
The requested written-spec audit corrected atomic admission/promotion across entry points, approval/native-custody enablement prerequisites, attempt-scoped tool ceilings versus existing restart recovery, revision-guarded UI clearing and attachment-prefix compatibility, upstream error propagation, demand-driven capture and raw performance accounting. No new schema or stronger cross-process transaction is claimed.
Files: Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md; backlog/decisions/225-console-send-preparation-and-io-ownership.md; Docs/Development/2026-10-06-console-send-architecture-review.md.
Validation: eight relative document links, placeholder and whitespace checks, source-based contract review and scoped Git diff checks. This is documentation-only work; runtime/unit/integration and performance acceptance belong to the implementation tasks and are not claimed here. No full suite was run. Existing diagnostic/product candidates remain outside this task.
<!-- SECTION:NOTES:END -->
