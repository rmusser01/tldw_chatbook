---
id: TASK-32022
title: Design scoped bidirectional agent messaging
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 16:19'
labels:
  - agents
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shipped messaging API provides supervisor-to-child steering and completion results, but no child-to-parent progress channel or sibling messaging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define concrete use cases for child progress and sibling coordination separately from the existing task store.
- [x] #2 Specify identity, conversation boundaries, permission checks, queue bounds, lifecycle, and visibility in an ADR before implementation.
- [ ] #3 Compare supervisor relay with direct peer messaging and record the selected minimal design.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md (Proposed)
Reason: A child-to-supervisor channel adds a cross-agent service contract and identity/permission boundary.

1. Trace current steering, shared task state, completion delivery, and runtime ownership; distinguish implemented behavior from proposed capabilities.
2. Compare task-store-only coordination, supervisor relay, and direct peers; draft the minimal relay contract, including bounds, consumption semantics, lifecycle, and no new automatic wake authority.
3. Write the proposed ADR and design spec with concrete use cases and a regression acceptance matrix. Reconcile stale implementation-status text in ADR-129/134/135 and the review ledger.
4. Self-review the design against current code and conservative defaults, validate documentation links/statuses, and present the concrete proposal for review before any runtime implementation.

No new runtime functionality is part of this design task. ADR numbering checked against all 454 local branch/remote refs and available worktree files; recheck at merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Drafted Proposed ADR-136 and the scoped messaging spec. The recommended first version adds child-only report_to_supervisor and primary-only read_agent_messages, with explicit relay through existing send_to_agent. It defines service-bound identity, causal-chain eligibility, conservative per-child/conversation/runtime bounds, whole-result sizing, ephemeral lifecycle, and honest queued/collected receipts. It adds no progress-triggered wake, direct peer access, or durable inbox.

ADR required: yes.
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md (Proposed; design review pending).
Spec: Docs/superpowers/specs/2026-09-08-scoped-agent-messaging-design.md.
No runtime implementation plan or runtime changes were made. The design-work plan was followed; the final selection/approval remains pending, so the task stays In Progress and AC3 remains unchecked.

Reconciled stale implementation-status and one-global-wake text in ADR-129/134/135, their index, and the review ledger. Self-review covers disclosure coupled to spawn allowance, post-dispatch truncation, blocked-parent waits, stale callbacks, old-chain consumption, terminal pruning, global retention, and metadata visibility. Local links/source paths and document whitespace/placeholder checks pass.

Targeted current-behavior verification: 216 passed, 6 failed before application execution at macOS multiprocessing SemLock allocation (errno 28). A standalone lock reproduces the failure despite 22.2 GiB disk free; sandbox-escalated rerun also fails all six at that prerequisite. These six remain unverified, not diagnosed as a product defect or proven sandbox-only issue. Output: /tmp/task32022-messaging-baseline.txt and /tmp/task32022-multiprocessing-regressions.txt. No full suite, live-provider run, commits, or staging. Added the evidence-backed verification lesson.
<!-- SECTION:NOTES:END -->
