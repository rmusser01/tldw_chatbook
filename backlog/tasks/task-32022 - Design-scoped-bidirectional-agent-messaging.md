---
id: TASK-32022
title: Design scoped bidirectional agent messaging
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-09 07:30'
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
- [x] #3 Compare supervisor relay with direct peer messaging and record the selected minimal design.
- [x] #4 Review the chosen contract against current runtime replay, cycle detection, storage, and UI behavior; record and resolve concrete design gaps before implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md
Reason: A child-to-supervisor channel adds a cross-agent service contract and identity/permission boundary.

1. Trace current steering, shared task state, completion delivery, and runtime ownership; distinguish implemented behavior from proposed capabilities.
2. Compare task-store-only coordination, supervisor relay, and direct peers; draft the minimal relay contract, including bounds, consumption semantics, lifecycle, and no new automatic wake authority.
3. Write the ADR and design spec with concrete use cases and a regression acceptance matrix. Reconcile stale implementation-status text in ADR-129/134/135 and the review ledger.
4. After the user's 2026-09-09 relay-first approval and request for further review, reproduce design/runtime incompatibilities, obtain an independent focused review, and revise the contract with an issue-by-issue disposition.
5. Validate final documentation links and contract examples, record review evidence and remaining limitations, then proceed to implementation planning under the existing approval.

No new runtime functionality is part of this design task. ADR numbering checked against all 454 local branch/remote refs and available worktree files on 2026-09-08; recheck at merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the scoped messaging design and the user's requested pre-implementation review. The user selected relay-first on 2026-09-09. ADR-136 is Accepted design; runtime implementation remains pending. Child-only report_to_supervisor and primary-only read_agent_messages use existing send_to_agent for explicit relay. No direct peer addressing, progress-triggered wakes, durable inbox, or new automatic allowance is added.

ADR required: yes.
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md.
Spec: Docs/superpowers/specs/2026-09-08-scoped-agent-messaging-design.md.
Review: Docs/superpowers/reviews/2026-09-09-scoped-agent-messaging-review.md.

Seven design findings were resolved: serialized input expansion could block FIFO; productive parameterless reads collide with loop detection; stale/oversized queues lacked user recovery; private continuation copies were under-disclosed; generic AgentSteps would leak bodies into automatic metadata; restored pending reads could acquire replacement inbox authority; collection guidance was insufficient for pull-only coordination. Corrections include exact envelope sizing, productive-read cycle handling, explicit user discard, truthful ADR-063 storage/replay semantics, body-free step projections, refusal of restored pending messaging calls, and capability-gated collection guidance. Queue lock/disposal ordering and primary-versus-child checkpoint scope are explicit. Independent review confirmed the continuation/authority/projection corrections and found no remaining blocker in those reviewed sections.

Validation: exact JSON probe produced 12,014 serialized characters for a draft-valid 2,000-character body; current cycle detector reports (1, 3) for three identical reader calls. Existing runtime/continuation tests passed 106 tests (one existing requests dependency warning); /tmp/task32022-design-runtime-evidence.txt. Local links, source references, whitespace, and placeholder checks pass. No runtime logic changed, so new messaging behavior is specified by the expanded future acceptance matrix, not claimed tested.

Earlier steering/mailbox/task-store baseline remains 216 passed and 6 environment prerequisite failures at macOS multiprocessing SemLock allocation. Standalone and escalated probes reproduced errno 28 despite free disk space; no store defect or sandbox-only cause was established. Those cases remain unverified and were not rerun in this documentation review.

The design-work plan is complete with no runtime implementation deviation. Review ledger and ADR index/status are updated; earlier stale ADR-129/134/135 delivery wording was corrected. The existing verification lesson records the semaphore incident. No commits, staging, provider network calls, or full suite. Next step is implementation planning under the user's existing approval.
<!-- SECTION:NOTES:END -->
