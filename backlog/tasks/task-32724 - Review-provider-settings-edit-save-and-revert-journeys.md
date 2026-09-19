---
id: TASK-32724
title: Review provider settings edit save and revert journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 16:29'
updated_date: '2026-09-17 16:32'
labels:
  - settings
  - providers
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through Providers and Models so keyboard editing, Save, Revert and navigation retain honest draft and saved state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider model and endpoint edits remain visibly keyboard accessible in dark/light at wide and compact sizes, with truthful staged and saved feedback.
- [x] #2 Revert cancellation preserves the draft, confirmed Revert restores saved values, and Save persists only the intended values with coherent state after navigation.
- [x] #3 Targeted tests and private native journeys establish no unintended provider requests; static checks, review, guide and audit evidence qualify any repairs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/012-provider-credential-settings-boundary.md, backlog/decisions/033-application-session-state-ownership.md, ADR-031 key conventions and ADR-150/161 design language. Reason: bounded review and routine repairs of the existing staged Settings form; revisit if a new boundary is needed.
1. Read existing provider edit/save/revert contracts and prior issues including TASK-15740, TASK-1560 and TASK-214. Add production-CSS keyboard probes for editing, Revert cancel/confirm, Save and return.
2. Diagnose and repair only confirmed defects, preserving credential masking and existing persistence ownership. Run targeted provider/form/navigation checks.
3. Verify with private native real-config journeys in dark/light at 170x48 and 80x24, inspect captures and lifecycle, run static checks and independent review, update guide/audit/task and commit locally. No full suite, push or merge.
Allocation: all reachable task paths across 308 refs and 27 worktrees had maximum 32720.
Closeout found a simultaneous uncommitted TASK-32721 in the main checkout (same created minute). Reallocated this workstream only; the other task is untouched. The CLI first offered 32722, also held by that batch. A fresh all-ref/worktree sweep found maximum 32723; this task moves to 32724. Native scratch paths retain the original ID as run provenance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified existing Providers and Models behavior with four new production-CSS keyboard journeys, real private config Save/Revert/navigation, eight inspected captures and exact persistence/deletion assertions. No production code or visual change was needed. Targeted selection: 228 passes; the four strengthened journey cases were then rechecked and passed (overlapping coverage). Independent-review reachability and persistence evidence gaps were resolved. Static checks, normal exit, eleven healthy private databases, zero chats/messages and unchanged default-profile fingerprints passed. Guide, audit and QA evidence: Docs/superpowers/qa/2026-09-17-provider-settings-journeys/README.md. ADR required: no; existing ADR-012, ADR-033, ADR-031, ADR-150 and ADR-161 apply. Tests qualify same-process return and actual file writes, not process restart or live provider connectivity. Concurrent task allocation was resolved by moving only this uncommitted review from 32721 through the CLI offer 32722 to verified free 32724; scratch path numbers remain provenance. No full suite, push or merge. No new general lesson beyond existing keyboard-evidence and task-allocation guidance.
<!-- SECTION:NOTES:END -->
