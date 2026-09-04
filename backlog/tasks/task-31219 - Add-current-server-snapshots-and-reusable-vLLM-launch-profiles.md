---
id: TASK-31219
title: Add current-server snapshots and reusable vLLM launch profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:34'
updated_date: '2026-09-04 04:42'
labels:
  - vllm
  - lab
  - profiles
dependencies:
  - TASK-31215
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make repeated vLLM operation efficient and honest by separating the immutable running configuration from editable restart intent and retaining reusable non-secret launch profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI displays an immutable current-server snapshot separately from the next-launch draft.
- [x] #2 Edits made while running are labeled as next-restart changes and can be applied with one Restart with draft action.
- [x] #3 Users can create, select, rename, duplicate, and delete named vLLM profiles containing only approved non-secret launch fields.
- [x] #4 The last selected vLLM view and profile restore across screen recomposition and application restart.
- [x] #5 Storage, migration if required, privacy, and profile lifecycle tests cover invalid, stale, and recovery states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict launch-profile schema validation and a versioned device-local repository with optimistic revisions, safe canonical names, a 32-profile cap, selected-profile restoration, and atomic writes.
2. Extend vLLM launch ownership with immutable current-server snapshots and restart sequencing that preserves generation fencing, claim ownership, and exact-process termination guarantees.
3. Add the profile/current-server/restart controls to the vLLM Lab, keeping editable next-restart configuration distinct from immutable current state and using thread workers for repository I/O.
4. Cover schema, migration/future-version behavior, atomicity, optimistic conflicts, name collisions, profile UX, snapshot privacy, and restart safety with focused tests.
5. Run focused verification, update the ADR-linked task notes and acceptance criteria, and record exact evidence in the implementation report.

Fix Round 1:
6. Add regression tests and strict source-specific model validation at construction, decode, and pre-write boundaries without disclosing rejected values.
7. Harden document and adjacent lock handling against symlinks using descriptor-based no-follow checks, and add a deterministic two-process same-revision race test.
8. Amend accepted ADR-115 to the approved Task 4 exact V1 schema and rerun all scoped verification gates.

ADR required: yes
ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-115 already governs durable launch-profile ownership, optimistic concurrency, privacy, and restart sequencing; this fix amends its detailed V1 schema to match the accepted implementation plan rather than creating a duplicate decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-115’s device-local vLLM profile and restart boundaries. Added exact strict V1 JSON profiles at get_user_data_dir()/vllm_launch_profiles.json with a 32-profile cap, canonical Unicode/casefold name uniqueness, deterministic duplicate suffixes, restrictive atomic writes, and cross-process portalocker CAS; future/corrupt documents and write failures remain byte-preserving and fail closed. Added selected-profile restoration through thread-worker I/O, launch-only raw-argument isolation, exact claim-bound immutable Current server snapshots, safe Next restart dirty labels, and two-generation restart sequencing that proves the old process dead and claim released before a new reservation. Termination failure retains the old snapshot and creates no second process. No DB/schema migration.

Fix Round 1 tightened the same ADR-115 boundary: model values now receive source-specific non-secret validation during construction, decode, and immediately before write; nonexistent safe absolute local paths remain repairable while repository-ID, option, traversal, control, and credential-URL violations fail generically. Profile document and adjacent lock leaves now use no-follow descriptor opens, regular/current-owner/private-mode and inode-identity checks, with no chmod of existing pathnames. A simultaneous two-process revision-0 barrier proves one successful CAS and one conflict. ADR-115 remains Accepted and is amended to the approved exact V1 key names, 120-code-point names, document selected_profile_id/revision, and intentional omission of updated_at. Focused profile/setup/connection/UI tests, the complete vLLM workflow, full Ruff on new files, scoped Ruff on all touched Python files, focused mypy, formatting of new files, py_compile, and diff checks pass. ADR: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md. Evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.
<!-- SECTION:NOTES:END -->
