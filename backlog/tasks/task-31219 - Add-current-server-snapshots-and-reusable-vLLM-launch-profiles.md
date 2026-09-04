---
id: TASK-31219
title: Add current-server snapshots and reusable vLLM launch profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:34'
updated_date: '2026-09-04 04:20'
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
1. Add RED-first repository contract tests for exact V1 schema, strict validation, collision handling, CAS, future-version preservation, atomic-write failure, lifecycle operations, and selection restoration.
2. Implement ADR-115's device-local vLLM profile repository at get_user_data_dir()/vllm_launch_profiles.json using the shared atomic JSON writer; no database/schema migration.
3. Add RED-first immutable current-server versus editable next-restart UI contracts, profile action messages, and selection behavior with repository I/O isolated in a thread worker.
4. Add RED-first two-generation restart sequencing that requires current successful preflight, confirms only allowlisted changed labels, proves the old exact process dead and claim released before reserving/launching the new generation, and fails closed on termination failure.
5. Run focused tests plus exact Ruff, focused mypy, py_compile, and git diff --check; complete ACs, ADR-115-linked notes, status, report, and commit.

ADR required: no
ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-115 already accepts device-local versioned JSON profile ownership and current-versus-next restart semantics; this task directly implements that decision without database/schema migration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-115’s device-local vLLM profile and restart boundaries. Added exact strict V1 JSON profiles at get_user_data_dir()/vllm_launch_profiles.json with a 32-profile cap, canonical Unicode/casefold name uniqueness, deterministic duplicate suffixes, restrictive atomic writes, and cross-process portalocker CAS; future/corrupt documents and write failures remain byte-preserving and fail closed. Added selected-profile restoration through thread-worker I/O, launch-only raw-argument isolation, exact claim-bound immutable Current server snapshots, safe Next restart dirty labels, and two-generation restart sequencing that proves the old process dead and claim released before a new reservation. Termination failure retains the old snapshot and creates no second process. No DB/schema migration. Focused repository/snapshot/restart tests, the complete vLLM workflow file, scoped Ruff, focused mypy, py_compile, and diff checks pass. ADR: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md. Files: vllm_profiles.py; vllm_setup.py; vllm_connection.py; vllm_setup_view.py; llm_screen.py; focused profile/connection/UI tests.
<!-- SECTION:NOTES:END -->
