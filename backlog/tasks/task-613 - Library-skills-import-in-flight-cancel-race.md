---
id: TASK-613
title: >-
  Library skills import: second submit cancels UI await but in-flight install still lands
status: In Progress
assignee:
  - codex
created_date: '2026-07-24 14:10'
updated_date: '2026-08-28 00:00'
labels:
  - skills
  - library
  - bug
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-library-skill-import-framework-classification.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All Library skill-import paths (loose file, folder, zip, URL) run the service call on a thread via the exclusive worker group; submitting a second import cancels the first path's coroutine, but the threaded service call runs to completion, so the first skill can land trust-pending silently while the UI reports only the second outcome. Pre-existing pattern; the URL path's network fetch widens the window from milliseconds to minutes (PR #831 final-review M4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the Import control is disabled while an import is in flight, or a cancellation check prevents the superseded install from landing after its UI await is cancelled.
- [ ] #2 Behavior is consistent across loose-file, folder, zip, and URL import paths.
- [ ] #3 A test covers the superseded-import scenario.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This closes an existing Library worker-ownership race while preserving ADR-009's local skill trust boundary and the incumbent import, storage, and remote-fetch contracts.

1. Reproduce the second-submit race with mounted, barrier-controlled tests for loose-file, folder, zip, and URL imports.
2. Give the Library screen one authoritative single-flight import state; disable every import control while accepted work runs, repeat the guard in handlers, and preserve truthful state across navigation.
3. Verify focused skill-import, remote-fetch, and directory-import behavior; update the Library skill guide and implementation notes without changing trust or execution policy.

## Implementation Notes

- Added synchronous screen-owned admission before worker scheduling, a non-exclusive accepted worker, and one `finally` release path so loose Markdown, folder, zip, and URL imports share the same single-flight lifecycle.
- Projected `Inspecting/importing…` and disabled import controls through the retained Skills canvas while leaving rail navigation available; accepted in-flight and terminal state survives leave/return, and forced repeat/cancel/browse actions fail closed.
- Added generation fencing for picker callbacks started before admission, generic unexpected-worker recovery, and barrier-controlled mounted coverage that keeps every successful import trust-pending under ADR-009.
- Updated the Library Skills guide and recorded the stale-widget harness incident in the testing-evidence lessons. No ADR was required because storage, trust, fetch policy, and runtime ownership boundaries are unchanged.
- Target verification: 87 passed across `test_skills_import.py`, `test_skills_library_flow.py`, `test_import_skill_directory.py`, and `test_skill_remote_fetch.py`; the known Requests dependency warning remains baseline. Task remains In Progress with acceptance criteria unchecked pending independent review.
- Scoped broader UI evidence: the import/per-click selection passed 12 tests; three unrelated pre-existing assertions remain outside TASK-613 (trust-posture fixture expectation, a bare-fake focus-generation attribute, and create-editor focus timing).
