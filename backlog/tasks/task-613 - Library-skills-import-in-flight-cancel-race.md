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
2. Give the app one authoritative single-flight import coordinator and snapshot; disable every import control while accepted work runs, repeat the guard in handlers, and preserve truthful state across routed screen replacement and rail navigation.
3. Verify focused skill-import, remote-fetch, and directory-import behavior; update the Library skill guide and implementation notes without changing trust or execution policy.

## Implementation Notes

- Review round 1 moved admission, the four-route mutation pipeline, live progress, and terminal receipt into one app-owned coordinator. Fresh routed Library screens hydrate the same immutable snapshot; the worker retains no stale screen/widget reference and publishes only by transiently looking up the current screen.
- `Inspecting/importing…` and disabled controls continue through the retained Skills canvas. Rail departure preserves terminal status/review, while explicit Cancel, Review, or a new draft dismisses it. Forced repeat/cancel/browse/path/review actions fail closed during accepted work.
- Cancel/reset/reopen/route changes advance the row generation; picker callbacks require the current open Skills row and live screen. `Input.Changed` also requires the currently mounted Input so a detached mount echo cannot erase a fast terminal receipt.
- Removed the parallel screen-owned import implementation; loose Markdown, folder, zip, and URL now share exactly one coordinator path. All success routes still pass `trust_approved=False`, preserving ADR-009, remote-fetch, storage, and runtime boundaries.
- Added routed replacement, completion-before-departure, completion-while-away, picker reset/reopen, cancellation settlement, and sole-owner tests. Updated the guide and testing-evidence lesson. No ADR was required because this directly implements the existing worker-ownership plan without changing a durable or security contract.
- Review-fix verification: complete `test_skills_import.py` passed 24 tests; focused Skills import/reset UI passed 9; four-route mounted matrix passed 4; replacement route passed 1; Ruff, diff check, and the Impeccable detector passed. The incumbent Requests warning and the task file's aggregate FD-growth warning remain. Task stays In Progress with acceptance criteria unchecked pending fresh independent review.
