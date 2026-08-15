---
id: TASK-16332
title: Deduplicate research service wiring in app.py
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:13'
updated_date: '2026-08-15 05:46'
labels:
  - research
  - tech-debt
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_wire_research_services (app.py:6621) duplicates verbatim the 47-line research wiring block embedded inside _wire_watchlists_and_notifications_services (app.py:6995-7041). The embedded copy runs first at startup and the method call then early-returns via its guard. Two copies of the same wiring is a divergence trap: a future edit to one silently misses the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The duplicated block inside _wire_watchlists_and_notifications_services is replaced with a call to _wire_research_services
- [x] #2 Startup behavior is unchanged: research services are wired exactly once with the same attributes set
- [x] #3 Existing app wiring tests and research service tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the duplicated 47-line research wiring block inside _wire_watchlists_and_notifications_services and the guard in _wire_research_services
2. Write a regression test pinning that the broad parity wiring path produces exactly one set of research service attributes (idempotence and attribute presence)
3. Replace the duplicated block with a call to _wire_research_services and keep the watchlists method ordering comment accurate
4. Run app wiring and research service tests plus lint
ADR required: no - mechanical refactor preserving existing boundaries
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- The 47-line embedded copy in `_wire_watchlists_and_notifications_services` is now `self._wire_research_services()`; the method's existing already-wired guard (`research_scope_service` AND `research_search_scope_service` present) makes the earlier-in-`__init__` bootstrap call equivalent to the old embedded block, and the later direct call at `__init__` line ~5729 still early-returns. One wiring path remains.
- New `Tests/Research/test_app_research_wiring.py` pins the contract that made the swap safe: boot wires the full six-service set, and a second `_wire_research_services()` call reuses the same instances (no torn/duplicate wiring). Both pins passed BEFORE the swap (they pin existing behavior) and stayed green after.
- Verified: Tests/Research/ + Tests/Research_Interop/ + Tests/test_application_state_ownership.py → 107 passed. One failure (`test_runtime_source_state_store_references_are_confined_to_owner_modules`) is pre-existing on the dev baseline — confirmed by stashing the change and re-running (fails identically). Ruff clean on app.py and the new test.
- Files: tldw_chatbook/app.py (47 lines removed, 9 added), Tests/Research/test_app_research_wiring.py (new).
<!-- SECTION:NOTES:END -->
