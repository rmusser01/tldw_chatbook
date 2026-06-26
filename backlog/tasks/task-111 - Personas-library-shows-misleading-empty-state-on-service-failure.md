---
id: TASK-111
title: Personas library shows misleading empty state on service failure
status: Done
assignee: []
created_date: '2026-06-11 03:01'
labels:
  - personas
  - recovery
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When refresh_persona_list fails or the backend lacks persona support, the workbench library renders the actionable 'No persona profiles yet' empty state instead of a recovery callout (DestinationRecoveryState). Surface service failures distinctly per the destination recovery pattern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Service failure renders recovery copy distinct from the true empty state,True empty state copy unchanged
<!-- AC:END -->

## Implementation Plan

1. Trace the Personas profile list refresh path and identify where failures are collapsed into the true-empty state.
2. Add a mounted regression that forces `refresh_persona_list()` to fail in Personas mode and asserts the library pane renders recovery copy instead of the true-empty copy.
3. Add a companion assertion that a successful empty result keeps the existing true-empty state unchanged.
4. Implement the smallest recovery-state path using the existing destination recovery vocabulary and stable selector.
5. Run focused Personas workbench tests plus diff hygiene, then update acceptance criteria and notes.

ADR required: no
ADR path: N/A
Reason: bounded UI recovery-state bugfix; no storage/schema, sync policy, service contract, or data ownership boundary changes.

## Implementation Notes

- Added strict persona profile list refresh handling for Personas workbench callers so missing or failing persona scope services produce a `DestinationRecoveryState` instead of being collapsed into an empty list.
- Updated the Personas library pane to render a dedicated recovery row with a stable `#personas-service-error` selector; recovery replaces stale rows and the true empty copy remains unchanged for successful empty lists.
- Added mounted regressions for service failure recovery, stale-row replacement, and true-empty state copy.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py Tests/UI/test_ccp_handlers.py::TestCCPPersonaHandler::test_refresh_persona_list_routes_through_scope_service --tb=short` passed with 130 tests; `git diff --check` passed.
- ADR check completed: no ADR required for this bounded UI recovery-state bugfix.
