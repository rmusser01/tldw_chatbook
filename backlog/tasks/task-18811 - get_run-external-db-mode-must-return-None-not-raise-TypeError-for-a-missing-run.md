---
id: TASK-18811
title: >-
  get_run external-db mode must return None, not raise TypeError, for a missing
  run
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-19 14:47'
labels:
  - research
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LocalResearchService.get_run promises dict-or-None, but its external-db branch only catches KeyError: an injected db that returns None for a missing run makes _as_local_run(None) raise dict(None) TypeError out of a lookup API. The new external-mode lease path (claim_run -> get_run) inherits this, so a missing run can surface as TypeError instead of the service's not-found contract. Found during PR #1822's external review round and adjudicated then as 'found, not fixed: worth its own task'. The path-backed branch is correct; only the external-db branch needs the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 get_run in external-db mode returns None when the injected db returns None for a missing run, matching the path-backed branch,The external-mode claim_run path resolves a missing run to its documented not-found error rather than a TypeError,Regression test covers an external db double whose get_run returns None
<!-- AC:END -->

## Implementation Plan

1. Verify the defect on current dev: LocalResearchService.get_run's external-db branch catches only KeyError; the external double (and the server db) return None for missing runs.
2. RED: two tests against FakeExternalResearchDB -- get_run("missing") must be None; claim_run("missing") must raise the documented ValueError("research run not found"), not TypeError.
3. Guard the branch: None from the db returns None before _as_local_run is called. The claim path then resolves through its existing ValueError.

ADR required: no
ADR path: N/A
Reason: Single-branch contract fix inside one method; no boundary or schema change.

## Implementation Notes

``get_run``'s external-db branch now returns None when the injected db returns None, matching the path-backed branch's dict-or-None contract, instead of passing it to ``_as_local_run`` and raising ``dict(None)`` TypeError. The KeyError catch (for dbs that signal not-found by raising) is preserved. ``claim_run``'s external path needed no separate change: it calls ``get_run`` and already raised ``ValueError("research run not found")`` on a None result -- the TypeError was pre-empting it.

TDD evidence: both new tests in Tests/Research/test_local_research_service.py failed on unmodified dev with TypeError; after the guard they pass. The existing FakeExternalResearchDB double already models the None-return contract, so no new stub was needed. Full Tests/Research/: 254 passed. Ruff: service file zero delta (5 pre-existing fixables both sides); test file formatted.

Modified: tldw_chatbook/Research_Interop/local_research_service.py, Tests/Research/test_local_research_service.py.
