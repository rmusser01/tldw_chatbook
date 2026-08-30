---
id: TASK-15261
title: Replace the fixed active-tool cap with token-budgeted discovery
status: Done
assignee: []
created_date: '2026-08-11'
updated_date: '2026-08-30 19:35'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent runtime currently uses two arbitrary counts to control tool
disclosure: catalogs above 16 tools switch to discovery, and a run may activate
at most 24 catalog tools. Activation only grows, so a successful run can
permanently exhaust its tool room and make a later, valid tool unreachable.
Provider registration order can therefore affect capability reachability even
though the full catalog is registered and permitted.

Replace those count gates with context-aware progressive disclosure. Catalog
schemas that fit inside a bounded fraction of the selected model's context are
disclosed directly. Larger catalogs remain fully searchable; search returns a
small ranked result set, and each load atomically selects a token-bounded
working set instead of growing a lifetime active set. A later load can replace
that set, so discovery never ends in a permanent `no room` state.

Keep the original production-shaped MCP concern as regression evidence: with
the shipped-size catalog and MCP registered last, a model must still find,
load, approve, and execute the MCP tool through the real permission path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Complete-catalog auto-disclosure requires the estimated token cost of the exact provider-visible schema set to be at or below 10% of the selected model context and a projected first request that fits after response reserve, never a fixed catalog count
- [x] #2 `find_tools` searches the complete allowed catalog with deterministic relevance ordering and returns at most eight results by default
- [x] #3 `load_tools` atomically replaces the catalog working set with valid requested schemas whose projected next request fits; a schema larger than the 10% auto-disclosure threshold remains loadable when the whole request fits, and earlier tools never consume permanent room
- [x] #4 Permission checks mirror the currently disclosed working set, so a replaced tool is not callable until loaded again and a newly loaded tool is callable immediately
- [x] #5 A production-shaped catalog test registers MCP last and proves find → load → ask/approve → execute reaches the MCP tool without provider-order truncation
- [x] #6 Small catalogs whose complete schemas and projected first request fit remain directly disclosed without extra discovery round trips; history pressure, estimator failures, and invalid model limits fail safely into discovery
- [x] #7 The obsolete `RunBudget.max_active_tools` and fixed direct-disclosure count are removed from live configuration, runtime code, tests, and normative documentation
- [x] #8 Invalid, budget-omitted, accepted, and mixed-batch load outcomes are deterministic; failed or non-exclusive loads preserve the previous working set
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add token-policy and structured load-selection primitives, then replace
   catalog substring ordering with deterministic allow-list-aware ranking.
2. Build provider-visible schema-set measurement and a shared first-request
   planner that requires both the 10% threshold and whole-request fit.
3. Replace append-only runtime loading with exclusive, structured working-set
   replacement and a lockstep permission-name commit.
4. Wire exact deferred-load request-fit planning through the service and
   Console preflight/live paths, including oversized-singleton reachability.
5. Rewrite obsolete count-pinned tests/comments, add the production-shaped MCP
   regression, run focused verification, and complete task/ADR documentation.

ADR required: yes
ADR path: `backlog/decisions/104-token-budgeted-agent-tool-disclosure.md`
Reason: This changes the provider request schema contract and the permission
boundary for dynamically loaded tools.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced fixed catalog/active-tool counts with provider-visible schema-token
  measurement, a 10% automatic-disclosure threshold, and exact first-request
  fit checks that include runtime schemas, response reserve, workspace context,
  named-agent roster, and fleet gates.
- Added deterministic eight-result catalog search plus exclusive `load_tools`
  batches whose accepted schemas atomically replace the model-visible working
  set and its permission-name set. Deferred loads use projected next-request
  headroom and preserve the prior set on invalid, omitted, mixed, or failed
  selections.
- Added count-independent direct/discovery, oversized-singleton, history
  pressure, replacement, permission-lockstep, continuation, child/fleet, and
  production-shaped MCP-last find → load → approve → execute regressions.
- Kept ADR-104 Accepted. No schema migration, setting, dependency, or persisted
  working-set state was added.
- Focused verification: 872 targeted agent/Console tests passed; the broader
  Console compatibility shard passed 413 tests with one independently
  reproduced `origin/dev` baseline regenerate test deselected. `py_compile`
  and `git diff --check` passed. Targeted mypy improved from the `origin/dev`
  baseline of 56 findings to 49; the remaining findings are pre-existing in
  the touched legacy modules.
<!-- SECTION:NOTES:END -->
