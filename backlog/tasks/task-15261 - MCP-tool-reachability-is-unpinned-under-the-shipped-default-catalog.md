---
id: TASK-15261
title: Replace the fixed active-tool cap with token-budgeted discovery
status: In Progress
assignee: []
created_date: '2026-08-11'
updated_date: '2026-08-30 17:54'
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
- [ ] #1 Catalog disclosure is selected by estimated schema-token cost against 10% of the selected model context, not by a fixed catalog count
- [ ] #2 `find_tools` searches the complete allowed catalog with deterministic relevance ordering and returns at most eight results by default
- [ ] #3 `load_tools` atomically replaces the catalog working set with valid requested schemas that fit the token allowance; later loads cannot fail because earlier tools permanently consumed room
- [ ] #4 Permission checks mirror the currently disclosed working set, so a replaced tool is not callable until loaded again and a newly loaded tool is callable immediately
- [ ] #5 A production-shaped catalog test registers MCP last and proves find → load → ask/approve → execute reaches the MCP tool without provider-order truncation
- [ ] #6 Small catalogs whose complete schemas fit the token allowance remain directly disclosed without extra discovery round trips; estimator/model-limit failures fail safely into discovery
- [ ] #7 The obsolete `RunBudget.max_active_tools` and fixed direct-disclosure count are removed from live configuration, runtime code, tests, and normative documentation
<!-- AC:END -->
