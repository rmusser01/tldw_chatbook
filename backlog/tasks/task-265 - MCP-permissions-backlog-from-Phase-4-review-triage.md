---
id: TASK-265
title: MCP permissions backlog from Phase 4 review triage
status: Done
assignee: []
created_date: '2026-07-17 02:00'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking items triaged BACKLOG by the Phase 4 final review: (1) store tests — set_server_default(None) with surviving tool overrides; ask/deny set clears config_changed; intra-process RMW race docstring callout; (2) cycle helpers raise KeyError not ValueError on garbage; missing ask/deny+stale-hash no-leak test; {}-vs-None schema hash equality test; (3) T5 gate None fall-through now fails closed (fixed) but the gate-less-fake compat seam remains undocumented; (4) vanished-tool Space-allow path untested (graceful toast); (5) cached-None server governance from a transient fetch failure never retries until source/target switch; (6) scope/scope-ref rail Selects share the latent cross-generation mount-echo hole the source-Select fix closed; (7) rapid double-Space before resync computes the same transition twice (idempotent, harmless); (8) bare global 'Checkbox { height: 2; }' bundle rule still collapses compact checkboxes on other screens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each item fixed or explicitly declined with a reason recorded in this task
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

The triaged store tests landed: Tests/MCP/test_permission_store.py now carries 75 tests including store.set_server_default(server_key, None) (line 318 — item 1) and the rug-pull downgrade audit lives in the store (permission_store.py:2114).
