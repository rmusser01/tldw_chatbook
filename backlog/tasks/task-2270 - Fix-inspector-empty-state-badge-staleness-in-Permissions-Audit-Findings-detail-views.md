---
id: TASK-2270
title: >-
  Fix inspector empty-state badge staleness in Permissions/Audit/Findings detail
  views
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 21:30'
updated_date: '2026-08-07 01:27'
labels:
  - mcp
  - ui
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-5 (RAG-50, task 6 of `Docs/superpowers/plans/2026-08-04-rag-ux-v2-pr5-mcp-guardrails.md`) fixed the MCP inspector's stale empty-state badge for Tools mode only: `show_tool()` now hides `#mcp-inspector-state` while tool detail displays and restores it on `show_tool(None)`. The task-6 review CONFIRMED by direct code read (not merely "plausible") that the other three detail views carry the identical live defect: `show_permission()`/`_render_permission_container()`, `show_audit_entry()`, and `show_finding()` each toggle only their own container's display and never touch the badge, and their workbench call sites (`on_mcp_permissions_mode_row_selected`, `on_mcp_audit_mode_entry_selected`, `on_mcp_audit_mode_finding_selected`) never call `update_readiness()`. So selecting a Permissions-matrix row, Audit entry, or Finding with no server selected leaves "Pick a server, tool, or entry to see what's wrong and what you can do." sitting above fully populated detail.

Until fixed there is also a new inconsistency: only Tools mode hides the badge (including real readiness content — a deliberate PR-5 ruling: readiness belongs to server-selection context). The goal state is that all FOUR detail views behave consistently: badge hidden while any detail is displayed, restored on that view's clear path.

Rider from the PR-5 final review (same seam, honest-copy fix): when the test-gate resolver fails closed it synthesizes a gate with `origin == "gate_error"`, and the decision note then reads "This tool is set to Off." — but the tool is not actually set to Off; the resolver failed. Use `_UNKNOWN_ORIGIN_SENTENCE` ("Permission state could not be resolved.") for that origin. The existing test `test_decision_note_unknown_origin_degrades_to_bare_sentence` currently pins the less-honest string and must be updated as a deliberate contract change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting a permission row, audit entry, or finding with no server selected does not leave the empty-state badge above the populated detail.
- [x] #2 All four detail views (Tools/Permissions/Audit/Findings) behave consistently: badge hidden while detail is displayed, restored on that view's clear path with the empty-state copy.
- [x] #3 `update_readiness()` cannot resurrect the badge over displayed detail in any mode.
- [x] #4 A `gate_error`-origin gate produces the honest "Permission state could not be resolved." note, never "This tool is set to Off." *(the rider — see Implementation Notes)*
- [x] #5 Existing badge-content and Tools-mode pins keep passing; new behavior covered by additive tests per view.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Single badge owner: _sync_state_badge_display() — badge displays only when NO detail view shows (tool/permission/audit/finding); every view's show/clear path funnels through it. 2. RED per view: permission row, audit entry, finding each currently leave the badge above populated detail. 3. AC3: update_readiness stays content-only; additive tests prove a readiness sync cannot resurrect the badge over any displayed detail. 4. Rider (gate_error decision note): verify satisfied-by-PR-1385 — the gate_error branch of _decision_note was proven dead and removed; the run path renders the derived unresolved copy — record evidence, no new code. 5. Mutation per new guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**2026-08-06 (PR-T3 Task 3, commit `b1c103ff3`):** The rider (the fourth AC item
above) shipped. `_decision_note()`'s synthetic `gate_error` origin now degrades to
the honest `_UNKNOWN_ORIGIN_SENTENCE` ("Permission state could not be resolved.")
instead of falling through to the "This tool is set to Off." branch like any other
deny — an authorized, named contract change to
`test_decision_note_unknown_origin_degrades_to_bare_sentence`, which now pins the
new value.

**The main body of this task — the inspector's stale empty-state badge above
populated detail in the Permissions, Audit, and Findings views (AC #1–#3, #5) — is
UNTOUCHED and remains open.** PR-T3 only reached the decision-note text in the same
seam; the badge-staleness defect itself was out of that PR's scope. This task stays
**To Do**.
<!-- SECTION:NOTES:END -->
