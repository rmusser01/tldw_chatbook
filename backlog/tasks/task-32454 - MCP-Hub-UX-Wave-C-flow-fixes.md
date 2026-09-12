---
id: TASK-32454
title: 'MCP Hub UX Wave C: flow fixes'
status: Done
assignee: []
created_date: '2026-09-11 19:51'
updated_date: '2026-09-11 20:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five flow improvements from the 2026-09-11 MCP screen UX review: first-run lands on the overview (selection/inspector detail preserved), a discovery breadcrumb naming zero-tool servers in Permissions, Save & connect on the add-server form plus lifecycle actions in the canvas detail toolbar, a profile-context hint line for non-default tool-policy profiles, and the step-by-step add-server tutorial in the user guide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh-install load shows the overview table while the problem server stays selected and the inspector explains it,Permissions legend names known servers with no discovered tools,Add-server form offers Save and connect; canvas detail toolbar offers Connect/Refresh tools for local profiles,Non-default tool-policy profiles show a one-line Console-context hint,mcp.md gains a step-by-step add-server walkthrough consistent with the new cycle order
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: TDD per item (six tests watched RED, then minimal implementation). Branch `feat/mcp-hub-ux-wave-a`, commits on top of Waves A/B.
- C1 overview hold: `_hold_canvas_overview` on the workbench, set by the F-054/task-2240 preselect, honored by `_show_selected_detail` (renders `show_detail(None)` — overview with fresh data), cleared by the first explicit selection (`_select_server_key`) or a restored view state. Inspector readiness for the preselected row is unchanged (task-2240's intent preserved).
- C2 discovery hint: `_undiscovered_servers_hint` (pure, local-source snapshots with `tool_count` None/0, names capped at 3) rendered as a third legend line via `update_matrix(discovery_hint=...)`.
- C3 Save and connect: `SubmitRequested.connect_after`; `_save_local_profile` dispatches `_start_lifecycle(..., "connect")` after the save+resync. Connect failures surface through the existing lifecycle notification/readiness path (e.g. missing env placeholder). Plain Save demoted to secondary styling; Save-and-connect is primary.
- C4 toolbar lifecycle: local detail toolbar gains Connect (primary, not connected) / Refresh tools (connected), posting the same `MCPInspector.HubActionRequested` the inspector's readiness buttons post — one execution path. Check-readiness stays inspector-only (toolbar economy); documented as a deliberate narrowing of the chat design.
- C5 profile hint: `#mcp-perm-profile-hint` Static + `set_profile_hint()`, shown only for non-default profiles ("Console agents run with this profile; persona policy may still floor some tools to Ask."). Per-tool effective-after-persona display remains future work (needs persona policy wired into resolution).
- C7 tutorial: "Adding your first MCP server (step by step)" section in Docs/User_Guide/mcp.md, consistent with this branch's behavior (overview landing, Save and connect, new cycle order, discovery hint). Doc-contract verified unchanged: 39 failures / 26 mcp.md-scoped before AND after — all pre-existing on origin/dev.
- F11 (gate-vs-permission pointer) was intentionally not implemented: subsumed by the existing gate breadcrumb, C2's discovery hint, and the Tools-mode diagnostic empty state; Wave F's bulk-action legend work will teach the server-default lever instead.
- ADR check: none required (flow fixes within existing seams; ADRs 148/149 own the structural decisions).
- Verification: 6 new tests GREEN; suites — test_mcp_workbench 340 passed, test_mcp_permissions_mode+servers+profile_form 149 passed, test_mcp_rail+tools 56 passed. Pre-existing/unrelated: test_destination_shells has 5 failures on clean origin/dev (library/schedules/models shells) plus flaky schedules timing (verified failing-then-passing in isolation on untouched code); test_mcp_documentation_contract 39 failures pre-existing.
<!-- SECTION:NOTES:END -->
