---
id: TASK-29231
title: 'MCP hub: human-facing view for server-initiated elicitation approval requests'
status: To Do
assignee: []
created_date: '2026-09-03 00:48'
labels:
  - mcp
  - interop
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server-initiated MCP elicitation requests (TASK-28228) are saved to the local approval-request store and exposed via control-plane actions (approval_requests.list / approve / deny), and build_live_elicit_fn polls the store until resolved. But there is no human-facing TUI surface in the MCP hub that lists pending elicitation approval requests and offers approve/deny, so today a real user can only answer a server's elicitation through the control-plane action, not a button. Without this view, an elicitation from a live server sits pending until it times out. TASK-28228 satisfied its ACs (the store IS the live approval surface, verified end to end); this is the last-mile usability follow-up so a person can actually see and answer these prompts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP hub lists pending server-initiated elicitation approval requests (message + server) in a human-reachable surface
- [ ] #2 A user can approve or deny a pending request from that surface, resolving it in the store so the waiting elicit_fn returns
- [ ] #3 A resolved or expired request leaves the pending list; a timed-out request cannot be approved after the fact
- [ ] #4 The view shows the approval deadline so a user knows the request will expire
<!-- AC:END -->


## Renumbering provenance

Created via the backlog CLI, which assigned TASK-28239 from its LOCAL view
(local max was 28238). A sweep across all remote refs and registered worktrees
found TASK-28239 already held by "Library media Reader Prev/Next item should
cross page boundaries" (origin/codex/task-20937-6-terminal-evidence + several
media-ux worktrees), and the true global max id was 29230. Per the owner rule
(TASK-19601: older id-holder keeps it; the younger renumbers with provenance),
this task renumbered 28239 -> 29231 (next free above the global max). Any
reference to TASK-28239 for the elicitation-approvals view means THIS task.
