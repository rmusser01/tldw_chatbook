---
id: TASK-28228
title: >-
  MCP: wire server-initiated sampling/elicitation to the live chat provider and
  approval surface
status: Done
assignee: []
created_date: '2026-09-01 23:28'
updated_date: '2026-09-03 00:51'
labels:
  - mcp
  - interop
dependencies:
  - TASK-26029
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26029 shipped the protocol-correct handler + connection dispatch for server-initiated sampling/elicitation, fail-closed and tested with injected callables. Production enablement requires the app to construct the dispatcher with the REAL chat provider (chat_api_call) as complete_fn and the REAL approval surface (async create-request then await resolution) as elicit_fn, source the per-server SamplingPolicy from config, and set client._server_request_dispatcher at the MCPClient creation site (MCP/local_control_service.py:778). This half is app-context and not headless-verifiable, so it was split out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server-initiated sampling request is fulfilled through the live chat provider (chat_api_call) and returned
- [x] #2 A server-initiated elicitation request is presented through the live approval surface and the response returned
- [x] #3 Per-server SamplingPolicy (allow + rate + token caps) is sourced from config, default deny
- [x] #4 client._server_request_dispatcher is set where MCPClient is created, so stdio and (future) remote servers both route through it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The build for all four ACs landed under this task's earlier TASK-27019 identity
in PR #2313 (renumbered to 28228 later; see provenance). Verified on dev
(66edc7109) that the wiring is live and unconditional:

- AC#1: `build_live_complete_fn` runs one bounded, non-streaming `chat_api_call`
  off the event loop and returns the extracted text
  (live_server_request_wiring.py).
- AC#2: `build_live_elicit_fn(store)` saves a `LocalApprovalRequest` and polls
  the store until it is resolved (approve -> accept, deny/vanish -> None,
  timeout -> expire-then-raise). The store IS the live approval surface, also
  exposed via control-plane actions `approval_requests.list/approve/deny`.
- AC#3: `sampling_policy_for_server` reads `[mcp] sampling_allowed_servers` +
  rate/token caps from config; default deny.
- AC#4: `local_control_service._get_client()` sets
  `client._server_request_dispatcher_factory` unconditionally at the single
  MCPClient creation site; `client.py` connect consumes it per-server
  (`factory(server_id).handle`), so stdio and future remote servers both route
  through it.

This task was split from TASK-26029 as "app-context, not headless-verifiable."
What was missing was an end-to-end test through the REAL factory: the prior
tests covered each builder in isolation and that `_get_client` sets the factory,
but nothing drove a factory-built dispatcher's `.handle()` for an actual
`sampling/createMessage` / `elicitation/create` request. Added three such tests
(Tests/MCP/test_live_server_request_wiring.py): sampling fulfilled via the live
provider and returned as a well-formed MCP result; default-deny stops an
unlisted server before the provider is reached; elicitation presented through
the store and the out-of-band approval returned. This closes the verification
gap headlessly. 16 tests pass.

Discovered gap (out of scope for these ACs, filed as TASK-29231): there is no
human-facing TUI view in the MCP hub that lists pending elicitation approval
requests with approve/deny. Today a real user can only answer via the
control-plane action, so a live server's elicitation would sit pending until it
times out. AC#2 is satisfied as written (the store/control-plane is the live
approval surface, verified end to end); the convenience view is the last-mile
usability follow-up.

Files: Tests/MCP/test_live_server_request_wiring.py (+3 end-to-end tests). No
production change needed — the wiring was already correct and live on dev.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task originally held id TASK-26042, colliding with the
"Console-Workspace-Files-read-only-inspector" task that arrived on origin/dev
first. It renumbered to TASK-27019 on 2026-09-02 after a sweep whose maximum was
27018.

TASK-27019 then collided with the older "Document Personal Context Profile for
Chatbook users and developers" task (created 2026-09-01 14:45); this MCP task
was created later, at 2026-09-01 23:28. Per the owner rule decided 2026-08-21
in TASK-19601 (**older id keeps it; the younger task renumbers with a provenance
note, regardless of status**), this task renumbered again to TASK-28228 after a
fresh sweep across the current tree, all refs, and all registered worktrees
found a maximum id of 28227 and no TASK-28228 claim.

Citations to TASK-26042 before the first renumber, or TASK-27019 between the two
renumbers, refer to THIS MCP task. The dev-resident TASK-26042 holder remains
the Workspace Files task; TASK-27019 remains the Personal Context documentation
task.
