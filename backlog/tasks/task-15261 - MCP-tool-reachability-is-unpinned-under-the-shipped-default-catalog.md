---
id: TASK-15261
title: MCP tool reachability is unpinned under the shipped default catalog
status: To Do
assignee: []
created_date: '2026-08-11'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing in the suite pins that an MCP tool is reachable **at all** under the
catalog size the app actually ships with.

Every MCP permission test in `Tests/Chat/test_console_agent_swap.py` runs with
a small catalog, so the model's direct fence call is direct-disclosed and the
permission gate is what gets exercised. Production is nothing like that: ~2
always-on builtins + 16 local tools + ~18 Library tools (`direct_library_tools`
defaults True, wired 2026-08-06/07 by task-1337) ≈ 36 entries, well past
`DIRECT_DISCLOSE_THRESHOLD` (16, `Agents/agent_models.py`). Past the threshold
`initial_disclosure` returns no schemas and offers `find_tools`/`load_tools`
instead — so find/load is the **live production disclosure mode**, and has been
since before PR #1474.

This was surfaced while repairing the dev baseline: PR #1474 (local tools on by
default) pushed the *test* catalog from 3 to 19 and all five MCP tests went red
at the disclosure gate, never reaching the permission gate they exist to test.
A fixture now keeps those tests' catalog small so they assert what they were
written to assert — which means the production-shaped path remains uncovered.

Related and worth doing in the same pass:
`test_mcp_review_hook_raise_fails_open_but_invoke_gate_still_refuses` was GREEN
but vacuous on dev for the same reason — it asserts `execute_calls == []` to
prove `invoke()`'s own gate refuses, while the refusal actually came from the
disclosure gate and `invoke()` was never reached. The fixture fix restores its
meaning; a production-shaped test would keep it honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test runs with a production-shaped catalog (> DIRECT_DISCLOSE_THRESHOLD entries) and asserts the model reaches an MCP tool via find_tools/load_tools and then executes it under the permission gate
- [ ] #2 The same test covers at least one gated verdict (ask → approve) so the permission path is exercised in the find/load disclosure mode, not only the direct-disclosure mode
- [ ] #3 The interaction with max_active_tools (24) and provider registration order is checked: MCP providers register last, so a large catalog must not silently truncate MCP tools out of reach
<!-- AC:END -->
