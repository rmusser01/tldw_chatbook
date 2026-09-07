---
id: TASK-2761
title: 'Home Details readiness: mcp_ready and rag_ready are constants no producer ever sets'
status: To Do
assignee: []
created_date: '2026-08-06'
labels: [home, bug, honesty]
dependencies: []
---
## Description (the why)

`HomeDashboardInput` declares `mcp_ready: bool = True` and
`rag_ready: bool = False` (`Home/dashboard_state.py:183-185`). A grep across
the package finds **no other occurrence of either name outside that file** —
`build_dashboard_input` never sets them, and no adapter produces them.
(Contrast `acp_ready`, which also defaults True but is genuinely set from the
ACP runtime manager in `home_screen.py` — live probes show `ACP: Blocked`.)

Consequences, all live-verified (dev @ 84e4b33f0, 2026-08-06):

- The Details section permanently shows `MCP: Ready` / "MCP ready", even with
  no MCP server configured or every server down.
- It permanently shows `RAG: Missing sources` / "RAG needs sources", no
  matter how complete the RAG setup is.
- `choose_next_best_action` branch 8 (`Search your Library` /
  "Search/RAG is ready over saved content.", `dashboard_state.py:384`) is
  gated on `rag_ready` and is therefore unreachable dead code.

Documented as a Quirk in `Docs/User_Guide/home.md`.

## Acceptance Criteria (the what)

- [ ] `mcp_ready` reflects real MCP state (or the field and its two Details
      strings are removed rather than shown as a constant).
- [ ] `rag_ready` reflects real RAG state (or likewise removed), and the
      "Search your Library" suggestion becomes reachable or is deleted.
- [ ] Tests cover at least one non-default value for each field reaching the
      Details strings.
