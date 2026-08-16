---
id: TASK-16788
title: Run-log tools are offered regardless of allowed_tools
status: In Progress
assignee: []
created_date: '2026-08-16'
updated_date: '2026-08-16 16:31'
labels:
  - agents
  - tools
dependencies: []
priority: low
---

## Description (the why)

Found during TASK-16174's oracle run (PR #1712): `search_run_log` and
`run_log_slice` are appended to the offered tool set as `runtime_schemas`
AFTER the `allowed_tools` filter is applied, so a caller that restricts
`allowed_tools` still sees them. Not a user-permission bypass (the
permission gate is a separate layer) — but it silently widens any
programmatic restriction: in the oracle run's tool-OFF arm, question q3
spent its agent steps on run-log tools and ended `stuck`, a confound for
any experiment that isolates arms by tool set, and a surprise for any
future embedder that passes `allowed_tools` expecting it to be exhaustive.

## Acceptance Criteria (the what)

- [ ] A decision is recorded: either `allowed_tools` also filters
      runtime schemas (with existing consumers checked for reliance on
      the current behaviour), or the parameter's docstring states
      explicitly that run-log tools are always offered
- [ ] A test pins whichever behaviour is chosen
- [ ] The oracle-run harness note in
      Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md is
      referenced so the confound stays discoverable
