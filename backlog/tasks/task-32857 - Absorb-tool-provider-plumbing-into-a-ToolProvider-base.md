---
id: TASK-32857
title: Absorb tool-provider plumbing into a ToolProvider base
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-32808.7
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Agents/tool_catalog.py:910` defines a 3-method `ToolProvider` Protocol, but all nine providers hand-roll the same four plumbing layers around it: the approval-stamp ledger (~531 LOC: `mcp_tool_provider.py:612-770`, `builtin_tool_gate.py:141-271`, `raw_shell_tool_provider.py:345-460`, `local_tool_provider.py:1230-1297`, `virtual_cli_provider.py:365-419`), pending-gate scaffolding (~250-330 LOC, structurally identical 5-step shape), gate micro-helpers (`_kill_switch_engaged` ×3, `_is_session_approved_safe` ×2, `_arg_rule_allows_safe` ×2 — virtual_cli's comment admits it "mirrors MCPToolProvider"), and the invoke prologue (~350 LOC). Consumer side pays too: 11+ `apply_batch_decisions` call sites in `Chat/console_chat_controller.py` plus five `getattr(provider, "stamp_scope")` compositions in `console_agent_bridge.py:6793-6805`.

A base owning ledger + pending-gate + helpers + prologue deletes ~850–980 LOC (~450–550 beyond TASK-32808.7's stamp store, which this builds on — do not re-implement it). ADR-032 mandates the shared seam; ADR-094's behavioral rules are parameterizable: peek (mcp/local/builtin) vs pop (raw_shell/virtual_cli), raw-shell `authority_generation` fencing, builtin never persisting `always_allow`. Characterization tests per provider are a precondition — the divergences are load-bearing and a naive base silently homogenizes them. ADR required: no — implements the ADR-032 seam; ADR-094 semantics preserved and pinned.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A ToolProvider base (or shared plumbing module) owns the ledger integration, pending-gate scaffolding, gate micro-helpers, and invoke prologue; providers override only genuinely provider-specific parts
- [ ] #2 Behavioral divergences are pinned by characterization tests BEFORE the refactor: peek-vs-pop, raw-shell authority-generation fencing, builtin's never-persist-`always_allow`, session-approved semantics
- [ ] #3 The consumer side routes through one coordinator API (the `apply_batch_decisions` sites and `stamp_scope` compositions collapse)
- [ ] #4 Builds on TASK-32808.7's unified stamp store; no second ledger implementation
- [ ] #5 Target net deletion ~450-550 LOC beyond 32808.7; provider and catalog test suites pass
<!-- AC:END -->
