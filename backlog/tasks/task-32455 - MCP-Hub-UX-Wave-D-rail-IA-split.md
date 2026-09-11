---
id: TASK-32455
title: 'MCP Hub UX Wave D: rail IA split'
status: Done
assignee: []
created_date: '2026-09-11 20:15'
updated_date: '2026-09-11 20:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-148 Wave D: split the agent tool catalog out of the built-in server rail row into its own Agent tools rail section keyed agent:builtin; move Tool gates to the agent detail; built-in detail points there. Plan: Docs/superpowers/plans/2026-09-11-mcp-hub-rail-ia-split.md; spec: Docs/superpowers/specs/2026-09-11-mcp-hub-rail-ia-split-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent tools rail section renders with its own heading and never shares a server_key,Tool gates render in the agent detail only; built-in detail points at the Agent tools row,Agent row routes end-to-end (rail click -> detail, inspector base actions, unscoped permissions preview),Overview table/callouts/preselection stay server-only,mcp.md gates section updated with doc-contract unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Executed inline per Docs/superpowers/plans/2026-09-11-mcp-hub-rail-ia-split.md (5 tasks, TDD per task). Branch `feat/mcp-hub-ux-wave-d` (stacked on `feat/mcp-hub-ux-wave-a`).
- `agent_tools_readiness` (readiness.py) keys the row `agent:builtin` via `AGENT_TOOLS_SERVER_KEY`, equality-pinned to permission_store's `BUILTIN_TOOL_SERVER_KEY` (readiness must not import the store).
- The agent snapshot rides BESIDE `_snapshots` (rail param + workbench state) — overview table, callouts, worst-state summary, and preselection heuristics are untouched and server-only; `_snapshot_for` resolves the agent key.
- Tool gates render for `source == "agent"` only; the built-in detail keeps `[mcp]` toggles plus the pointer line "Agent tool gates live under Agent tools in the rail." Inspector `_wired_actions` needed no change (base-only already).
- Plan deviations: (1) `LOCAL_TOOLS_MASTER_KEY` was NOT imported in mcp_workbench (plan's grep assumption half-wrong) — import added; the NameError was caught by the new tests immediately. (2) A blanket click-re-target over-matched two builtin-`[mcp]`-toggle tests (they legitimately keep the built-in row) — reverted; suite green after. (3) The 100x30 layout test selects via `_select_server_key` (fourth rail row can clip at 30 rows; test is about layout, not routing).
- Re-targeted tests: servers-mode gate render/toggle/master-off/master-on/restart-marker/geometry tests now use the agent snapshot; workbench gate-path tests click the agent rail row; rail-row count pin 3→4.
- Deferred (recorded in plan self-review): the spec's `(external MCP)` / `(Console agents)` label suffixes for the duplicate matrix rows — follow-up with the duplication-disambiguation work.
- ADR: covered by ADR-148 (linked); no additional ADR.
- Verification: 777 passed across test_mcp_workbench + rail + servers + permissions + inspector; destination-shell MCP tests 22 passed; doc-contract unchanged at the pre-existing 39 failures (26 mcp.md-scoped, all on origin/dev before this branch); ruff no new findings (before=after=2 on touched files).
<!-- SECTION:NOTES:END -->
