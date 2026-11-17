---
id: TASK-41308
title: "When the MCP store is unbound, a Send runs without MCP tools and neither the user nor the model is told"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- mcp
- windows
- ux
priority: medium
references:
- tldw_chatbook/Chat/console_display_state.py
- tldw_chatbook/MCP/console_snapshot.py
- tldw_chatbook/Backup_Recovery/mcp_source_participants.py
- https://github.com/rmusser01/tldw_chatbook/pull/3541
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40714 (PR #3541) found it and left it unfixed. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** TASK-40714 made an unbound MCP store count as "no MCP source": the Send goes through with an empty MCP maximum, so the model is offered no MCP tools. The store is unbound on every Windows install and with a non-default data directory. A user with MCP servers configured gets replies that silently ignore those tools, and nothing says why. The model is not told either, so it may claim the tools do not exist.

**Evidence.**
- `Backup_Recovery/mcp_source_participants.py:105-111`: the store's bind returns early without pinned I/O, so a Windows store is never bound.
- `MCP/console_snapshot.py:284` (`unbound_console_sources`) accepts such a store as a source with no MCP tools.
- The inspector's MCP row, `_mcp_inspector_row` (`Chat/console_display_state.py:222-255`, called at `:1992`), shows only "N tools ready" or "M servers enabled, not connected". It returns `None`, so the row disappears, when the count is `None` or both counts are zero. It has no "unavailable, and why" state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the MCP store is unbound and MCP servers are configured, the Console shows the user that MCP tools are off for this Send and why (e.g. in the inspector's MCP row), shown by a test that is red on dev
- [ ] #2 The model is told, in the request, that MCP tools are unavailable for this turn, or the task notes record why it should not be
- [ ] #3 With a bound store, the MCP row and the request are unchanged (existing tests still pass)
<!-- AC:END -->

## Implementation Plan

DRAFT (design round, 2026-11-16): options for the owner; nothing is implemented until a decision comes back.

ADR required: no. The change is display copy plus, at most, one model-facing line. It adds no authority: the unbound store still is not read for the turn.

**Evidence (fbf1c9ea74).**
- No config-level count of MCP servers exists. `[mcp]` in `config.toml` holds only server-side switches. The user's servers live in `local_mcp_store.json`, which `LocalMCPControlService.get_external_servers()` reads from disk on every call. Knowing "servers are configured" therefore needs a disk read of the unbound store, off the UI thread.
- Knowing "the store is unbound" needs no I/O. `MCP/console_snapshot.unbound_console_sources(service)` is a pure identity check, and it is the exact predicate that makes the Send drop MCP. The reason splits on `raw_participants._pinned_io_available()`: False means Windows. True means the store is outside the profile's data folder. That second case is unproven in production, but it is how `Tests/UI/app_factory._build_test_app()` builds every bootstrap-profile test app (probe: `unbound=True`).
- The unbound case loses all MCP tools: the user's servers and, on the full surface, chatbook's built-in MCP tools (`_capture_mcp_definition_selection` freezes `{}`). The compose then publishes `(None, None)`, so `_mcp_inspector_row` returns None and the row disappears. Even when bound, the row appears only after the first Send composes.
- Places to tell the user: the inspector's MCP row (`Chat/console_display_state._mcp_inspector_row`, fed by `ChatScreen._derive_console_inspector_state`); a transcript notice (none exists for MCP); the MCP hub (it reads the store and shows nothing about the Console).
- Telling the model: every Console request goes through `build_console_first_request_plan` (`Chat/console_agent_bridge.py`), and live dispatch and both previews share it. Precedent: TASK-35080's workspace note, an always-sent suffix on the system prompt (`Agents/agent_request_execution.py`, `_append_workspace_context_note`), added because a silent prompt let the model make things up. Counter-precedent: V2-F2/TASK-34100.5.3 (`first_request_window.PLAIN_CHAT_SYSTEM_PROMPT`, `agents.console_light_surface_hint`): a tool-less request carries only the session prompt, and the tool hint is "never sent when no tool is offered". Binding never changes after the store is built, so a note would be a constant suffix. It costs about 25 tokens on every request, but it does not churn the prompt-cache prefix from turn to turn.
- Pre-existing red, to file separately: the 5 mounted MCP-row tests in `Tests/UI/test_console_internals_decomposition.py` (`-k mcp`) fail on base with `RecoveryRequired: raw_source_selection_changed`. They have no `bootstrap_profile` marker.

**Options.**
- A (recommended): show the row, do not tell the model. `console_snapshot` gets a reason helper (about 10 lines; "" when bound). `ChatScreen._derive_console_inspector_state` passes the reason. `_mcp_inspector_row` returns `MCP: off · <reason>` (status blocked) when a reason is given, before its count branches, so a bound store renders as before. The row shows from mount, before the first Send. Task notes record why the model is not told: the row is the place that explains why; "I don't have that tool" is true for the turn; and it keeps V2-F2.
- B: A plus a one-line model note. A new PromptSpec, `agents.console_mcp_unavailable_note`, appended in `build_console_first_request_plan` next to the workspace note, only when unbound. It also needs payload tests: present under Windows, absent when bound.
- C: A plus a one-time transcript notice on a session's first Send. This needs more wiring and adds persistent noise. Not recommended.

**Steps (option A).**
1. Failing test (red on base), in `Tests/UI/test_console_send_without_mcp_store.py`. Mount the screen under the bound, Windows (`_bind_as_on_windows`) and non-default-data-dir profiles. Open the inspector and read `#console-inspector-mcp`: present with the off copy when unbound; when bound, unchanged and absent before a Send. Add unit rows in `Tests/Chat/test_console_display_state.py` (the reason wins; the count branches are unchanged).
2. Make the code change in `MCP/console_snapshot.py`, `Chat/console_display_state.py` and `UI/Screens/chat_screen.py`. Update `Docs/User_Guide/console/agent-runs-and-tools.md`.
3. Mutation check (drop the reason pass-through: red). Run `Tests/UI/test_console_run_inspector.py`, `test_console_right_rail.py`, `test_product_maturity_phase3_layout_contracts.py` and the display-state tests on base and on head; bootstrap-profile test apps are unbound, so pins on inspector rows may move. Then preflight, and `Tests/Architecture` run serially.
