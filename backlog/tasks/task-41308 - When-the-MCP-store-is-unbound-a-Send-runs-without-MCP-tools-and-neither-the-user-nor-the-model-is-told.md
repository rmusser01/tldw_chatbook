---
id: TASK-41308
title: "When the MCP store is unbound, a Send runs without MCP tools and neither the user nor the model is told"
status: Done
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
- [x] #1 When the MCP store is unbound and MCP servers are configured, the Console shows the user that MCP tools are off for this Send and why (e.g. in the inspector's MCP row), shown by a test that is red on dev
- [x] #2 The model is told, in the request, that MCP tools are unavailable for this turn, or the task notes record why it should not be
- [x] #3 With a bound store, the MCP row and the request are unchanged (existing tests still pass)
<!-- AC:END -->

## Implementation Plan

Owner decision (2026-11-16): option A. Do not tell the model; show the row whenever the store is unbound, from mount, with no disk read; status blocked; an MCP hub warning is out of scope (filed separately). The Windows copy was dropped after the premise correction below (see Implementation Notes).

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

## Implementation Notes

**Premise corrected.** The "every Windows install" premise, carried over from TASK-40714, is false. `raw_participants`, `mcp_source_participants` and `private_paths` import `os` from `Utils/platform_files.py`. On `nt` that `os` is the `WindowsOS()` facade (`Utils/windows_files.py`), which provides `O_DIRECTORY`, `O_NOFOLLOW`, `supports_dir_fd` and `supports_follow_symlinks`. So `_pinned_io_available()` is True on Windows, and the store binds at the default data path. The coordinator found this on fu-base; TASK-41306 stopped on the same finding. The case that remains is a store outside the profile's data folder (`mcp_source_participants.bind`'s path check). Every bootstrap-profile `_build_test_app()` builds one, because the factory picks its own data folder. The Description is left as filed.

**Change.**
- `MCP/console_snapshot.console_mcp_unavailable_reason(service)` returns "store isn't in this profile's data folder" when `unbound_console_sources(service)`, the same pure predicate that makes the Send drop MCP, and "" otherwise. It reads no file and costs about 0.07-0.4 ms a call at load 70.
- `ChatScreen._derive_console_inspector_state` passes the reason (lazy import, no new method) to `ConsoleInspectorState.from_values(mcp_unavailable_reason=...)`.
- `_mcp_inspector_row` returns `MCP: off · <reason>`, status blocked, ahead of its count branches. An empty reason keeps every existing branch, so a bound store renders as before.
- The row shows from mount, before the first Send, and stays after it.
- The `unbound_console_sources` docstring no longer says "every Windows install".

**AC #2: the model is not told.** Owner ruling: the row is the place that explains why. "I have no such tool" is true for the turn. V2-F2 / TASK-34100.5.3 (a tool-less request carries only the session prompt) stays intact.

**Tests.**
- `Tests/UI/test_console_send_without_mcp_store.py::test_inspector_mcp_row_says_off_and_why_when_the_store_is_unbound[bound|no-pinned-io|non-default-data-dir]` presses the real Send button and reads `#console-inspector-mcp` before and after the reply.
  - On base, both unbound profiles timed out waiting for the row; bound passed.
  - On head all three pass.
  - Mutation checks:
    - Dropping the ChatScreen pass-through: 2 failed, 1 passed.
    - Making the helper return "": 2 failed, 1 passed.
- `Tests/Chat/test_console_display_state.py::test_console_inspector_state_mcp_off_reason_wins_over_counts` (4 count states) fails on base with a TypeError.
- That file's TASK-40714 docstring and its `_bind_as_on_windows` helper wrongly claimed Windows. The helper is now `_bind_without_pinned_io`, and the parametrize id is `no-pinned-io`. Behaviour is unchanged.
- The same correction applies to the docstring of `Tests/Chat/test_console_configuration_preparation.py::test_unbound_mcp_store_is_no_mcp_source_on_the_runtime_route`.

**AC #3 evidence: the 5 mounted MCP-row tests, red on dev.** All 5 in `Tests/UI/test_console_internals_decomposition.py` (`-k mcp`) failed on dev fbf1c9ea74 with `RecoveryRequired: raw_source_selection_changed`. They read config under the per-test profile redirect without a `bootstrap_profile` marker, the trap TASK-38111 fixed elsewhere.
- Each now carries the per-node marker.
- With only the marker, all 5 pass on base. Under this change, the 3 that pin bound rows (tools ready, not connected, omitted by default) would see the off row, because a bootstrap test app's store is unbound. So they build their app at the profile's own data folder (`_build_bound_mcp_test_app`), which asserts the store is bound.
- The 2 compose tests replace the service with `FakeMCPService` and are unchanged.
- With head tests on base product code, all 5 passed in a serial run. Under `-n 4` at load 60-85, `_open_console_inspector`'s 2 s rail-open deadline timed out on 1-2 of them on either code.

**Covering files, base (fu-base) vs head, same 22 files, `-n 4`, run together at load 64-147.** The 22 files are the Inspector-deriving UI/Backup_Recovery files, `test_console_run_inspector`, `test_product_maturity_phase3_layout_contracts`, the display-state and send-without-store files, `Tests/MCP/test_console_snapshot_source_contracts.py` and `Tests/Chat/test_console_async_mcp_snapshot.py`.
- Base: 347 failed, 738 passed. Head: 342 failed, 751 passed (8 more nodes).
- No node fails only on head. The 5 nodes that fail only on base are the 5 MCP-row tests above.
- The 342 shared reds are pre-existing: `test_console_internals_decomposition` 122, `test_console_session_settings` 97, `test_console_right_rail` 33, `test_console_async_mcp_snapshot` 32, `test_console_rail_reconciliation` 27, `test_console_snapshot_source_contracts` 16, and others.
- This run predates the premise correction, which changed only copy, docstrings, the dropped Windows unit test and test ids. The touched files were rerun after it: 75 passed.

**Files.**
- Product: `tldw_chatbook/MCP/console_snapshot.py`, `tldw_chatbook/Chat/console_display_state.py`, `tldw_chatbook/UI/Screens/chat_screen.py` (+6 lines, no method; within the size ratchet).
- Tests: `Tests/UI/test_console_send_without_mcp_store.py`, `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_internals_decomposition.py`, `Tests/Chat/test_console_configuration_preparation.py` (docstring).
- Docs: `Docs/User_Guide/console/agent-runs-and-tools.md` (MCP tools section).

