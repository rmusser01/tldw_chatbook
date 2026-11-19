---
id: TASK-41900
title: "With the agent runtime on, an unbound MCP store refuses every Console Send"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 20:31
updated_date: 2026-11-19 16:13
labels:
- follow-up
- console
- mcp
priority: high
references:
- tldw_chatbook/Agents/native_source_storage.py
- tldw_chatbook/MCP/console_snapshot.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-16 Console stream (TASK-41507/41511). With the agent runtime on (the Console default), a Send whose MCP store is not bound refuses with `mcp_source_not_installed`, raised by `_source_identity` (`MCP/console_snapshot.py` ~137), called from `_AgentSources.__init__` (`Agents/native_source_storage.py` ~107). The refusal is silent (no notice). TASK-40714's guarantee that an ordinary Send reaches the provider holds only because its tests turn the agent runtime off: turning it on fails `test_ordinary_send_reaches…[no-pinned-io|non-default-data-dir]`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes name which real configurations leave the MCP store unbound at Send time (TASK-41308 called the non-default data folder case unproven in production)
- [x] #2 With the agent runtime on and the MCP store unbound, the turn runs without MCP tools instead of being refused, as TASK-40714 AC#2 requires
- [x] #3 TASK-40714's ordinary-send tests run with the agent runtime on, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce: run TASK-40714's ordinary-send test with `[console] agent_runtime` on (the app's `app_config` reloaded after the builder writes False), and find each place an unbound store stops the turn (temporary stack traces, reverted).
2. AC#1: trace start-up (`app_service_wiring` builds `LocalMCPStore(get_user_data_dir() / ...)`, its `__init__` calls `mcp_source_participants.bind`) and probe `bind`'s skips under real configurations: default profile, `[paths] data_dir` (absolute and `~`), `[general] users_name`, and an unreadable config.
3. Fix at the root: `_AgentSources.__init__` pins the MCP permission receiver the turn may use. For an unbound stock store (`unbound_console_sources`, the predicate the capture already uses) select the service's own permission store before the pin, so the turn's lazy kill-switch and tool-state readers find the pinned object; take no `_source_identity` proof (none can exist). Bound and non-stock stores keep today's route.
4. Tests in `Tests/UI/test_console_send_without_mcp_store.py`: the ordinary-send test parametrized over the gateway and the agent runtime (red before), asserting the agent runtime really ran; an in-process `_AgentSources` test showing a bound store keeps its prospective identity, an unbound stock store selects its permission store, and a replaced reader still refuses with `mcp_source_not_installed`.
5. Covering files base vs head, preflight, ratchets.

## Implementation Notes

**AC#1: which real configurations leave the store unbound at Send time.** None that this trace or the probes found. The app builds one store, `LocalMCPStore(get_user_data_dir() / "local_mcp_store.json")` (`app_service_wiring.py`), and its `__init__` calls `mcp_source_participants.bind`, which skips when there is:
- no pinned I/O: every supported platform has it (Windows through `Utils/windows_files.WindowsOS`, TASK-41308);
- no config cache, or a cache loaded from another path than the effective config path;
- a store path other than `profile_paths.user_data_dir(cache)/local_mcp_store.json`.

`get_user_data_dir()` resolves `[paths] data_dir` and `[general] users_name` from the same config, so the last two agree in the app. Probes through the wrapper (temporary test, deleted) built the store the app's way and read `_BINDINGS`:
- default profile: bound;
- `[paths] data_dir` absolute, and as `~/...`: bound;
- `[general] users_name = "alice"`: bound;
- an unreadable config.toml: `get_user_data_dir()` itself refuses (`RecoveryRequired: config_directory_selection_unavailable`), so the app never builds a store.

So TASK-41308's "non-default data folder" is bound when it is configured, and unbound only when a store is built at a folder its config does not select. Test factories do that (`_build_test_app(user_data_dir=<factory folder>)`, round 16's `_held_console`). The fix below is defensive for production, and needed by those harnesses.

**Cause, traced with temporary stack traces (reverted).** With the agent runtime on, a Send builds the turn's native sources (`_AgentSources.__init__`, from `prepare_agent_sources`). For a stock service with no permission store yet, it took `_source_identity(store)` as the store's prospective identity; unbound, that raised `mcp_source_not_installed` before the provider. Skipping only that proof was not enough: the turn's own readers read `service.permission_store` (seen: the MCP composition's anchor, and with that skipped the Tools chip's kill-switch read; by reading the code, also the local tools' kill switch and the built-in gate), whose getter creates the store lazily, after the turn had pinned `permission = None`. `_check_original_bindings` then failed at settlement with `OriginalSourceChanged` and no reply.

**Fix (`Agents/native_source_storage.py`, +9).** For an unbound stock store (`unbound_console_sources`, the predicate the capture uses since TASK-40714), `_AgentSources.__init__` selects the service's own permission store before it pins it, so every lazy reader in the turn finds the pinned object; no identity proof is taken, since none can exist. MCP tools stay off: the capture already froze an empty MCP maximum, and `MCPToolProvider` offers nothing outside it. A bound store, or a non-stock (replaced) service, takes the same route as before; a replaced reader still refuses with `mcp_source_not_installed`. Decision: the turn reads the unbound permission file for the kill switch and tool states, as the MCP hub in the same app does; the alternative, treating the service as absent for the turn, would also drop local tools and change built-in tool gating, and needed guards at every reader.

**Tests (`Tests/UI/test_console_send_without_mcp_store.py`).**
- `test_ordinary_send_reaches_the_provider_and_saves_the_reply` is parametrized over `runtime` (`gateway`, `agent`). The agent case reloads `app.app_config` after turning the runtime on and asserts one `_AgentSources` was built for the turn, so it cannot pass on the gateway path. It also pins the turn's permission store and that it is bound exactly when the store is. Red at base: `[no-pinned-io-agent]` and `[non-default-data-dir-agent]` ("no reply"; `RecoveryRequired` at `controller_submit`).
- New `test_turn_sources_prove_a_bound_store_and_select_an_unbound_ones_permissions[bound|no-pinned-io|replaced-reader]` builds the turn's sources in process. Bound: the permission store stays unbuilt and the prospective identity equals `_source_identity(store)`, as before. Unbound: the service's permission store is pinned, and a kill-switch read afterwards finds that same object. Replaced reader: refuses with `mcp_source_not_installed`. Red at base: `[no-pinned-io]` (`mcp_source_not_installed`). Mutation check: selecting the permission store for every store fails `[bound]` and `[replaced-reader]`.
- First run of the parametrization passed 6 of 6 at base because the setting was saved after the app's config snapshot was taken; lesson added.

**Covering runs, one command at base (c176f553b3's three product files) and at head, `-n 4`, load 18-73.** Both stream fixes were in the same run (TASK-41903, TASK-41904).
- Chat/MCP/Performance (`test_console_ordinary_tree_widening`, `test_console_workflow_scheduling`, `MCP/test_console_snapshot_source_contracts`, `test_console_goal_dispatch`, `test_console_goal_authority`, `test_console_light_surface_goal_runs`, `Performance/test_console_received_import_contract`): base 49 failed / 91 passed, head 48 / 92. Only at base: TASK-41903's new test. The 48 shared reds are pre-existing (workflow scheduling 26, snapshot source contracts 16, goal authority 6).
- UI (`test_console_send_without_mcp_store`, `test_console_goal_controls`, `test_console_send_start_window`, `test_console_runtime_ownership`, `test_console_send_acknowledgement`, `test_console_send_draft_snapshot`, `test_console_received_intent_feedback`, `test_console_raw_cli_send`, `test_console_send_admission_off_pump`, `test_console_resume_send_guard`, `test_console_fallback_input_boundaries`): base 38 failed / 237 passed, head 35 / 240. Only at base: the 4 new red nodes (this task's 3, TASK-41904's 1). Only at head: `test_console_send_admission_off_pump.py::test_a_second_enter_after_an_unknown_command_hint_sends_it_as_text`, a pre-existing load flake: interleaved serial runs at load 42-55 failed 2 of 4 at base and 2 of 4 at head. The 34 shared reds are pre-existing (draft snapshot 21, admission off pump 8, fallback input boundaries 3, goal controls 2).
- `./scripts/preflight.sh` passed. `Tests/Architecture/test_module_size_ratchet.py` and `test_screen_size_ratchet.py`: 234 passed. Ruff on the changed files reports only findings that are already on base lines.

Files: `tldw_chatbook/Agents/native_source_storage.py`, `Tests/UI/test_console_send_without_mcp_store.py`, `backlog/docs/lessons-testing-evidence.md`. No User Guide change: `console/agent-runs-and-tools.md` already says a Send with an unusable MCP settings file carries no MCP tools, which is now true with the agent runtime on too.

**Lane.** `Tests/UI/test_console_send_without_mcp_store.py` is already a whole UI census entry. Serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`) at load 41-48: 20 passed in 184 s (TASK-41308 recorded 98 s at load 15). The six new nodes took 41 s of it: the three agent-runtime sends 11-14 s each, the three in-process cases about 1.3 s each.
