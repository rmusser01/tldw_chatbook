---
id: TASK-40714
title: "Console Send is refused with 'Console preparation source unavailable' when the MCP store is not bound, as on every Windows install"
status: Done
created_date: 2026-11-13 06:28
labels:
- follow-up
- console
- bug
- windows
priority: high
references:
- tldw_chatbook/UI/Console_Modules/turn_admission.py
- tldw_chatbook/Chat/console_configuration_preparation.py
- tldw_chatbook/Backup_Recovery/mcp_source_participants.py
- Tests/UI/test_console_send_draft_snapshot.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** On Windows, every ordinary Console Send is refused with "Console preparation source unavailable; Send again." The provider is never called, and Send again gives the same refusal.

**Path.**
- Send reaches `UI/Console_Modules/wiring.py:3737` (`_capture_console_configuration_async`), then `Chat/console_configuration_preparation.py:2806`, which asks `standard_console_sources(service)`.
- That is False when the MCP store was never bound: `MCP/console_snapshot.py:135-137` raises `mcp_source_not_installed`, which is caught at :220-223.
- So line 2905 sets `captured = None`, and the stock screen falls back to `capture_authority` (:2929).
- `UI/Console_Modules/turn_admission.py:184-189` then raises whenever the screen has a Console runtime, which a real ChatScreen always has.
- On a stock screen, every non-standard path ends in that refusal. No fallback exists.

**When the MCP store is unbound.**
- `Backup_Recovery/mcp_source_participants.py:105-112` returns from `bind()` without binding when `_pinned_io_available()` is False. `Backup_Recovery/raw_participants.py:281-289` needs POSIX `O_DIRECTORY`/`O_NOFOLLOW` and `dir_fd` support, so the store is never bound on Windows.
- `bind()` also skips (:113-121) when the config cache is missing or the store path is not `user_data_dir/<leaf>`. Whether a non-default data directory reaches that branch has not been re-proven.

**Tests.** On 2026-11-12 (dev e0186fa10a), simulating Windows made 18 of 32 tests in `Tests/UI/test_console_send_draft_snapshot.py` fail, 8 of them with this refusal. #3480's own mounted-Send tests (`Tests/UI/test_console_streaming_default_screen.py`, `Tests/UI/test_console_refused_stream_retry_screen.py`, `Tests/ProductionApp/test_provider_selection_ownership.py`) pass on dev only because commit 022fc46a86 gave them owned preparation sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With pinned I/O unavailable, as on Windows, an ordinary Console Send reaches the provider and its reply is saved.
- [x] #2 With the MCP store unbound, the turn runs without MCP tools instead of being refused.
- [x] #3 The Send draft snapshot sends that failed only because the MCP store was unbound (queued, armed tokenless, spoken) pass under simulated Windows and under a non-default data directory.
<!-- AC:END -->

## Implementation Plan

ADR required: no. The fix narrows an existing refusal to its intended cases; it adds no new authority (an unbound MCP store is read by nothing, the turn simply carries no MCP tools).

1. Premise check: reproduce through the mounted ChatScreen Send button with a fake gateway, under whole-process pinned-I/O-off (Windows) and with a non-default data directory; check #3497's Windows serve run against the merge order of #3480.
2. Treat a stock MCP control plane whose store never bound to backup custody as "no MCP sources" in the owned Console configuration capture: the configuration references carry no MCP service, so the snapshot's MCP maximum is empty. Route both the mounted (stock screen) and received (queued) captures through the owned path for that case; keep every other non-standard source refused.
3. Regression test in Tests/UI/test_console_send_without_mcp_store.py (bound, Windows, non-default data dir), mutation check, draft-snapshot file under both simulations, covering files base vs head, preflight, Tests/Architecture.

## Implementation Notes

**Premise.** Confirmed on dev 1455ceeb96 through the mounted ChatScreen Send button with a fake gateway: with the MCP store unbound (pinned I/O off at its bind, and separately a non-default data directory) Send showed "Console preparation source unavailable; Send again." and the provider was never called. A whole-process run with `raw_participants._pinned_io_available` patched to False at import gave the same refusal. The binding has one route (`mcp_source_participants.bind`, from the store's `__init__`), which returns early without pinned I/O, so a Windows store is never bound. PR #3497's Windows serve run does not contradict this: its body (2026-11-12 02:22Z) predates #3480's merge (03:49Z), and the refusal came in with #3480's 11d71e5db5.

**Fix.** `MCP/console_snapshot.unbound_console_sources(service)` is `standard_console_sources(service, _bound=False)` plus "the store has no backup binding". `_bound=False` keeps every type, method and receiver check (permission store, execution log, `get_kill_switch`, `effective_tool_states`, `get_inventory`, the catalog readers) and skips only the `_source_identity` binding proofs. So a stock-typed service with a replaced reader or receiver is still refused. `_source_references` then carries no MCP service, and the owned configuration capture runs with an empty MCP maximum (no MCP tools), as it already did with no MCP service. The stock-screen route (`_controller_capture_turn_configuration_snapshot`) and the received route (`capture_received_configuration_owned`) accept that case instead of refusing. Nothing reads the unbound store, so no custody is weakened. #3497's WorkspaceDB scoping in `capture()` is untouched, and `git merge-tree` with #3497 is clean.

**Tests.**
- `Tests/UI/test_console_send_without_mcp_store.py`:
  - A real Send button press under bound, Windows-bind and non-default-data-dir profiles reaches the gateway and saves the reply. The owned capture froze the built-in MCP definitions only when bound, `{}` otherwise, and the gateway was offered no tools.
  - The same unbound store with an instance-replaced `get_kill_switch` is still refused, and the gateway is not called.
  - The three draft-snapshot sends that failed only because the store was unbound (queued, armed tokenless, spoken/received) pass under both unbound profiles.
- `Tests/Chat/test_console_configuration_preparation.py::test_unbound_mcp_store_is_no_mcp_source_on_the_runtime_route`: on the runtime route, a store built without pinned I/O at its bind is an eligible source and captures `{}`, `()` and complete. A replaced reader is not eligible.
- Mutation checks:
  - Base code fails the unbound cases.
  - Reverting each of the three preparation edits on its own fails 4-8 nodes.
  - The pre-review type-only `unbound_console_sources` fails both replaced-reader tests.
- The new file passed 3 of 3 runs with `-n 4` (10 of 10 each).

**AC #3, narrowed.** The draft-snapshot sends that failed only because the MCP store was unbound now pass under simulated Windows and a non-default data directory. Each simulation added 4 failures at base: `active_run_queues`, `armed_unknown_tokenless` and `spoken_send_sends`, plus the flaky `custody_refusal[True-False]`, which also fails on the default path. The first three are pinned by the new file under both profiles. **Pre-existing, to be filed separately:** 20 of the file's 32 nodes already fail on the bound default data path at base, and the same on e0186fa10a, independent of the MCP binding. Most replace prompt-queue or controller seams and are then refused by `turn_admission.capture_authority`. Only 2 of the file's nodes are in the UI PR gate.

**Queue scenario flake (review M4).** `test_console_active_run_queues_without_a_screen_worker` fakes an active run straight after the composer mounts. When that dispatch runs before the Console attach reconciliation finishes, the skill sources are still cold. The send then takes the deferred skill capture, and `seal_pending_queue_capture` refuses with "Original queued skill preparation changed; send again." This is not specific to an unbound store: it also failed once on the bound default path. Delaying `_reconcile_console_after_attach` by 2 s made it fail 2 of 2 times. The test now waits for the attach to settle, as other mounted tests do, and the same delayed probe then passed 2 of 2 times. No timeout was changed.

**Not fixed (follow-ups).**
- Nothing tells the user or the model that MCP tools are off because the store is not bound. The inspector's MCP row (`console_display_state._mcp_inspector_row`) only renders counts and omits itself for "absent", so saying why would need new row copy and wiring.
- With pinned I/O off for the whole process, `ConsoleReadinessConfigProjection.read_current` (`UI/Console_Modules/console_spend_projection.py`, `raw.participant_identity(participant)` on a config operation with no participant) raises `raw_participant_not_installed`. The readiness display never warms and the control bar never renders. In that run the reply was saved, but the transcript still read "Sending…".

**Covering files, base vs head (same files, -n 4, load 40-55).** 31 files: the 17 Chat/MCP/Plugins files that reference the preparation module, plus 14 UI/ProductionApp/Performance Send files. Non-UI: 108 failed at head and 109 at base, with the same set apart from load flakes. The two nodes that failed only at head both gave identical results when rerun serially on both trees. Both trees also hang on the same `test_console_ordinary_tree.py` nodes. UI: base had 17 failures (9 pre-existing plus 8 of the new file), head had the same 9 pre-existing. After the review fixes, `test_console_configuration_preparation.py`, `test_console_snapshot_source_contracts.py` and `test_console_send_draft_snapshot.py` fail only nodes that also fail at base (1, 16 and 17 of the 20 respectively). `./scripts/preflight.sh` passed. `Tests/Architecture` (serial, rerun after the review fixes): 1348 passed, 3 failed. The 3 also fail on base (`test_reviewed_diagnostic_changes_are_metadata_only`, which names `Agents/agent_service.py`, and two timer-path nodes, which name `project_index_binding_card.py`).

Files: `tldw_chatbook/MCP/console_snapshot.py`, `tldw_chatbook/Chat/console_configuration_preparation.py`, `Tests/UI/test_console_send_without_mcp_store.py`, `Tests/Chat/test_console_configuration_preparation.py`, `Tests/UI/test_console_send_draft_snapshot.py` (queue scenario wait only).

