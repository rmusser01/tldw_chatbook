---
id: TASK-40714
title: "Console Send is refused with 'Console preparation source unavailable' when the MCP store is not bound, as on every Windows install"
status: In Progress
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
- [ ] #3 The Send draft snapshot tests pass under simulated Windows and under a non-default data directory, not only on the default data path.
<!-- AC:END -->

## Implementation Plan

ADR required: no. The fix narrows an existing refusal to its intended cases; it adds no new authority (an unbound MCP store is read by nothing, the turn simply carries no MCP tools).

1. Premise check: reproduce through the mounted ChatScreen Send button with a fake gateway, under whole-process pinned-I/O-off (Windows) and with a non-default data directory; check #3497's Windows serve run against the merge order of #3480.
2. Treat a stock MCP control plane whose store never bound to backup custody as "no MCP sources" in the owned Console configuration capture: the configuration references carry no MCP service, so the snapshot's MCP maximum is empty. Route both the mounted (stock screen) and received (queued) captures through the owned path for that case; keep every other non-standard source refused.
3. Regression test in Tests/UI/test_console_send_without_mcp_store.py (bound, Windows, non-default data dir), mutation check, draft-snapshot file under both simulations, covering files base vs head, preflight, Tests/Architecture.

## Implementation Notes

**Premise.** Confirmed on dev 1455ceeb96 through the mounted ChatScreen Send button with a fake gateway: with the MCP store unbound (pinned I/O off at its bind, and separately a non-default data directory) Send showed "Console preparation source unavailable; Send again." and the provider was never called. A whole-process run with `raw_participants._pinned_io_available` patched to False at import gave the same refusal. The binding has one route (`mcp_source_participants.bind`, from the store's `__init__`), which returns early without pinned I/O, so a Windows store is never bound. PR #3497's Windows serve run does not contradict this: its body (2026-11-12 02:22Z) predates #3480's merge (03:49Z), and the refusal came in with #3480's 11d71e5db5.

**Fix.** `MCP/console_snapshot.unbound_console_sources(service)` names a stock MCP control plane whose `LocalMCPStore` has no backup binding. `_source_references` then carries no MCP service, so the owned configuration capture runs with an empty MCP maximum (no MCP tools), as it already did with no MCP service. The stock-screen route in `_controller_capture_turn_configuration_snapshot` and the received route in `capture_received_configuration_owned` accept that case instead of refusing. Every other non-standard MCP source (custom types, replaced methods, a bound store whose selection changed) is still refused. Nothing reads the unbound store, so no custody is weakened. #3497's WorkspaceDB scoping in `capture()` is untouched.

**Tests.** `Tests/UI/test_console_send_without_mcp_store.py`: (a) a real Send button press under bound, Windows-bind and non-default-data-dir profiles reaches the gateway and saves the reply; (b) the three draft-snapshot sends that only fail because of the unbound store (queued, armed tokenless, spoken/received) under the Windows-bind and non-default-data-dir profiles. Base: 8 of 9 fail. Head: 9 pass. Reverting each of the three edits on its own fails 4-8 of them.

**AC #3 is not met as written.** `Tests/UI/test_console_send_draft_snapshot.py` has 20 of 32 nodes failing on the default (bound) data path at base, and the same on e0186fa10a. Those failures do not depend on the MCP binding. Most call `_dispatch_console_draft_send`/`handle_console_send_message` after replacing prompt-queue or controller seams, and are then refused by `turn_admission.capture_authority` with the same message. Only 2 of the file's nodes are in the UI PR gate. Each simulation adds 4 failures at base (`active_run_queues`, `armed_unknown_tokenless`, `spoken_send_sends`, and the flaky `custody_refusal[True-False]`, which also fails on the default path). At head, both simulations fail only the default-path set.

**Also found (not fixed).** With pinned I/O off for the whole process, `ConsoleReadinessConfigProjection.read_current` (`UI/Console_Modules/console_spend_projection.py`, `raw.participant_identity(participant)` on a config operation that has no participant) raises `raw_participant_not_installed`. The readiness display never warms and the control bar never renders. In that run the reply was saved, but the transcript still read "Sending…".

**Covering files, base vs head (same files, -n 4, load 40-55).** 31 files: the 17 Chat/MCP/Plugins files that reference the preparation module, plus 14 UI/ProductionApp/Performance Send files. Non-UI: 108 failed at head and 109 at base, with the same set apart from load flakes. The two nodes that failed only at head both gave identical results when rerun serially on both trees. Both trees also hang on the same `test_console_ordinary_tree.py` nodes. UI: base had 17 failures (9 pre-existing plus 8 of the new file), head had the same 9 pre-existing. `./scripts/preflight.sh` passed. `Tests/Architecture` (serial): 1348 passed, 3 failed. The 3 also fail on base (`test_reviewed_diagnostic_changes_are_metadata_only`, which names `Agents/agent_service.py`, and two timer-path nodes, which name `project_index_binding_card.py`).

Files: `tldw_chatbook/MCP/console_snapshot.py`, `tldw_chatbook/Chat/console_configuration_preparation.py`, `Tests/UI/test_console_send_without_mcp_store.py`.

