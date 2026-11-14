# Task 17: current-head Stop CI preflight

Read-only source diagnosis at `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036`. No source/index/HEAD edits, installs, cohort runs, new exclusions, warning changes or timeout changes.

## Actual results

PR Fast Lane run 37244353688 / job 111559166890 failed exactly `Tests/UI/test_console_runtime_ownership.py::test_accepted_agent_chat_start_has_visible_stop_in_mounted_target` at line 3141: `assert stop.display and stop.region.width > 0`. `stop.display` was False and the mounted button retained `console-stop-idle console-action-disabled`. The preceding assertion had established `controller.run_state.status is STREAMING`. The failed run therefore established native acceptance/provider entry and a stale Stop projection at the immediate assertion; it did not reach the click/cancellation/custody checks. Its receipt remains `task-17-PR-fast-lane-failure-receipt.json`: 1 failed / 221 passed / 1 pre-existing xfail / 5 warnings. Raw CI log remains private and is not copied into this report.

The single permitted unchanged-node reproduction passed: exit 0, 1 passed in 30.06 s (call 20.16 s); subprocess elapsed 35.96 s. Shared Python 3.12.11, worktree PYTHONPATH, canonical unmodified bootstrap with original bootstrap_profile marker, fresh private basetemp, unchanged existing 300 s pytest timeout and 300 s subprocess bound. `task-17-stop-safe-evidence/original-node-{argv,result}.json`, `.log`, and `.xml` capture this execution. Source hashes stayed equal. A green isolated run does not resolve the CI failure.

## Native path and ownership

`_native_start_rig` uses the existing scripted `_Gateway`, real file-backed ChaChaNotes/AgentRuns databases, live source run/cancel event, real pending target handoff and real library policy capture. `_controller` constructs the native bridge/controller. The mounted app factory retains its original startup ownership, DB/service fakes, splash config and ready provider configuration. The test substitutes that actual controller/store/bridge into the app runtime after attach reconciliation; ChatScreen accessors read the current runtime, so Stop does not consult an old cached controller.

Production `_start_created_chat` is the one shipping caller of coordinator `start` (the test calls it directly). `start` registers an owned task and shields the acceptance outcome. `_prepare` owns ledger preparation; `submit_draft` passes the real ledger `accept` cutoff, retains/shields the physical durable-turn commit, publishes consumed handoff, and `confirm_receipt` resolves `started` only after both acceptance fences. The continuation installs the target assistant/cancel event and STREAMING state before running the physical provider worker. The test's paused bridge sets `entered` and retains the original 120 s release guard, so an entered assertion is physical-worker evidence rather than a fabricated run-state transition.

The single production `stop_active_run` caller is ChatScreen's visible action; button, Ctrl+G and slash Stop route there through composer_run_controls. The action rejects a blocking setup surface, acknowledges the button and calls the current controller. Native Stop selects the viewed target session, pauses its queue, sets that target's cancel event and cancels the logical stream task; `_signal_stop` also withdraws only unaccepted starts. An accepted start remains independent of source Stop. `run_provider_worker` shields its exact provider task and `_run` drains physical commit/provider workers before ledger settlement and releasing its automatic primary claim. The test keeps the original click observation, 0.5 s shield timeout proving custody remains live, claimed target, release, STOPPED state and one used generation. Those all passed in the unchanged local reproduction.

## Projection boundary and demonstrated evidence gap

Stop rendering is `ChatScreen._sync_console_rail_and_controls` -> `_sync_console_control_bar_under_config` -> `_sync_console_composer_action_state` -> `ConsoleComposerBar.sync_action_state`. The screen derives availability from the viewed controller's stop-allowed state or accepted start; STREAMING permits Stop. Composer stores `_stop_available`, then publishes the active/idle classes and display synchronously. Its keystroke/resize replay uses that cached availability, so it cannot itself discover a newly started run.

The test currently awaits `_sync_native_console_chat_ui()` then calls one `pilot.pause()`. That await is not a publication-completion contract: if a sync, maintenance pause, or whole-sync replay is already in progress, it sets `_console_sync_requested=True` and returns immediately. Even a full pass awaits retrieval/character scope reads before publishing controls. Textual `pilot.pause()` waits for messages queued at entry and CPU idle; it does not wait for arbitrary background storage workers or future timer replay. Thus the assertion can run with STREAMING controller truth and the previous idle button projection. CI contains no sync flags or scheduling timeline at the assertion, so which of these interleavings occurred is not established. This is a concrete synchronization evidence gap, not a demonstrated lost-request product defect.

For an ordinary live mounted pass the pending request is retained: its `finally` clears in-progress and starts a fresh exclusive console-sync worker when requested. If config admission deferred the whole pass, `_run_coalesced_control_bar_sync` retries the live fresh pass; maintenance resume rearms pending work when admission opens. Teardown intentionally drops work and live exceptions surface rather than guarantee success. The existing overlap test checks rearming but remains an unchanged strict xfail because its bare-screen fixture cannot satisfy whole-screen owners; it is not passing race evidence. No new skip/xfail is proposed.

## Smallest correction proposal

Use the existing `_wait_for_selector` already imported by this test to await the published active class:

```python
await _wait_for_selector(chat, pilot, "#console-stop-generation.console-stop-active")
```

Place it after the current awaited native sync, before reading/asserting Stop. Keep the helper's existing 2.0 s deadline and every original assertion, physical worker, acceptance fence, 120 s release guard, 0.5 s custody check and click observation. The class and display are updated in one synchronous composer call; the helper's trailing `pilot.pause()` supplies the next layout refresh for the original width/right-edge checks. This waits for a concrete control publication without new infrastructure or expanded timeout. It remains a proposal until independently authorized/verified. If publication never reaches that selector within the existing bound, retain the failure and collect `_console_sync_in_progress`, `_console_sync_requested`, maintenance/replay/scheduled flags and owned worker state at that deadline to distinguish a lost request from slow scope work.

## Frozen evidence and incoming dev

`focused-frozen-comparison.json` compares current source with Task 14's frozen reconciliation argv hashes and Task 16's pre-edit tracked map. The whole runtime-ownership test, chat controller, screen and composer are byte-identical to both receipts. `console_chat_start.py` differs from Task 14 only in cleanup hardening; start/accept/confirm_receipt/run_provider_worker/is_accepted/authorizes ASTs all match both frozen heads. Task 16's start file matches the current file. The earlier receipts remain valid for their recorded selections; neither is claimed to be a new Stop execution.

Incoming origin/dev `8c4dfe59a243ce0cec8e131aff3935646c64b298` touches the same chat_screen file for slash-command handoff/output routing and command send comments. It changes none of native acceptance, Stop projection, visible click, cancellation or worker-custody seams exercised here. No rebase or incoming qualification occurred in this preflight.

## Post-run metadata currentness

The run belongs to source HEAD `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036`. Root subsequently committed requirement-only metadata at `13d1668d4eedbdfcab50fff28a47a8a68346f033` (plan, ADR-220 and Backlog task). `post-run-currentness.json` confirms every captured test/fixture/native-path source hash still matches the run and the subprocess closed with exit 0. No second execution occurred.
