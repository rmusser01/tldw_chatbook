# Task 1 report — the pure ownership move (task-15860, headless wake)

**Zero semantic change is the whole deliverable.** Only WHO constructs the
Console runtime moved. Its lifetime, its teardown order, its hook binding
and the `_shutdown_requested` gate that refuses a headless wake are all
exactly as they were on `origin/dev` `97f643816`.

- Branch: `feat/task-15860-console-runtime`, worktree
  `.worktrees/headless-runtime`, base `origin/dev` `97f643816`.
- Owner's staging condition (`DECISIONS.md`, answer (3)): Task 1 is a pure
  ownership move, gated by the full battery, separately reviewable and
  separately revertable from Tasks 2–3, with
  `Tests/UI/test_screen_residency.py` green throughout. All four hold.

## What shipped

New: `tldw_chatbook/Chat/console_runtime.py` — `ConsoleRuntime`, an
app-lifetime holder that lazily constructs and owns

| Object | Was built in | Now built in |
|---|---|---|
| `ConsoleChatStore` (+ its `ChatPersistenceService`) | `ChatScreen._ensure_console_chat_store` | `ConsoleRuntime.ensure_chat_store` |
| `ConsoleProviderGateway` (incl. the `console_provider_gateway_factory` seam) | `ChatScreen._ensure_console_provider_gateway` | `ConsoleRuntime.ensure_provider_gateway` |
| `ConsoleAgentBridge` + its sibling `AgentRunsDB` + `register_fleet_attention` | `ConsoleAgentController._ensure_console_agent_bridge` | `ConsoleRuntime.ensure_agent_bridge` |
| `ConsoleChatController` (and therefore `_register_fleet_wake`/`_register_fleet_usage_reattach`, which run inside its `__init__`) | `ChatScreen._ensure_console_chat_controller` | `ConsoleRuntime.ensure_chat_controller` |

Constructed at `app.py` `TldwCli.__init__` Phase 1, immediately after
`console_image_edit_operations` (the precedent this follows), and
re-created lazily by `ensure_console_runtime(app, view=…)` for app objects
that never had one — the shape `ChatScreen._h3_image_edit_registry`
already uses.

Disposed at `ChatScreen.on_unmount` via `dispose_console_runtime(app,
view=self)`, **after** the unchanged teardown sequence
(`fleet_teardown_split()` snapshot → `await controller.shutdown()` →
`await gateway.aclose()`). `dispose()` is a reference drop and a
generation bump; it shuts nothing down. That call is what Task 2 removes.

The four `_ensure_*` seams keep their names, their laziness, their return
types and their patchability. Three test files replace
`_ensure_console_agent_bridge` by name on the screen instance
(`Tests/Chat/test_change_turn_tracking.py` ×2,
`Tests/Chat/test_console_agent_swap.py`) and ~65 sites reach
`_ensure_console_chat_store`; nothing was renamed.

`register_fleet_attention` moved with the bridge, unchanged, still passing
the APP object and never a screen. The wake coordinator's registration
travels inside `ConsoleChatController.__init__`, so it moved with the
controller by construction.

## Two ordering hazards found and closed, not discovered later

1. **The no-agent-runtime probe must stay first.** The bridge's durable-DB
   probe returns `None` before the store or the gateway is touched. Passing
   them as values would have built both on an in-memory harness. Every
   view-supplied dependency therefore crosses as a callable
   (`store_factory`, `provider_gateway_factory`,
   `native_tools_enabled_factory`).

2. **Two `ChatScreen`s are briefly alive at once.**
   `_complete_screen_navigation` (`app.py`) constructs the incoming screen
   and calls `restore_state` **before** `switch_screen` unmounts the
   outgoing one, and `_restore_native_console_state` reaches
   `_ensure_console_chat_store`. A naive app-owned holder would have handed
   the incoming screen the outgoing screen's controller, which
   `on_unmount` then shuts down underneath it — a dead Console after a
   same-target navigation. `ConsoleRuntime.view` closes it: a runtime
   claimed by a different view is replaced with a fresh one, which is
   exactly today's one-runtime-per-screen semantics, and `dispose` skips a
   runtime claimed by someone else. This is Task 1 lifetime preservation,
   **not** Task 2's attach/detach seam, and it is documented as such.

## The two pins (`Tests/UI/test_console_runtime_ownership.py`)

Both are labelled characterization pins in their docstrings. Task 1 has no
user-visible behaviour change, so there is no defect red to reproduce;
dressing these as defect reds would be a lie about what they measure.

1. `test_console_runtime_is_the_single_construction_site` — greps the
   shipped package: `ConsoleChatStore(` and `ConsoleChatController(` appear
   only in `Chat/console_runtime.py`. (`ConsoleProviderGateway(` is
   deliberately excluded: the Personas preview controller builds its own.)
   **Mutation:** re-added a direct `ConsoleChatStore(...)` in
   `_ensure_console_chat_store` → the pin failed, naming
   `chat_screen.py:5118`. Restored by Edit; `grep MUTATION` clean.

2. `test_second_console_visit_gets_a_new_runtime` — a real mount, a real
   `handle_screen_navigation` away and back. Asserts the runtime owns what
   the screen holds, that leaving still sets `_shutdown_requested`, that
   `on_unmount` disposes (generation 0 → 1, app attribute cleared), and
   that the second visit's runtime, controller, store and bridge are all
   different objects. **Mutation:** made `dispose_console_runtime` a no-op
   → the pin failed at `assert runtime_one.generation == 1`. Restored by
   Edit.

Pin 2 is the one that matters: it makes Task 2's change land as a visible
diff to a test rather than a diff to nothing. When Task 2 gives the
runtime cross-navigation lifetime this test must go red and be rewritten,
and its docstring says so.

## Gate — baseline (untouched branch) vs final, same commands, same order

Runner: `.venv/bin/pytest <paths> -p no:randomly -q --no-header -rf`,
cwd = the worktree.

| Suite | Baseline | Final | Delta |
|---|---|---|---|
| `Tests/test_probe_import_provenance.py` | 1 passed | 1 passed | — |
| `Tests/UI/test_screen_residency.py` | 7 passed | 7 passed | — |
| wake (Chat: `test_console_fleet_wake*.py` + `test_fleet_*.py`) | 65 passed | 65 passed | — |
| wake (UI: `test_console_fleet_wake*.py`) | 19 passed | 19 passed | — |
| `Tests/Agents/` | 1409 passed | 1409 passed | — |
| `Tests/Chat/` | 18 failed, 5445 passed, 66 skipped | 18 failed, 5445 passed, 66 skipped | — (failure sets byte-identical by `diff`) |
| `Tests/UI/test_console*.py` | 32 failed, 3039 passed | 32 failed, 3041 passed | **+2 passed** = the two new pins; failure count identical, set swapped 2-for-2 |

The `Tests/Chat/` 18 are pre-existing on this branch's base (measured
before any edit) and include the known `test_console_visual_evaluation`
and the 15663 marks-migration red.

**The ui_console 2-for-2 swap is flake noise, established rather than
assumed.** Went green: `test_console_agent_rail::test_drilldown_falls_back
_to_overview_after_conversation_switch`, `test_console_rewind_restore::
test_summarize_up_to_choice_blocked_while_a_run_is_streaming`. Went red:
`test_console_control_bar_coalescing::test_requested_sync_still_executes`,
`test_console_session_settings::test_console_settings_modal_refreshes_
readiness_after_returning_to_model_list`. Evidence:

- both newly-red tests pass in isolation AND with their whole module on
  the changed branch (`238 passed` across all four modules, with a *third*
  rewind test flipping red in that run instead — the cluster flips
  independently of the change);
- the settings-modal one drives a `ModalHarness` + `ConsoleSettingsModal`
  and never constructs a `ChatScreen`, so no `ConsoleRuntime` exists in it
  at all — it is structurally untouchable by this change;
- the coalescing one asserts a debounce collapsed three requests into one
  and observed three, i.e. timer timing, on a machine that was running six
  other agents' pytest processes concurrently throughout;
- the flip happened in both directions and left the count identical.

Pre-existing red NOT caused here and not in the mandated battery:
`Tests/Architecture/test_screen_size_ratchet.py` is already red on
`origin/dev` (`chat_screen.py` measures 23,466 lines against a 17,727
budget). This change leaves it at 23,470 (+4, docstrings) with the
`ChatScreen` method count unchanged — no method was added or removed.

## Deliberately not done in Task 1

- The runtime does **not** join `_shutdown_app_owned_lifecycles`. That
  hook runs before Textual closes screen state, and `dispose()` has no
  durable work to settle, so joining it would only reorder Console's quit
  path. Textual's exit unmount already disposes it. Add an exit-time
  settle in Task 2, deliberately, with the quit ordering re-verified.
- `ChatScreen` still holds `_console_chat_store` / `_console_provider_
  gateway` / `_console_chat_controller` as its own handles, set from the
  runtime by the `_ensure_*` seams. ~40 call sites read them directly as
  "has this been built yet" probes, and 59 test sites assign them.
  Repointing those to the runtime is Task 2's job, when the runtime starts
  outliving the screen and the two can actually disagree.
- Hooks are still bound to screen methods (`on_scope_flushed`, the
  dictionary/world-info appliers, the wake probes). Task 0's P3 found all
  five slots a viewless wake touches still pointing at a dead screen;
  rebinding is Task 4.
- `_attempt`'s `_shutdown_requested` gate — Task 0's P2 proved it is the
  single line refusing a headless wake — is untouched.

## Note on plan numbering

The merged plan's "Task 1" is the teardown split. The coordinator
re-sequenced per the owner's staging condition: this task is the pure
ownership move, and the teardown split now follows it. The plan document
was not edited here.
