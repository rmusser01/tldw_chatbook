# Lifetime report — the Console runtime survives unmount (task-15860)

Plan Task 1 (teardown split) + plan Task 2's remaining lifetime half, per
the plan's "EXECUTION ORDER — reconciliation" block.

- Branch: `feat/task-15860-runtime-lifetime`, worktree
  `.worktrees/headless-lifetime`, base `origin/dev` `dcdcf2925` (the merged
  ownership move, PR #1648).
- Predecessors: `2026-08-14-headless-wake-task-0-report.md` (the executed
  probes), `2026-08-14-headless-wake-task-1-report.md` (the pure ownership
  move).

**One sentence:** the Console runtime now outlives every `ChatScreen`;
`ChatScreen.on_unmount` ends a VISIT (`leave_console_runtime`) instead of
destroying the runtime, and destruction moved to app exit
(`dispose_console_runtime`, registered in
`TldwCli._shutdown_app_owned_lifecycles`).

---

## 1. How `ConsoleRuntime.view` was replaced, and the restore-before-unmount order

Task 1 shipped `ConsoleRuntime.view` as an explicitly-labelled **stand-in**:
a runtime claimed by a different view was *replaced with a fresh one*, which
is simply dispose-at-unmount wearing a different hat. It protected the
overlapping-screens window by making sure the two screens never shared
anything.

That is now **removed, not extended**. `ensure_console_runtime` no longer
builds a second runtime for a second view; there is one runtime per app and
views attach to it:

| Was (Task 1) | Now |
|---|---|
| `runtime.view is not view` → build a **new** `ConsoleRuntime`, re-attach to the app | `runtime.view is not view` → `runtime.attach_view(view)` on the SAME runtime |
| `dispose_console_runtime(view=…)` skipped a runtime claimed by someone else | `detach_view(view)` returns `False` and touches nothing when claimed by someone else |
| lifetime preserved by not sharing | lifetime preserved by an explicit, ordered claim |

### The order, stated and tested

`_complete_screen_navigation` (`app.py`) constructs the incoming screen and
calls `restore_state` **before** `switch_screen` unmounts the outgoing one,
and `_restore_native_console_state` reaches `_ensure_console_chat_store`.
So on a same-target navigation (reachable through the `coding` route, which
aliases to Chat) the real sequence is:

1. **incoming** `restore_state` → `_ensure_console_chat_store` →
   `ensure_console_runtime(app, view=incoming)` → **`attach_view(incoming)`**:
   claims the runtime (`runtime.view = incoming`), calls
   `controller.begin_visit()` (fresh per-visit cancellation Event, prompt-queue
   admission re-opened), and binds every slot in `CONSOLE_VIEW_HOOK_SLOTS` to
   the incoming screen.
2. `switch_screen` mounts incoming, unmounts outgoing.
3. **outgoing** `on_unmount` → `leave_console_runtime(app, view=outgoing)` →
   `detach_view(outgoing)` sees `runtime.view is incoming` → **returns `False`
   and does nothing at all**: no hook is cleared, no turn is cancelled, no
   Event is set.

**The successor's claim wins.** A superseded screen can neither clear a hook
its successor bound nor poison its visit. Pinned by
`test_a_superseded_screen_never_detaches_the_successors_runtime`, which drives
the real `coding` alias navigation and asserts all three: `runtime.view is
chat_two`, `controller.notify_run_outcome is not None`, and
`not controller._shutdown_requested.is_set()`.

The deliberate consequence, recorded rather than hidden: on a **same-target**
navigation the previous screen's streaming turn is no longer cancelled. Today
it is, only because the two screens owned two different controllers. With one
runtime the user never left Console, so cancelling would be wrong *and* would
hand the incoming visit a set cancellation Event — a dead Console, which is
hazard #1 exactly.

Screen-owned **timers** (transcript sync, fleet survivor tick, cost TTL) are
not runtime state, are not in the slot list, and stay stopped at unmount.

---

## 2. Handle repointing (piece (a)) — scope

The moment the runtime outlives the screen, a fresh `ChatScreen`'s own `None`
handle SHADOWS a live runtime object until `_ensure_*` happens to run. Four
handles were repointed at the runtime as properties; **~400 production read
sites and ~59 test assignment sites kept working unchanged**, because the
properties are read-write.

| Handle | Holder | Backing |
|---|---|---|
| `_console_chat_store` | `ChatScreen` | `runtime.chat_store` / `set_chat_store` |
| `_console_provider_gateway` | `ChatScreen` | `runtime.provider_gateway` / `set_provider_gateway` |
| `_console_chat_controller` | `ChatScreen` | `runtime.chat_controller` / `set_chat_controller` |
| `_console_agent_bridge` | `ConsoleAgentController` (proxied by `ChatScreen`) | `runtime.agent_bridge` / `set_agent_bridge` |

Three `__init__` slots were **deleted** (`chat_screen.py`) rather than kept:
assigning `None` in `__init__` would have cleared the surviving runtime's
store the instant a fresh `ChatScreen` was constructed — which
`_complete_screen_navigation` does *before* the outgoing screen unmounts.
The same reasoning removed `on_unmount`'s two `= None` assignments.

`ChatScreen._console_runtime()` memoises its runtime and **never re-resolves**.
That is load-bearing: a read must never re-claim, so an already-superseded
screen reads its own runtime and can never reach through to the successor's.
`ensure_console_runtime` also gained a view-held fallback for app objects that
cannot hold the attribute (`None`, read-only doubles), because bare
`ChatScreen.__new__` fixtures reach these handles and would otherwise get a
brand-new runtime per call, losing every write.

`_console_agent_bridge` was included beyond the brief's named three: it has the
identical shadowing shape, and `chat_screen.py:12556` reads it directly as a
"built yet?" probe for `change_tracking_enabled`.

---

## 3. The view seam (piece (b))

**One enumerated list, both directions:** `CONSOLE_VIEW_HOOK_SLOTS` in
`Chat/console_runtime.py`. `attach_view` sets every entry from
`ChatScreen.console_view_hooks()`; `detach_view` restores every entry's
`viewless_default`. `test_attach_and_detach_cover_exactly_the_same_slot_set`
asserts the two key sets are equal.

19 slots: 17 on the controller, 1 on the store (`on_scope_flushed`), 1 on the
wake coordinator (`delivery_ui_hook`).

**Four slots were NOT in Task 0's P3 list of fifteen**, because P3 only wrapped
callables that were `ChatScreen` methods:

- `_default_session_settings` and `_turn_context_provider` — bound to the
  screen's `ConsoleSessionController`, not the screen;
- `prompt_history` — a value built by the screen's prompts controller;
- `on_scope_flushed` — lives on the STORE, not the controller.

Each still holds the dead screen transitively. Worse, all seven
constructor-supplied callables (`_chat_dictionary_applier`,
`_world_info_applier`, `_rag_capture_provider`, `_default_session_settings`,
`_library_provider_factory`, `_global_user_display_name`,
`_turn_context_provider`) are read at construction ONLY — with a surviving
runtime they would have stayed bound to **visit 1's screen for the whole app's
life**. The old wiring block re-applied only the post-construction slots.

Two deliberate exclusions: `controller.app` (the APP outlives every view, and
clearing it breaks the `call_from_thread` bridge a surviving turn still needs)
and `wake.wire(app=…)`, for the same reason.

`_global_user_display_name`'s viewless default is **not `None`**: the
constructor does `global_user_display_name or (lambda: "User")`, so clearing it
to `None` would turn every read into a `TypeError`. It restores
`viewless_user_display_name`.

### A hazard this landing created and closed

`ConsoleHandsFreeController._install_console_hands_free_store_tap` monkeypatches
five `ConsoleChatStore` methods with closures over the screen, and its docstring
said it was "installed once and never uninstalled" — correct while the store
died with the screen. With the store surviving, that tap would strand a dead
screen through five closures **and be re-wrapped once per Console visit,
forever**. `uninstall_console_hands_free_store_tap` now restores the captured
originals at `on_unmount`, conservatively (a seam no longer holding *this*
screen's wrapper is left alone).

### `wake_conversation_in_view` — deliberately left at `None`

Detach clears it to `None`, its documented "unwired" value, which keeps the
historical clear-on-delivery. Making a viewless runtime report *not-in-view*
(so the `◈` mark is KEPT) is plan Task 4 and is **not** done here — the brief
forbids touching the 15971 off-view ruling in this landing. What this landing
does guarantee is that the slot no longer points at a **dead screen**, which is
the silent-wrong-answer hazard.

---

## 4. The teardown split (piece (c))

| Call | When | Behaviour |
|---|---|---|
| `ConsoleRuntime.leave_console` → `ConsoleChatController.leave_console` | every navigation away | ends ONE visit |
| `ConsoleRuntime.dispose` → `ConsoleChatController.shutdown` | app exit only | today's behaviour, exactly |

`leave_console()`, in order:

1. `prompt_queue_coordinator.shutdown()` — tombstone this visit's chains
   **before any cancellation**, preserving `begin_shutdown`'s ordering contract.
2. `self._shutdown_requested.set()` — this visit's Event. Denies every parked
   approval/confirm round armed during the visit, and keeps
   `_attempt`'s wake gate refusing.
3. cancel + await this visit's **USER** stream tasks, stamping
   `cancel_reason="shutdown"` on each one's in-flight citation repair.

`dispose()` additionally closes the provider gateway — a step **moved out of**
`on_unmount`, since an app-owned gateway must not be closed on a navigation and
then reused. `test_leaving_console_does_not_close_the_provider_gateway` pins it.

`_shutdown_app_owned_lifecycles` now awaits `_shutdown_console_runtime()`. Task
1 deliberately did not join that hook, on the reasoning that `dispose()` was a
reference drop with nothing to settle; that reasoning expired the moment
`dispose()` started running `controller.shutdown()` and `gateway.aclose()`.

### The quit-time resurrection window (found by review, not by a test)

Joining that hook created a defect this landing had to close. The hook runs
**before** Textual closes screen state, so a Console screen — and its 0.2s
timers — can still be live while `dispose()` runs, and there are ~75
`_ensure_console_chat_*` call sites reachable from them. As first written,
`dispose()` dropped every reference, so any one of those would have **built a
fresh controller (and store, and `ChatPersistenceService`) mid-quit, which
nothing would ever shut down.** Returning `None` instead would only have moved
the failure into a tick that has never had to handle it.

`dispose()` therefore **keeps its references** and latches `_disposed`; every
`ensure_*` returns what it already holds and builds nothing new. What it hands
back is genuinely torn down, so it refuses work through its permanently-set
cancellation Event — the right answer at exit.

### The permanence trap

`_shutdown_requested` was permanent, justified **solely** by `shutdown()`'s own
docstring: *"`ChatScreen` never reuses an instance after unmounting it."*
App-owned lifetime falsifies that premise — the same controller now serves
visit after visit.

**`_shutdown_requested` is therefore per-VISIT.** The Event object is
*replaced*, not cleared:

- `leave_console()` **sets** the current Event (it stays set forever);
- `begin_visit()` — called by `attach_view` on a NEW claim — **installs a
  fresh, unset Event**;
- `begin_shutdown()` sets `_disposed`, after which `begin_visit()` refuses to
  install anything.

**The fresh Event is installed at ATTACH, not at the end of `leave_console`.**
This is a deliberate deviation from the brief's wording and it is what keeps
"do not relax `_attempt`'s `_shutdown_requested` gate" true: installing it at
the end of `leave` would leave the flag *unset* between visits, and
`console_fleet_wake.py`'s gate — which this landing does not touch, by a single
byte — would stop refusing, silently shipping the headless wake that the
wake-fires-headless task owns. With install-at-attach, the flag is set between
visits exactly as it is today. `test_the_wake_gate_still_refuses_between_visits`
pins both halves.

### The visit Event captured at arm time

Because the attribute is *replaced*, a poll site that re-read
`self._shutdown_requested` would answer with the **next** visit's fresh, unset
Event and resurrect a round the previous visit already denied.

`_bind_visit_cancel_signal()` captures it **once, at arm time** — the same
discipline, for the same reason, as `_bind_round_cancel_signal`'s arm-time
binding of the per-run cancel Event. All three arm sites
(`request_mcp_approvals`, `request_skill_install_confirm`,
`request_skill_script_confirm`) capture it; all three poll sites read
`visit_event`, never `self._shutdown_requested`. `_is_session_cancelled`'s
`visit_event` parameter is keyword-only **with no default**, so a future bridge
cannot silently inherit "no signal". Every site still fails closed.

### The owner ruling, applied

`leave_console()` does **not** cancel an in-flight `AGENT_WAKE` turn.
`submit_draft`'s `AGENT_WAKE` branch registers `session.id` in
`_agent_wake_turn_sessions` and releases it in the same `try`'s `finally`, so
the exemption cannot outlive the turn. `shutdown()` (app exit) still takes
everything. Two tests:
`test_leaving_console_does_not_cancel_an_in_flight_wake_turn` and
`test_the_wake_exemption_never_outlives_its_turn`.

Known cosmetic consequence: `fleet_teardown_split()`'s `killed` count can
over-report by one while a wake turn is mid-flight at nav-away. Its contract is
AC#2's existing one and is deliberately left untouched.

---

## 5. Per-visit queue admission (piece (d))

The coordinator's `shutdown()` is a permanent `_shutting_down` latch — correct
while every navigation built a new controller, fatal once one app-owned
coordinator serves every visit: **the prompt queue would be dead for the rest
of the app's life after the first nav-away.**

`ConsolePromptQueueCoordinator.reopen()` (and `ConsolePromptQueueRegistry.
reopen()`) clear the latch only. The chains, snapshots and queued prompts stay
cleared by the tombstone — that is the pre-existing, AC#2-required leaving-
Console semantics. Called from `begin_visit()`, never on a disposed controller.

`test_the_prompt_queue_admits_again_on_the_next_visit` walks all three states
through the public `admit` API: `REROUTE_NORMAL_SEND` → `SHUTTING_DOWN` →
`REROUTE_NORMAL_SEND`.

---

## 6. The two AC#2 reds, and proof they failed before the fix

Both live in `Tests/Chat/test_console_runtime_lifetime.py`:

- `test_leaving_console_still_cancels_a_streaming_user_turn`
- `test_leaving_console_still_denies_a_parked_approval_round`

The red state is not `origin/dev` (where the runtime is destroyed, so the
assertion cannot even be posed) — it is **the intermediate state this landing
passes through: the runtime survives, and nothing ends the visit.** Reproduced
by Edit-mutating `ConsoleRuntime.leave_console` so it never calls
`controller.leave_console()`:

```
FAILED Tests/Chat/test_console_runtime_lifetime.py::test_leaving_console_still_cancels_a_streaming_user_turn
FAILED Tests/Chat/test_console_runtime_lifetime.py::test_leaving_console_still_denies_a_parked_approval_round
2 failed, 1 warning in 11.72s
```

Restored by Edit. Both green with the split wired.

The third red — the attach/detach seam's defect red,
`test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen` —
drives the production path (`_set_run_state` into a terminal status for a
non-active session) on the controller that survives the navigation, and asserts
the unmounted screen's toast never fires. Before the seam, that slot was still
bound to the dead screen (Task 0's P3 measured five such slots, none raising).

---

## 7. Mutations run and killed

Every restore was **Edit-based**; the final tree is byte-identical to `HEAD`
(`git diff` CLEAN, `grep -rn "MUTATION-" tldw_chatbook/` → none).

| # | Mutation | Expected to die | Actually died |
|---|---|---|---|
| 1 | `leave_console` never calls `controller.leave_console()` | both AC#2 reds | `test_leaving_console_still_cancels_a_streaming_user_turn`, `test_leaving_console_still_denies_a_parked_approval_round` ✅ |
| 2 | visit Event never `.set()` in `leave_console` | both AC#2 reds | `..._denies_a_parked_approval_round`, `test_a_round_from_the_previous_visit_is_not_resurrected`, `test_the_wake_gate_still_refuses_between_visits`, `test_second_console_visit_reuses_the_runtime` ✅ (4) |
| 3 | drop `notify_run_outcome` from `CONSOLE_VIEW_HOOK_SLOTS` | the attach-set == detach-set test | `test_attach_and_detach_cover_exactly_the_same_slot_set` **and** `test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen` ✅ |
| 4 | `detach_view` is a no-op | the dead-screen test | `test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen`, `test_second_console_visit_reuses_the_runtime` ✅ |
| 5 | a poll site re-reads `self._shutdown_requested` instead of its captured Event | a resurrected-round test | `test_a_round_from_the_previous_visit_is_not_resurrected`, and **only** that one ✅ |
| 6 | remove the `_disposed` latch from `ensure_*` | the quit-time rebuild test | see below — **survived at first**, then killed after the test was fixed ✅ |

Mutation 2 is worth a note: the brief predicted it would kill "both AC#2 reds".
It killed the parked-approval red but **not** the streaming-turn red — that one
survives because cancelling a stream goes through `_signal_stop` +
`task.cancel()`, which does not consult the Event at all. Recorded as measured,
not as predicted.

**Mutation 6 caught a test that passed for the wrong reason**, which is the
whole reason for the discipline. `test_dispose_does_not_let_a_late_ensure_
rebuild_the_runtime` did **not** die when the latch was removed: once
`dispose()` keeps its references, `ensure_chat_controller` returns the existing
object via the plain `is not None` check, latch or no latch. The latch's real
job is the slot that was **never built** before quit and so has no reference to
hand back. `test_dispose_does_not_let_a_late_ensure_build_an_unbuilt_slot` was
added for exactly that case and does die under mutation 6 (verified both
directions). Both tests are kept: they cover different halves.

### Incident: a checkpoint taken mid-mutation

Two transient 529s killed this session. The coordinator checkpointed the tree as
`0a79c4d1c` — **while mutation 2 was installed** — and reported "4 tests red".
All four were that one removed line (`self._shutdown_requested.set()`), and the
four names match mutation 2's kill list above exactly. Restored by Edit,
verified with a one-line `git diff`, 19/19 green, committed as `7ed52e071`.

**None of the four was a genuine defect.** In particular
`test_second_console_visit_reuses_the_runtime` is *both* things at once, and the
distinction matters: it is the **intentional inversion** of Task 1's
`test_second_console_visit_gets_a_new_runtime` (whose own docstring said "when
Task 2 makes the runtime survive a navigation, this test must go red and be
rewritten"), and it was *additionally* failing under the mutation because it
asserts the previous visit's Event stays set. The pin was rewritten
deliberately; it was not catching a defect.

---

## 8. Gate — baseline (untouched branch) vs final

Runner: `.venv/bin/pytest <paths> -p no:randomly -q --no-header -rf`,
cwd = the worktree. `Tests/test_probe_import_provenance.py` is in every gate:
the venv's editable install resolves `tldw_chatbook` to a foreign worktree and
loses only by `sys.meta_path` ordering.

<!--GATE_TABLE-->

---

## 9. Deliberately not done

- **`_attempt`'s `_shutdown_requested` gate is untouched, byte for byte.**
  A wake still does not fire headless: between visits the Event is set exactly
  as it is today. That is the wake-fires-headless task.
- **Continuity is still screen-owned.** Console message state still travels
  through `ScreenStateStore.native_console_state`; the app-owned store is not
  yet the continuity owner (plan Task 3, blocked on owner call #3). Task 0's
  P3b consequence therefore still stands: a wake that ran while Console was
  unmounted would be invisible on return.
- **The 15971 off-view ruling is untouched.** `wake_conversation_in_view`'s
  viewless default stays `None` (plan Task 4 changes it).
- **`fleet_teardown_split`'s contract** is unchanged (see §4).

## 10. Concerns

1. **Same-target navigation no longer cancels the outgoing turn** (§1). Forced
   by one-runtime semantics and pinned, but it is a real behaviour change on a
   live path (`coding` → Chat) and deserves an owner eyeball.
2. **`Tests/Chat/test_console_video_capacity.py::test_real_unmount_path_
   invokes_pending_artifact_drain` is red in the baseline** and exercises the
   unmount path this landing rewrites. It was already red before any edit here;
   worth confirming it stays red for its original reason rather than a new one.
3. **The hands-free tap** now has an uninstall path that did not exist before.
   It is conservative, but it is new code on a rarely-exercised surface.
6. **`dispose()` now leaves torn-down objects reachable** (§4). That is the
   safest of the three options at quit, but it does mean
   `runtime.chat_controller` is non-`None` after dispose — a reader expecting
   the Task-1 reference-drop semantics will be surprised. `generation` and
   `_disposed` are the honest signals.
4. **`prompt_history` rebinds per visit**, so two `PromptHistory` instances can
   briefly exist over the same JSONL path across a navigation. That was already
   true per-screen before this landing; it is not made worse, but it is not
   fixed either.
5. **Process note.** Pieces (a)–(d) were verified individually but landed in
   one implementation commit rather than four: two infrastructure kills forced a
   checkpoint-and-recover, and separating them after the fact would have meant
   hunk-level staging of two heavily-interleaved files. Piece (a) does have its
   own isolated evidence (347 passed / 1 known flake across the three
   handle-heaviest suites, run before (b)–(d) were written).

---

# Regression round — the five "regressions" were a stale base, not this landing

The coordinator ran the seven-file `Tests/Chat/` cluster on both sides of the
branch and measured **`origin/dev` 4 failed / 531 passed** vs **this branch 9
failed / 525 passed**, and handed over five named failures as regressions from
the lifetime landing.

**All five are `origin/dev` repairs that landed AFTER this branch's base.** Not
one of them is caused by a line this branch wrote. The branch was cut at
`dcdcf2925` (PR #1648); PR #1647 — the ~200-commit "reconcile current dev
diagnostics" batch — merged into `dev` *after* it, and three of its commits fix
exactly these five tests.

## The decisive measurement

Run the five node-ids at the **merge-base itself** — `dcdcf2925`, pristine, with
zero bytes of this branch applied — in a throwaway detached worktree:

```
5 failed, 1 passed, 1 warning in 1.71s
FAILED …test_save_history_soft_deletes_messages_removed_from_resave
FAILED …test_save_history_without_ids_skips_variant_rows_in_positional_fallback
FAILED …TestChatHistorySaving::test_resave_chat_history
FAILED …test_context_change_before_first_admission_pauses_for_explicit_review
FAILED …test_local_marks_migrate_from_v16_to_v17_with_expected_schema
```

The same six node-ids at `origin/dev` `239ca3f33`: **`6 passed`**.

A failure that reproduces with the branch's changes *absent* cannot have been
introduced by them. The comparison the coordinator ran was branch-vs-`dev`-tip,
which conflates "what this branch changed" with "what `dev` gained meanwhile";
the branch-vs-**base** comparison separates them, and every one of the five
lands on the `dev`-gained side.

## Mechanism, per regression

| # | Test | Fixed on `dev` by | Mechanism |
|---|---|---|---|
| 1 | `test_save_history_soft_deletes_messages_removed_from_resave` | `63a1e6023` (production) | `save_history` updating a parent's content ran `update_message`'s recursive-descendants tombstone, soft-deleting `msg-assistant-1`; the loop's next iteration then hit `ValueError: Message msg-assistant-1 not found`. `dev` added `preserve_descendants=True` for the authoritative bulk resave. |
| 2 | `test_save_history_without_ids_skips_variant_rows_in_positional_fallback` | `63a1e6023` (production) | Same tombstone, reached through the positional-fallback branch. |
| 3 | `TestChatHistorySaving::test_resave_chat_history` | `63a1e6023` (production) | Same tombstone, through `Chat_Functions`' resave path. |
| 4 | `test_context_change_before_first_admission_pauses_for_explicit_review` | `e5d2acf6a` (test fixture) | The fixture drove a context change via `store.update_message_content(...)`, which is no longer a context-summary mutation; `dev` repointed it at `store.set_session_context_summary(...)`. |
| 5 | `test_local_marks_migrate_from_v16_to_v17_with_expected_schema` | `5300077fd` (test fixture) | The V16 fixture builds a current-schema DB then strips it back to V16; it did not strip the `note_folders` / `note_folder_memberships` tables introduced at V35→V36, so the migration assertion saw tables a real V16 DB cannot have. `dev` added the two `DROP TABLE IF EXISTS`. |

Regressions 1–3 are **production** bugs (fixed on `dev`, in a file this branch
never touches: `chat_persistence_service.py` / `ChaChaNotes_DB.py`).
Regressions 4–5 are **test fixtures** that had gone stale against `dev`
production, and `dev` repaired the fixtures. In neither case was the old
lifetime encoded, and no assertion was weakened anywhere.

## The #5 verdict: stale base, not order-dependence

`test_local_marks_migrate_from_v16_to_v17_with_expected_schema` was flagged as
possibly order-dependent (seen red on `dev` earlier in the session, green in the
seven-file run). It is neither ordering nor flake: it fails **alone** at
`dcdcf2925` and passes **alone** at `239ca3f33`, and the mechanism is a fixture
that must be taught about every table added after V16. The earlier "pre-existing
`dev` red" observation is the same staleness seen from a `dev` checkout that
predated `5300077fd`.

## The fix

`git merge origin/dev` into `feat/task-15860-runtime-lifetime` (`f6d31e76f`).
The merge is a **clean auto-merge** — `git merge-tree --write-tree HEAD
origin/dev` returned a tree with no conflict output before the merge was run —
despite `dev` having extracted `UI/Console_Modules/image.py` and `video.py` out
of `chat_screen.py` in the meantime. Verified after the merge:

- no `_console_chat_store = None` / `_console_provider_gateway = None` /
  `_console_chat_controller = None` / `_console_agent_bridge = None` assignment
  was reintroduced anywhere in `tldw_chatbook/` (the §2 hazard);
- every §2/§3/§4 seam survived: `ChatScreen._console_runtime`, the four
  read-write handle properties, `console_view_hooks`,
  `leave_console_runtime` at `on_unmount`,
  `uninstall_console_hands_free_store_tap`, and `app.py`'s
  `_shutdown_console_runtime` → `dispose_console_runtime`;
- `dev`'s two new Console modules read the repointed handles and never assign
  them, so the properties serve them unchanged.

No production code and no test was changed by this round. There was nothing to
fix in this branch.

## Mutations (attribution, not new tests)

No test was added or changed, so there is no new assertion to mutate. Instead
the three `dev` fixes were **reverse-mutated** on the merged branch to prove
they, and not anything in this branch, are what carries the five green. Every
restore was `sed`-based in place and proven with an empty `git diff`.

| # | Mutation | Died |
|---|---|---|
| A | `save_history`'s two `preserve_descendants=True` → `False` | regressions 1, 2, 3 — `3 failed` ✅ |
| B | prompt-queue fixture back to `store.update_message_content(...)` | regression 4 ✅ |
| C | V16 fixture's two `DROP TABLE IF EXISTS` removed | regression 5 ✅ |

B and C were installed together and killed both their targets in one run
(`2 failed`). `git diff` is empty and `grep -c "MUTATION-"` is `0` in all three
files.

## Gate

Runner both sides: `.venv/bin/pytest <paths> -p no:randomly -q --no-header -rf`.
`Tests/test_probe_import_provenance.py` is in every group it can be added to.
**Branch** = `f6d31e76f` (this branch merged with `dev` `239ca3f33`);
**dev** = `239ca3f33` in a throwaway detached worktree, same invocation, same
machine.

| Gate | dev `239ca3f33` | branch `f6d31e76f` | Verdict |
|---|---|---|---|
| the seven `Tests/Chat/` files + probe | 4 failed / 532 passed | **4 failed / 532 passed** | identical; the four are the known pre-existing reds |
| `Tests/Chat/` in full + probe | 14 failed / 5561 passed / 66 skipped (18:56) | **14 failed / 5575 passed / 66 skipped** (17:09) | **failure sets byte-identical** (`comm` both directions empty); +14 passed = this branch's new `test_console_runtime_lifetime.py` |
| `Tests/Agents/` + probe | 1418 + probe | **1419 passed / 0 failed** | identical |
| `Tests/UI/test_screen_residency.py` | 7 passed | **7 passed** | identical |
| `test_console_runtime_ownership.py` + `test_console_runtime_lifetime.py` | 2 passed (dev has neither the 4 added cases nor the file) | **6 + 14 = 20 passed** | as specified |
| wake suites — the specified glob, 16 files + probe | 110 passed | **110 passed** | identical; **the known `[size0]` red is GREEN on both** |
| wake-adjacent files the glob misses (`test_console_agent_fleet_sync_coalescing.py`, `test_probe_headless_wake_p1_continuity.py`, `test_probe_headless_wake_p2_p3_p4.py`) | 9 passed / 1 skipped | **9 passed / 1 skipped** | identical |
| `Tests/UI/test_console_mcp_approval.py` | 69 passed | **69 passed** | identical |

Not one failure anywhere in the battery is attributable to this branch.

The dev-baseline worktree (`.worktrees/devbase-regr`) was removed after the run.

`origin/dev` moved twice during this round: to `5e8b3724b` (docs-only, two
backlog files) and then to `a99aaf2fd` (media reading scope, selection dialogs,
console workspace — real code). **The gate above is pinned to `239ca3f33`** so
both columns describe the same tree; merging a moving tip mid-gate would have
invalidated an 18-minute measurement. Bringing the branch to the newer tip is a
pre-merge step for whoever opens the PR, not part of this attribution round.

Two report claims from §10 are also resolved by the merge: concern 2's
`test_console_video_capacity.py::test_real_unmount_path_invokes_pending_
artifact_drain` is in **neither** side's failure list (dev's Console
video-controller extraction fixed it), and the `[size0]` wake red is green.

## Concerns from this round

1. **A branch-vs-`dev`-tip test comparison cannot attribute a failure.** It
   measures the union of "what the branch broke" and "what `dev` fixed since the
   branch was cut". Five confident regression reports came out of that
   conflation. The attributing measurement is branch-vs-**merge-base**, and it
   costs one throwaway detached worktree.
2. The merge pulled in ~200 `dev` commits including a Console decomposition
   wave (`image.py`, `video.py`), the trajectory feature (ChaChaNotes v38), and
   the modal-dismissal boundary. It auto-merged cleanly and the gate below is
   green, but this branch has not been re-reviewed against that much new
   neighbouring code.
3. The wake-suite gate as specified (`Tests/Chat/test_fleet_*.py
   Tests/Chat/test_console_fleet_*.py Tests/UI/test_console_fleet_*.py`)
   collects **109** tests here, not the 177 quoted; the quoted figure must have
   come from a wider file set. Counts for the three wake-adjacent files that the
   glob misses are recorded in the gate table so the set is unambiguous.
4. The known `[size0]` red in `test_console_fleet_discoverability.py` is
   **green** after the merge — `dev`'s task-16220 compact-grid work fixed it.
