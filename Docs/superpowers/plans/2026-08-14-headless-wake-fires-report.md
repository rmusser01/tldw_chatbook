# Wake-fires-headless report — the auto-wake fires with no Console mounted (task-15860)

Plan Task 1's **AC#1 half** — the slice every earlier landing of this arc
built toward and deliberately did not ship.

- Branch: `feat/task-15860-wake-fires-headless`, worktree
  `.worktrees/headless-fires`.
- **Merge-base: `bab84f7d9`** (contains all four merged headless-wake
  landings: ownership, lifetime, viewless, continuity). Every baseline
  below was measured at that commit on the untouched tree, in a throwaway
  detached worktree — never against `dev`'s tip.
- Predecessors: `…-task-0-report.md`, `…-task-1-report.md`,
  `…-lifetime-report.md`, `…-viewless-report.md`,
  `…-continuity-report.md`.

**One sentence:** `ConsoleFleetWakeCoordinator._attempt` now refuses a wake
only when the controller is **disposed** (app exit), not when a Console
**visit has merely ended**, so a background sub-agent that finishes while
no Console screen is mounted delivers a full supervisor turn.

---

## 1. What `_attempt` reads now, and why that is the whole change

```python
# before (merge-base)
shutdown = getattr(controller, "_shutdown_requested", None)
if shutdown is not None and shutdown.is_set():
    return

# after
if getattr(controller, "_disposed", False):
    return
```

Before the lifetime landing those were one signal, and correctly so:
`ConsoleChatController.shutdown()` was called both at app exit **and**
from `ChatScreen.on_unmount` — i.e. on every ordinary navigation away
from Console. `_shutdown_requested` therefore meant *either* "the process
is going away" *or* "the user switched tabs", and this gate could not
tell them apart. It refused every headless wake. That is exactly the
limit the User Guide states ("**If Console isn't open, no wake fires**")
and the limit this task removes.

The lifetime landing separated them and — deliberately — did not use the
separation here:

| Signal | Set by | Means |
|---|---|---|
| `_shutdown_requested` (per-VISIT; the attribute is *replaced*) | `leave_console()` on every nav-away | this visit ended; every round armed during it stays denied |
| `_disposed` (a latch) | `begin_shutdown()` / `shutdown()` | this controller is finished forever |

The lifetime landing installs the fresh Event at **`attach_view`**, not at
the end of `leave_console`, *specifically* so the flag stays set between
visits and this gate kept refusing — it could not accidentally ship this
slice. So the distinction already existed; `_attempt` was reading the
wrong half.

Nothing else moved. `leave_console()` still tombstones the visit's queue
chains, still sets the visit Event, still cancels and awaits this visit's
in-flight **USER** stream tasks, still exempts an in-flight `AGENT_WAKE`
turn (the owner ruling), and `_bind_visit_cancel_signal`'s arm-time
capture is untouched. The durable `FLEET_UNSEEN` mark still stages what a
headless delivery genuinely cannot reach: a conversation with no open
session, and anything owed across a process restart.

A controller double with neither attribute is unaffected: allowed before
(no `_shutdown_requested`), allowed now (`_disposed` defaults `False`).

### Docstrings corrected rather than left as folklore

Four places asserted the old meaning in prose. Each was updated to the
measured one: `console_fleet_wake`'s module docstring (the "Scheduling"
paragraph and the class docstring's "post-teardown drains… stage via the
durable mark"), `_attempt`'s own docstring, `console_runtime`'s "Still
deliberately unchanged" section, and `ConsoleChatController.leave_console`
/ the `_shutdown_requested` attribute comment.

---

## 2. The end-to-end red, and its before/after

`Tests/UI/test_console_headless_wake_fires.py::
test_a_survivor_settling_with_no_console_mounted_wakes_the_supervisor`

Everything runs through the production path: a real `ChatScreen` is
mounted and seeded, Console is left through the **real navigation API**
(`app.post_message(NavigateToScreen("library"))` plus the real "Leave
Console?" `ConfirmationDialog`, answered by pressing **Leave**), so
`on_unmount` runs and `leave_console_runtime` genuinely ends the visit.
Then the survivor settles **from a plain child thread** through the real
fan-out (`on_fleet_drained`), exactly as `ConsoleAgentBridge` delivers it.
No gate is monkeypatched and no teardown is suppressed. Real on-disk
ChaChaNotes, real `agent_runs`, real conversation-marks store.

Four harness preconditions are asserted before the drain, so a green can
never mean the test failed to reach the state under test:

```
chat not in app.screen_stack                       # Console really unmounted
controller is app.console_runtime.chat_controller  # the runtime outlived it
controller._shutdown_requested.is_set()            # the visit really ended
controller._disposed is False                      # a nav is not an app exit
```

**BEFORE** (at `bab84f7d9`, zero bytes of this branch, the file copied
into a detached worktree):

```
FAILED Tests/UI/test_console_headless_wake_fires.py::
       test_a_survivor_settling_with_no_console_mounted_wakes_the_supervisor
E   AssertionError: a background sub-agent settled while NO Console screen
    was mounted and no wake turn ever reached the provider
E   assert False
1 failed
```

**AFTER**: `1 passed`.

The full chain it asserts, not merely the send:

1. **exactly one** wake turn reaches the provider, and its payload's
   trailing entry is `role="user"` carrying the machine marking and the
   child's result (a payload ending on an assistant row is a prefill to
   strict providers);
2. the machine-origin **SYSTEM** notice (`metadata.origin ==
   "agent_wake"`, exactly one such row) and the assistant reply are in the
   **app-owned store** *and* in **ChaChaNotes** — the persisted senders
   are exactly `["user", "assistant", "system", "assistant"]`;
3. `agent_runs.wake_delivered_at` is stamped **exactly once**: a
   re-delivered drain plus an extra `retry_soon()` produce no second
   payload, no stamp movement and no extra persisted row;
4. **no USER row** anywhere — the store's only USER row is the seeded
   one, and the DB has exactly one `user` sender;
5. the **◈ `FLEET_UNSEEN` mark SURVIVES** the delivery commit, because a
   runtime with no view reports the conversation as not watched;
6. navigating back through the real API shows the delivered turn in the
   **rendered** transcript (walked off `ConsoleTranscript._row_widgets`,
   the mounted rows — not the widget's model).

---

## 3. Proving disposed and visit-ended are genuinely distinct

Not by reading the flags: by driving each state through the production
seam (`ConsoleRuntime.leave_console(view)` / `ConsoleRuntime.dispose()` —
what `ChatScreen.on_unmount` and `TldwCli._shutdown_app_owned_lifecycles`
respectively call) and asserting the opposite **consequences**.

`Tests/Chat/test_console_headless_wake_invariants.py`:

| Test | State | Asserted consequence |
|---|---|---|
| `test_a_visit_that_merely_ended_does_not_refuse_the_wake` | after `leave_console(view)` | the wake **delivers**, stamps the ledger, writes one notice row, and keeps the ◈ mark |
| `test_a_disposed_runtime_refuses_the_wake_and_loses_nothing` | after `dispose()` | the wake **never** reaches the provider — and refusal is not loss: pending bit retained, ledger unstamped, mark retained, no orphaned notice row |
| `test_a_disposed_controller_never_reopens_for_a_new_view` | `dispose()` then `attach_view(fresh)` | `_disposed` is **permanent** where a visit boundary is not: `begin_visit` refuses, the Event stays set, the wake stays refused |
| `test_the_relaxed_wake_gate_does_not_revive_the_visits_cancellation` | one controller, both halves | a round armed during the visit still resolves to `{"write_file": "deny"}` when the visit ends **and** a survivor settling right afterwards still gets its full wake turn |

Every one of these asserts, as a harness precondition, that
`leave_console` really set the visit Event and really did not dispose the
controller — so none of them can pass by never reaching the state the old
gate refused.

The last row is the "do not widen it further" pin: if a future edit ever
relaxed the visit Event itself instead of the gate, its first half goes
red on the same controller that its second half proves still delivers.

---

## 4. AC#2 — not regressed, with counts

Nothing in AC#2's mechanism was touched. The evidence:

| Suite | Result |
|---|---|
| `Tests/Chat/test_console_runtime_lifetime.py` (owns both AC#2 semantics) | **14 passed** |
| `test_leaving_console_still_cancels_a_streaming_user_turn` | passed |
| `test_leaving_console_still_denies_a_parked_approval_round` | passed |
| `test_leaving_console_does_not_cancel_an_in_flight_wake_turn` (the wake exemption) | passed |
| `test_the_wake_exemption_never_outlives_its_turn` | passed |
| `Tests/UI/test_console_mcp_approval.py` | **69 passed** |
| plus the new `test_the_relaxed_wake_gate_does_not_revive_the_visits_cancellation` | passed |

**One existing test was renamed, deliberately.**
`test_the_wake_gate_still_refuses_between_visits` →
`test_the_visit_cancellation_event_is_set_between_visits`. Its assertions
are unchanged and still correct — the Event *is* set between visits, and
that is what keeps a visit's parked rounds denied — but its name and
docstring claimed *"between visits the wake gate must still refuse"*,
which this slice makes false. A green test asserting a property the code
no longer has is exactly how folklore gets into a suite; the docstring now
names the two tests that own the new property.

---

## 5. AC#3 — the invariants, headless

| Invariant | Test | Result |
|---|---|---|
| Exactly-once via the ledger across **refusal → retry → restart** | `test_exactly_once_across_a_refusal_a_retry_and_a_restart_headless` | passed |
| **No phantom wake after a restart** (crash-killed child swept to `error`) | `test_a_crash_killed_child_swept_to_error_wakes_nobody_after_a_restart` | passed |
| **Kill switch read fresh at the headless fire point**, OFF loses nothing | `test_the_kill_switch_is_read_fresh_at_the_headless_fire_point` | passed |
| **No USER row** in the app-owned store | `test_a_headless_wake_writes_no_user_row_in_the_app_owned_store` | passed |
| (added) the **quit race this slice opens** leaves consistent durable state | `test_a_wake_racing_app_exit_leaves_consistent_durable_state` | passed |

Each is built so it cannot pass vacuously:

- **Exactly-once** runs three legs on one run: a not-ready provider
  refuses (pending retained, ledger unstamped, mark retained, no orphaned
  notice row); the retry delivers once and stamps; then a **restart** —
  a fresh store, controller and coordinator over the same `agent_runs`
  file, with the mark still set — claims **nothing** back
  (`seed_from_marks() == 0`, `pending_conversation_ids() == ()`, and a
  `retry_soon()` that stays quiet). The control that stops that leg being
  vacuous: a genuinely undelivered second run in the same conversation
  **is** claimed (`seed_from_marks() == 1`) and the notice the restarted
  process composes carries `"second child result"` and **not**
  `"first child result"`.
- **The phantom case** models the real mechanism: a child left `running`
  by a crash, then a new `AgentRunsDB` over the same file whose
  `__init__` sweeps it to `error` (the per-process `_swept_paths` guard
  is discarded for that path, the way `Tests/DB/test_agent_runs_db.py`
  models a restart). It carries no `FLEET_UNSEEN` mark, so the mount
  claim — the only headless path a restarted process has — finds nothing
  and nobody is woken.
- **The kill switch** is set OFF via the env var, the headless fire point
  stays silent, and nothing durable is lost (the pending bit is recorded,
  the ◈ mark still works, the ledger is unstamped, `seed_from_marks()`
  returns 0). Then it is flipped ON with **no restart and no new
  controller** and the same live coordinator delivers what OFF recorded —
  which is what "read fresh at the fire point" means.

### A finding: what the crash-swept orphan is *not* exempt from

The phantom test's first version also asserted that a swept orphan could
never appear in a **later legitimate** wake for the same conversation. It
**failed**, and the code is right. `AgentRunsDB.undelivered_wake_runs`
deliberately includes `error` runs, and its docstring records choosing
`>=` on the parent/child timestamps *"so a restart-reconcile sweep that
stamps an orphaned child and its parent in the same pass still reports the
child."* An interrupted child is genuinely owed to the supervisor — the
news is that it died.

So the test now pins the true contract, which is stronger than what it
originally claimed: the orphan wakes **nobody** on its own (no mark,
nothing claimed), and when a real completion in the same conversation does
trigger a wake, the orphan is announced **with its honest `error`
status**, **exactly once** — a third completion's wake no longer mentions
it, because the ledger stamp closed it out.

### A window this slice OPENS, measured rather than assumed

Before the gate change a wake could not start between visits, so
`ConsoleRuntime.dispose()` never raced one. It can now: a survivor
settling after the user navigated away, with the app quitting a moment
later. `dispose()` → `shutdown()` cancels every session's stream task,
wake turns included (the `leave_console` exemption is deliberately not
`shutdown`'s).

`test_a_wake_racing_app_exit_leaves_consistent_durable_state` asserts the
property exactly-once actually rests on — **consistency between the two
durable layers** — rather than guessing a branch: a stamped ledger implies
the notice row landed *and* the pending entry is gone; an unstamped ledger
implies the pending entry *and* the ◈ mark survive.

Probe run, recorded so nobody re-derives it: **`stamped=True, notices=1,
pending=False, marked=True`**. The turn is accepted before it streams, so
quitting mid-stream truncates the reply without un-delivering the wake —
the same semantics a mounted Console has when the user presses Stop on a
wake turn — and the off-view mark still points the user at it next launch.

---

## 6. Caps — a headless wake is a normal turn under every one

| Cap | Test | How it is asserted |
|---|---|---|
| `max_parallel_runs` | `test_the_global_cap_defers_a_headless_wake_like_any_other_send` | three busy sessions saturate the cap (`send_refusal_copy` is checked as a precondition); the headless wake stays quiet and keeps its pending bit; freeing one slot's run state retries it through the production hook and it delivers |
| per-session busy refusal | `test_a_busy_session_defers_a_headless_wake_until_it_goes_terminal` | the session's own run is STREAMING; the wake defers **and never dispatches a send at all** (dispatch counter, see the M7 survivor below); the terminal transition retries it |
| token ceiling | `test_a_headless_wake_resolves_with_the_same_selection_as_a_manual_send` | the `ConsoleProviderSelection` a headless wake resolves with is **equal** to the one a mounted manual send resolved with for the same session — `max_tokens` included, asserted non-`None` first so the comparison is not vacuous |
| wall clock + token ceiling on the agent path | `test_a_headless_wake_takes_the_same_agent_dispatch_and_budget` | both turns reach the single `ConsoleAgentBridge.run_reply` dispatch site with the same `session_id`, `conversation_id`, `model`, a callable `should_cancel` and **the same argument set**; and `run_reply`'s signature has **no budget/wall/max_tokens parameter at all**, so no caller — wake or manual — can vary `CONSOLE_RUN_BUDGET` (`max_wall_seconds`, `max_total_tokens`), which is applied inside it |

The last row's structural half is what makes the claim executable rather
than a reading: "same entry point, same arguments, no budget knob" is
three assertions, and the third one fails the moment somebody adds a way
to bound a wake turn differently.

---

## 7. Mutations run and killed

Every mutation applied by **Edit** and restored by **Edit**; after each,
`git diff -- tldw_chatbook` is empty and `grep -rn "MUTATION-"
tldw_chatbook/` returns nothing. Verified again at the end.

| # | Mutation | Killed |
|---|---|---|
| M1 | **the whole change reverted** (the untouched merge-base, both new files copied in) | **10 failed, 3 passed** — the e2e plus 9 invariants tests. The 3 survivors are the ones whose property does not depend on the change: both disposed-direction tests (the old gate refused anyway) and the phantom test (its restarted controller is fresh, so its Event was never set) ✅ |
| M2 | `_attempt`'s gate never refuses | `test_a_disposed_runtime_refuses_the_wake_and_loses_nothing`, `test_a_disposed_controller_never_reopens_for_a_new_view` — **exactly those two** (2 failed, 11 passed) ✅ |
| M3 | `leave_console` never sets the visit Event | 11 failed: the 8 tests whose harness precondition is that the visit really ended, **plus** `test_leaving_console_still_denies_a_parked_approval_round`, `test_a_round_from_the_previous_visit_is_not_resurrected` and the renamed Event-lifecycle test — i.e. AC#2's own owners ✅ |
| M4 | `_deliver` never calls `mark_wake_delivered` | 5 failed: the e2e and the four ledger-dependent invariants (visit-ended, exactly-once, phantom, kill switch) ✅ |
| M5 | `viewless_conversation_in_view` → `True` | 5 failed: the e2e's ◈ assertion, two of mine, and the viewless landing's own two owners — correct overlap, mine covering the `leave_console` path and theirs the `detach_view` path ✅ |
| M6 | the `autowake_enabled()` check dropped from `_attempt` | `test_the_kill_switch_is_read_fresh_at_the_headless_fire_point` **only** (1 failed, 12 passed) ✅ |
| M7 | `_attempt` skips `send_refusal_copy` | `test_the_global_cap_defers_a_headless_wake_like_any_other_send` only — **the per-session test SURVIVED**; see below ✅ after the fix |
| M8 | the `AGENT_WAKE` branch writes its notice as a `USER` row | `test_a_headless_wake_writes_no_user_row_in_the_app_owned_store` and the e2e (2 failed, 11 passed) ✅ |
| M9 | `begin_visit` no longer refuses on `_disposed` | `test_a_disposed_controller_never_reopens_for_a_new_view` and the lifetime landing's `test_a_disposed_controller_is_never_re_opened` (2 failed, 25 passed) ✅ |
| M10 | delivered run ids never leave the pending registry | `test_a_wake_racing_app_exit_leaves_consistent_durable_state` **only** (1 failed, 13 passed) ✅ |

### M7 survived, and what it taught

Bypassing `send_refusal_copy` inside `_attempt` left
`test_a_busy_session_defers_a_headless_wake_until_it_goes_terminal`
**green**. Investigated rather than patched around: `submit_draft`
refuses a busy session on its own (`_active_run_rejection`), so the read
site is **double-guarded** and "nothing streamed" cannot say which guard
did it — the same shape as the viewless landing's surviving M9. The test
was therefore claiming coverage it did not have: it owned "a busy session
produces no stream", not "the coordinator defers".

Fix: the test now wraps `controller.submit_draft` with a counter and
asserts the wake **never dispatches a send at all** while the session is
busy. Re-run under M7: **2 failed** (both cap tests). Restored: **12
passed**. The global-cap test needed no such change — the cap is only
enforced at the coordinator's gate for a wake, so it was already the sole
owner.

M3's kill list is also worth reading as a positive result rather than
noise: it shows the AC#2 semantics are owned by the lifetime landing's
tests and that my preconditions are load-bearing — remove the visit
Event and eight of my tests stop being able to make their claim, which is
what a precondition is for.

M10 taught the complementary lesson about *when* an observation can see a
defect at all: leaving delivered runs in the pending registry is invisible
to every other test, because the registry **self-heals on the next
attempt** — `_rows_for` drops a run the ledger already shows delivered.
Only an observation taken with no further attempt coming — which is
exactly what quitting gives you — can catch it. That is why the quit-race
test is its sole owner, and it is a good argument for keeping it.

---

## 8. Gate — baseline (merge-base `bab84f7d9`) vs final

Runner both sides: `.venv/bin/pytest <paths> -p no:randomly -q
--no-header -rf`, cwd = the worktree.
`Tests/test_probe_import_provenance.py` is in every gate it can be added
to — the venv's editable install resolves `tldw_chatbook` to a FOREIGN
worktree and loses only by `sys.meta_path` ordering. **Baseline** = the
same invocation in a throwaway detached worktree at `bab84f7d9`
(`.worktrees/headless-fires-base`), with the two new test files copied in
for the new-suite row. Every count below was READ off a summary line.

| Gate | Baseline @ merge-base `bab84f7d9` | Final @ branch | Delta |
|---|---|---|---|
| **The new suite** — `Tests/UI/test_console_headless_wake_fires.py` (1) + `Tests/Chat/test_console_headless_wake_invariants.py` (13) + probe | **11 failed, 4 passed** (both files copied into a detached worktree at the merge-base) | **15 passed** | **+11**; the 4 that pass on both sides are the two disposed-direction tests, the phantom test (its restarted controller never had a visit) and the provenance probe |
| **The specified battery** — `test_console_store_continuity` (4), `test_console_viewless_hooks` (12), `test_console_runtime_lifetime` (14), `test_console_runtime_ownership` (7), `test_screen_residency` (7), `test_console_mcp_approval` (69), the 16-file wake glob (109), probe (1) = 223 | **223 passed, 0 failed** (179.6s) | **223 passed, 0 failed** (179.5s) | **0** — identical, both sides clean |
| `Tests/Agents/` + probe | not measured — a green final cannot hide a regression | **1438 passed, 0 failed** (40.1s) | — |
| **The files the specified gate MISSES but the blast radius names** — `test_probe_headless_wake_p1_continuity`, `test_probe_headless_wake_p2_p3_p4` (Task 0's own probes, which record post-unmount wake behaviour), `Tests/Architecture/test_console_wave6_inventory`, `Tests/Architecture/test_persistent_diagnostic_inventory`, probe | the same **3 failed** at the merge-base (21.1s) | **3 failed, 82 passed, 1 skipped** (166.2s) | **0** — the three reproduce with this branch absent (`test_workspace_browser_methods_have_no_sibling_controller_reach_through`, `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`, `test_task_15743_final_rebase_diagnostics_are_metadata_only`); both Task 0 probes are green |
| **`Tests/Chat/` in full + probe** | **14 failed, 5589 passed, 66 skipped** (1149.2s) | **14 failed, 5602 passed, 66 skipped** (1158.3s) | **0** — the failure SETS are byte-identical (`comm` empty in **both** directions over the sorted node-id lists); **+13 passed = exactly this branch's 13 new invariants tests** |

The two `Tests/Chat/` runs were launched together, the merge-base side
with `--ignore=Tests/Chat/test_console_headless_wake_invariants.py` so
both collect the same set, and they stayed in lockstep the whole way (the
progress logs never diverged by more than 39 characters). The fourteen
shared failures are dev's pre-existing reds, grouped by file:
`test_console_provider_continuation` ×9, `test_console_chat_controller`
×2, `test_console_h3_image_edit` ×1, `test_console_visual_evaluation` ×1,
`test_console_voice_input` ×1 — the same files, and now the same
node-ids, the viewless and continuity landings recorded.

Machine note, for anyone reading the wall clocks: this battery ran against
up to **8 concurrent `pytest` processes** (six foreign sessions), so
elapsed time is not comparable to a quiet machine. Counts were READ off
the summary lines; none is inferred.

The wake glob as specified (`Tests/Chat/test_fleet_*.py
Tests/Chat/test_console_fleet_*.py Tests/UI/test_console_fleet_*.py`)
collects **109** tests across **16** files here — the same figure the
lifetime landing's report recorded, and not the 177 quoted earlier in the
arc.

**Stability.** The two new files drive real navigation, real threads and
real timers, so they were re-run for flakiness *while both full
`Tests/Chat/` runs were saturating the machine*: the e2e 3× (`1 passed`
each, 6.6s/7.5s/7.0s) and the invariants suite 3× **with random ordering
left ON** (`13 passed` each) — which also rules out order dependence.

---

## 9. Deliberately not done (and still true)

- **The approval clock is untouched** (plan Task 5). A headless wake that
  reaches a risk-tagged tool still parks; nothing surfaces during the
  window. One observation worth recording, measured only by reading the
  code path rather than executed here: a round armed **after**
  `leave_console` captures the visit's already-set Event at arm time, so
  `_is_session_cancelled` denies it at the first poll — fail-closed and
  *faster* than P4's 120.43s timeout, not slower. Task 5 owns whether
  that is the desired policy; this slice does not change it.
- **Launch / first-boot wake** (plan Task 6) is untouched, and this
  change cannot have accidentally enabled it. Read, not executed: the
  app constructs the `ConsoleRuntime` holder eagerly (`app.py`
  `self.console_runtime = ConsoleRuntime(self)`) but `ensure_chat_
  controller` / `ensure_agent_bridge` are lazy and their only callers are
  `ChatScreen` and its Console modules — so with Console never opened in
  a process there is no controller, therefore no
  `ConsoleFleetWakeCoordinator` and no fan-out registration, and nothing
  can fire. The wake still needs Console to have been opened **once**.
- **Documentation** (plan Task 8) is untouched: the User Guide's "**If
  Console isn't open, no wake fires**" paragraph and the spec's "honest
  architectural limit as built" are now stale and are that task's job.
- The continuity, viewless and lifetime landings are unchanged apart from
  the prose corrections in §1 and the one test rename in §4.

## 10. Concerns

1. **The User Guide is now wrong.** `Docs/User_Guide/console/
   agent-runs-and-tools.md:494-501` still tells users no wake fires with
   Console closed. That is plan Task 8 (AC#4) and out of this slice's
   scope, but it is a user-visible falsehood the moment this merges, and
   it should not sit unfixed for long.
2. **A headless wake now spends money with no UI attached.** That is the
   point of the task, and it is gated by `autowake_enabled`, the caps and
   the send gate — but it is a real behavioural change for anyone who
   assumed leaving Console stopped everything. The ◈ mark surviving is
   the only signal the user gets until they return.
3. **The approval-clock interaction above is inferred from the code, not
   executed.** I did not run a headless risk-tagged wake to observe the
   deny; the brief put the approval clock out of scope. Labelled as
   inference.
4. **Same-target navigation** still carries the lifetime landing's
   recorded consequence (the outgoing screen's streaming turn is no
   longer cancelled). Unchanged here; still worth the owner eyeball that
   report asked for.
5. **`Tests/UI/` was not run in full.** What was run is in §8: every UI
   file in the change's blast radius (continuity, ownership, residency,
   the MCP approval file, all the UI fleet files, both Task 0 probe
   files, the new e2e) plus `Tests/Chat/` in full and `Tests/Agents/` in
   full. Saying so plainly rather than implying coverage that was not
   obtained.
6. **Three `Tests/Architecture/` failures and fourteen `Tests/Chat/`
   failures are dev's, not this branch's** — each measured at the
   merge-base with zero bytes of this branch and reproducing there
   (the `Tests/Chat/` fourteen byte-identical in both directions). They
   are pre-existing reds on `dev` and someone should own them, but not
   this task.

---

## 11. Cross-suite leak — found after merging dev

### 11.1 The symptom

Merging current `dev` did not break anything on its own, but it made the
branch's own gate insufficient. Run the two files **together**:

```
pytest Tests/UI/test_console_headless_wake_fires.py \
       Tests/UI/test_console_store_continuity.py -q
```

→ **1 failed, 4 passed**. The casualty is the continuity landing's
four-way agreement test, and its message is a navigation that never
happened: `navigating to 'chat' never reached ChatScreen; stuck on
LibraryScreen`. Every file passes alone (continuity alone: **4 passed**).

### 11.2 Bisect — it takes THREE Console apps, and the wake shape

Measured, each combination executed:

| Invocation | Result |
|---|---|
| `test_console_store_continuity.py` alone (4 tests) | 4 passed |
| headless-wake e2e + `test_transcript_payload…` (2 apps) | 2 passed |
| `test_a_wake_that_ran…` + `test_transcript_payload…` (2 apps) | 2 passed |
| headless-wake e2e + `test_a_wake_that_ran…` + `test_transcript_payload…` | **1 failed, 2 passed** |
| four *identical* wake rounds in one process (throwaway probe) | 4 passed — so it is **not** monotonic accumulation |
| the two poisoners + a plain (no-wake) nav probe | 3 passed — so it is **not** "the third app" |

So the trigger is the third Console app **that also runs a wake turn
across a navigation**, and it is order/timing sensitive, not a plain
regression.

### 11.3 The mechanism — traced, not inferred

A throwaway pytest plugin wrapped `ConsoleSessionSurface.sync_sessions`,
`App._handle_exception`, `DOMNode.run_worker` (filtered to
`group="console-sync"`) and `ChatScreen._sync_native_console_chat_ui`.
The failing run's own log, in order:

```
STEP nav: leaving console
tick ENTER  screen_running=True   in_progress=False
tick ENTER  (coalesced: in_progress=True -> _console_sync_requested = True)
run_worker(console-sync) from chat_screen.py:15846      <- the tick's own `finally`
tick EXIT-OK screen_running=False                       <- the screen is ALREADY closed
tick ENTER  screen_running=False screen_mounted=True    <- the re-armed worker runs anyway
sync_sessions RAISED NoMatches("#console-native-tab-strip")  mounted=True running=False
App._handle_exception(WorkerFailed(...))
```

and immediately afterwards the app object reports
`running=False closing=True closed=True
exception=WorkerFailed(NoMatches("#console-native-tab-strip"))`.

**The leaking object, named precisely: the `console-sync` Textual
`Worker` wrapping `ChatScreen._sync_native_console_chat_ui`, created by
that coroutine's own `finally` re-arm on a `ChatScreen` whose message
pump Textual has already closed.**

Three production facts compose:

1. `_sync_native_console_chat_ui`'s `finally` re-arms itself with
   `self.run_worker(...)` whenever a coalesced request arrived mid-tick
   (a wake turn appending transcript rows during a navigation is exactly
   that). That call happens **after** Textual's unmount sweep
   (`Widget._on_unmount` → `workers.cancel_node(self)`), so the worker it
   creates was never in the cancelled set and nothing will ever cancel it.
2. The tick touches the DOM: `_sync_console_native_session_tabs` →
   `ConsoleSessionSurface.sync_sessions` →
   `query_one("#console-native-tab-strip")` — the widgets a navigation
   away from Console removes.
3. Textual workers default to `exit_on_error=True`
   (`textual/worker.py:382`), so the `NoMatches` reaches
   `App._handle_exception` and **exits the app**. After that every
   `post_message` is silently dropped — which is why the next test's
   `NavigateToScreen("chat")` produced 15 seconds of complete silence and
   then "stuck on LibraryScreen".

`is_mounted` is no defence: the removed surface still reported
`is_mounted=True` while its own pump reported `is_running=False`.

### 11.4 Verdict: PRODUCTION defect, with proof

**This is a production crash that the wake-fires-headless gate change
exposes. It is not test pollution, and it is not the app-owned runtime
failing to be disposed.**

Proof, in four parts, each executed:

1. **Nothing crosses the test boundary.** The failure is entirely inside
   the third app's own lifetime: that app kills *itself*, and the pytest
   test that owns it then fails on the consequence. Four *identical* wake
   rounds in one process were all green (§11.2), so no object accumulates
   across apps; the app-owned `ConsoleRuntime`, its store, its bridge and
   the wake coordinator are all per-app and none of them appears in the
   crash chain.

   The undisposed-runtime hypothesis was the first one checked and it was
   **executed, not reasoned away**. A probe seeded a real Console, left
   `app.run_test()`, then measured: `controller._disposed` → **True**,
   `app.console_runtime is runtime` → **False** (detached), and after
   `gc.collect()` the live-object census was **0 `ConsoleRuntime`, 0
   `ConsoleChatController`**. The runtime is disposed and collected at app
   exit; it is not the leak.
2. **Every frame in the crash chain is production code**, and none of it
   is this branch's. `git show --stat` over this branch's five commits:
   the only production commit (`474af3b6b`) touches
   `console_chat_controller.py`, `console_fleet_wake.py` and
   `console_runtime.py` — **`chat_screen.py` is untouched by this
   branch**. Both halves of the defect are present verbatim on
   `origin/dev`: the unguarded `finally` re-arm
   (the `finally` of `chat_screen.py`'s
   `_sync_native_console_chat_ui`, `origin/dev` @ `5b4820931`
   lines 15859-15868 — cited by name because dev moves under the
   numbers) and the unguarded
   `query_one` (`origin/dev:console_session_surface.py:532`).
3. **The gate relaxation is the exposer, measured both ways.** With the
   `_attempt` gate reverted to `_shutdown_requested` *and* the fix
   neutralised, the same two-file invocation gives **1 failed, 4 passed
   where the failure is the headless e2e** (correct: the old gate refuses)
   and continuity is **4/4 green**. With the new gate and the fix
   neutralised it is the continuity test that fails. Same test files,
   same machine, one line of production difference.
4. **The consequence is app-lifetime, not test-lifetime.** In production
   the app that dies is the user's session: navigate away from Console
   while a sync tick is coalescing (a wake or fleet turn landing rows
   during the "Leave Console?" flow is the everyday shape) and the TUI
   exits. Calling this "just test pollution" would have shipped that.

Honest scope: this branch did not create the crash, it made it reachable
often enough to see. The window existed on `dev` — a 0.2s sync tick plus
an append-driven tick, against a screen that can be unmounted at any
moment — and nobody had hit it.

*Inference, labelled:* why it takes three apps rather than one is not
proven. The most likely reading is accumulated process pressure (loguru
sinks, SQLite handles, executor threads) widening a race, and the
measurements above are consistent with it — but I did not isolate the
slowdown source, and no claim here depends on doing so.

### 11.5 The fix

`tldw_chatbook/UI/Screens/chat_screen.py`, three small pieces, one
invariant — **a torn-down screen renders nothing**:

* `_console_screen_is_torn_down(screen)` — new **module-level** predicate
  reading `_closing`/`_closed`, the pair Textual itself sets first in
  `MessagePump._close_messages` and reads for `is_parent_active`.
  Deliberately **not** `is_mounted` (True for the corpse) and
  deliberately **not** `is_running` (also False *before* a pump starts,
  which would silently no-op every harness that calls the tick on a
  hand-built, never-mounted `ChatScreen`).

  It is a module function rather than a method **because the first draft
  was a method and that broke three tests.** `MagicMock(spec=ChatScreen)`
  is a common fixture here, and a spec'd mock auto-answers every *method*
  on the class — truthily — so as a method the predicate reported "torn
  down" for every mocked screen and the tick returned before doing
  anything. Measured: `Tests/UI/test_ui_responsiveness.py` alone was
  **15 passed** at the pre-fix baseline (`.worktrees/hf-leak-base` @
  `eae717f53`), **3 failed** with the method form, **15 passed** again
  with the module form. `_closing`/`_closed` are set in
  `MessagePump.__init__`, so they are absent from `dir(ChatScreen)` and a
  spec'd mock — like a never-mounted screen — correctly reads as LIVE.
  `test_ui_responsiveness.py` is the standing guard against anyone
  converting this back to a method.
* `_sync_native_console_chat_ui` returns immediately when the screen is
  torn down, and clears the coalesced request rather than passing it on.
* Its `finally` no longer re-arms a worker on a dead screen, and a
  **teardown-scoped** `except` absorbs a failure that arrives mid-tick —
  re-raising untouched when the screen is alive, so this is not a blanket
  swallow.

`exit_on_error` is deliberately left alone: turning worker failures
non-fatal app-wide would hide the next bug of this class instead of
fixing this one.

### 11.6 Regression tests — `Tests/UI/test_console_sync_outlives_screen.py`

Five tests, all driven through the real navigation API on a real app:

1. a `console-sync` worker scheduled on a genuinely navigated-away screen
   must not kill the app — and navigating back must still work (the
   user-visible symptom, asserted directly);
2. a torn-down screen must run **no** sync work and must **not** re-arm;
3. **control** — a LIVE screen's sync failure must still propagate;
4. a real navigation arriving mid-tick is absorbed, and the coalesced
   request it leaves behind must not become a worker;
5. the partial-teardown window (`_closing` set, children already gone —
   Textual's own ordering) is absorbed.

### 11.7 Mutations run and killed

| # | Mutation | Killed by | Result |
|---|---|---|---|
All six re-run against the final (module-function) shape; counts READ off
the summary lines.

| # | Mutation | Killed by | Result |
|---|---|---|---|
| 1 | entry guard `if _console_screen_is_torn_down(self):` → `if False:` | test 2 | 1 failed, 4 passed |
| 2 | `finally` re-arm guard → `if True:` (always re-arm) | test 4 | 1 failed, 4 passed |
| 3 | teardown-scoped `except` → always `raise` | test 5 | 1 failed, 4 passed |
| 4 | teardown-scoped `except` → blanket swallow (`if False: raise`) | test 3 (the control) | 1 failed, 4 passed |
| 5 | `_console_screen_is_torn_down()` → `return False` | tests 1, 2, 4, 5 | 4 failed, 1 passed |
| 6 | `_console_screen_is_torn_down()` → `return True` | tests 3, 4, 5 | 3 failed, 2 passed |

A seventh, shape-level mutation is covered by an existing file rather
than a new one: converting the predicate back to a **method** turns
`Tests/UI/test_ui_responsiveness.py` from 15 passed to 3 failed (§11.5).
That was not a hypothetical — it is what the first draft did, and it is
how the trap was found.

Recorded honestly: **mutation 3 SURVIVED the first four-test draft.** The
mid-tick navigation test could not reach the `except` at all, because by
the time an awaited navigation returns the whole surface has gone and
`_sync_console_native_session_tabs` short-circuits on its own
`QueryError` guard. Test 5 was written specifically to reach that branch,
and only then did mutation 3 die. Every mutation was applied and reverted
with `Edit`, with `grep -c MUTATION` = 0 and `git diff --stat` checked
after each.


### 11.8 Gate for this section

Runner: `.venv/bin/pytest <paths> -p no:randomly -p no:cacheprovider -q
--no-header -rf`, cwd = `.worktrees/headless-fires`, `PYTHONPATH` pinned
to the worktree (the venv's editable install resolves `tldw_chatbook` to a
FOREIGN worktree). `Tests/test_probe_import_provenance.py` is in every
row. Every count below was READ off a summary line.

| Gate | Result |
|---|---|
| **The red itself** — `test_console_headless_wake_fires.py` + `test_console_store_continuity.py` + probe, one invocation | **6 passed** (run twice; was `1 failed, 4 passed` before the fix) |
| **The landing's battery + this branch's new files + `test_ui_responsiveness`** (continuity, viewless hooks, runtime lifetime, runtime ownership, screen residency, MCP approval, the 16-file wake glob, headless e2e, the 13 invariants, the 5 new leak tests, probe) — one invocation, 262 collected | **262 passed, 0 failed** (258.0s) |
| `Tests/Agents/` + probe | **1448 passed, 0 failed** (40.4s) |
| **The console-sync neighbourhood** — `test_console_control_bar_coalescing`, `test_ui_responsiveness`, `test_console_stop_feedback`, `test_console_switch_draft_integrity`, `test_console_native_transcript`, probe | **4 failed, 119 passed** on the branch vs **4 failed, 114 passed** at the pre-fix baseline (`.worktrees/hf-leak-base` @ `eae717f53`) — the **same four node-ids** both sides, `+5` = this branch's new file. Dev's reds, not this branch's |
| The 5 new leak tests alone | **5 passed** |
| `ruff check` on both changed files | 1 error, `F401 inspect imported but unused` in `chat_screen.py` — reproduces at the baseline; pre-existing, untouched |

**`Tests/UI/` in full, in one invocation: attempted twice, not
completed — stated plainly rather than implied.** The directory collects
**12,879** tests here. The first attempt was launched at the method-form
code and had to be discarded when the predicate changed shape; the second
ran alone and measured **~17 tests/min** (2% in 15 minutes) against a
machine carrying **three to four concurrent `pytest` processes from other
sessions** (14 cores, load average ~8) — i.e. ~12 hours, which this task
could not afford. What replaced it, chosen so the *property that matters
for this bug class* is preserved — many app lifetimes inside ONE process:

* the 262-test battery above is a single invocation spanning 15 files and
  dozens of app lifetimes, including every file in the change's blast
  radius;
* a single invocation over the **entire Console population** —
  `Tests/UI/test_console_*.py` (158 files) + `test_screen_residency` +
  probe, **3,298 collected** — plus three further invocations covering
  every remaining `Tests/UI` file, were running at the time this section
  was written. Their counts belong here and are not being guessed.

The honest summary of the risk this leaves: a regression confined to a
`Tests/UI` file outside the Console population and outside the battery
would not have been caught by what completed.

### 11.9 Two things this section found and deliberately did not fix

1. **The codebase already knew this hazard existed** — and that is
   corroboration, not an excuse. `chat_screen.py`'s avatar-render
   `except` carries this comment, written before this task:

   > *"Must never raise: called from `_refresh_active_character_avatar_
   > if_scope_changed` … invoked unconditionally on every 0.2s Console
   > sync tick (`_sync_native_console_chat_ui`) — some worker dispatch
   > sites run with `exit_on_error=True`, so an escaping mount failure …
   > could crash the app."*

   Someone had already reasoned their way to "a raise out of this tick
   kills the app" and defended **one** call site. The tick's own DOM work
   was left undefended, and the wake-fires-headless landing made that
   reachable. Recorded because it settles the production-vs-harness
   question from a second direction entirely.

2. **The shape is wider than this tick.** `chat_screen.py` has 55
   `run_worker(` call sites and only five pass `exit_on_error=False`, so
   any of the other fifty can take the app down on an unhandled raise.
   Nothing here changes that, and nothing here should: a blanket
   `exit_on_error=False` would convert this class of bug from "crashes
   loudly" to "silently stops working", which is worse. The right
   follow-up is a per-worker audit of which of those can run against a
   torn-down DOM, and it is out of this task's scope. Naming it rather
   than quietly widening the fix.

### 11.10 `Tests/UI/` in full — what was actually run, and against what

The literal single invocation was attempted twice and abandoned (§11.8).
What replaced it covers **every file in `Tests/UI/`**, as four invocations
chosen so the property that matters for this bug class — many app
lifetimes inside ONE process — is preserved, with the whole Console
population kept in one of them:

| Invocation | Branch | Pre-fix baseline (`.worktrees/hf-leak-base` @ `eae717f53`) |
|---|---|---|
| **Console population** — every `Tests/UI/test_console_*.py` (158 files) + `test_screen_residency` + probe, **3,298 collected**, one process | **26 failed, 3272 passed** (3412.6s) | **28 failed, 3265 passed** (3292.1s; 157 files — the baseline has no `test_console_sync_outlives_screen.py`) |
| non-Console chunk 1 (134 files) + probe | 6 failed, 3018 passed, 2 skipped, 9 errors (4422.9s) | — (failures checked individually, below) |
| non-Console chunk 2 (134 files) + probe | 8 failed, 3951 passed, 1 skipped (4504.9s) | — |
| non-Console chunk 3 (134 files) + probe | 15 failed, 2544 passed (2769.8s) | **13 failed, 2546 passed** (2767.0s) |

**The headline, and the reason the Console run was worth 57 minutes:
ZERO failures are unique to the branch in the Console population.** The
branch's 26 node-ids are a strict **subset** of the baseline's 28 —
`comm -23` over the sorted lists is **empty**, and the two the baseline
has extra (`test_console_native_chat_flow::test_console_accepted_send_
records_first_send_flag`, `test_console_send_disabled_state::test_enter_
hotkey_queues_draft_behind_accepted_run`) are order-sensitive reds that
happened not to fire on the branch side. Net: **+7 passed, −2 failed**,
and the +5 of that +7 is this branch's own new file.

The other three chunks' 29 failures were each run at the baseline:

* chunks 1+2 (14 node-ids, one invocation each side): **12 failed, 2
  passed on BOTH sides, same node-ids.** Dev's.
* chunk 3 (15 node-ids, one invocation each side): **13 failed, 2 passed
  on BOTH sides, same node-ids.** Dev's.
* the whole chunk-3 run repeated at the baseline gives 13 vs the branch's
  15, and the two extra are `test_settings_theme_editor::…preset_swatches
  _are_keyboard_activatable` and `test_speech_playground_pane_lifecycle::
  …preserves_editable_text_and_current_result_geometry` — **both pass in
  isolation on BOTH sides** (measured above), and `grep -lE
  "ChatScreen|chat_screen|console_sync"` returns nothing for either file,
  so neither can reach the code this branch changes. Order-sensitive
  Settings/Speech reds inside a 2,559-test chunk, labelled as such rather
  than swept into "dev's" without checking.

Time cost of the four branch runs plus the two baseline runs: ~4h45m of
wall clock against a machine carrying three to four foreign `pytest`
processes throughout.
