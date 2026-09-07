# Viewless report — every hook slot's viewless default (task-15860, plan Task 4)

Branch `feat/task-15860-viewless-hooks`, worktree `.worktrees/headless-viewless`,
merge-base `84a8f3ffc` (dev at start; dev moved to `48ad9e7de` during the work —
every failure below is measured at the MERGE-BASE, never at dev's tip).

Plan: `Docs/superpowers/plans/2026-08-14-headless-wake.md` Task 4.
Predecessors this builds on, unchanged: Task 0's probes
(`…-task-0-report.md`, P3) and the lifetime landing (`…-lifetime-report.md` §3,
which shipped `attach_view`/`detach_view` over one enumerated
`CONSOLE_VIEW_HOOK_SLOTS` list and deliberately left
`wake_conversation_in_view` at `None`, naming this task as its owner).

## What changed, in one sentence

The lifetime landing made every slot **clearable**; this landing makes each
cleared value **semantically correct**, records *why* next to the value, and
adds the one thing a cleared value cannot express — a re-arm.

## The viewless default for EVERY slot, and its justification

`ConsoleViewHookSlot` gained a `why` field: the production read site that makes
the value correct. The rule it encodes: **`None` is allowed only where the read
site's own guard turns `None` into the semantically right behaviour (inert, or
fail-closed). Where the guard's fallback is a WRONG answer, the default must be
an explicit callable.** A test asserts every slot carries a `why`.

| Slot | Target | Viewless default | Why that is right |
|---|---|---|---|
| `wake_conversation_in_view` | controller | **`viewless_conversation_in_view` → False** | **CHANGED.** `_conversation_in_view` reads an *unwired* probe as **IN VIEW** and CLEARS the ◈ FLEET_UNSEEN mark. `None` therefore made a delivery nobody could have watched commit as "seen". Safety red #1. |
| `wake_user_priority_probe` | controller | **`viewless_user_priority_probe` → False** | **CHANGED.** No composer exists, so no user can be mid-thought and no wake may defer. `None` gave the same outcome *by accident of `_attempt`'s `callable()` check*; the sibling probe uses the opposite unwired convention (uncertainty defers), so leaving it resting on that guard is a coincidence, not a default. Safety red #2 (see the honesty note below). |
| `_global_user_display_name` | controller | `viewless_user_display_name` → `"User"` (unchanged from the lifetime landing) | `_presentation_context_for` CALLS this slot with no `is None` guard. `None` ⇒ `TypeError` on every read, swallowed by the broad `except` into a per-read warning + `"User"`: a silently degraded slot, which is worse than a raise. |
| `_chat_dictionary_applier` | controller | `None` | `_apply_chat_dictionaries` early-returns the payload unchanged. Dictionaries are a per-conversation view concern; a wake notice is machine text. |
| `_world_info_applier` | controller | `None` | `_apply_world_info` early-returns the payload unchanged. Same reasoning. |
| `_rag_capture_provider` | controller | `None` | `_resolve_staged_rag_context` returns the empty 4-tuple. Staged RAG is composed IN the composer; with no composer nothing is staged. |
| `_default_session_settings` | controller | `None` | Both read sites are `is not None` guards; a viewless session falls back to the controller's own defaults instead of a dead screen's widget state. |
| `_library_provider_factory` | controller | `None` | `_library_provider_for_context` returns `None`, and the send path already degrades a missing factory to "no Library tools this run". The factory reads Settings THROUGH the screen, so a viewless turn must not use a stale one. |
| `_turn_context_provider` | controller | `None` | Guarded; the controller re-derives from its own selection state. The provider reads live WIDGET state (mode toggles, tool switches) — precisely what a viewless turn must not consult. |
| `on_submission_accepted` | controller | `None` | Guarded, and MANUAL-origin only. Its whole job is clearing a composer that does not exist. |
| `prompt_history` | controller | `None` | Guarded. Prompt history records what the USER typed; a wake notice is not user input (the wake invariant), so a viewless turn must record nothing. |
| `set_pending_approval` | controller | `None` | Inert but **not lossy**: `request_mcp_approvals` calls `add_pending_round` and retains the round's payload in `_parked_approval_payloads` BEFORE consulting this hook, unconditionally. A round armed headless is still registered and still claimable at the next mount. (Surfacing it app-wide and the 120s clock are plan Task 5 — untouched.) |
| `park_pending_approval` | controller | `None` | The badge/toast half of the same round; guarded at both call sites; the registry write that makes the round recoverable does not depend on it. |
| `notify_run_outcome` | controller | `None` | Guarded everywhere. A toast with no screen to toast into is the exact dead-screen call P3 found. |
| `notify_run_failure` | controller | `None` | Guarded everywhere. The run's terminal state and its DB row are unaffected; nothing durable is lost. |
| `set_pending_skill_install` | controller | `None` | `None` makes `request_skill_install_confirm` **fail closed immediately** — its read site says so in as many words (nothing could ever set the Event, so denying at once beats blocking for the full timeout with no way to be resolved). |
| `set_pending_skill_script` | controller | `None` | Same fail-closed-at-once contract (`allow=False, remember=False`). |
| `on_scope_flushed` | store | `None` | Guarded. It repaints the scope chip; the flush itself already happened in the store. |
| `delivery_ui_hook` | wake | `None` **+ a re-arm at attach** | The read site's `callable()` guard makes `None` exactly "no repaint target", which is correct while detached. The hazard is not the inert value but the MISSING RE-ARM — see below. |

Two slots that are **not** in the list stay out, unchanged: `controller.app` and
`wake.wire(app=…)` (the APP outlives every view; clearing either breaks the
`call_from_thread` bridge a surviving turn still needs).

### Where `None` turned out to be unsafe

Exactly one slot beyond the lifetime landing's own finding:
`wake_conversation_in_view`. Its read site's unwired fallback is a **wrong
answer**, not an inert one. `_global_user_display_name` remains the other (found
and fixed by the lifetime landing; re-pinned here by consequence).
`wake_user_priority_probe` was *not* unsafe today — it is now explicit for
robustness, not for a live defect (stated plainly rather than dressed up).

Every other read site was read and confirmed guarded before its `None` was
accepted; the two skill confirms go further and fail **closed**.

## The two safety reds, and their observable-consequence proof

Both are asserted on what the system DOES, never on the callable.

**Red 1 — `wake_conversation_in_view`: the ◈ mark must survive.**
`Tests/Chat/test_console_viewless_hooks.py::test_a_wake_delivered_with_no_view_keeps_the_unseen_mark`
attaches a view, detaches it through the production `ConsoleRuntime.detach_view`
seam, then drives a real survivor settle through the production chain
(`on_fleet_drained → _attempt → _deliver → submit_draft`, real controller, real
`AgentRunsDB`, real marks service) and asserts the FLEET_UNSEEN mark is still
set after the delivery commit.

RED, measured before any production edit:

```
Tests/Chat/test_console_viewless_hooks.py:145: AssertionError: a wake delivered
with NO Console attached cleared the ◈ mark: the user has no way to learn the
supervisor turn ever ran
```

The preserved side is pinned too
(`test_a_wake_delivered_to_the_attached_view_still_clears_the_mark`), so "keep
the mark" cannot be implemented by ignoring the probe.

**Red 2 — `wake_user_priority_probe`: no view ⇒ no user claim ⇒ no deferral.**
`test_a_wake_is_not_deferred_by_a_user_claim_once_the_view_is_gone` attaches a
view whose probe claims the user IS mid-thought, proves the wake defers (a
`_quiet` window with no provider payload — the harness precondition), detaches,
and asserts the next retry delivers.

**Honesty note: this red did NOT reproduce as a failure.** It passed on the
untouched branch. The reason is mechanical and worth recording: `_attempt`
guards with `callable(probe)`, so a `None` probe is skipped and the wake was
already not deferred. The property held *by accident of that guard's
direction*, in a coordinator whose OTHER probe treats an unwired slot as a
reason to fail toward the badge. The fix taken is to make the answer the slot's
own (`viewless_user_priority_probe`), which is what makes it mutation-killable
(M2 below) instead of contingent on one `callable()` check. Reported as a
green-on-arrival property, not as a red I fixed.

## The `delivery_ui_hook` re-arm pin

`ConsoleFleetWakeCoordinator._attempt` fires `delivery_ui_hook` **once**, at
delivery start — that hook is the only thing that arms the screen's 0.2s
transcript poll for a wake turn (a wake bypasses the user-send worker that
normally arms it). With a runtime that outlives the screen, "delivery start" and
"view attach" became independent events, so a Console opened *during* a
headless delivery got a live turn and no poll: the 4+ minute frozen transcript
PR 3a-2 Task 7 measured live.

- `ConsoleFleetWakeCoordinator` now tracks `_delivering_session` in lockstep
  with `_delivering` (a conversation id is not a session id) and exposes
  `delivering_session_id()`.
- `ConsoleRuntime.attach_view` calls `_rearm_delivery_ui_hook()` after binding:
  if a delivery is in flight it fires the freshly-bound hook with that session.
  Best-effort — a raising hook is logged, never propagated into the attach.
- Deliberately **not** gated on "the view changed": `attach_view` runs on every
  `_ensure_console_chat_controller()` call and the production hook is idempotent
  (`_start_console_transcript_sync_timer` early-returns when a timer exists), so
  an extra re-arm costs one pump hop while a missed one costs the freeze.

Pinned by `test_a_delivery_started_with_no_view_re_arms_at_the_next_attach`
(RED before the change: `opening Console during a wake delivery left the
transcript poll unarmed (the live 4-minute freeze); got []`), with the
conditional half pinned separately by
`test_attaching_with_no_delivery_in_flight_arms_nothing` — a poll armed with
nothing to repaint is the recurring-idle-repaint regression 15664 AC#2 forbids.

## Viewless from BIRTH, not only after a detach

`ensure_chat_controller` now applies the viewless defaults when nothing has
claimed the runtime (`self.view is None`). Without it, a runtime that never had
a view — the wake-at-launch shape plan Task 6 owns — kept the constructor's
`None`s and inherited the exact silent wrong answer this task removes. RED
before the change (`a runtime built with no view reported the conversation as
watched`).

The STORE is deliberately excluded from that construction-time clear
(`_clear_view_hooks(only=…)`): its one slot, `on_scope_flushed`, is a
CONSTRUCTOR parameter the caller just supplied, and nulling it during the
restore-before-attach window would drop scope flushes. Production attaches one
line later, so a mounted Console pays nothing either way.

## Mutations run and killed

Every mutation applied by Edit and restored by Edit; `git diff` verified empty
for `console_chat_controller.py` after each of its three.

| # | Mutation | Killed |
|---|---|---|
| M1 | `viewless_conversation_in_view` → `True` | `test_a_wake_delivered_with_no_view_keeps_the_unseen_mark`, `test_a_runtime_that_never_had_a_view_answers_viewless` (2 failed, 48 passed) |
| M2 | `viewless_user_priority_probe` → `True` | `test_a_wake_is_not_deferred_by_a_user_claim_once_the_view_is_gone` (its owner) + the mark and re-arm tests, whose precondition is that a viewless wake can deliver at all (3 failed, 47 passed) |
| M3 | drop `_rearm_delivery_ui_hook()` from `attach_view` | `test_a_delivery_started_with_no_view_re_arms_at_the_next_attach` only (1 failed, 49 passed) |
| M4 | drop the `if not session_id: return` guard in `_rearm_delivery_ui_hook` | `test_attaching_with_no_delivery_in_flight_arms_nothing` (owner) + the re-arm test, which then sees a spurious `[None]` arm (2 failed, 48 passed) |
| M5 | `_global_user_display_name` viewless default → `None` | `test_the_display_name_slot_is_never_cleared_to_none` only (1 failed, 49 passed) |
| M6 | gate the approval-round registry write behind `set_pending_approval is not None` (the "swallow the card" defect) | `test_an_approval_round_armed_with_no_view_is_not_lost` only (1 failed, 24 passed) |
| M7 | drop `or self.set_pending_skill_install is None` from the fail-closed early return | `test_a_skill_install_confirm_with_no_view_fails_closed_at_once` only (1 failed, 10 passed) |
| M8 | drop the viewless branch in `ensure_chat_controller` | `test_a_runtime_that_never_had_a_view_answers_viewless` only (1 failed, 24 passed) |
| M9 | remove `_apply_world_info`'s `if applier is None` guard | **SURVIVED — see below** |
| M10 | `delivery_ui_hook` slot target `"wake"` → `"controller"` | `test_every_slot_names_a_real_attribute_on_the_target_it_declares` + the re-arm test (2 failed, 11 passed) |
| M11 | remove `on_submission_accepted` from the slot list | `test_attach_and_detach_cover_exactly_the_same_slot_set` only (1 failed, 12 passed) |
| M12 | keep the slot in the list but skip `on_submission_accepted`/`prompt_history` in `_clear_view_hooks` (bound and never cleared) | `test_a_viewless_turn_calls_none_of_the_departed_views_hooks` only (1 failed, 12 passed) |

### M9 survived, and the fix it forced

`test_a_whole_turn_runs_with_no_view_attached` did not kill an unguarded
`_apply_world_info`. Investigated rather than patched around: the applier is
unreachable in that rig (the session has no `persisted_conversation_id`, so the
method returns before touching the slot) **and** the call is wrapped in a broad
`except` that returns the payload unchanged. The read site is double-guarded, so
no single-point mutation there is killable — which means the whole-turn test
owns "a viewless turn runs end to end", not "this slot is guarded", and saying
otherwise would have been a false claim of coverage.

The fix was to add the test that owns the property P3 actually found: **a turn
running after the view is gone calls NOTHING the view supplied**
(`test_a_viewless_turn_calls_none_of_the_departed_views_hooks`). It runs two
turns on one controller — attached (recorders MUST fire, so the assertion cannot
be vacuous) then detached (recorders must be silent) — and M12 kills exactly it.
M11 showed the complementary ownership: deleting a slot from the list is caught
by the attach-set/detach-set equality test, not by this one, which is the
correct split.

## Gate — baseline (untouched branch) vs final

Runner: `/Users/…/tldw_chatbook/.venv/bin/pytest <paths> -p no:randomly -q
--no-header -rf`, cwd = this worktree. `Tests/test_probe_import_provenance.py`
is in the gate: the venv's editable install resolves `tldw_chatbook` to a
FOREIGN worktree and loses only by `sys.meta_path` ordering.

| Suite | Baseline (untouched, `84a8f3ffc`) | Final |
|---|---|---|
| `Tests/Chat/` (full) | **14 failed, 5574 passed, 66 skipped** (23:09) | see below |
| `Tests/UI/test_screen_residency.py` + `Tests/Agents/` + provenance | — (green at final; a green run cannot hide a regression) | **1426 passed** |
| `Tests/UI/test_console_runtime_ownership.py` | 6 passed | 6 passed (one assertion intentionally rewritten, below) |
| `Tests/Chat/test_console_runtime_lifetime.py` + the 5 Chat wake suites | 45 passed | 45 passed |
| The 7 UI wake suites + `Tests/UI/test_console_mcp_approval.py` + provenance | — | **114 passed** |
| `Tests/Chat/test_console_viewless_hooks.py` (new) | 3 failed, 8 passed (the reds) | **12 passed** |

The 14 `Tests/Chat/` failures are measured **on the untouched branch at the
merge-base**, before any edit: `test_console_chat_controller` ×2,
`test_console_h3_image_edit` ×1, `test_console_provider_continuation` ×8,
`test_console_visual_evaluation` ×1, `test_console_voice_input` ×1. None of them
touch the runtime hook seam. (The final full-`Tests/Chat/` number is appended
below when that run lands.)

**One existing assertion was rewritten, deliberately.**
`Tests/UI/test_console_runtime_ownership.py::test_second_console_visit_reuses_the_runtime`
asserted `controller_one.wake_conversation_in_view is None` after a real
navigation — the exact value this task changes. It now asserts the **decision**
the production path makes after that same real navigation
(`wake._conversation_in_view(...) is False`), which is the property that
mattered all along; the callable-identity assertion never was.

## Deliberately not done

- `_attempt`'s `_shutdown_requested` gate is untouched, byte for byte. A real
  headless wake still does not FIRE — `leave_console` still sets that Event and
  only the next `attach_view` replaces it. That is the wake-fires-headless
  slice, not this one. Consequence for how the tests are built: the viewless
  state under test is produced through `detach_view` (the same seam
  `leave_console` calls first), not through `leave_console`, and the test module
  says so.
- The `ScreenStateStore` snapshot (Task 3) and the approval clock (Task 5) are
  untouched. The approval work here is strictly "the viewless default does not
  LOSE the round".

## Concerns

1. **A headless MCP approval still burns the full 120s.** Task 4 proves the
   round is registered and its payload retained, so a mount can claim it — but
   nothing surfaces during the window and, if no one mounts, it verdicts
   `timeout` (fail-closed) at ~120.4s, exactly as P4 measured. That is Task 5's
   whole subject; flagged here only so the "not lost" claim is not read as
   "handled".
2. **A round armed while detached and still armed at attach is not yet
   re-mounted.** `attach_view` re-arms `delivery_ui_hook` but does not push a
   retained approval payload at the newly attached view. Plan Task 5 owns it,
   and the payload-slot caveat at `console_chat_controller.py:4103-4128` is the
   known limitation it must assert *against*.
3. **`_conversation_in_view`'s own unwired fallback (`True`) is unchanged.**
   Deliberate: `test_an_unwired_view_probe_keeps_the_historical_clear` pins it
   for controller doubles and the pre-screen rig. The correction lives at the
   runtime seam, so a controller built outside the runtime keeps its documented
   behaviour. If a future task ever wants "unwired means unwatched" globally,
   that read site is the one line to change — and that existing test is the one
   that must be rewritten with it.
4. **`ensure_chat_controller`'s construction-time clear excludes the store.**
   Justified above, but it is an asymmetry in a list whose whole value is being
   run in both directions with no exceptions. It is documented at the call site
   and covered by the born-viewless test; a reviewer should confirm the
   reasoning rather than assume the symmetry.

## Final `Tests/Chat/` run (coordinator-run, after the agent hit its session limit)

**14 failed, 5586 passed, 66 skipped (16:54)** against the merge-base baseline of
**14 failed, 5574 passed, 66 skipped**. Same failure COUNT; +12 passed = exactly this
landing's new `test_console_viewless_hooks.py` suite. The 14 group by file as
`test_console_provider_continuation` ×9, `test_console_chat_controller` ×2,
`test_console_h3_image_edit` ×1, `test_console_visual_evaluation` ×1,
`test_console_voice_input` ×1 — the same files as the baseline, none of which touch the
runtime hook seam. (The baseline table above lists provider_continuation as ×8; the
per-file breakdown was off by one against its own total of 14. The totals are what match.)
