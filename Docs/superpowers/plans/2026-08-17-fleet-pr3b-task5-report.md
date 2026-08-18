# Fleet PR 3b — Task 5 landing report: Stop semantics + "Cancel all agents"

Branch `feat/fleet-3b-stop-semantics`, from origin/dev `98a189015` (the
merge of PR #1799 — the coordinator's restart instruction moved this
task's base forward from `e49c5dba8`; verified: `git diff
e49c5dba8..98a189015 -- tldw_chatbook/Agents/
tldw_chatbook/Chat/console_agent_bridge.py` is EMPTY, so every seam this
task edits was byte-identical across the move). Plan:
`2026-08-17-fleet-pr3b-steering.md`, Task 5 — the plan's named
highest-risk task. Spec §8 (Stop semantics move here), §3 invariant 4
(steering never cancels), honoring Task 4's two binding concerns (any
direct-finish status checked against `RETAINED_TRANSCRIPT_STATUSES`;
the first-writer-wins retention pin as tripwire) and Task 3's two
(the steering bar/Cancel-all visibility cross-pin; the layered-guard
mutation lesson).

## What the change is

Branched entirely on the existing `[agents] subagents_outlive_turn` key
(shipped default ON). Three seams in `agent_service.py`:

1. **`child_should_cancel`** — with the key ON a child polls ONLY its
   own cancel Event; the parent-poll term (`should_cancel()`) is gone,
   so a user Stop — whose probe stays flipped forever — no longer kills
   background work at the child's next loop boundary. The key is read
   ONCE, at spawn (`_launch_fleet_child`): a child's Stop-coupling
   contract is fixed at launch, while `_surviving_handles` still reads
   at settle time, so flipping the switch OFF mid-run still lets the
   very next settle cancel a previously-decoupled child through its
   Event. With the key OFF the closure is the pre-Task-5 line verbatim.
   A RESUMED child (Task 4's continuation) launches through the same
   `_launch_fleet_child` and inherits the same semantics for free.
2. **`wait_agents`' cancel branch** — ON: Stop releases the WAIT
   without cancelling; the note reads "(The run was cancelled;
   sub-agents continue in the background and their results will be
   delivered when they finish.)" and the unwind-grace drain is gated on
   an actual cancel (`stopped_children`), so the stopped turn is not
   held open for children that are not stopping. OFF: the exact
   "(…sub-agents were stopped.)" cancel path, and the budget branch is
   untouched in both directions (a parent's time budget expiring is not
   a Stop).
3. **`_surviving_handles`** — the user-cancel settle branch is deleted
   (it was reachable only with the key ON) and the now-unused
   `should_cancel` parameter dropped from the signature (one caller,
   `_settle_fleet`, updated). OFF returns `set()` before ever reaching
   the deleted line — byte-identical.

No new terminal status was introduced anywhere, so Task 4's
`RETAINED_TRANSCRIPT_STATUSES` concern is discharged by construction: a
Cancel-all'd or settle-cancelled child still finishes `RUN_CANCELLED`
through `run_child`'s existing `finally`, and the first-writer-wins
retention pin (`test_finish_with_transcript_respects_first_writer_wins`)
stayed green through the whole gate — survivor-race ownership is
unchanged.

Two pre-existing tests whose SUBJECT this task changes were re-pinned
turn-scoped with the reason in-line rather than deleted:
`test_stopping_the_turn_still_stops_its_children` (whose docstring
literally reserved this change for PR 3b) and
`test_wait_agents_cancellation_stops_children_and_ends_the_run` — whose
script, verified while designing the probes, flips its cancel BEFORE
the wait fence returns, so the parent dies at the loop's pre-dispatch
gate and `wait_agents`' own cancel branch never actually ran in it; its
docstring now records that honestly. `fleet_teardown_split`'s docstring
(console_chat_controller.py) was updated: "killed" now describes the
session's RUN; its children die with it only under the kill switch.

## Merge-base probes (C1 style), committed red/green as evidence

`Tests/Agents/test_fleet_stop_semantics.py`, measured at untouched
`98a189015` and committed failing (`d8ddd8534`) before any change:

| Probe | At the merge-base | After the change |
|---|---|---|
| (a1) outlive ON, Stop mid-turn → child survives, finishes DONE | **RED** — the stopped turn held open 5.18s by the settle join; child came back `cancelled` | green |
| (a2) outlive ON, Stop during `wait_agents` → wait releases, note says "continue in the background", child unharmed | **RED** — held open 10.14s (cancel-branch + drain grace + join); children cancelled | green |
| (a3) a stopped parent's survivor still steerable AND still drains (delivered at its next boundary) | **RED** — `assert 'cancelled' == 'running'`: the child was already dead | green |
| (b1) outlive OFF, Stop → whole tree dies THROUGH THE CANCEL-EVENT PATH (child's Event asserted set) | **GREEN** | green, file untouched through the change commit |
| (b2) outlive OFF, Stop during `wait_agents` → children cancelled, exact "(…sub-agents were stopped.)" note in the step log | **GREEN** | green, untouched |

Probe (b2) is, as far as the suite knows, the FIRST test to genuinely
exercise `wait_agents`' cancel branch: the deterministic in-wait trigger
flips `should_cancel` on its SECOND call after the wait fence returns
(call one is the loop's per-call pre-dispatch gate at
`agent_runtime.py`'s cancellation checkpoint) — flipping any earlier
kills the parent before `wait_agents` dispatches, which is exactly what
the pre-existing test's script does.

A sixth probe, (b3), was added before the mutation round when analysis
showed the OFF closure's parent-poll term is outcome-redundant with the
settle in every existing test (a mutant dropping it would have
survived): the parent is held INSIDE its model call after Stop — no
settle, no Event — so the released child's death at its next boundary
can only be the poll (cancelled after exactly ONE model call). Verified
GREEN at the untouched merge-base in the baseline worktree (see gate).

## The byte-identical proof for OFF

Three layers of evidence: (1) probes (b1)/(b2) were committed green at
the merge-base and stayed green through the change with their file
untouched (the only later edit to the file ADDED tests); (2) the OFF
arm of each branch is the pre-change line verbatim (`should_cancel() or
child_cancel.is_set()`; the `_cancel_fleet_handles(pending)` +
`stopped_children` cancel path; `_surviving_handles` returning `set()`
at the kill-switch check); (3) probe (b3), which isolates the one OFF
behaviour no pre-existing test owned, passes identically at the
merge-base and on the branch.

## The teardown / app-exit audit — the plan's premise was WRONG, recorded honestly

The plan's audit bullet assumed "conversation-deletion /
ephemeral-close teardown still cancels the fleet — it goes through
cancel Events, not the parent poll." **Measured at the merge-base, that
premise is false**: NO conversation-deletion or ephemeral-close path
sets per-child cancel Events. The complete inventory of
`_cancel_fleet_handles` callers is: the end-of-turn settle
(non-survivors), `wait_agents`' cancel and budget branches, and the
per-row `AgentService.cancel_subagent`. Session close
(`ConsoleChatController.close_session`) and both controller teardowns
(`shutdown`, `leave_console`) reach an in-flight fleet only through
`_signal_stop` → the PARENT's cancel probe.

Consequence under this change, stated plainly: closing a session (or
navigating away, or controller shutdown) mid-turn cancels the TURN; its
children now survive those teardowns — which is CONSISTENT, not novel:
`leave_console`'s own docstring already promised "cross-turn fleet
SURVIVORS keep running, untouched", and `_teardown_fleet_service`
retains a survivor-owning service regardless of how its turn ended, so
the panel keeps seeing and stopping them. The audit's two load-bearing
consequences are pinned in the suite
(`test_fleet_stop_semantics.py`'s audit section):

* **The Event path survives the decoupling** — a stopped turn's
  survivor still dies on its OWN Event through the per-row cancel seam
  (the mechanism Cancel-all, the settle, and any future teardown all
  speak). Kills any mutant that drops the Event term rather than just
  the parent term.
* **App exit takes everything** — every fleet child runs on a
  `daemon=True` thread (`agent_service.py`'s spawn tail), so process
  exit reaps them; `ConsoleRuntime.dispose` latches `_disposed` so
  nothing rebuilds. This was ALREADY the only exit guarantee earlier
  turns' survivors had before this task; the pin stops a future
  thread-pool refactor from silently changing it.

## Steering-interaction pins

* A stopped parent's surviving child **still drains steering and is
  still steerable**: probe (a3) posts USER steering AFTER the Stop and
  asserts the exact `[Steering from user] …` message in the child's
  next model payload.
* A child cancelled by Cancel-all **refuses further steering with the
  terminal copy**: `steer_subagent`/`post_steering` refuse terminal
  handles (Task 3's pins, all green), and the send_to_agent side is the
  retention pin below.
* Task 3's requested cross-pin: the steering bar and the Cancel-all
  affordance **leave the live surface on the SAME sync** after a
  cancel-all (`test_cancel_all_and_the_steering_bar_hide_together_…`,
  painted-frame).

## "Cancel all agents"

`ConsoleAgentBridge.cancel_all_subagents(conversation_id) -> int`
(`console_agent_bridge.py`, directly after `cancel_subagent`):
enumerates live handles from the same two tiers `fleet_snapshot` reads
(published service's coordinator view, else the retained survivor
owners; de-duplicated), then cancels each THROUGH the existing
per-handle `cancel_subagent` — the current-service-then-retained-owners
walk, approval-card revocation, and honest lost-race misses all ride
along; the count returned is "children actually cancelled by this
press". No second mechanism, pinned at the method seam by a delegating
spy (Task 3's layered-guard lesson: the outcome alone would not kill a
parallel second mechanism).

Bridge suite (`Tests/Chat/test_console_agent_bridge_cancel_all.py`, 4
tests, committed red at `AttributeError` on the missing method): the
full walk in one press (turn 2 in flight with its own child + turn 1's
retained survivor → count 2, both rows `cancelled`), the
retained-owner-only tier (no run in flight — the mandated mutation's
owner), the delegation spy (one call per live handle), and the
unknown-conversation 0. Real `run_reply` turns, real gated children on
real threads.

Panel affordance: a `Button` ("Cancel all agents",
`#console-agent-cancel-all`) in the rail's agent body after the fleet
section (`left_rail.py`), visibility computed by
`ConsoleAgentController._console_agent_cancel_all_visible` from the
SAME `fleet_snapshot` live source the rows and the steering-bar
visibility read — the three surfaces cannot disagree. The section
payload grew a 9th element rather than a second equality guard (Task
3's landing note). Screen delta: mechanical unpack/apply + construction
kwarg + ONE `@on` delegation in the row-cancel handler's exact grammar
(+27 lines; the size ratchet was already red on dev at the merge-base —
20,861 lines vs its 17,727 budget, documented inside the ratchet test
itself — and remains a pre-existing dev red).

Painted-frame evidence (`Tests/UI/test_console_agent_cancel_all.py`, 5
tests, committed red at `NoMatches`): paints at its own region with a
live child (`_assert_painted_at_own_region`, compositor hit-test); does
NOT paint for a finished-only fleet (rows on screen, nothing live — the
mandated does-not-paint case) nor an empty one; a REAL click reaches
the bridge seam exactly once and the next sync hides the affordance;
and the bar/Cancel-all cross-pin above.

## Retention interaction (Task 4)

A cancel-all'd child is never retained (cancelled is not a retained
status) and must draw the HONEST refusal, not the unknown-id copy:

* Coordinator seam (bridge suite): after cancel-all + unwind,
  `get_retained(handle_id)` is None and a second press counts 0.
* Service seam (`test_a_cancel_alled_child_draws_the_not_retained_
  refusal_not_unknown`): the child is cancelled mid-turn through the
  exact per-handle seam the delegation spy pins, then the supervisor's
  `send_to_agent` in the same turn draws "has finished (cancelled) and
  no retained transcript is available … cannot be resumed", never "no
  sub-agent matches id".

One honest limitation, pre-existing (Task 4's resolution order, not
this task's): after the NEXT turn's `prune_terminal`, a cancelled
child's HANDLE id falls through to the unknown-id copy (only retention
and the DB's run-id tier survive the prune, and cancelled children are
in neither by handle id). Flagged for Task 6's docs pass rather than
patched here — changing the resolution ladder is Task 4 territory.

## Mutations — six runs, zero survivors, Edit-based restores (`git diff -- tldw_chatbook` read 0 after each)

| Mutation | Kills |
|---|---|
| M1 parent-poll term re-added to the ON closure (mandated) | 3 — probe (a1) the survivor-on-Stop owner, (a2), (a3) |
| M2 cancel-all skips retained owners (mandated) | 3 — the foreign-survivor owner, the full-walk test, the delegation spy |
| M3a the closure's key branch flipped (OFF altered, mandated) | 4 — probe (b3) owns the OFF direction + the 3 ON owners |
| M3b `wait_agents`' outlive read forced True (OFF altered, mandated) | exactly 1 — probe (b2) |
| M3c `_surviving_handles`' kill-switch check deleted | 11 — probes (b1)+(b2) + 9 pre-existing kill-switch guards in `test_fleet_runtime.py` |
| M4 affordance visibility inverted (work-exposed) | 5 — every UI owner, both painted-frame tests included |

The near-survivor lesson (caught by analysis before it survived): the
OFF parent-poll term was outcome-redundant with the settle in every
pre-existing test — dropping it changed nothing any of them measured.
Probe (b3) was added FIRST and M3a then died on it in the OFF
direction. The general form matches this programme's eighth-survivor
history: a term with a redundant backstop needs a test that disables
the backstop.

## Gate (read counts; baselines on the untouched merge-base first)

| Suite | Baseline (98a189015) | Final |
|---|---|---|
| `Tests/Agents/test_fleet_stop_semantics.py` (new) | — (3 failed, 2 passed at stage 0) | **9 passed** |
| `Tests/Chat/test_console_agent_bridge_cancel_all.py` (new) | — (4 failed, AttributeError) | **4 passed** |
| `Tests/UI/test_console_agent_cancel_all.py` (new) | — (5 failed, NoMatches) | **5 passed** |
| `Tests/Agents/test_fleet_runtime.py` | 107 passed | **107 passed** |
| `Tests/Agents/` (full) | 1534 passed | **1543 passed** (= 1534 + 9) |
| `Tests/Agents/test_fleet_steering_mailbox.py` | 22 passed | **22 passed** |
| `Tests/Agents/test_fleet_send_to_agent.py` | 15 passed | **15 passed** |
| `Tests/Agents/test_fleet_continuation.py` | 35 passed | **35 passed** |
| `Tests/Chat/test_console_agent_bridge.py` | 214 passed | **214 passed** |
| `Tests/Chat/test_console_agent_bridge_steering.py` | 10 passed | **10 passed** |
| `Tests/Chat/test_console_runtime_lifetime.py` | 14 passed | **14 passed** |
| `Tests/UI/test_console_fleet_panel.py` | 9 passed | **9 passed** |
| `Tests/UI/test_console_agent_rail.py` | 33 passed | **33 passed** |
| `Tests/UI/test_console_agent_steering_bar.py` | 14 passed | **14 passed** |
| `Tests/UI/test_console_agent_controller.py` | 7 passed | **7 passed** |
| `Tests/UI/test_console_reaction_picker.py` (rail-constructor consumer) | 38 passed | **38 passed** |
| `Tests/UI/test_console_mcp_approval.py` | see attribution note | see attribution note |
| `Tests/test_probe_import_provenance.py` | 1 passed (names this worktree) | **1 passed** |

**Two attribution rulings, both measured at the untouched merge-base
before being called anything:**

1. `Tests/UI/test_console_agent_fleet_sync_coalescing.py` is a TIMING
   FLAKE on dev: across 5 file-level runs at untouched `98a189015` it
   failed twice (alternating owners, `assert 5 == (2 + 1)` — extra
   coalesced sync runs) and once even in isolation. Pre-existing; not
   this branch. It passed 3/3 in this branch's own gate batch.
2. `Tests/UI/test_console_mcp_approval.py` fails the SAME 3
   deny-deadline tests (`…cancellation_denies_undecided`,
   `…records_denied_decision_to_execution_log`,
   `test_shutdown_still_denies_a_survivors_round`) on BOTH the branch
   and the untouched merge-base when the machine is loaded (measured
   back-to-back while the one-process baseline run was executing;
   identical failure sets), and passes on both when quiet (see the
   population-gate section). Load-sensitive deadlines; not this branch.

### The one-process app-lifetime population gate

<!-- POPULATION_GATE_RESULTS -->

## Concerns for Task 6

* **Docs must teach the new Stop contract**: Stop stops the supervisor;
  survivors continue; the children's kill switches are per-row Cancel,
  "Cancel all agents", and `[agents] subagents_outlive_turn = false`.
  The `wait_agents` cancel note's new copy promises "results will be
  delivered when they finish" — that delivery is the auto-wake, so the
  User Guide should connect the two (and note the wake kill switch).
* **The teardown notice under-reports by design**: a "killed" session's
  surviving children go unmentioned in the next-mount notice
  (`fleet_teardown_split`'s updated docstring records this); their own
  settle toasts still report them. If the live pass finds that
  confusing, the notice copy is the place, not the split.
* **The pruned-cancelled-handle-id refusal gap** (retention section
  above): after the next turn starts, a cancel-all'd child's handle id
  draws the unknown-id copy. Cheap fix if wanted: a DB tier for handle
  ids, or teaching the refusal to mention run ids — Task 4's ladder,
  file a task rather than patch in Task 6.
* **task-18055** (`send_to_agent` missing from the skills shadow-name
  set) was deliberately left whole for Task 6 — nothing in this task
  touched the tool-registration surface.
* **Live verification**: the plan's §6 exercise should now include one
  real Stop mid-turn with a survivor (watch it keep working, steer it,
  then Cancel-all it) — the panel affordance's enabled/disabled edge is
  painted-frame-tested but deserves one real-tty look.
