# Headless Wake (task-15860) — app-owned Console runtime

Task: `backlog/tasks/task-15860 - Headless-wake-fire-the-supervisor-auto-wake-with-no-Console-screen-mounted.md`
(4 binding ACs, restated in "What done means" below).
Spec: `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
§3 invariant 5 (`:104`, auto-wake REQUIRED) and §7 (`:371`), whose
"Where the wake itself runs — the honest architectural limit as built"
paragraph (`:424-437`) names this task as the follow-up that removes the
limit.
Predecessor: `Docs/superpowers/plans/2026-08-13-supervisor-fleet-pr3a2-autowake.md`
(PR 3a-2, merged) built every artifact headless wake needs; the only delta
is WHERE the wake runs.
Coordinator rulings and the two escalated owner calls:
`.superpowers/sdd/2026-08-14-headless-wake/DECISIONS.md`.
Task 0's executed evidence: `Docs/superpowers/plans/2026-08-14-headless-wake-task-0-report.md`.

## The design, and why it is A

**Design A: the Console runtime (agent bridge + `ConsoleChatController` +
the message store) is owned by the app, not by `ChatScreen`. The screen
attaches to it as a VIEW at mount and detaches at unmount.**

Three designs were on the table. B and C both keep per-screen ownership and
make a headless wake write only to ChaChaNotes (B: a throwaway headless
controller; C: a DB-only delivery path). Task 0 executed the premise both
depend on, and it does not hold:

- **P1.** With Console unmounted, SYSTEM + ASSISTANT rows written to the
  conversation through the production `ChatPersistenceService` were, at the
  next Console mount: absent from the transcript, absent from the rendered
  widgets, absent from the next send's provider payload — and the next
  persisted append **forked the tree**, parenting itself to the pre-nav
  assistant and leaving the headless rows on a dead branch. Maintaining the
  durable active-leaf pointer as well did not change any of it (the
  snapshot restore does not consult the pointer, and the next send
  overwrote it). Two runs, identical.
- **P3b.** The composition that decides it: a wake turn that RAN headlessly
  (surviving controller, real `submit_draft`, ledger stamped) persisted all
  four rows to ChaChaNotes — and the user returning to Console saw only the
  two that predated the snapshot. The wake notice and its reply were
  invisible.

So "DB-only headless writes are invisible and divergent at remount" is now
executed, not read. B and C are rejected on evidence. The remaining question
is not whether to move ownership but how little else has to move with it —
and Task 0 answered that too:

- **P2.** After a real `on_unmount` + `controller.shutdown()`, the bridge
  fan-out is still registered (`['fleet-attention', 'usage-reattach',
  'fleet-wake']`), `on_fleet_drained` still records into the pending
  registry from a child thread, and the coordinator's captured loop is the
  app loop, open and running. Nothing about the signal path dies with the
  screen. The only thing that stops delivery is
  `_attempt`'s `_shutdown_requested` gate.
- **P3.** With `shutdown()` suppressed and the screen genuinely unmounted,
  the wake reached `submit_draft`, the payload's trailing `user` entry
  carried the child's result, and `agent_runs.wake_delivered_at` was
  stamped — the full turn, with no view. Of 15 screen-wired hook slots only
  5 were touched, and 3 of those are the wake's own probes.

**The delta is therefore: (1) don't destroy the runtime when the view
dies, (2) make the app-owned store the continuity owner instead of the
screen snapshot, (3) rebind the five hook slots that a viewless turn
touches.** Everything else already works.

## Verified seam map — cite these, do not re-derive

Every line number below was read in this worktree at `origin/dev`
(`7a6e8f804`).

| Seam | Where | Fact |
|---|---|---|
| Screens are never cached | `app.py:8128` `_create_navigation_screen` | Every navigation builds a fresh instance; `ScreenStateStore` is documented there as the continuity mechanism |
| Navigation body | `app.py:8700` `_complete_screen_navigation` | `TAB_CHAT` snapshot is **discarded then re-saved** (`:8724-8736`); the incoming screen is restored BEFORE `switch_screen` (`:8791-8798`) |
| Snapshot store | `app.py:5458` `self.screen_state_store = ScreenStateStore()` | App-owned already; only its CONTENT is screen-shaped |
| Console snapshot write | `chat_screen.py:15781` `save_state` → `:15784` `_serialize_native_console_state` (`:15576`) | Writes `sessions` (`:15606`) + `messages_by_session` (`:15610`) |
| Console snapshot read | `chat_screen.py:15800` `restore_state` → `:15646` `_restore_native_console_state`, `store.restore_state(...)` at `:15693` | Rebuilds the store from the payload; **never re-reads ChaChaNotes** (P1 confirms by execution) |
| The DB resume path (not used by restore) | `console_chat_store.py:1136` `restore_persisted_session` | The only DB→store path; reached by opening a conversation, not by a tab switch |
| Screen teardown | `chat_screen.py:15392` `on_unmount` → `:15470` `_record_console_fleet_teardown` → `:15486` `await controller.shutdown()`, then `:15467` drops the reference | The one line that has to change ownership |
| What shutdown means today | `console_chat_controller.py:5621` `shutdown` | Sets `_shutdown_requested` unconditionally and first, then cancels EVERY session's stream task. Its own docstring records that `on_unmount` is the frequent caller, not process exit |
| Teardown accounting | `console_chat_controller.py:2281` `fleet_teardown_split` | Already partitions killed-vs-surviving; AC#2's existing contract |
| Fan-out | `console_agent_bridge.py:1112` `FleetDrainFanout`, registration `:3714` `on_fleet_drained` | Bridge-lifetime, child-thread, per-consumer isolation. P2: survives teardown intact |
| Wake coordinator | `console_fleet_wake.py:275` | `on_fleet_drained` `:386`, `retry_soon` `:422`, `_attempt` `:446`, `_deliver` `:523`, `_conversation_in_view` `:601`, `seed_from_marks` `:636` |
| The gate that stops a headless wake | `console_fleet_wake.py:460-462` | `_shutdown_requested.is_set()` → return. P2: this is the ONLY thing stopping delivery post-unmount |
| Wake send authority | `console_chat_controller.py:2727` `submit_draft`, `AGENT_WAKE` branch `:2795`, no USER row `:2831` | `ConsoleSubmissionOrigin.AGENT_WAKE` = `console_chat_models.py:54` |
| Wake send gate | `console_chat_controller.py:2422` `send_refusal_copy` | The same gate a manual send passes |
| Screen→controller wake wiring | `chat_screen.py:5579-5601` | `wake.wire(app=...)`, `wake.delivery_ui_hook = _on_console_wake_delivery_started` (`:5587`, impl `:15304`), `wake_user_priority_probe` (`:5588`, impl `:15177`), `wake_conversation_in_view` (`:5599`, impl `:15262`) |
| Mount claim | `chat_screen.py:15145` `_claim_console_fleet_wake_marks` (called synchronously in `on_mount`, `:15050`) → `seed_from_marks` | Ordering hazard: must precede the first tab sync |
| Active leaf | `console_chat_store.py:696` `_active_leaf_by_session`, write-through `:6680` `_persist_active_leaf`; DB `ChaChaNotes_DB.py:8313` `set_conversation_active_leaf` / `:8330` `get_conversation_active_leaf` | Local-only column; P1: the snapshot restore ignores it entirely |
| Out-of-band append | `chat_persistence_service.py:878` `create_message` (`sender=` is the role) | What a DB-only headless writer would use |
| Approval clock | `console_chat_controller.py:4325` `_resolve_mcp_approval_timeout_seconds`, default `:259` `= 120.0`, deadline `:3955`, `timeout` verdict `:4051-4053`, poll granularity `:262` `= 1.0` | P4 measured 120.43s to verdict with no UI wired |
| Honest-limits copy (AC#4) | `Docs/User_Guide/console/agent-runs-and-tools.md:494-501` | "**If Console isn't open, no wake fires**… follow-up work (task-15860)" |
| Spec copy to correct | spec `:424-437` | Same limit, same follow-up id |

## What done means (the task's 4 ACs, verbatim intent)

1. A finished background sub-agent wakes its supervisor while no Console
   screen is mounted, under the same `autowake_enabled` gate, caps, and
   approval floor as the mounted case.
2. The ownership move does not regress documented screen-scoped semantics
   (leaving Console still cancels streaming turns and denies parked
   approvals; survivors keep running).
3. Every wake invariant holds headless: no USER transcript row, exactly-once
   via the `wake_delivered_at` ledger, no phantom wake after restart.
4. The User Guide's honest-limits paragraph is removed or rewritten.

## Global constraints

- Env rules as 3a-2: no `git stash`; Edit-based restores only; python ONLY
  via pytest (a bare `python -c` importing `tldw_chatbook.config` rewrites
  the live config); never touch `~/.config/tldw_cli`; no worktrees under
  `/private/tmp`; regenerate the CSS bundle, never hand-edit.
- **The venv's editable install resolves `tldw_chatbook` to a FOREIGN
  worktree** (`.worktrees/task-2512-mcp-unified`). Run pytest with the
  worktree as cwd and keep `Tests/test_probe_import_provenance.py` (or an
  equivalent assertion) in the gate — it is the only thing that proves the
  code under test is this branch.
- **Reproduce red before fixing; mutation-test every new test.** Seven
  confident readings have now been refuted by execution in this programme.
- A wake notice is never user input and never approval, mounted or headless.
- Nothing in this plan may add a second source of truth for conversation
  history. Design A's whole justification is that it REMOVES one.

## Escalated to the owner — blocking status

Both are recorded in `DECISIONS.md`. Neither blocks Tasks 1–2.

- **Owner call #2 — wake at app launch with no Console ever opened?**
  Blocks Task 6 only. Recommendation stands: YES, gated by the existing
  `autowake_enabled` AND triggered only by an existing `FLEET_UNSEEN` mark.
- **Owner call #3 — does Console message state stop travelling through
  `ScreenStateStore`?** Blocks Task 3, which is the structural half of
  design A. Task 0's P3b is the argument FOR: with the runtime app-owned
  but the snapshot still screen-owned, a wake that genuinely ran is
  invisible on return. Task 3 is where the freeze-incident risk lives; do
  not start it without the ruling.

---

## EXECUTION ORDER — reconciliation (coordinator, 2026-08-14)

**The executed order differs from the numbering below. Read this before picking up work.**

The owner approved design A on a staging condition: *the pure ownership move lands first,
with ZERO semantic change, separately reviewable and separately revertable from any lifetime
semantics.* That inverts the plan's Task 1/Task 2 order, so the coordinator dispatched the
ownership move first.

| Shipped | Commit | = plan task | State |
|---|---|---|---|
| Four execution probes | `25bc081b2` (PR #1641) | Task 0 | DONE |
| App-owned `ConsoleRuntime` holder, **still disposed at unmount** (zero semantic change) | `f09bb991d` | Task 2's ownership half ONLY | DONE |
| Teardown split (`leave_console` vs `dispose`) + the runtime SURVIVING unmount | — | Task 1 + Task 2's lifetime half | NEXT |
| Tasks 3-8 | — | unchanged | pending |

**Two hazards the seam map did not name, found while executing the ownership move — the next
task must handle both:**

1. **`_complete_screen_navigation` constructs the incoming screen and calls `restore_state`
   BEFORE `switch_screen` unmounts the outgoing one**, and `_restore_native_console_state`
   reaches `_ensure_console_chat_store`. A naive app-owned holder therefore hands the
   incoming screen the OUTGOING screen's controller, which `on_unmount` then shuts down
   underneath it — a dead Console after a same-target navigation (reachable via the
   `coding → chat` alias). `ConsoleRuntime.view` currently closes this by replacing a runtime
   claimed by a different view. **That is a lifetime-preservation device standing in for
   today's semantics; the attach/detach seam must REPLACE it, not build on it.**
2. **`ChatScreen` still holds its own `_console_chat_store` / `_console_provider_gateway` /
   `_console_chat_controller` handles** — ~40 sites read them as "built yet?" probes and 59
   test sites assign them. Identical lifetimes today, so they cannot diverge. The moment the
   runtime outlives the screen, a fresh screen's `None` handle SHADOWS a live runtime object
   until `_ensure_*` runs. **Repointing them is the next task's FIRST move, not an
   afterthought.**

### Task 0 — the four execution probes (DONE)

Delivered: `Docs/superpowers/plans/2026-08-14-headless-wake-task-0-report.md`,
probes `Tests/UI/test_probe_headless_wake_p1_continuity.py`,
`Tests/UI/test_probe_headless_wake_p2_p3_p4.py`,
`Tests/test_probe_import_provenance.py`. No production code. Verdict:
**design A survives P1**; the off-ramp was not taken.

### Task 1 — split teardown: cancel the user's work vs destroy the runtime

`shutdown()` currently does both (`console_chat_controller.py:5621`), and
`_shutdown_requested` is the exact flag `_attempt` reads to refuse a wake
(`console_fleet_wake.py:460-462`). AC#2 requires the first half to keep
happening on nav-away; AC#1 requires the second half to stop.

- [ ] Red: a test that navigates away from Console with (a) a streaming
  user turn and (b) a parked approval round, and asserts both are still
  cancelled/denied — while a survivor keeps running and the coordinator's
  `_attempt` is NOT refused by a teardown flag. Fails today because one
  flag governs both.
- [ ] Introduce a view-detach path distinct from runtime shutdown:
  detach cancels this view's in-flight USER turns and revokes parked
  approval rounds (reusing `fleet_teardown_split`'s existing partition,
  `:2281`); runtime shutdown keeps its current meaning and is called only
  at app exit.
- [ ] Mutation-test: flip the detach path to also set `_shutdown_requested`
  and prove the AC#1 test goes red.
- [ ] Gate + commit.

### Task 2 — move bridge + controller ownership to the app

- [ ] Red: after a real navigation away from Console (no `shutdown()`
  suppression, no monkeypatching — the production path), a survivor's
  settle delivers a wake turn that reaches the provider and stamps
  `wake_delivered_at`. This is exactly P3's assertion with the artificial
  survival removed; it fails on dev today.
- [ ] The app constructs and owns the bridge + controller (+ store) lazily,
  one per runtime identity (`app.py:8211` `_current_runtime_identity`);
  `ChatScreen._ensure_console_chat_controller` returns the app-owned
  instance instead of building one. `on_unmount` (`chat_screen.py:15392`)
  detaches the view (Task 1's path) and no longer drops the runtime.
- [ ] The coordinator's `wire(app=...)` and captured loop move to
  construction time. P2 already proves the app loop is the right one and
  that it outlives the screen — do not add a second loop.
- [ ] Keep `seed_from_marks`'s mount-claim ordering intact
  (`chat_screen.py:15050`): with an app-owned runtime the claim happens at
  runtime construction, and the first tab sync's view-clear must still come
  after it. Pin that ordering with its own test, as PR 3a-2 did.
- [ ] Gate + commit.

### Task 3 — the app-owned store becomes the continuity owner

**Blocked on owner call #3.**

- [ ] Red: P3b as a regression test — a wake turn that ran while Console
  was unmounted is present in the transcript the user sees on returning.
  Fails today (executed: the user sees 2 of 4 rows).
- [ ] Console message state stops travelling through
  `ScreenStateStore.native_console_state`; `_serialize_native_console_state`
  /`_restore_native_console_state` (`chat_screen.py:15576`/`:15646`) are
  retired or reduced to view-only state (draft text, image view modes,
  focus), never `sessions`/`messages_by_session`.
- [ ] The 2026-07-11 freeze incident hardened the mechanism being changed.
  Add a rapid-switch soak (the shape `Tests/UI/run_workbench_soak.py`
  already uses) to the gate for this task specifically.
- [ ] Prove no second source of truth remains: a test that appends through
  the runtime and asserts transcript, provider payload, DB rows and the
  active-leaf pointer all agree — the four things P1 found disagreeing.
- [ ] Gate + commit.

### Task 4 — rebind the five hook slots a viewless wake turn touches

P3 executed which slots a wake turn with no view actually touches:
`wake.delivery_ui_hook`, `wake_conversation_in_view`,
`wake_user_priority_probe`, `_chat_dictionary_applier`,
`_world_info_applier`. In P3 all five were still bound to methods of a
DEAD screen and none raised — which is worse than raising, because a
silent wrong answer from `wake_conversation_in_view` decides whether the
`◈` mark survives (task-15971's whole point) and a silent wrong answer
from `wake_user_priority_probe` decides whether the user wins a tie.

- [ ] Red: with no view attached, `wake_conversation_in_view` reports
  not-in-view (so the mark is KEPT) and `wake_user_priority_probe` reports
  no user claim — each proven by the observable consequence (mark set;
  wake not deferred), not by asserting on the callable.
- [ ] Attach/detach rebinds all five: a view sets them, detaching restores
  viewless defaults. No slot may keep pointing at an unmounted screen.
- [ ] `delivery_ui_hook` becomes a no-op when detached, and is re-armed by
  the next attach if a wake is still delivering
  (`delivering_conversation_id`, `console_fleet_wake.py:370`) — the
  4-minute freeze PR 3a-2 Task 7 found is the cost of getting this wrong.
- [ ] The 10 untouched slots (`set_pending_approval`,
  `park_pending_approval`, `notify_run_outcome`, `notify_run_failure`,
  `on_submission_accepted`, the two skill-confirm slots, and three
  providers) get viewless defaults too, with a test per safety-relevant
  one — "not touched in P3's turn" is a fact about one turn, not a proof
  they cannot be reached.
- [ ] Gate + commit.

### Task 5 — headless approval: park, notify app-wide, keep the clock

Coordinator decision 4. P4 measured the real cost with no UI wired: the
round runs to the shipped `[mcp] approval_timeout_seconds` = 120.0
(`console_chat_controller.py:259`) and verdicts `timeout` after **120.43s**,
fail-closed. Nothing surfaced to the user during that window — P3 showed
`set_pending_approval` and `park_pending_approval` are untouched by a
viewless turn.

- [ ] Red: a risk-tagged tool in a headless wake turn raises an app-wide
  notification (toast + badge) at the moment the round arms, and the round
  is resolvable by opening Console within the deadline.
- [ ] Do NOT pause or extend the deadline while detached — that is a change
  to a fail-closed safety gate and is deliberately deferred. Document the
  120s cost in the User Guide instead of hiding it.
- [ ] A round armed while detached and still armed at attach must mount its
  card, not be silently re-parked (the payload-slot caveat at
  `console_chat_controller.py:4103-4128` is a known limitation — assert
  against it rather than around it).
- [ ] Gate + commit.

### Task 6 — wake at launch / first boot

**Blocked on owner call #2.** This is the case the 2026-08-14 task-16300
correction left as genuinely headless alongside ordinary navigation.

- [ ] Red: with `FLEET_UNSEEN` marks and undelivered ledger rows present at
  process start and Console never opened, the wake fires.
- [ ] Gated by `autowake_enabled` AND an existing mark, per the
  recommendation — never a bare "wake on launch".
- [ ] AC#3's phantom-wake case: a run already stamped `wake_delivered_at`
  must not be re-announced across a restart (`_rows_for`'s stale drop,
  `console_fleet_wake.py:757-772`, already does this — pin it headless).
- [ ] Unsaved (ephemeral) conversations wake headlessly while the process
  lives and are gone after a restart — accepted (coordinator decision 5),
  stated in the docs.
- [ ] Gate + commit.

### Task 7 — invariants under headless (AC#3) as one gate

- [ ] No USER transcript row in a headless wake (the `AGENT_WAKE` branch,
  `console_chat_controller.py:2831`) — asserted on the DB rows, not only
  the in-memory store, because headless is exactly the case where the two
  used to disagree.
- [ ] Exactly-once across a restart mid-commit, via the ledger.
- [ ] `autowake_enabled = false` silences the headless fire point too, and
  loses nothing durable.
- [ ] Deliveries stay serialized app-wide with the runtime app-owned (one
  `_delivering` for one runtime, not one per screen).
- [ ] Gate + commit.

### Task 8 — documentation (AC#4)

- [ ] `Docs/User_Guide/console/agent-runs-and-tools.md:494-501` — the
  "**If Console isn't open, no wake fires**" paragraph is rewritten to what
  now happens, including the 120s headless-approval cost from Task 5.
- [ ] Spec `:424-437` — the "honest architectural limit as built" paragraph
  gets its superseding note; §10's follow-up row for task-15860 closes.
- [ ] Update the page's "Verified against" stamp.
- [ ] `backlog/docs/lessons-*.md`: the import-provenance trap (the venv's
  editable install pointing at a foreign worktree) and the P1 finding
  (a DB append is invisible to a live Console *and* to the next mount —
  the store, not the DB, is what the transcript and the payload are built
  from) both generalise beyond this task.
- [ ] Gate + commit.
