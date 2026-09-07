# Fleet PR 3b — Task 3 landing report: panel steering input + queued state

Branch `feat/fleet-3b-panel-steering`, from origin/dev `22d156155` (the
Task 2 merge, PR #1777; merge-base verified identical to origin/dev tip
at start). Plan: `2026-08-17-fleet-pr3b-steering.md`, Task 3 (the USER's
path into Task 1's per-child mailbox). Spec bindings: §6 (two paths one
mechanism, latency honesty), §7 ("phase 3 adds the steering input +
mailbox 'queued' state"), §1's owner pin (the panel watches/steers,
never launches), §3 invariant 4 (steering never cancels). Task 2's two
binding concerns honored: the id-resolution SHAPE is reused with this
producer's own validation copy, and submit is disabled on an empty
target — an empty id never draws an unknown-id refusal naming `''`.

## What landed (four commits, pushed incrementally)

1. `b54c9a407` — the red suites
   (`Tests/Chat/test_console_agent_bridge_steering.py`,
   `Tests/UI/test_console_agent_steering_bar.py`), committed failing.
2. `a90a12133` — `console_agent_bridge.py`: `steer_subagent(
   conversation_id, row_id, text) -> bool`. Coordinator lookup via
   `_fleet_coordinators` — **no service hop, deliberately** (unlike
   `cancel_subagent` directly above it, whose retained-owner walk exists
   only because cancel Events are service-local); UI-thread safe (the
   coordinator's own brief lock is the only lock taken; `snapshot()` and
   `post_steering()` are each one short locked section). Live-handle
   resolution in Task 2's pinned order — handle id FIRST, then a live
   handle's run id. Validation at THIS boundary (non-empty after strip,
   `MAX_STEERING_CHARS`) — the mailbox does not validate (Task 1's
   pinned decision). Posts `STEERING_SOURCE_USER`. A lost post race
   (child terminal between snapshot and post) returns `post_steering`'s
   own honest False.
3. `fe2e1d2f7` — the widget + the wiring:
   * `Widgets/Console/console_agent_steering_bar.py`
     (`ConsoleAgentSteeringBar`): compact `Input` + queued-count line +
     its own refusal note line; posts typed
     `SteeringSubmitted(target_id, text)`. **Explicit width/height on the
     bar and every child** (bar `width:100%`/`height:auto`; Input
     `width:100%`/`height:3`; Statics `width:100%`/`height:auto`) — the
     1fr-default-pushes-siblings-off-screen trap is named in the module
     docstring. Three submit guards run BEFORE the message posts: empty
     text → inert; empty target → inert; oversize → refused with this
     bar's own copy ("Steering is too long (N chars; the cap is 4000).
     Shorten it and press Enter again."), draft kept in the input.
     `ConsoleAgentSteeringState` is a frozen all-fields-required
     dataclass (the `ConsoleInspectorSectionState` anti-drift
     discipline), so the Agent-section payload's `==` equality guard
     keeps working unchanged.
   * `Console_Modules/agent.py`: `_console_agent_steering_state()`
     (derivation — see "visibility" below),
     `_steer_console_agent_drilldown_child(target_id, text)` (the exact
     `_cancel_console_agent_fleet_row` shape: getattr-guarded, degrades
     to False, no resolution here), `_fleet_row_from_handle` appends
     `· steering queued (N)` after the token segment, and
     `_console_agent_section_payload` grows an 8th element (the steering
     state).
   * `left_rail.py`: the bar mounts in the agent body's drill-in chrome,
     between the fleet section and the Back button, constructed from the
     controller's CURRENT derivation — so a rail recompose mid-drill-in
     paints correctly without waiting for the next equality-guarded sync
     tick (the same reason `agent_drilldown_active` is a constructor
     value). The kwarg defaults to `None` (hidden) so pre-existing bare
     constructions (`test_console_reaction_picker.py`) stay valid.
   * `chat_screen.py`: ONE new `@on` handler
     (`on_console_agent_steering_submitted`), in the row-cancel
     handler's exact grammar (delegate to the controller; a successful
     post requests one coalesced fleet resync), plus the mechanical
     payload unpack/apply lines and the construction kwarg.
4. `440d73e47` — the mutation-survivor fix (below) + the strengthened
   red.

## Where the bar mounts, and how visibility is decided

Mounted at `#console-agent-steering-bar` inside
`#console-rail-section-body-agent` (the rail's agent body), immediately
before the Back button — the drill-in region's chrome.
`ConsoleAgentController._console_agent_steering_state` decides
visibility; the widget only applies it (at construction and on every
`_sync_console_agent_section` apply):

* not drilled in (`_console_agent_drilldown_run_id` empty) → hidden
  (the overview never steers);
* drill-in scoped to another conversation → hidden (the same Finding-C
  self-heal scope check `_console_agent_section_lines` runs);
* the drill target is matched against `bridge.fleet_snapshot` handles in
  BOTH vocabularies (the drill state holds a RUN id; a row click's
  identity is a handle id) — no match → hidden (historical/resumed
  children have no live handle, so they are hidden by construction);
* matched but `status in TERMINAL_RUN_STATUSES` → hidden (a finished
  child takes no more model turns; continuation is supervisor-only,
  spec §1);
* otherwise visible, with `target_id` = the matched handle's HANDLE id
  (the mailbox's own key; the vocabulary the bridge resolves first) and
  `queued` = the copy's `queued_steering`.

## The queued-count path

One source of truth end to end: the coordinator's mailbox dict →
`FleetHandle.queued_steering`, COMPUTED onto every `get()`/`snapshot()`
copy (Task 1's design — never stored on the live handle) →
`bridge.fleet_snapshot` returns those copies → two independent surfaces
render it: `_fleet_row_from_handle` appends `· steering queued (N)` to
the row's secondary line (overview), and
`_console_agent_steering_state.queued` feeds the bar's own
`steering queued (N)` line (drill-in). The child's drain empties the
mailbox; the next sync's snapshot copies read 0 and both surfaces clear.
No cache, no decrement bookkeeping — the count can never disagree with
what the drain would deliver.

## The reds — before/after, painted-frame-asserted

Stage 0 (untouched branch, measured before any implementation): the
bridge suite fails **10/10** at `AttributeError: 'ConsoleAgentBridge'
object has no attribute 'steer_subagent'`; the UI suite dies at
collection (`ModuleNotFoundError: No module named
'tldw_chatbook.Widgets.Console.console_agent_steering_bar'`). After
commit 2 the bridge suite reads **10 passed**; after commit 3 the UI
suite reads **10 passed**.

| Red | Before | After |
|---|---|---|
| input paints in a LIVE child's drill-in | collection ImportError | pass — `_assert_painted_at_own_region` on the Input itself (compositor hit-test), full display-chain walk |
| does NOT paint for a finished child's drill-in | ImportError | pass — the drill-in itself is real (status header asserted) but `bar.display` is False |
| does NOT paint in the overview | ImportError | pass — live child on the fleet list, no drill-in, bar hidden |
| submit → mailbox holds the exact USER-labeled entry | ImportError | pass — REAL Enter keypress through the REAL screen handler; `drain_steering == [("user", text)]` asserted with the literal; the delegating fake runs the REAL `ConsoleAgentBridge.steer_subagent` over a REAL coordinator |
| queued count paints on the fleet row, clears after a drain | ImportError | pass — row secondary painted at own region with `steering queued (2)`; after `drain_steering` + next sync the segment is gone |
| queued line paints in the drill-in, clears after a drain | ImportError | pass — bar's own line painted (`steering queued (1)`), hidden after drain + sync |
| empty submit inert (never a refusal naming `''`) | ImportError | pass — no steer call, nothing queued, note line empty and hidden |
| empty-TARGET submit posts no message at all | ImportError (strengthened mid-round, see mutations) | pass — post_message spy sees zero `SteeringSubmitted` |
| oversize refused with this bar's own painted copy | ImportError | pass — note painted at own region, names "too long" + 4000, draft kept, nothing posted anywhere |
| at-cap accepted | ImportError | pass — exactly `MAX_STEERING_CHARS` chars queued |

Bridge-level reds (all from the same stage-0 AttributeError to pass):
exact-label by handle id; run id reaches the same mailbox; handle id
beats a colliding run id; text posted stripped; empty/whitespace
refused; oversize refused + at-cap accepted (boundary-exact); empty
row_id refused; terminal target refused by both vocabularies; unknown
conversation/id refused; steering never cancels (status/finished_at/
result/error untouched, queued computed onto the next copy).

## Mutations — seven runs, six mutants, ONE first-round survivor

| Mutation | Result |
|---|---|
| M1 label user→supervisor (mandated) | killed — 5 bridge + 2 UI deaths, every exact-label assertion incl. the UI end-to-end |
| M2 visibility inverted (mandated) | killed — 6 UI deaths incl. BOTH owners (live-paints and finished-does-not-paint) |
| M3a row queued read from `total_tokens` (mandated, row surface) | killed — exactly the fleet-row queued test |
| M3b bar queued read from `total_tokens` (mandated, bar surface) | killed — exactly the drill-in queued test |
| M4 empty-target guard dropped (mandated) | **SURVIVED first round** — then killed exactly by the strengthened test |
| M5 cap `>`→`>=` | killed — exactly the two at-cap boundary tests |
| M6 run-id-before-handle-id | killed — exactly the collision test |

**The survivor's lesson.** The empty-target defense is LAYERED: the
widget guard, the controller's `not target_id` arm, and the bridge's own
refusal. My original red asserted the shared OUTCOME ("nothing reaches
the bridge") — which the controller layer satisfies with the widget
guard deleted, so the mandated mutant survived. An outcome-level
assertion is vacuous for any single layer of a layered guard; the red
now pins the widget layer itself (a `post_message` spy asserts NO
`SteeringSubmitted` posts at all), was run RED against the live mutant
before restoring, and the re-applied mutant dies exactly there. All
restores were Edit-based with `git diff` proof (0 files changed after
each restore; the only deliberate production delta after the round is
the guard's comment now citing its pinning test).

## Gate (read counts; baselines on the untouched branch first)

| Suite | Baseline | Final |
|---|---|---|
| `Tests/Chat/test_console_agent_bridge_steering.py` (new) | — (10 failed, AttributeError) | **10 passed** |
| `Tests/UI/test_console_agent_steering_bar.py` (new) | — (collection ImportError) | **10 passed** |
| `Tests/UI/test_console_fleet_panel.py` | 9 passed | **9 passed** |
| `Tests/UI/test_console_agent_rail.py` | 33 passed | **33 passed** |
| `Tests/UI/test_console_agent_fleet_sync_coalescing.py` | 3 passed | **3 passed** |
| `Tests/Agents/test_fleet_steering_mailbox.py` | 22 passed | **22 passed** |
| `Tests/Agents/test_fleet_send_to_agent.py` | 15 passed | **15 passed** |
| `Tests/Chat/test_console_agent_bridge.py` | 197 passed | **197 passed** |
| `Tests/Agents/` (full) | 1499 passed | **1499 passed** |
| `Tests/UI/test_console_reaction_picker.py` (rail-constructor consumer) | — | **38 passed** |
| `Tests/test_probe_import_provenance.py` | 1 passed (probe names this worktree) | **1 passed** |

No TCSS was touched (the widget owns explicit inline styles), so the CSS
bundle-sync guard was not in scope; the bundle is untouched in the diff.

**Pre-existing dev red, measured at the merge-base before any change of
mine (attribution rule):**
`Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_
grow_past_its_budget` fails on untouched `22d156155` — `chat_screen.py`
is at 20,367 lines against the 17,727 budget, i.e. dev is ~2,640 lines
over its own ratchet BEFORE this task. This task's screen delta is the
one `@on` handler plus mechanical unpack/apply/kwarg lines (~50 lines);
all behavior lives in `UI/Console_Modules/` and `Widgets/Console/` per
the ratchet's own guidance. The ratchet stays red either way; the
breach is dev's and predates this branch.

### The one-process app-lifetime gate

The WHOLE Console test population in ONE pytest invocation —
`Tests/UI/test_console_*.py` (169 files) + `Tests/Chat/test_console_*.py`
(100 files) + `Tests/Agents` + `Tests/test_probe_import_provenance.py` —
run twice: once in a REAL pre-change baseline worktree
(`.worktrees/steer-t3-base`, checked out at the merge-base
`22d156155`), once on the final branch. Verdict by failure-SET
comparison, not raw totals.

<!-- ONE_PROCESS_GATE_RESULTS -->

## Notes and concerns for Tasks 4–5

- **Task 4 (continuation)**: `steer_subagent` resolves over
  `coordinator.snapshot()` LIVE handles only and returns a silent bool —
  when continuation lands, the PANEL still must not grow a resume path
  (spec §1: supervisor-only). If Task 4's retention changes what
  `fleet_snapshot` returns for finished children, re-check
  `_console_agent_steering_state`'s terminal arm: it hides the bar for a
  TERMINAL status, and a retained-but-terminal handle must keep hitting
  that arm, not the visible one. The finished-child UI red
  (`test_steering_input_does_not_paint_for_a_finished_childs_drill_in`)
  is the tripwire.
- **Task 4**: `retain_transcript` claims the undelivered mailbox remnant
  — the row's `· steering queued (N)` segment reads the SAME mailbox, so
  after retention claims it the count correctly reads 0; no UI change
  needed, but a test asserting a finished child's row never shows a
  stale queued count would be a cheap pin.
- **Task 5 (Stop semantics)**: the steering bar targets whatever
  `fleet_snapshot` says is live. When Stop stops cancelling survivors
  (outlive ON), a drilled-in survivor keeps its bar mid-Stop — that is
  correct (it is still running) but worth one painted-frame assertion in
  Task 5's suite so "Cancel all" and the bar's visibility are proven to
  move together.
- **Layered guards** (this task's mutation lesson, generalizable to Task
  5's cancel-all): when the same refusal exists at widget, controller,
  and bridge layers, every layer's test must observe THAT layer's own
  seam (message posted / method called), never just the shared outcome —
  or its mutation will survive.
- The `_console_agent_section_last` payload is now an 8-tuple; anything
  Task 5 adds to the Agent section sync (e.g. a Cancel-all affordance
  state) should extend the same payload rather than adding a second
  equality guard.
