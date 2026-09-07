# Fleet PR 3b — steering: send_to_agent, mailboxes, panel steering, finished-agent continuation

**Every file:line citation in this document was read at origin/dev `a2b621c80`
(merge of PR #1771). Re-verify against that commit, not against any older
checkout.**

Task: parent `backlog/tasks/task-13154 - Supervisor-agent-fleet-program.md`;
subtask 13154.3 (filed with this plan).
Spec: `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md`
— §6 "Steering and approvals" is the contract (two paths one mechanism;
protocol-coherent drain; source labels; latency honesty; finished-agent
continuation with in-memory retention and `resumed_from_run_id`;
cross-restart resurrection explicitly out of scope), §3 invariant 4
(steering never cancels), §5 (containment), §7 (drill-in: "phase 3 adds the
steering input + mailbox 'queued' state"), §8 (Stop semantics move here;
3b audits the cost ticker), §10 (the PR row this plan implements).
Predecessors: the PR 2b, 3a-1, 3a-2 plans and the eight
`2026-08-14-headless-wake-*` landing reports (app-owned `ConsoleRuntime`,
store as continuity owner, per-visit cancellation, `_attempt` refusing only
DISPOSED).

## The design, and why

**Mailboxes live on `FleetCoordinator`. The drain is a per-child callable on
`LoopDeps`, built by `spawn`'s fleet branch. The drain point is the single
protocol-coherent line before each model call. Continuation of a finished
child is a NEW fleet child launched through the existing spawn tail, seeded
from an in-memory retained transcript captured at that same coherent
boundary.** Three verified facts decided each piece:

1. **Lifetime and reach.** The bridge owns one `FleetCoordinator` per
   conversation for the app's life and injects it into every turn's service
   (`console_agent_bridge.py:2451`, `:4025-4085`, injection `:3352`). Cancel
   Events are service-local and need the retained-owner walk
   (`agent_service.py:1982`; bridge `:4318-4332`) — but a mailbox on the
   coordinator is reachable from the UI thread, from any turn's supervisor
   (including for a live survivor another turn spawned — steering, unlike
   cancel, needs no per-service state), and from the child's thread, under
   the lock the coordinator already holds (`fleet_coordinator.py:95`).
2. **One coherent point, both protocols.** Nothing today injects into a
   running loop: `messages` is a local (`agent_runtime.py:727`) and
   `LoopDeps` (`:261-384`) carries no message-mutation callable — verified
   at dev tip. Every batch's results are fully appended by the end of the
   `for call in calls:` body via `_append_tool_result` (`:627-652`, call
   site `:1486`) — the native `role:"tool"`-paired message and the fence
   protocol's user-role line alike. The next model call happens only in the
   non-restoring branch (`:901-911`). A drain THERE can never split a
   `tool_calls` ↔ `role:"tool"` pair (the mid-batch returns that would make
   any other point unsafe: `:1068-1069` cancellation, `:1109-1110`
   cycle-stuck), never touches the continuation-restore machinery
   (`expand_restore_history` rewrites `messages[start:]`, `:810-824`,
   `:1482-1484` — the restoring branch is structurally skipped), and is
   safe under an ACTIVE checkpoint (`agent_service.py:908-917`;
   `agent_runtime.py:463-494`).
3. **Continuation rides existing machinery or re-derives all of it.**
   `run_child`'s `finally` (`agent_service.py:2049-2107`) does
   `fleet.finish` → first-writer-wins DB terminal fallback →
   `_on_child_settled`, which the bridge feeds into `FleetDrainFanout` and
   the wake coordinator (`console_agent_bridge.py:3339-3344`,
   `console_fleet_wake.py:479-520`). A resumed child launched through the
   same reserve→thread→handle tail (`agent_service.py:1955-2154`) inherits
   containment (`contain_child_budget`), panel visibility, settle, and the
   wake with zero new wiring.

Two spec-anchored scope pins:

- **Panel steering targets LIVE children only.** Continuation of a finished
  child is supervisor-only (`send_to_agent`): the panel watches/steers,
  never launches (spec §1 owner decision).
- **The "notification chip" in §10's 3b row is already shipped** (PR 3a-2:
  toast + `FLEET_UNSEEN` + `◈`). 3b adds only the mailbox "queued" surface.

## Verified seam map — cite these, do not re-derive

All at `a2b621c80`.

| Seam | Where | Fact |
|---|---|---|
| Loop history is loop-local | `Agents/agent_runtime.py:727` | `messages = list(initial_messages)`; no external mutation path exists |
| LoopDeps | `agent_runtime.py:261-384` | No message hook; `wait_agents`/`check_agents` fields `:344-365` are the threading precedent |
| Drain point | `agent_runtime.py:895-911` | Non-restoring branch, before the model call; budget/cancel checks `:880-893` run first (a dead run never consumes a mailbox) |
| Batch-complete boundary | `agent_runtime.py:627-652`, `:1486` | `_append_tool_result` — native pairing AND fence convention, one source of truth |
| Mid-batch terminal returns | `agent_runtime.py:1068-1069`, `:1109-1110` | Cancel / cycle-stuck return with a possibly split batch — why transcript capture must be boundary-indexed |
| Done-return excludes final text | `agent_runtime.py:1020-1025` | `RUN_DONE` returns BEFORE the assistant append; a retained transcript must append `final_text` itself |
| Restore-slice rewrite | `agent_runtime.py:810-824`, `:1482-1484` | Drains must be structurally impossible mid-restore |
| Fleet spawn tail | `agent_service.py:1955-2154` | reserve `:1956` → cancel Event `:1980-1986` → `run_child` `:1988` → thread `:2109-2145` → handle result `:2147-2154`; `on_run_id` precedent `:1433`, `:2029-2031` |
| Settle chain a resumed child inherits | `agent_service.py:2049-2107` → bridge `:3339-3344` | `fleet.finish` → DB fallback → `_on_child_settled` → fan-out → wake |
| Child budget branch | `agent_service.py:1786-1802` | `turn_scoped = fleet is None or inline`; threaded children get `contain_child_budget` |
| Definition resolution + composition | `agent_service.py:1736-1752`, `:1840-1856`, roster `:3187-3190` | Per-turn roster; instructions APPEND; intersection-never-union |
| Fleet tool gate + wiring | `agent_service.py:1534-1539`, `:2948-2949` | Schemas pinned + deps wired under the SAME `fleet_active` predicate |
| Tool schemas to copy | `Agents/tool_catalog.py:68-150` | `build_spawn_schema`, `WAIT_AGENTS_SCHEMA`, `CHECK_AGENTS_SCHEMA` |
| Identity vocabularies | `agent_service.py:2300-2304` vs `Chat/console_fleet_wake.py:202-209` | check_agents speaks HANDLE ids; the wake notice speaks RUN ids — `send_to_agent` must resolve both |
| Not-user-input / never-approval pin | `console_fleet_wake.py:142-147` | `WAKE_NOTICE_DISCLAIMER` — the guarantee steering labels mirror |
| Approval wait shape | `Agents/human_input_wait.py`; approval round in `console_chat_controller.py` | A pending card blocks the child inside its tool call; "queued" is the honest state |
| Coordinator lock + copies | `Agents/fleet_coordinator.py:95`, `:223-250` | Every public method locked; `get`/`snapshot` return copies |
| prune_terminal | `fleet_coordinator.py:284-313`, caller `console_agent_bridge.py:4084` | Terminal handles dropped at TURN START — retention must not live on handles |
| Cancel thread-path (panel) | `Console_Modules/agent.py:1210-1244` → bridge `:4291-4332` → `agent_service.py:3322-3381` | The path `steer` mirrors (minus the service hop) |
| Rail composition | `UI/Console_Modules/left_rail.py:500-574` | Where the steering bar mounts |
| Row grammar | `Widgets/Console/console_inspector_section.py:60-95`; builders `Console_Modules/agent.py:319-365` | Live row_id IS the handle id |
| Screen handler grammar | `UI/Screens/chat_screen.py:2083-2120` | Minimal `@on` delegation only (ratchet) |
| Screen ratchet | `Tests/Architecture/test_screen_size_ratchet.py:65-67` | Over budget; new code in `UI/Console_Modules/` / `Widgets/Console/` |
| Painted-frame idiom | `Tests/UI/test_console_fleet_panel.py:31` | `_assert_painted_at_own_region` |
| Migration precedent | `DB/AgentRuns_DB.py:290-303`, `:334-345`, drift `:38` | wake-ledger idempotent ALTER; `_CURRENT_SCHEMA_VERSION` still 3 (task-15669); `resumed_from_run_id` does NOT exist (verified) |
| `SELECT *` row dicts | `AgentRuns_DB.py:857`, `:892`, `:916` | A new column flows to reads for free |
| Stop coupling today | `agent_service.py:1985-1986`, `:2220-2223`, `:1290-1296` | `child_should_cancel = should_cancel() or child_cancel.is_set()`; comments cite "spec Sec 10 keeps Stop-semantics changes in PR 3b" |
| Cost rollup | `Console_Modules/agent.py:1176-1208`; `fleet_coordinator.py:54-63` | `total_tokens` set only at finish; §8's 3b audit target |
| App-lifetime gate | `…headless-wake-closeout-report.md:263-272` | The one-process combined suite list |

## What done means

1. `send_to_agent(id, message)` exists as a primary-only, fleet-gated
   runtime tool; a live child sees the message as a
   `[Steering from supervisor]`-labeled user-role message at its next model
   turn, at a boundary that never splits native pairing; the run-log records
   it; the run is never cancelled or restarted by steering.
2. The fleet drill-in offers a steering input for a LIVE child; entries are
   labeled `[Steering from user]`; the panel shows "queued (N)" until the
   child consumes them; painted-frame-asserted.
3. `send_to_agent` to a finished child starts a NEW run seeded with the
   retained transcript (+ undelivered queued steering) + the new message,
   linked via `resumed_from_run_id`, contained like any threaded child,
   visible in the panel, waking the supervisor through the existing settle
   path. After a restart the transcript is gone and the error says so.
4. Steering never satisfies an approval — pinned, mirroring the wake's
   not-user-input guarantee.
5. Stop cancels the supervisor's turn only (when `subagents_outlive_turn`
   is on); the panel offers "Cancel all agents"; turn-scoped mode stays
   byte-identical.
6. §8's cost-ticker audit executed and recorded; docs + stamps updated; a
   live steering + continuation exercise ran against a real provider.

## Global constraints (bind every task)

- Reproduce red before fixing; **measure the red at the MERGE-BASE**.
  Mutation-test every new test; a survivor is a finding to investigate.
- Python ONLY via `pytest`. Never read/write `~/.config/tldw_cli`. Keep
  `Tests/test_probe_import_provenance.py` in every gate.
- Pure modules stay pure: `agent_runtime.py` / `agent_models.py` /
  `fleet_coordinator.py` gain no I/O, no config reads.
- `chat_screen.py` is over its ratchet — new UI code in
  `UI/Console_Modules/` or `Widgets/Console/`; screen deltas are minimal
  `@on` delegations.
- UI assertions are painted-frame, never DOM presence.
- Tasks 3 and 5 run the whole-Console-population-in-one-process gate.
- Steering text validated at both boundaries (non-empty, `MAX_STEERING_CHARS`
  cap); the label is prepended by the mechanism, never trusted from input.
- Regenerate the CSS bundle for any TCSS change; never hand-edit.

## Coordinator rulings on the design pass's escalations (2026-08-17)

1. **Resume vs changed/vanished definition — per recommendation.** A
   definition that still exists re-resolves to its CURRENT form (the new
   row's `definition_fingerprint` records the change — that audit exists
   for exactly this); a deleted/disabled definition refuses with a clear
   error suggesting a fresh spawn, mirroring the existing unknown-agent
   refusal shape. Rationale: silent downgrade to a generic child would be
   the only WRONG option, and both chosen behaviours reuse existing
   patterns.
2. **Oversize transcript — do not retain; error honestly.** Truncation
   risks splitting native pairs and silently changing the child's memory —
   the owner's stability-over-quick-wins ruling decides this.
3. **Fold task-15669's fix into the v8 migration — YES.** Set
   `_CURRENT_SCHEMA_VERSION = 8`, add the agreement test, record the
   resolution; every prior migration has widened the drift and this closes
   a filed task for four lines plus one test.

---

### Task 1 — mailbox + drain seam (pure core, service threading)

The child-side plumbing, complete and tested, with no producer yet.

- [ ] `agent_models.py` (pure): `STEP_STEERING = "steering"`;
  `STEERING_SOURCE_SUPERVISOR` / `STEERING_SOURCE_USER`;
  `MAX_STEERING_CHARS` (4000, the `max_subagent_result_chars` shape); pure
  `format_steering_message(source, text)` →
  `"[Steering from {supervisor|user}] {text}"` — one formatter so the loop,
  the run log, and tests can never drift.
- [ ] `fleet_coordinator.py` (pure, locked): `post_steering(handle_id,
  source, text) -> bool` (False for unknown/terminal), `drain_steering(
  handle_id) -> list[tuple[str, str]]` (return-and-clear under the lock),
  `FleetHandle.queued_steering: int = 0` on copies. Mailboxes die with
  `prune_terminal` (Task 4 claims remnants at retention time).
- [ ] `agent_runtime.py`: `LoopDeps.drain_mailbox: Callable | None = None`.
  In the non-restoring branch (`:901-911`), immediately before the model
  call: drain; per entry append the labeled user-role message,
  `add(STEP_STEERING, …)`, `_emit_record(deps, "steering", …)`. Wrapped
  never-raise (the `on_step` containment rule `:739-742`).
- [ ] `agent_service.py`: `_run_one` gains `drain_mailbox=None`; `spawn`'s
  fleet branch supplies `lambda: fleet.drain_steering(handle_id)` in
  `child_kwargs` (`:1884-1909`); primaries and inline children get `None`.
- [ ] Reds: (a) mid-batch post delivers only at the next boundary, exact
  `messages` sequence asserted for BOTH protocols; (b) a multi-call native
  batch never interleaves the injected message among `role:"tool"` results;
  (c) the restore path never drains; (d) drain under an ACTIVE checkpoint
  produces no `continuation_error`; (e) a raising drain does not abort the
  run; (f) concurrent post/drain under threads; (g) a cancelled/stuck/
  budget-exhausted run leaves entries queued.
- [ ] Mutations: drain after the assistant append → (a)/(b) red; label
  dropped → label red; drain in the restoring branch → (c) red; drain
  before the budget checks → (g) red.

### Task 2 — `send_to_agent` for live children (supervisor path)

- [ ] `agent_models.py`: `SEND_TO_AGENT_TOOL_NAME`, added to
  `RUNTIME_TOOL_NAMES` (`:63-77`).
- [ ] `tool_catalog.py`: `SEND_TO_AGENT_SCHEMA` beside `WAIT_AGENTS_SCHEMA`:
  `{id, message}` required; description teaches both id vocabularies,
  delivery latency, and never-cancels.
- [ ] `agent_service.py`: closure beside `wait_agents` (`:2156`): resolve
  live handle id, else a live handle's `run_id`, over the WHOLE coordinator
  (foreign survivors steerable); validate; `post_steering(…, SUPERVISOR,
  …)`; ok-copy states "queued; delivered before its next model turn".
  Terminal/unknown → error naming known live ids (Task 4 upgrades this
  branch). Schema + dep under the exact `fleet_active` predicate; in-loop
  dispatch in the `wait_agents` shape (`:1252-1273`).
- [ ] Reds: schema absent without a fleet / for a subagent; end-to-end via a
  fake provider (child's next payload contains the labeled message);
  refusal shapes; **steering never cancels** (status/row untouched, no
  Event set); **steering never satisfies an approval** (round armed and
  held → verdict pending, tool not executed, entry queued).
- [ ] Mutations: run-id-before-handle-id with a colliding fixture →
  resolution red; wired under `agent_kind == SUBAGENT` → gate red.

### Task 3 — panel steering input + queued state (user path)

- [ ] `console_agent_bridge.py`: `steer_subagent(conversation_id, row_id,
  text) -> bool` — coordinator lookup, live-handle resolution (both
  vocabularies), `post_steering(…, USER, …)`. No service hop; UI-thread
  safe.
- [ ] New `Widgets/Console/console_agent_steering_bar.py`
  (`ConsoleAgentSteeringBar`): compact `Input` + queued-count line; posts
  `SteeringSubmitted`. Mounted in the drill-in region of the rail's agent
  body; visible only while drilled into a LIVE child.
- [ ] `Console_Modules/agent.py`: controller method resolving the drill-in
  target → `bridge.steer_subagent`; sync payload gains visibility + queued
  count; `_fleet_row_from_handle` appends `· steering queued (N)`.
  `chat_screen.py` gets ONE minimal `@on` delegation.
- [ ] Reds: painted-frame — input paints in a live drill-in, does NOT paint
  for finished/overview; submit → USER-labeled entry; queued count paints
  and clears after a simulated drain; mislabeling caught.
- [ ] Gate: this PR's suites + fleet panel + coalescing + the one-process
  app-lifetime gate.

### Task 4 — finished-agent retention + continuation (+ v8 migration)

- [ ] `agent_runtime.py` (pure): track `coherent_len` at each drain
  boundary; `RunOutcome.final_messages: list[dict] | None = None`; every
  terminal return sets `messages[:coherent_len]`, plus the final assistant
  append on `RUN_DONE` (`:1020-1025`). `_persist` untouched —
  `final_messages` never reaches the DB.
- [ ] `fleet_coordinator.py` (pure): retention keyed by BOTH handle id and
  run id, separate from `_handles` (survives `prune_terminal`); caps via
  constructor + `set_retention_caps` (the `set_max_live` shape); config
  `[agents] retained_transcripts` (5) / `retained_transcript_max_chars`
  (200_000) read beside `max_live` in the bridge. `retain_transcript`
  claims the undelivered mailbox remnant. Retained: done/stuck/error;
  cancelled/superseded are not. Oversize → not retained (ruling #2).
  Oldest-evicted-first.
- [ ] `agent_service.py`: `run_child`'s finally calls `retain_transcript`
  after `fleet.finish`. Extract spawn's reserve→Event→thread→handle tail
  into `_launch_fleet_child` (mechanical; spawn byte-identical).
  `send_to_agent`'s terminal branch becomes continuation: spawn-slot +
  live-cap accounting, definition re-resolution per ruling #1,
  `contain_child_budget`, seed = retained + undelivered-queued (original
  labels) + new supervisor-labeled message, `parent_run_id` = current
  primary, `resumed_from_run_id` = old run id, launched via
  `_launch_fleet_child`.
- [ ] `AgentRuns_DB.py`: `resumed_from_run_id TEXT` — idempotent ALTER in
  the wake-ledger shape, version row 8, `create_run` kwarg. Ruling #3:
  `_CURRENT_SCHEMA_VERSION = 8`, agreement test, docstring note — closing
  task-15669. Drill-in header gains "resumed from <id>".
- [ ] Reds: coherent-boundary property (every terminal path yields no
  unpaired native batch — Hypothesis-worthy); retention caps + eviction +
  oversize refusal; post-restart error copy; resumed row lineage +
  fresh `definition_fingerprint`; undelivered queued entries ride the
  seed; migration idempotency (open a pre-v8 DB twice); live-cap and
  spawn-budget refusals for resume.
- [ ] Mutations: capture full `messages` at the cancel return → coherence
  red; retain under `prune_terminal`'s dict → prune red; skip the
  spawn-slot consume → budget red.

### Task 5 — Stop semantics + "Cancel all agents"

The highest-risk task. Probe first, at the merge-base.

- [ ] Merge-base probes: (a) outlive ON — a user Stop today cancels a child
  that should survive (`child_should_cancel`'s `should_cancel()` term
  `:1985-1986`; `_surviving_handles` `:1290-1296`); (b) outlive OFF — Stop
  kills everything via cancel Events; must stay byte-identical.
- [ ] Change, branched on the existing key: outlive ON →
  `child_should_cancel` drops the parent-poll term; `wait_agents`' cancel
  branch stops waiting WITHOUT cancelling ("sub-agents continue in the
  background"); `_surviving_handles` keeps survivors on user cancel.
  Outlive OFF → byte-identical.
- [ ] Audit: conversation-deletion / ephemeral-close teardown still cancels
  the fleet (cancel Events, not the poll); pin it.
- [ ] Bridge: `cancel_all_subagents(conversation_id)` — current service +
  retained owners, every live handle, count returned. Panel "Cancel all"
  affordance, enabled only with live rows; painted-frame test; the
  per-handle revocation path reused, no second mechanism.
- [ ] Mutations: re-add the parent-poll term → survivor-on-Stop red;
  cancel-all skips retained owners → survivor-cancel red.
- [ ] Gate: the one-process app-lifetime gate.

### Task 6 — cost-ticker audit, docs, live verification

- [ ] §8's audit, executed: (a) a resumed child's `total_tokens` reaches
  the fleet rollup at finish; (b) record the honest gap — a finished
  survivor's spend leaves `fleet_snapshot` between turns; FILE the
  follow-up, do not patch it here.
- [ ] User Guide: steering (two paths, labels, queued/latency, continuation
  + restart limit, Stop semantics + Cancel all) + stamp. Spec §7/§10
  shipped-notes.
- [ ] Backlog: 13154.3 notes; task-15669 closed per ruling #3; lessons only
  if earned.
- [ ] Live (before merge): steering mid-run (panel + send_to_agent), one
  finished-agent continuation, one steer-while-approval-pending (card
  unaffected; steering delivered after the round resolves).

## Deliberately NOT in this plan

- Cross-restart resurrection of finished agents (spec §6: out of scope).
- Panel-initiated continuation (the panel watches/steers, never launches).
- Steering the PRIMARY mid-turn (the user steers it by talking to it).
- A new completion chip (shipped in 3a-2).
- Steering for inline children (no handle, no mailbox; input hidden).
- Pausing/extending the approval deadline (fail-closed gate untouched).
- Per-definition `max_wall_seconds` + starter library (phase 4).
- A check_agents "continuable roster" (a third id surface without need).
- Fixing the finished-survivor spend-attribution gap (audited + filed).
