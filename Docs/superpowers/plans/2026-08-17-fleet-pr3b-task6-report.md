# Fleet PR 3b — Task 6 landing report: cost-ticker audit, docs, live pass

Branch `feat/fleet-3b-closeout`, from origin/dev `cf5db6f50` (the Task 5
merge, PR #1808, plus the Qodo-round repairs #1799/#1811). Plan:
`2026-08-17-fleet-pr3b-steering.md`, Task 6 — the close-out. This lands
PR 3b's last row and closes subtask 13154.3.

**Nothing in this task changes production code.** Its deliverables are an
executed audit (with a test), documentation, backlog hygiene, and the
live pass. Everything the audit and the live pass found is FILED, per the
plan's own scope pin.

## Part A — spec §8's cost-ticker audit, executed

**Positive half (verified, pinned, mutation-checked).** A resumed child's
`total_tokens` reaches the fleet rollup through the same seam as any
child: `run_child`'s `finally` calls `fleet.finish(..., total_tokens=…)`
(`agent_service.py`), and a resumed child launches through the shared
`_launch_fleet_child` tail, so it inherits that call with zero new
wiring. The rollup source is
`ConsoleAgentController._console_agent_fleet_token_total`
(`UI/Console_Modules/agent.py`): `sum(handle.total_tokens for handle in
bridge.fleet_snapshot(conversation_id))`.

New test:
`Tests/Agents/test_fleet_continuation.py::test_a_resumed_childs_spend_
reaches_the_fleet_rollup_at_finish` — two real turns, a real resume, then
the exact `_console_agent_fleet_token_total` summation asserted at the
coordinator seam (`rollup == old_spend + resumed_handle.total_tokens`).

**Mutation:** `total_tokens=total_tokens_spent` → `total_tokens=0` at
`fleet.finish`. Killed the new pin AND PR2b's
`test_finished_children_record_their_measured_token_spend_on_the_handle`
(2 failed + 1 error). Restore was Edit-based; `git diff -- tldw_chatbook`
read empty afterwards.

**Negative half — recorded, characterized in the same test, FILED not
patched (TASK-18311):**

1. A finished survivor's per-child spend LEAVES `fleet_snapshot` at the
   next turn's `prune_terminal` (the bridge prunes at turn start), so the
   figure is visible only from finish until the next turn begins. The
   test's tail asserts exactly this (`prune_terminal() >= 2`, rollup back
   to 0) with a comment naming the task, so a future reader cannot
   "fix" the assertion by accident.
2. A continued task's aggregate spend spans two runs and NO surface can
   join them: `agent_runs` persists no token column at all (verified —
   `PRAGMA table_info(agent_runs)` has no token field), so the DB can
   join lineage via `resumed_from_run_id` but not spend, and
   `fleet_snapshot` can do neither.

The chip-level money story is separately covered and was NOT re-litigated
here: `cced002ab`'s `unattributed_fleet_tokens` ("Sub-agents: N tok (not
priced)") plus 3a-2's `FleetDrained` re-attach fold (task-15660).

## Part B — inherited items

### task-18055 — already landed on dev; verified by mutation, not by claim

The plan and Task 5's report both carried this as Task 6's one production
fix. **It was already on dev at this branch's base**: commit `48fd9c623`
("fix(library): cover send_to_agent in the shadow-name set (TASK-18055)",
merged as PR #1789), and the task file reads `status: Done`. `send_to_agent`
sits in `_SHADOWED_BUILTIN_NAMES` (`Library/library_skills_state.py`)
with its reason in place.

Rather than take that on faith, the guard was mutation-checked: deleting
the `"send_to_agent",` entry makes
`Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_
in_sync_with_real_sources` fail naming exactly it —

    AssertionError: Shadow-guard drift -- ALL gaps across ALL sources
    (nothing masked): {'RUNTIME_TOOL_NAMES': ['send_to_agent']}

— and restoring it (Edit-based; `git diff` empty afterwards) returns the
suite to **16 passed**. So the fix is real, the guard bites, and no
production change was owed here.

### The pruned-cancelled-handle refusal gap — reproduced, then FILED (TASK-18312)

Task 5 flagged it; this task **verified it still reproduces at
`cf5db6f50`** with a throwaway probe placed inside `Tests/Agents/`
(sandbox-active placement per `lessons-live-verification.md`), then
deleted the probe. A child cancelled mid-turn, then `prune_terminal`,
then `send_to_agent` by its handle id:

    ERROR: send_to_agent: no sub-agent matches id '292a9e3c…'
    (checked handle ids and run ids). Live sub-agent ids: none.

That is the unknown-id copy for a child that really existed. Filed with
the transcript; **not patched** — the resolution ladder is Task 4's
shipped design (six order-pinning tests) and changing it deserves its own
review, exactly as Task 5 recommended.

### The two controller gateway reds — attributed twice, then FILED (TASK-18313)

`test_controller_real_gateway_budgets_active_continuation_owner_atomically`
and `test_controller_bridge_agent_service_bound_private_history_on_real_send`
fail on a clean `cf5db6f50` tree:

* the first `assert ['old', 'old answer', 'current'] == ['current']`;
* the second `AttributeError: 'list' object has no attribute 'messages_payload'`.

Mechanism found while filing: the shared `ContinuationHistoryGateway`
fake records `self.prepared = messages` — whatever it is handed. The
direct-controller path hands it a `PreparedProviderRequest`; the
agent-bridge path hands it a raw list. That is the same family TASK-16077
reconciled on 2026-08-13 (`347f20ca0`), so the drift re-arrived between
2026-08-14 and 2026-08-17 (**inference** from two dated measurements; not
bisected, and stated as inference in the task). The task's first AC is
to decide whether the bridge/direct divergence is a production contract
bug or a fixture bug.

### Backlog hygiene

IDs swept across all remote refs **and** every worktree before filing
(`git for-each-ref` + `git ls-tree -r -z`, numeric sort), the CLI's own
next-id probed with a throwaway task and deleted, and each subject
ghost-checked against `backlog/tasks/` first. Filed: **18311, 18312,
18313** (one commit) and **18414** (the live-pass find).

## Part C — documentation

`Docs/User_Guide/console/agent-runs-and-tools.md`, every sentence checked
against shipped code rather than the plan:

* **Steering a running sub-agent** — the two paths and their exact
  labels; queued honesty (`· steering queued (N)` on the row, the bar's
  own line) and the delivery-latency truth (next model turn; a long tool
  call delays it); the three never-does (never cancels, never satisfies
  an approval, prefix applied by the mechanism); the 4,000-char cap; the
  primary and inline children excluded.
* **Continuing a finished sub-agent** — supervisor-only; a NEW run with
  the `· resumed from <id>` header; costs a spawn slot; the retention
  caps `[agents] retained_transcripts` (5) / `retained_transcript_max_chars`
  (200 000); cancelled/superseded never retained; restarts forget; the
  second-resume fork.
* **Stopping a run vs. stopping its sub-agents** — a new section: Stop
  cancels the supervisor's turn only, the exact survivor note, the
  connection to auto-wake AND what `autowake_enabled=false` changes about
  what "continue" yields, and the four kill switches (per-row Delete,
  Cancel all agents, session close — now takes the fleet with it, #1808's
  Qodo round — and `subagents_outlive_turn = false`).
* The two now-wrong Stop paragraphs were rewritten, "Cancel all agents"
  documented beside per-row Delete, and the related-settings list gained
  the retention keys and the Stop-semantics role of
  `subagents_outlive_turn`.
* One honest cross-reference added: a resumed child's token figure is the
  new run's own (task-18311).

**Spec** (`2026-08-08-supervisor-agent-fleet-design.md`): §7's drill-in
row and §10's 3b row carry shipped-notes naming the six landings; §8's
audit bullet is marked EXECUTED with the test name and the filed gap; the
config surface now lists the shipped keys.

**Stamp** moved only for what was actually driven live (below).

## Part D — the live pass

tmux, `-L steert6`, 235×52, isolated scratch profile
(`HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`TLDW_CONFIG_PATH` all under the
session scratchpad, `[paths] data_dir` scratch), repo-root Anthropic key,
real model. Isolation verified three ways: `ps -wwE` showed the scratch
`TLDW_CONFIG_PATH`; `lsof` on the PID counted **0** handles under
`default_user`; and after the run the real profile has no `steer_t6_live`
directory and `~/.config/tldw_cli/config.toml`'s mtime never moved off
2026-08-17. 22 panes captured under `…/scratchpad/steer-t6-live/panes/`.

### The headline: the first real send 400'd (FILED as TASK-18414)

Scratch Console configured `provider="anthropic"`, `model="claude-opus-5"`,
shipped default temperature. Every send failed
(`panes/A1-sent.txt`):

    Agent run failed: provider returned HTTP 400 (Provider error from
    anthropic: bad request. Status: 400. Selected model: claude-opus-5.
    The provider rejected this request. Confirm the model is still
    available, or choose another model from the model picker.)

Switching only the model to `claude-sonnet-5` made the same session work
immediately. Reading the request builder afterwards found two
hand-maintained model-name gates in `LLM_Calls/LLM_API_Calls.py`:
`_anthropic_is_sonnet_5()` is the sole suppressor of
`temperature`/`top_p`/`top_k` and matches only `claude-sonnet-5*`, and
`_ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS` omits the Opus 5 tier so it
falls through to the legacy `budget_tokens` branch — both parameters are
400-rejected on the current Opus/Fable tier. **Honest limits of this
finding:** the 400 body was not captured, so which parameter the provider
named is unestablished; the code analysis is a read, not a bisection.
Filed with both halves labeled; **not fixed** — out of Task 6's scope.
The live pass then ran on `claude-sonnet-5`, the family both gates know.

### Scenario results

| # | Scenario | Verdict | Evidence (one line) |
|---|---|---|---|
| 1a | Panel steering of a live child | **PASS** | Typed into the drill-in bar; the child's run record carries `"kind": "steering", "summary": "[Steering from user] STEERING: end your closing paragraph with the exact phrase LE CARRE EST PARFAIT"` and its answer ends `… LE CARRE EST PARFAIT` |
| 1b | Supervisor `send_to_agent` | **PASS** | Transcript shows `⚙ send_to_agent → Steering for 79bc17b3… queued; it will be delivered before its next model turn…`; the child's record carries `[Steering from supervisor] …` and the phrase `DER WURFEL IST PERFEKT` |
| 1c | `queued (N)` surface | **PASS** | `steering queued (1)` painted in the rail (`panes/S4-steered-pending.txt`), and cleared once the child drained |
| 2a | Stop with a survivor | **PASS** | `Response stopped by user.` + `(The run was cancelled; sub-agents continue in the background.)`; the survivor finished **`done`** (never `cancelled`) and its completion auto-woke the supervisor with the `[Background sub-agent completion — automated notice]` row |
| 2b | Steer a live survivor | **PASS** | 1a's child was a survivor of an already-returned turn (spawned no-wait), steered live and delivered |
| 2c | Cancel all agents | **PASS** | Two live children (`01f88fe3`, `b08d3bbb`) → one press → both `cancelled` (cooperatively, ~10 s later); the affordance left the rail on the next sync (grep count 1 → 0) |
| 3 | Finished-agent continuation | **PASS** | `send_to_agent` to a finished child answered `resumed 4303e9a1… as a NEW run: started 46bceec1…, seeded with its retained transcript (35 messages)…`; a SECOND resume forked from the same snapshot; both rows carry `resumed_from_run_id` and the requested phrase; the live drill-in header read **`Sub-agent · running · resumed from …`** (`panes/S3-resumed-header.txt`) |
| 4 | Steer while an approval pends | **PASS** | One frame holds all three: `Approval required` card, `steering queued (1)`, `Approvals: 1 pending`. The steering stayed undelivered across ~90 s and a SECOND card; only after both rounds were approved did the child consume it — `[Steering from user] STEERING WHILE APPROVAL PENDS: mention the word HYPOTENUSE…` and `Closing line: Like calculating the HYPOTENUSE …` |
| 5 | Session close takes the fleet | **PASS** | Confirm dialog `Close Console session? … Live agent turns: 1`; on Close, the live child `747162b1` went `running` → **`cancelled`** on the first poll and the tab left the strip |

### What the live pass did NOT establish

* **Steering a survivor of a *stopped* turn specifically.** Every Stop
  scenario's survivor finished before a steer could land (they complete
  in 10–20 s once released). Steering a live survivor is proven (2b) and
  the stopped-turn variant is unit-pinned by Task 5's probe (a3); the
  seam is identical (the bridge resolves live handles on the
  conversation-lifetime coordinator and cannot see how the turn ended),
  but it was not observed end-to-end live. Stated as a gap, not glossed.
* **`queued (N)` on the *overview row*.** Observed on the drill-in bar;
  on the row the count usually cleared within one sync because the child
  drains at its next model turn. The row surface is painted-frame-tested
  (Task 3) and reads the same mailbox.

### Two live-only observations worth recording (neither filed as defects)

1. **The fleet panel scrolls out of view inside the Agent section body.**
   At 235×52 with all rail sections expanded, `Sub-agents` and
   `Cancel all agents` sit below the fold of the Agent section's own
   scroll area — reachable only by collapsing the sibling sections or
   wheel-scrolling inside the body. That is the shipped known gap the
   User Guide already documents ("There is no 'View all' tail, and
   expanding the panel does not scroll it into view", task-15201); this
   run is a second live sighting of it, not a new defect.
2. **Synthetic SGR clicks are region-selective** (task-17500's lesson,
   third sighting). The tab strip, the rail sections, the panel chevron,
   the steering bar, the approval card's buttons and the close-confirm
   dialog all accepted clicks; the tab's `✕` accepted one only at exactly
   one column (67, not the 68 the character index implies) and the
   drill-in's `(Back)` text never did — Back was reached by tab-hopping
   instead. Harness quirk; the controls are fine.

## Gate (READ counts)

| Suite | Count |
|---|---|
| `Tests/Agents/` + `Tests/test_probe_import_provenance.py` (one invocation) | **1545 passed** (1543 Task-5 baseline + this task's 1 new audit test + the provenance probe) |
| `Tests/Chat/test_console_agent_bridge.py` + `…_steering.py` + `…_cancel_all.py` + `test_console_runtime_lifetime.py` + `test_console_close_session_fleet.py` + `Tests/Library/test_library_skills_state.py` | **262 passed** |
| `Tests/UI/test_console_fleet_panel.py` + `test_console_agent_rail.py` + `test_console_agent_steering_bar.py` + `test_console_agent_controller.py` + `test_console_agent_cancel_all.py` + `test_console_mcp_approval.py` | **142 passed** |
| Per-file confirmations (collect-only) | mcp-approval **74**, close-session-fleet **4**, continuation **36** (35 + 1 new), stop-semantics **9**, mailbox **22**, send_to_agent **15**, shadow-guard **16** |

Every count matches the baselines Task 5's report recorded, plus exactly
the one test this task adds. `test_probe_import_provenance.py` was in the
first invocation and printed this worktree's path.

Owner directive honored: targeted suites only — the fleet/steering
cluster, the bridge and lifetime suites, the Console UI suites this PR's
surfaces live in, the shadow-guard suite touched by the 18055
verification, and the provenance probe. No population run.

## Concerns / open items for whoever picks this up

* **TASK-18414 is the highest-severity thing this task found** and it is
  not a fleet bug at all — it is a request-builder gate that silently
  excludes most of the current Anthropic top tier from Console. It wants
  its own arc.
* **TASK-18311** (cost rollup) needs an owner decision as much as code:
  is per-run spend deliberately ephemeral, or does `agent_runs` gain a
  token column? The audit deliberately did not choose.
* **TASK-18312/18313** are both small and both carry their reproduction
  in the task.
* The live pass ran on `claude-sonnet-5`. Anyone re-running it against
  the Opus tier will hit 18414 first.
