# Fleet PR 3b — Task 2 landing report: `send_to_agent` for live children

Branch `feat/fleet-3b-send-to-agent`, from origin/dev `3b069d0be` (the
Task 1 mailbox merge, PR #1776). Plan:
`2026-08-17-fleet-pr3b-steering.md`, Task 2 (the supervisor producer for
Task 1's per-child mailbox). Spec bindings: §6 (two paths one mechanism,
latency honesty), §3 invariant 4 (steering never cancels). Task 1's
binding decision honored: **`post_steering` does not validate — the
producer does**, with its own refusal copy per shape.

## The surviving red suite

A prior agent's untracked `Tests/Agents/test_fleet_send_to_agent.py`
survived in the worktree. Verdict: **usable, and used** — every
plan-mandated red was present and correctly constructed against the real
fixtures (`make_fleet_service`/`FleetChat`/`CARD_CFG`), including the
armed-and-held approval round and the forged handle/run-id collision.
One repair was needed (below); everything else landed verbatim. It was
committed in the first push for loss protection.

## What landed (four commits, pushed incrementally)

1. `81e273606` — `agent_models.py` (pure): `SEND_TO_AGENT_TOOL_NAME =
   "send_to_agent"` joins `RUNTIME_TOOL_NAMES`; the two exact-set pins
   (`test_agent_models.py`, `test_install_skill_runtime_tool.py`) grow
   the name; the red suite lands.
2. `661476d07` — `tool_catalog.py`: `SEND_TO_AGENT_SCHEMA` beside the
   fleet schemas. `{id, message}`, BOTH required (unlike `wait_agents`'
   optional `ids` — an id-less or message-less steer has no meaning).
   The description teaches both id vocabularies (handle ids from spawn
   results/`check_agents`; run ids from completion notices), honest
   latency (next model turn; a long tool call delays it), and
   never-cancels/never-restarts.
3. `d6000451f` — `agent_runtime.py` (pure): `LoopDeps.send_to_agent`
   (default `None`) + an in-loop dispatch branch in the `wait_agents`
   shape with the same defensive coercion — non-strings become `str()`,
   missing/null become `""`, so junk gets the service's refusal copy,
   never a loop crash. No new imports beyond the `agent_models` name.
4. `e1467b313` — `agent_service.py`: the closure beside `wait_agents`,
   schema pinned + dep wired under the EXACT `fleet_active` predicate
   (primary-only is inherited: `fleet` is `None` for every sub-agent).
   Producer-boundary validation (non-empty after strip;
   `MAX_STEERING_CHARS` cap, boundary-exact). Ok copy: "queued; …
   delivered before its next model turn", naming the RESOLVED handle id.
   Terminal refusal states the child finished and names the live ids;
   the branch carries Task 4's continuation-seam comment. Unknown-id and
   junk refusals name the offending id and the live ids.

## Id resolution: handle id FIRST, then a live handle's run id, whole coordinator

- **Order.** The coordinator minted the handle id as the primary
  vocabulary — spawn results, `check_agents`, and the panel rows all
  speak it — so a pathological collision (child A's run id equals child
  B's handle id) must land on B, the handle-id owner. Mutation 1 proves
  the order is load-bearing.
- **Reach.** Resolution runs over `fleet.snapshot()`, never
  `my_handle_ids`: a live survivor another turn's service spawned is
  steerable, because the mailbox lives on the conversation-lifetime
  coordinator and needs no per-service state — deliberately unlike
  `cancel_subagent`, whose retained-owner walk exists only because
  cancel Events are service-local. Mutation 4 proves the reach is
  load-bearing.
- **Race.** If the target goes terminal between the snapshot and the
  post (`post_steering` returns False), the closure re-snapshots and
  falls into the terminal refusal instead of claiming success.

## The reds — before/after, measured in stages

Stage 0 (untouched branch): the whole suite dies at collection —
`ImportError: cannot import name 'SEND_TO_AGENT_TOOL_NAME'`. After
commit 1: collection ImportError on `SEND_TO_AGENT_SCHEMA`. After
commit 2: **13 failed, 2 passed** (the two registration tests go green).
After commit 3: unchanged 13/2 (the dep is still None everywhere).
After commit 4: **15 passed**.

| Red | Before (post-schema stage) | After |
|---|---|---|
| schema absent without a fleet / present with one | `assert 'send_to_agent' in <fleet primary's system prompt>` fails | pass |
| primary-only + child's hallucinated call refused | child-prompt/parent-prompt assertions fail | pass — child never sees it; its call → "Tool not permitted" |
| end-to-end via fake provider | child's 2nd payload ends at the tool result, no labeled message | pass — `[Steering from supervisor]`-labeled user message LAST, after the tool result |
| empty-message refusal | no ERROR record | pass — own copy ("empty"), nothing queued |
| oversize refusal + at-cap acceptance | no ERROR record | pass — "too long" names 4000; at-cap queued |
| unknown-id refusal | no ERROR record | pass — names `'nope'` AND the live ids |
| terminal id (handle AND run id) | no ERROR record | pass — "finished", "Live sub-agent ids: none" |
| run id reaches the same mailbox | no labeled message | pass — ok copy names the resolved HANDLE id |
| handle id beats colliding run id | no post at all | pass — B's mailbox (0, 1) |
| junk args never crash | crash-adjacent path unwired | pass — coerced `'123'` named in the refusal |
| steering never cancels | no post to measure | pass — mid-turn status/row/Event untouched; ends DONE |
| steering never satisfies an approval | no post to measure | pass — verdict pending, tool unexecuted, entry queued; delivered only at the post-approval boundary |
| foreign live survivor steerable | no post | pass — fresh service, shared coordinator, RUNNING after the post |

## One test repair — attributed at origin/dev BEFORE touching it

`test_steering_never_cancels_the_child` asserted the child's raw cancel
Event is unset AFTER `run_turn` returns. That failed — and the failure
is **origin/dev behavior, not this branch's**: `_settle_fleet` sets
every settling child's Event unconditionally at end of turn, documented
in-line as "for an already-finished fleet every Event set here is
inert". The settle path is byte-identical to origin/dev in this branch's
diff (verified: no settle/cancel lines in `git diff origin/dev`). The
assertion pinned a promise dev never made; it was replaced with the
honest end-state (coordinator DONE + run row DONE, i.e. not CANCELLED)
with a comment citing `_settle_fleet`. The invariant's real measurement
— the mid-turn probe that the POST itself touched neither status, row,
nor Event — stands unchanged.

## Mutations — six, all killed, restores Edit-based with `git diff` proof

| Mutation | Kills |
|---|---|
| 1. run-id resolved BEFORE handle-id (mandated) | exactly `test_a_live_handle_id_beats_a_colliding_run_id` |
| 2. schema wired under `agent_kind == SUBAGENT` (mandated) | primary-only gate test + schema-offered test |
| 3. posted with source `"user"` instead of SUPERVISOR | 5 died — every exact-label payload assertion (end-to-end, run-id, never-cancels, approval, foreign-survivor) |
| 4. resolution scoped to `my_handle_ids` | exactly `test_a_foreign_live_survivor_is_steerable` |
| 5. cap check `>=` instead of `>` | exactly the oversize/at-cap boundary test |
| 6. empty-message refusal dropped | exactly the empty-message test |

No survivors. The registration tests are killed by the stage reds
(collection ImportError before their pieces landed).

## Gate (read counts; baselines on the untouched branch first)

| Suite | Baseline | Final |
|---|---|---|
| `Tests/Agents/test_fleet_send_to_agent.py` (new) | — (ImportError) | **15 passed** |
| `Tests/Agents/test_fleet_steering_mailbox.py` (Task 1) | 22 passed | **22 passed** |
| `Tests/Agents/test_fleet_runtime.py` | 107 passed | **107 passed** |
| `Tests/Agents/` (full) | 1484 passed (new suite excluded) | **1499 passed** (= 1484 + 15) |
| `Tests/Chat/test_console_agent_bridge.py` | 197 passed | **197 passed** |
| `Tests/UI/test_console_mcp_approval.py` | 74 passed | **74 passed** |
| `Tests/test_probe_import_provenance.py` | 1 passed | **1 passed** |

## Notes and concerns for Tasks 3–4

- **Task 3 (panel steering)**: `bridge.steer_subagent` should reuse this
  closure's resolution SHAPE (handle-then-run-id over the whole
  coordinator) but must re-implement validation at ITS boundary with
  panel-facing copy — the same rule Task 1 pinned for this task. The
  `queued_steering` count on `get()`/`snapshot()` copies is already
  computed for its "queued (N)" surface.
- **Task 4 (continuation)**: the terminal branch to upgrade is marked
  in-line in the closure ("PR3b Task 4's continuation seam"). Two facts
  it inherits: (a) the terminal lookup already speaks BOTH vocabularies
  (the terminal-refusal test steers by the finished child's run id too —
  keep that working when the branch becomes a resume); (b) the lookup
  only sees handles still on the coordinator — after `prune_terminal`
  at next turn-start, a finished child's ids fall through to the
  UNKNOWN-id refusal, so continuation must resolve against the retention
  store FIRST or restarted-away children will get the wrong copy.
- **Settle sets Events on finished children** (the repaired test's
  lesson): any future "steering never cancels" or Stop-semantics
  assertion (Task 5) must probe Events MID-TURN or assert
  status/row end-state — a post-`run_turn` raw-Event read measures
  `_settle_fleet`, not the feature under test.
- The empty-string id coercion in the dispatch branch means a missing
  `id` arg reaches the closure as `""` and gets the unknown-id refusal
  naming `''` — acceptable copy, but Task 3's input path should simply
  disable submit on an empty target instead.
