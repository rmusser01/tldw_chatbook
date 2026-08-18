# Fleet PR 3b — Task 4 landing report: finished-agent retention + continuation (+ v11 migration)

Branch `feat/fleet-3b-continuation`, from origin/dev `01ac5d8ad` (the
Task 3 merge, PR #1793). Plan: `2026-08-17-fleet-pr3b-steering.md`,
Task 4, under its three coordinator rulings; Task 2's binding concern
(a) — the terminal-id lookup only sees coordinator handles and
`prune_terminal` drops them at turn start — is closed here by resolving
continuation against the retention store FIRST. One mid-task coordinator
instruction (the Qodo retention-race finding on plan PR #1773) is
honored with a design change recorded below.

## What landed (seven commits, pushed incrementally)

1. `b78586756` — the red suites, committed failing (stage-0 measured on
   the untouched branch): `Tests/Agents/test_fleet_continuation.py`
   (collection ImportError), 3 DB reds (constant 3 vs version-table 10),
   2 bridge-factory reds, 1 drill-in-header red.
2. `0d1cd36a7` — `agent_runtime.py` (pure): `coherent_len` tracked at
   the drain boundary; `RunOutcome.final_messages` captured on every
   terminal return as `messages[:coherent_len]`, plus the final
   assistant append on RUN_DONE only. `_persist` untouched.
3. `83cb2f221` — `fleet_coordinator.py` (pure, locked): the retention
   store (`RetainedTranscript`, `retain_transcript`, `get_retained`,
   `set_retention_caps`, cap properties), **retention atomic with
   `finish(..., transcript=...)`** (the race closure), separate from
   `_handles` so it survives `prune_terminal`.
4. `618a156b6` — `AgentRuns_DB.py`: `resumed_from_run_id TEXT` (DDL +
   idempotent ALTER + version row **11**), `create_run` kwarg,
   `_CURRENT_SCHEMA_VERSION` 3 → **11**, class-docstring contract —
   closing task-15669.
5. `b6ba82733` — mechanical `_launch_fleet_child` extraction (pure code
   motion; spawn behavior byte-identical, fleet suites 144 = baseline).
6. `dced5aa57` — the continuation: `send_to_agent`'s terminal branch
   resumes retained children; `run_child` passes
   `transcript=final_messages` to `finish`; two Task 2 tests updated for
   the designed behavior change (reasons in-line); schema description
   teaches the new finished-child behavior.
7. `8e223b7b3` — bridge coordinator factory reads
   `[agents] retained_transcripts` (5) / `retained_transcript_max_chars`
   (200_000) beside `max_live` (construct new / `set_retention_caps` in
   place); drill-in header gains `· resumed from <id>`.

## Deviations from the plan, each with its reason

- **v11, not v8.** The plan's "v8 migration" was written when the
  version table topped at 7; TASK-16800's change_notes work consumed
  rows 8–10 before this task ran. Ruling #3's substance (fold the
  task-15669 constant fix into THIS migration, agreement test, docstring
  contract) is intact; only the number moved with dev.
- **Retention is atomic with `finish`, not a separate post-`finish`
  call.** Mid-task coordinator instruction (Qodo race finding on plan PR
  #1773): the plan's `run_child`-finally-calls-`retain_transcript`
  ordering leaves a window where a child answers as terminal but
  `get_retained` misses, and a racing continuation would refuse a child
  that is continuable microseconds later. Retain-BEFORE-finish was
  rejected as unsound: retainability depends on the first-writer-wins
  terminal status only `finish` establishes — a settle-cancel that wins
  must veto retention (pinned:
  `test_finish_with_transcript_respects_first_writer_wins`). Chosen
  design: `finish(..., transcript=...)` performs the transition AND the
  retention in ONE critical section, so the window is unreachable by
  construction — pinned deterministically by a counting-lock test
  (`test_finish_with_transcript_retains_atomically_in_one_critical_
  section`, which kills the two-step mutant at exactly 2 acquisitions),
  plus the honest-refusal half:
  `test_a_cancelled_child_draws_the_honest_not_retained_refusal_not_
  unknown` (a real child never draws the unknown-id copy).
  `retain_transcript` remains public as the standalone seam over the
  same `_retain_locked` rules.
- **Loop-top terminal returns carry the boundary BEFORE the final
  completed round.** `coherent_len` advances only at the drain boundary
  (after the loop-top budget/cancel checks), so a budget-exhausted or
  loop-top-cancelled run retains exactly what the model saw at its LAST
  call — at most one fully-appended round is dropped. Deliberate: the
  one blessed capture point is the same protocol-coherent line the
  steering drain earned; a second capture point would need its own
  restore-machinery proof. Pinned with the reason in-line
  (`test_budget_exhausted_at_loop_top_yields_the_last_boundary`).
- **Two Task 2 tests updated** (designed behavior changes, reasons
  in-line): the terminal-refusal test now pins the not-retained refusal
  (retention caps zeroed — a retainable finished child now RESUMES, and
  the continuation suite owns that path); the collision test's tail now
  asserts the undelivered entry was CLAIMED into retention at finish
  (Task 1's pinned window names Task 4 as the claimant) instead of
  lingering in the mailbox.
- **A retained entry is not consumed by a resume.** The plan is silent;
  entries age out by the count cap. A second resume of the same old id
  forks from the same snapshot — the ok copy names the NEW handle id the
  supervisor should address next.
- One sentence added to `SEND_TO_AGENT_SCHEMA`'s description (the tool
  now resumes finished children; costs a spawn slot; in-memory only) —
  the schema is the supervisor's whole curriculum, and teaching the old
  refusal would be dishonest.

## Resolution order across live / retained / unknown

`send_to_agent` resolves, in order: (1) LIVE handles, handle id then a
live handle's run id, over the whole coordinator — Task 2's behavior
byte-identical (steer, never resume); (2) the RETENTION store, same
vocabulary order (`get_retained`: handle id first, then run id) — a
retained finished child resumes, and because retention survives
`prune_terminal`, this works after the turn-start prune (Task 2 concern
(a), pinned at both the coordinator and service seams); (3) a REAL
terminal handle still on the coordinator with nothing retained →
honest "finished, no retained transcript (cancelled/superseded are never
retained; oversize/evicted are gone), spawn a fresh sub-agent" — never
the unknown copy; (4) a run id only the DATABASE remembers (terminal
subagent row, THIS conversation) → the post-restart copy: "finished in
an earlier session … transcripts live in memory and do not survive an
app restart. Spawn a fresh sub-agent instead."; (5) only then the
unknown-id refusal. Mutation 7 (retention resolved after the terminal
handles) killed six tests — the ordering is load-bearing.

## The migration: before/after

| | Before | After |
|---|---|---|
| `_CURRENT_SCHEMA_VERSION` | 3 | **11** |
| Version rows on a fresh DB | 4,5,6,7,8,9,10 | 4..**11** |
| `agent_runs.resumed_from_run_id` | absent | present (DDL + guarded ALTER; NULL for every pre-existing row) |
| Constant vs table | disagree (the task-15669 drift, widened to 10) | agree, with a tripwire test and a docstring contract |

A pre-v11 file opens twice idempotently (hand-built legacy fixture);
`create_run(resumed_from_run_id=...)` round-trips and flows through
every `SELECT *` read for free.

## Reds — before/after (staged, measured at each stage)

Stage 0 (untouched branch): the continuation suite dies at collection
(`ImportError: DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS`); DB 3 failed
(constant 3 != 10; `create_run` TypeError); bridge 2 failed (missing
keys); header 1 failed. After the runtime commit: still collection
ImportError (staged). After the coordinator commit: **25 passed** —
runtime + coordinator sections green, 9 failed + 7 errors on the
service seam (no retention wiring, no continuation). After the
continuation commit: **35 passed**. DB: 50 → **53**. Bridge: 212 →
**214**. Controller: 6 → **7**.

Highlights per plan-mandated red: the Hypothesis coherence property
(80 examples over random batch shapes × terminal modes, invariant:
`final_messages` == the last model-call payload, +assistant on DONE,
pairing-complete) — red by construction at stage 0, and under mutation
1 it shrank to the MINIMAL counterexample `([(False, 1)], None, False,
True, None)`: one fence round + step-budget exhaustion, where full
capture ≠ boundary. Its strongest class: 3 identical native calls in
ONE batch trip the cycle detector mid-batch with partial `role:"tool"`
results appended — the slice discards the split batch wholesale.

## Mutations — seven runs, zero survivors, Edit-based restores (git diff 0 after each)

| Mutation | Kills |
|---|---|
| 1. full `messages` instead of the coherent slice (mandated) | 4 — the property (falsifying example above) + both mid-batch owners + the budget pin |
| 2. retention dropped by `prune_terminal` (mandated: "retain under prune's dict") | exactly 2 — the prune-window owners (coordinator + service) |
| 3. spawn-slot consume skipped (mandated) | exactly 1 — the budget-refusal owner (a third row appears, no refusal) |
| 4. retention as a SECOND critical section after finish | exactly 1 — the counting-lock atomicity pin (2 acquisitions), deterministic |
| 5. new supervisor message seeded BEFORE the queued remnant | exactly 1 — the seed-order owner |
| 6. definition re-resolution dropped | exactly 2 — both ruling-#1 owners (fresh fingerprint + deleted-definition refusal) |
| 7. terminal handles resolved before the retention store | 6 — every un-pruned resume path (the prune-window test survives BY CONSTRUCTION: its handle was pruned, so order is moot there) |

## Gate (read counts; baselines on the untouched branch first)

| Suite | Baseline | Final |
|---|---|---|
| `Tests/Agents/test_fleet_continuation.py` (new) | — (collection ImportError) | **35 passed** |
| `Tests/Agents/test_fleet_steering_mailbox.py` | 22 passed | **22 passed** |
| `Tests/Agents/test_fleet_send_to_agent.py` | 15 passed | **15 passed** |
| `Tests/Agents/test_fleet_runtime.py` | 107 passed | **107 passed** |
| `Tests/Agents/` (full) | 1499 passed | **1534 passed** (= 1499 + 35) |
| `Tests/DB/test_agent_runs_db.py` | 50 passed | **53 passed** |
| `Tests/Chat/test_console_agent_bridge.py` | 212 passed | **214 passed** |
| `Tests/UI/test_console_agent_controller.py` | 6 passed | **7 passed** |
| `Tests/test_probe_import_provenance.py` | 1 passed (names this worktree) | **1 passed** |

The one-process Console-population gate was NOT run — this task has no
screen wiring (the header change is a text composition inside the
existing controller path, covered by its own harness test), and the
plan assigns that gate to Tasks 3 and 5.

## Notes and concerns for Tasks 5–6

- **Task 5 (Stop semantics)**: `child_should_cancel` still carries the
  parent-poll term; when it drops (outlive ON), a user Stop will leave
  survivors running — their eventual finishes retain normally. But the
  OTHER half matters for retention: "Cancel all agents" and the
  conversation-teardown audit cancel via Events → the children return
  RUN_CANCELLED → `finish(CANCELLED, transcript=...)` refuses retention
  by status. If Task 5 adds any path that finishes handles DIRECTLY
  with a non-cancelled status (e.g. a timeout/abandonment status), check
  it against `RETAINED_TRANSCRIPT_STATUSES` deliberately — an
  unrecognized terminal status is silently not retained.
- **Task 5**: `_settle_fleet`'s abandonment path calls
  `fleet.finish(..., RUN_CANCELLED)` without a transcript; if it wins
  the race against a still-unwinding child, that child is (correctly)
  not retained. If Task 5 changes who wins that race for survivors, the
  first-writer-wins retention pin
  (`test_finish_with_transcript_respects_first_writer_wins`) is the
  tripwire.
- **Task 6 (cost audit)**: a RESUMED child's `total_tokens` reaches the
  rollup through the same `finish` call as any child — but its spend is
  the NEW run's only; the plan's §8 audit should note a continued task's
  aggregate spend spans two runs linked by `resumed_from_run_id` (the
  DB can join them; `fleet_snapshot` cannot).
- **Task 6 (docs)**: the User Guide steering page should state the
  continuation contract exactly as the refusals do: resume costs a
  spawn slot; retention is per-conversation, in-memory, capped at
  `[agents] retained_transcripts` (5) / `retained_transcript_max_chars`
  (200_000), 0 disables; cancelled/superseded never resume; restarts
  forget transcripts.
- **Live verification (Task 6)**: the finished-agent continuation
  exercise should include one resume-by-run-id AFTER at least one new
  turn (so `prune_terminal` has really run) — that is the path Task 2
  predicted would break and this task's retention store exists to serve.
- The retained-entry-not-consumed choice (fork-on-second-resume) is
  deliberate but undocumented to users; if Task 6's live pass finds it
  confusing, a one-line "already resumed as <id>" hint in the ok copy
  of a SECOND resume would be a cheap follow-up.
