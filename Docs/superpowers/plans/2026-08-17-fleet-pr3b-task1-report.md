# Fleet PR 3b — Task 1 landing report: mailbox + drain seam

Branch `feat/fleet-3b-mailbox`, from origin/dev `e0d47f4be`. Plan:
`2026-08-17-fleet-pr3b-steering.md`, Task 1 (child-side plumbing, no
producer). Spec bindings: §6 (protocol-coherent drain, source labels),
§3 invariant 4 (steering never cancels).

## Citation-base verification

Every seam-map line this task relies on was re-verified at
`e0d47f4be`. `git diff a2b621c80..e0d47f4be -- tldw_chatbook/Agents/
Tests/Agents/` is EMPTY (the only commits between are the plan-docs
merge), so the plan's `a2b621c80` citations hold byte-for-byte at this
branch's base. **The code had not moved; the drain landed exactly where
the plan reasoned it must** — the non-restoring branch (`agent_runtime.py`
`:901-911` at the citation base), immediately before the model call,
after the budget/cancel checks (`:880-893`).

## What landed (four commits, pushed incrementally)

1. `804e8a5d5` — `agent_models.py` (pure): `STEP_STEERING`,
   `STEERING_SOURCE_SUPERVISOR`/`STEERING_SOURCE_USER`,
   `MAX_STEERING_CHARS = 4000`, and the ONE formatter
   `format_steering_message(source, text)` →
   `"[Steering from {source}] {text}"`.
2. `744936225` — `fleet_coordinator.py` (pure, locked):
   `post_steering(handle_id, source, text) -> bool` (False for
   unknown/terminal), `drain_steering(handle_id)` (one locked
   return-and-clear pop), `FleetHandle.queued_steering` COMPUTED onto the
   copies `get()`/`snapshot()` return (stored state lives only in the
   mailbox dict, so a copy can never disagree with the mailbox).
   Mailboxes die with `prune_terminal`; undrained remnants survive
   `finish()` until then (Task 4 claims them at retention time — pinned).
3. `fe55b5cd8` — `agent_runtime.py` (pure):
   `LoopDeps.drain_mailbox: Callable[[], list[tuple[str, str]]] | None`.
   The drain sits in the non-restoring branch immediately before the
   model call; per entry it appends the labeled user-role message, adds a
   `STEP_STEERING` step, and emits a `"steering"` run-log record with the
   source as `status`. Wrapped never-raise (the `on_step` containment
   rule).
4. `a63f011ce` — `agent_service.py`: `_run_one` gains
   `drain_mailbox=None` (the `on_run_id` threading precedent); spawn's
   fleet branch supplies
   `lambda handle_id=handle.handle_id: fleet.drain_steering(handle_id)`
   via `child_kwargs`; primaries and inline children keep `None`.

Suite: `Tests/Agents/test_fleet_steering_mailbox.py` (22 tests).

## The seven reds — each measured failing before its piece landed

| Red | Test(s) | Before | After |
|---|---|---|---|
| (a) boundary delivery, exact `messages` sequence, BOTH protocols | `test_red_a_fence_…`, `test_red_a_native_…` | `TypeError: LoopDeps.__init__() got an unexpected keyword argument 'drain_mailbox'` | pass — exact 4-/5-message sequences asserted, steering last, after every tool result |
| (b) never interleaves among `role:"tool"` results | `test_red_b_two_native_batches_…` (two consecutive 2-call batches, posts between dispatches, plus a pairing-scan over EVERY payload) | same `TypeError` | pass — exact 9-message final payload |
| (c) restore-batch path never drains | `test_red_c_restore_batch_path_never_drains` | same `TypeError` | pass — `order == ["invoke", "drain", "model"]`; entry delivered after the restored rows, RUN_DONE |
| (d) drain under ACTIVE checkpoint → no `continuation_error` | `test_red_d_…` | same `TypeError` | pass — full barrier cycle `[ToolBatchReady, Executing, Finished, Final]`, RUN_DONE, no error step |
| (e) raising drain never aborts | `test_red_e_a_raising_drain_…` | same `TypeError` | pass — RUN_DONE, drain retried at every boundary |
| (f) concurrent post/drain thread-safety | `test_red_f_concurrent_post_and_drain_…` (4 posters × 50 entries vs 2 drainers) | `AttributeError: 'FleetCoordinator' object has no attribute 'post_steering'` | pass — 200/200 delivered exactly once |
| (g) a dead run never consumes a mailbox | 4 tests: loop-top cancel, mid-batch cancel, budget-exhausted, cycle-stuck | same `TypeError` | pass — zero drains for pre-posted entries on dead runs; late posts stay queued on the coordinator |

Service-wiring reds (Task 1's `agent_service` deliverable): the
end-to-end delivery test failed with the child's second payload ending
at the tool result (no steering message), and the `run_agent_loop` spy
saw `drain_mailbox=None` on the fleet child. One deliberate
characterization PIN (not a red, labeled as such in the test):
`test_inline_children_and_their_primary_stay_unwired` — current behavior
is already correct; wiring them would be the regression it catches.

## The four mutations — all killed, restores Edit-based with `git diff` proof

| Mutation | Kills (owner in bold) |
|---|---|
| 1. drain moved to AFTER the assistant-echo append | 9 died: **(a) fence, (a) native, (b)** + (c), (d), (e), both late-post (g)s, end-to-end |
| 2. label dropped from the formatter call (`steer_message = steer_text`) | 6 died: **both (a)s' and (d)'s label assertions** + (b), (c), end-to-end |
| 3. drain added in the restoring branch | exactly 1 died: **(c)** |
| 4. drain moved BEFORE the budget/cancel checks (non-restoring-guarded, isolating the ordering claim) | 2 died: **(g) loop-top-cancelled, (g) budget-exhausted** |

No survivors. Note on mutation 4: the two late-post (g) variants
(mid-batch cancel, cycle-stuck) pass under it BY CONSTRUCTION — their
entries are posted after the loop-top boundary, so no pre-check drain
can reach them; their claim (mid-batch death leaves the mailbox intact)
is untouched by this mutant and is killed by mutation 1 instead. Every
test in the suite is killed by at least one of the four mutants except
the pure constant/coordinator pins, which the stage reds covered
(TypeError/AttributeError before their pieces landed).

## Purity

`agent_models.py`, `fleet_coordinator.py`, `agent_runtime.py`: the
branch diff adds NO imports of any kind to any of them (verified by
grep over `git diff e0d47f4be..HEAD`) — the runtime's two new names come
through its existing `from .agent_models import` block. No I/O, no
config reads, stdlib only.

## Gate (read counts; baselines taken on the untouched branch first)

| Suite | Baseline | Final |
|---|---|---|
| `Tests/Agents/test_fleet_steering_mailbox.py` (new) | — | **22 passed** |
| `Tests/Agents/test_fleet_runtime.py` | 107 passed | **107 passed** |
| `Tests/Agents/` (full) | 1462 passed | **1484 passed** (= 1462 + 22 new) |
| `Tests/Chat/test_console_agent_bridge.py` | 197 passed | **197 passed** |
| `Tests/Chat/test_fleet_settle_fanout.py` | 7 passed | **7 passed** |
| `Tests/test_probe_import_provenance.py` | 1 passed | **1 passed** |

## Decisions and notes for Tasks 2–4

- **Text validation (non-empty, `MAX_STEERING_CHARS`) is NOT in
  `post_steering`.** The plan's global constraint places it "at both
  boundaries" — the producers (Task 2's `send_to_agent`, Task 3's panel
  input), each of which needs its own user-facing refusal copy that a
  silent bool cannot carry. `post_steering`'s contract is exactly the
  plan's: False for unknown/terminal, True otherwise. Task 2 must not
  assume the mailbox validates for it.
- **`queued_steering` is computed at copy time**, not stored on the live
  handle — one source of truth (the mailbox dict), so Task 3's
  "queued (N)" surface can trust any `get()`/`snapshot()` copy.
- **Undrained remnants survive `finish()` until `prune_terminal`** —
  pinned by `test_undrained_entries_survive_finish_until_prune_terminal`
  so Task 4's `retain_transcript` has the window the plan promises it.
- The drain consumes step budget (each delivery adds a `STEP_STEERING`
  step). At default budgets this is negligible (steering is rare and
  capped by the producers), and it keeps `max_steps` a true ceiling on
  step-log growth.
