# Fleet reliability repairs implementation plan

**Goal:** Repair the six validated reliability defects from the orchestration review.

**Architecture:** Keep the existing conversation coordinator, source-labeled
mailboxes, explicit continuation, and wake authority. Bound retained state and
retry scheduling, and expose unread terminal steering through handle snapshots.

**Tech stack:** Python, stdlib threading/asyncio, Textual, SQLite, pytest.

**Spec:** `backlog/decisions/129-fleet-mailbox-and-wake-reliability.md`

ADR required: yes
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: bounded admission, snapshot state, and retry policy cross the coordinator,
service, and Console interfaces.

## Global constraints

- Preserve approval authority, tool-result pairing, FIFO delivery, per-run
  budgets, source labels, and explicit continuation.
- No new dependency, schema migration, or default sibling communication.
- Run targeted tests only. Preserve unrelated working-tree changes.
- Every behavioral change gets a failing test before production edits.

## TASK-32485 — restore safety coverage

- [x] Reproduce the existing three failures (already verified at working tree
  and clean HEAD in the review).
- [x] Import the existing legacy-session fixture into
  `Tests/Chat/test_console_fleet_wake_safety.py` so pytest discovers it.
- [x] Run that complete three-test module; require all original assertions.

## TASK-32018 — wake retry and fairness

- [x] Add a refused-provider test counting `resolve_for_send` attempts over a
  bounded window; require at most one attempt before the retry delay.
- [x] Add a two-session test: readiness refuses A, accepts B; B must complete
  while A stays pending. Pin delayed recovery of A and no work after disposal.
- [x] In `Chat/console_fleet_wake.py`, track per-conversation retry deadlines,
  check them before dispatch, and keep one delayed attempt timer. A refusal
  records the next eligible time; success removes its deadline.
- [x] Run wake scheduling, safety, view-mark, and staleness modules.

## TASK-32483 — bounded steering

- [x] Add tests rejecting entry 33 and a 64,001-character aggregate; drain
  frees admission and previously accepted entries remain FIFO.
- [x] Add a retention test where the transcript alone fits but adding unread
  steering exceeds the configured ceiling; no truncated retention is returned.
- [x] Add a service test checking that a live full queue is reported as full,
  never as a finished agent. Verify the Console preserves a refused draft.
- [x] Add limits inside `FleetCoordinator.post_steering`, include unread
  steering in retention sizing, and update producer refusal copy.
- [x] Run mailbox, send-to-agent, continuation, bridge-steering and steering UI tests.

## TASK-32484 — unread terminal steering

- [x] Add a real-loop test posting during the final model call: the model
  never sees the message, and the terminal snapshot reports one unread entry.
- [x] Verify both retained and unretained outcomes in the painted fleet row.
- [x] Expose terminal unread count and retention availability on snapshot
  copies; render recovery guidance without adding an automatic continuation.
- [x] Run continuation, mailbox, and steering UI tests.

## TASK-32486 — obsolete events

- [x] Add a repeated reserve/finish/prune test: no obsolete events remain;
  a live survivor's started event still drains once.
- [x] Remove pruned handles' events within the existing prune critical section.
- [x] Run the coordinator module.

## TASK-32487 — defensive transcript copies

- [x] Add native-tool nested mutation tests against both the input passed to
  finish and the snapshot returned by get_retained.
- [x] Deep-copy retained messages at those two boundaries with stdlib copy.
- [x] Run continuation and mailbox modules.

## Completion

- [x] Self-review the combined patch for races, behavior drift, and accurate copy.
- [x] Run the union of targeted suites and applicable lint/format checks.
- [x] Record exact evidence, unresolved environment limits, and per-task outcomes.
- [x] Mark only fully verified tasks Done through Backlog CLI.

## Verification outcome

- Combined targeted suite: 295 passed in 101.23 seconds.
- Final retention failure-path refinement: 84 affected tests passed.
- Stronger compositor checks: 3 passed; recovery/refusal words are painted in
  the widgets' own regions. These runs overlap.
- New test lint/format, scoped whitespace, and diagnostic inventory checks pass.
  Legacy-file lint counts introduce no new diagnostics; unrelated in-flight
  Console extraction and existing baseline findings were preserved.
- Self-review added a red/green regression for uncopyable provider content:
  retention refusal must precede claiming unread steering. This refines the
  existing retention failure contract without changing the plan's authority.
- TASK-32483 through TASK-32018 have checked criteria, implementation notes,
  ADR disposition, and Done status through the Backlog CLI.
- TASK-32019 through TASK-32022 remain design proposals, not implemented.
- No full suite, real provider, commit, or integration action was performed.
  The expanded review's six semaphore-construction failures are recorded in
  the review ledger as an independently reproduced environment limitation.
