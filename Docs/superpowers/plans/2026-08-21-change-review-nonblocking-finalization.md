# Change Review nonblocking finalization implementation plan

Date: 2026-08-21
Task: TASK-19502
Design: `Docs/superpowers/specs/2026-08-21-console-file-review-performance-recovery-design.md` §§4.2–4.3

## Goal

Release a terminal Console turn and drain its prompt queue as soon as the
assistant outcome is durable, without waiting for Change Review baseline/end
Git work. Preserve per-canonical-root FIFO attribution, survivor windows,
durable marker placement, bounded resources, and shutdown safety.

## Architecture

Add one `ChangeReviewFinalizationCoordinator` owned by `ConsoleRuntime` beside
the bridge and `AgentRunsDB`. Registration atomically appends one reservation
to every canonical-root lane before model execution. A reservation starts only
when it is the head of every lane it needs. Fixed daemon filesystem workers run
synchronous tracker baseline/end operations and return immutable results to a
single publisher queue. The publisher alone writes durable change rows and
advances a content-free publication signal; workers never receive the DB,
store, widgets, or callbacks.

Nested repositories discovered by the bounded baseline scan are not allowed to
bypass those lanes. Before their first snapshot, the coordinator atomically
enrolls each canonical nested root into the same reservation; enrollment either
succeeds for the complete discovered set or records an honest tracking error
without touching any newly requested lane. Capacity is defined over admitted
reservations, so registration reserves capacity before it mutates any lane and
rolls back as one operation.

The bridge requests finalization in its terminal `finally` and returns
immediately. It no longer appends live change-summary/failure markers. The
mounted transcript and resume path both derive those markers by joining the
run's durable assistant anchor with durable change rows. The transcript poll
stays alive while coordinator work is pending and performs a DB re-derive only
when the publication signal revision changes.

Live re-derivation is a temporary render projection over the current message
list; it never applies a resume overlay to, or otherwise mutates, the
conversation store.

ADR required: no
ADR path: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md`
Reason: ADR-077 already defines app ownership, per-root FIFO, immutable pure
worker inputs/results, durable publication races, and shutdown order.

## Files

- Create `tldw_chatbook/Workspaces/change_review_finalization.py`
- Modify `tldw_chatbook/Workspaces/change_turn_tracker.py`
- Modify `tldw_chatbook/Workspaces/__init__.py`
- Modify `tldw_chatbook/Chat/console_runtime.py`
- Modify `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify `tldw_chatbook/Chat/console_chat_controller.py`
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`
- Create `Tests/Workspaces/test_change_review_finalization.py`
- Modify `Tests/Chat/test_change_turn_tracking.py`
- Modify `Tests/Chat/test_console_runtime_lifetime.py`
- Modify `Tests/UI/test_console_native_chat_flow.py`
- Modify `backlog/tasks/task-19502 - Decouple-Change-Review-finalization-from-Console-turn-completion.md`

## Task 1: Extract synchronous tracker operations

1. Add RED tracker tests proving a caller-owned worker can create a handle,
   populate B synchronously, and take E synchronously without an inner thread.
2. Extract `new_turn_handle`, `populate_baseline`, and `finish_turn` from the
   existing `begin_turn`/`end_turn` bodies. Preserve `begin_turn` as the legacy
   compatibility wrapper used by existing tests; it may start its existing
   thread, while the new coordinator never calls it.
3. Replace `TurnHandle`'s thread-only readiness with an event-backed bounded
   wait that works for both legacy and coordinator ownership. A timeout marks
   unresolved roots once and invalidates a late B result for dispatch gating.
4. Run `Tests/Chat/test_change_turn_tracking.py -k 'begin or baseline or tracker'`.

## Task 2: Build bounded per-root FIFO admission

1. Add RED real-Git/barrier tests for:
   - disjoint roots running concurrently;
   - two conversations sharing one root executing `B1,E1,B2,E2`;
   - a multi-root reservation starting only when it heads every required lane;
   - bounded-capacity rejection leaving no partial lane entries;
   - tombstoned reservations waking successors.
2. Implement immutable `ChangeReviewReservation`, operation/result records,
   lane deques, one coordinator lock, fixed daemon filesystem workers, bounded
   operation/result queues, and all-or-nothing registration.
   Capacity counts admitted lane reservations exactly. A rejected turn is a
   caller-owned lightweight token and may reserve one slot in a separate
   bounded error-publication channel; channel saturation returns a typed
   visible fallback rather than creating a second reservation pool.
3. When bounded baseline discovery finds nested repositories, enroll their
   canonical roots into the reservation before snapshotting them. Test both
   shared nested-root FIFO and capacity rejection without partial enrollment.
4. Workers receive only reservation ID, canonical roots, tracker filesystem
   inputs, and touched-path strings. They never receive DB/store/UI objects.
5. Run `Tests/Workspaces/test_change_review_finalization.py -k 'fifo or multi_root or capacity or nested'`.

## Task 3: Decouple bridge completion

1. Add a RED bridge test with E held on an event. Assert `run_reply` returns its
   terminal outcome before E is released and that no change row exists yet.
2. Register the reservation where the bridge currently calls `begin_turn`.
   Keep B in parallel with the model and keep the existing permission wrapper
   calling the reservation's bounded baseline wait.
3. In the terminal `finally`, capture touched paths and terminal metadata,
   request coordinator finalization, and return. Remove synchronous
   `end_turn`, direct `_record_change_snapshots`, and live marker append for
   the turn window.
4. Persist honest tracking-error results for baseline/end failures and for
   failed/cancelled turns exactly as for successful turns.
5. Run `Tests/Chat/test_change_turn_tracking.py -k 'nonblocking or failed_run or tracking'`.

## Task 4: Preserve survivor-window ordering

1. Add RED tests for a child surviving primary completion, a successor turn
   registering before the survivor window is published, and the last child
   settling with no successor. Assert each write belongs to exactly one window.
2. Move survivor lineage from bridge-local `TurnHandle` maps into coordinator
   reservation metadata. A finalization result that observes live children
   opens the post-turn lane window at its E SHA. The already-queued successor
   B closes it at that same SHA; without a successor, last-child settlement
   requests its E.
3. Bridge child-settlement code sends content-free reservation/run IDs only;
   it never performs Git or DB work itself.
4. Run the survivor subset in `Tests/Chat/test_change_turn_tracking.py`.

## Task 5: Publish and render by durable join

1. Add RED result-first, anchor-first, and remount tests. A marker appears only
   after both `agent_runs.assistant_message_id` and `change_snapshots` exist,
   appears once, and sits immediately after its assistant anchor.
2. Add `ChangeReviewPublicationSignal`, an atomic revision/pending snapshot
   with no payload. Add one `AgentRunsDB` batch method so the publisher commits
   every row for one completed window in one transaction and never retries a
   partially published window. It then increments the signal. Anchor writes
   increment the same signal.
   If the atomic batch raises, make exactly one second batch attempt containing
   per-root tracking-error rows; never retry that terminal attempt.
3. Remove live change marker insertion. Extend the transcript poll to keep
   running while finalization is pending and, only when the signal revision
   changes, re-derive a temporary active-session render projection with
   `resume_marker_messages`/`inject_resume_agent_markers`. Do not call
   `apply_resume_marker_overlay` during live polling: that mutates view state
   and can discard ephemeral tool, task, alias-warning, or streaming rows.
4. Ensure repeated refreshes are idempotent and alias-only readiness warnings
   remain outside `change_snapshots`.
5. Run bridge marker/resume tests and mounted transcript tests.

## Task 6: Own shutdown and prove the third turn starts

1. Add a RED mounted three-turn test: hold turn two's E, wait until its
   assistant is durable, submit turn three, and prove its provider starts
   before E is released.
2. `ConsoleRuntime.dispose` orders teardown as: stop controller admission and
   settle turns; coordinator bounded shutdown/drain; close the publisher and
   runtime/UI thread-local AgentRunsDB connections on their owning threads;
   then provider gateway close. Late-generation worker results are rejected
   before DB access.
3. Add shutdown tests with a blocked filesystem worker proving bounded return,
   queued tombstones, no DB read after disposal, and no live non-daemon review
   threads.
4. Run runtime lifetime, three-turn, and shutdown suites.

## Task 7: Verify and close TASK-19502

1. Run focused tests:

   ```bash
   /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
     Tests/Workspaces/test_change_review_finalization.py \
     Tests/Chat/test_change_turn_tracking.py \
     Tests/Chat/test_console_turn_execution_context.py \
     Tests/Chat/test_console_runtime_lifetime.py \
     Tests/UI/test_console_native_chat_flow.py -k 'change_review or three_turn or runtime_lifetime'
   ```

2. Run Ruff on every changed Python file and `git diff --check`.
3. Mutation-check lane-head ordering, all-or-nothing rollback, nonblocking
   bridge return, durable join idempotency, and late-generation shutdown.
4. Check all TASK-19502 acceptance criteria, add implementation notes, and set
   status Done only after the evidence is green.

## Review notes before implementation

- The coordinator must not use `ThreadPoolExecutor`: its non-daemon threads and
  unbounded internal queue violate the shutdown/resource contract.
- Never submit E by spawning from the bridge's `finally`; one app-owned bounded
  queue is the only filesystem-work ingress.
- Do not key FIFO lanes by conversation. Canonical root is the attribution and
  repository-integrity boundary.
- Do not hold the coordinator lock during filesystem/Git/DB work or while
  waiting on a baseline event.
- Do not append a marker optimistically. Durable rows and durable assistant
  anchors are the only publication truth, so both completion races and remount
  have one rendering path.
- A baseline timeout must invalidate its late success for the current dispatch
  decision; otherwise a tool can proceed untracked and later publish a
  misleading apparently-complete review.
- Survivor settlement may race the primary E operation. Record settlement as
  coordinator state and decide whether to open/close the follow-on window only
  while advancing the reservation; never infer it from a bridge-local map
  after E returns.
- A publisher failure produces one terminal tracking-error publication attempt;
  it must not blindly retry non-idempotent inserts. One-window batch commit is
  the atomicity boundary.
