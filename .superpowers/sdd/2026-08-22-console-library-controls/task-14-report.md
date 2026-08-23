# Task 14 implementation report

## Outcome

Task 14 replaces durable manual and queued acceptance with one caller-owned
`BEGIN IMMEDIATE` transaction.  The transaction creates or validates the
conversation and exact Library policy, writes the captured USER and attachments,
creates one empty assistant in `accepted`, inserts the v45 dispatch checkpoint,
and applies Task-12 contributions through the restricted shared transaction
writer.  Conversation identity/title and every session, transcript, queue,
workspace, hook, and staged-input publication remain outside the transaction.

The store returns one immutable preparation-keyed durable commit and owns a
bounded postcommit completion ledger.  The controller claims and marks each
effect only after success, resumes missing effects by preparation ID, and crosses
`accepted -> dispatch_started` before provider entry.  Real durable adapters that
do not expose this atomic contract now fail closed before provider dispatch;
`ConsoleChatPersistence` explicitly declares the method.  There is no db-shape
heuristic or legacy durable fallback: only explicitly ephemeral sessions may use
the volatile compatibility path.

Precommit exceptions roll back every row, pause the same preparation as
Persistence with Retry/Cancel only, retain all staged input, and publish only a
bounded generic error.  Postcommit exceptions retain the same durable owner and
never create a second turn.  Successful live completion settles only Task-13's
volatile preparation so a following queued turn may run; the durable checkpoint
intentionally remains `dispatch_started` for Task 15.

## TDD and verification

- Exact clean-head Task-13 baseline: 628 passed; runtime/queue/UI companions:
  100 passed.  Both had the inherited Requests dependency warning.
- Both complete Task-14 test files were written before production.  Initial RED:
  24 failed because the atomic adapter, immutable commit, postcommit ledger, and
  controller path did not exist.
- Fresh Task-14 GREEN: 25 passed (including the explicit no-legacy-fallback
  adapter ratchet).
- Task-6 repositories/contribution plus Task-7 policy/promotion and the original
  Task-14 battery: 177 passed.  Full Task-7 Store gate plus that battery:
  351 passed.
- Fresh exact Task-13 affected gate: 628 passed in 36.55s.  Fresh Task-13
  runtime/queue/UI companion gate: 100 passed in 14.94s.
- Migration/schema/message-generation-state companions: 62 passed.
- Every completed run emitted only the inherited `RequestsDependencyWarning`.
  No full repository sweep was run.

The fresh exact Task-13 run initially exposed one meaningful RED rather than an
environmental hang: after a successful durable provider return, the Task-13
volatile preparation remained `dispatch_started`, so a queued cross-process
policy test returned the next entry to pending.  The durable path now performs
the same volatile preparation settlement as the legacy successful path while
leaving checkpoint terminal settlement out of scope.  The isolated regression
plus Task-14 controller suite then passed 14 tests before the exact 628-test gate.

## Fault and mutation matrix

The real SQLite fault matrix covers first conversation insert, policy
insert/validation, USER insert, attachment insert, assistant accepted insert,
checkpoint insert, contribution sequence allocation, contribution insert, and
outer commit.  Each failure compares exact database and live-memory snapshots;
clean retry reuses the preparation and staged identity and creates one owner.
Existing-conversation sequence overflow and policy mismatch remain byte/version
identical.

Nine independently applied and restored mutants were killed:

1. publish conversation identity/title before commit;
2. commit an atomic row outside the caller transaction;
3. omit the assistant `accepted` generation state;
4. swallow a contribution exception;
5. clear the staged draft before commit;
6. mark a postcommit effect complete before its callback succeeds;
7. remove preparation-keyed completion/idempotence;
8. enter the provider without the checkpoint CAS;
9. return an accepted queued entry to pending.

The postcommit test injects failure for identity publication, durable owner
publication, staged-input clearing, workspace projection, queue acknowledgement,
accepted hook, prompt history, preparation publication, checkpoint transition,
and provider entry.  Each callback executes once after retry and is never marked
complete on failure.

## Static, formatting, and privacy evidence

Scoped Ruff passed for every changed production and test file.  Ruff format
checks pass for the controller, queue coordinator, and both new tests.  The
repository's unchanged whole-file formatting drift in
`chat_persistence_service.py` and `console_chat_store.py` remains; their changed
ranges were reviewed and kept formatter-consistent without retaining unrelated
whole-file formatting churn.  `git diff --check` passed.

Checkpoint construction stores only frozen typed authority, credential-free
destination, reconstructability flags, identifiers, versions, hashes, origin,
queue identity, and revisions.  The real-database privacy assertion and source
review found no draft, prefill, evidence/query/source body or identity,
attachment bytes/path, API key, provider request, or arbitrary exception text in
checkpoint JSON or new logs.  The new persistence log uses a fixed event name.

## Files and governance

- `tldw_chatbook/Chat/chat_persistence_service.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- `Tests/Chat/test_console_durable_turn_acceptance.py`
- `Tests/Chat/test_console_first_send_atomicity.py`
- TASK-19900.3 plan/notes, this report, and the shared progress ledger

ADR required: no.  ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`.  ADR-079 owns
the implemented transaction, privacy, recovery-owner, workspace-projection, and
queue boundaries.

TASK-19900.3 remains In Progress with all 22 criteria unchecked.  Task 15 still
owns restart hydration, recovery actions/UI, queue restart reconciliation,
durable terminal settlement/checkpoint deletion, and Discard.  Task 16 still
owns continuation handoff and later projections/history.  Until Task 15 lands,
successful/failed provider terminal message updates use the existing live path
while the v45 checkpoint honestly remains `dispatch_started`; this task makes no
terminal-cleanup or restart-recovery claim.

## Review fix round 1

The review-fix RED was 9 failed/9 passed before production edits. Durable
eligibility now depends only on session/preparation authority and a callable
atomic adapter capability; a real SQLite wrapper with `db=None` but no capability
fails before transcript/provider/checkpoint mutation, while a `db=None` atomic
wrapper succeeds. Durable test adapters explicitly implement atomic and repository
contracts; intentionally db-less fixtures explicitly create ephemeral sessions.

One preparation RLock now owns a global preparation-id index, immutable
acceptance fingerprints, staged identities, commits, effect ledgers, and
controller continuations. A two-session/two-thread duplicate-id race admits one
owner, separate concurrent identifiers both proceed, and forged cached-acceptance
reuse fails closed. Queue acknowledgement uses exact registry settlement keyed by
session, entry, and preparation; it works after chain teardown, never settles a
mismatched claim, pauses later entries, and is idempotent only for the exact owner.

Checkpoint reconstructability is conservative for every automatic or explicitly
staged frozen EvidenceBundle/ConsoleLiveWorkLaunch and every transient prefill;
the stored projection remains body-free. Successful provider handoff, terminal
volatile failure, Cancel, and close remove body-bearing recovery state. A bounded
128-entry body-free fingerprint/effect tombstone is the only retained success
cache; 1,000 real turns and close/failure/resume probes verify the bound and
privacy shape. Eviction makes no process-cache exactly-once claim: deterministic
durable IDs and database uniqueness remain authoritative.

Production fault injection covers identity publication, owner hydration, staged
input clearing, workspace projection, queue acknowledgement, accepted hook,
prompt history, preparation publication, checkpoint CAS, and provider entry.
Every bit remains absent until its real seam succeeds, re-entry uses the same
durable IDs/checkpoint, and each injected target records two attempts/one success
while provider entry remains once. The COMMIT case is an actual SQLite `commit()`
failure: a deferred foreign-key violation is inserted from the outer transaction
`__exit__` after all writes, proving exact rollback before clean retry.

Fresh verification after restoration: Task-14 battery 46 passed; exact Task-13
affected gate 628 passed; queue/runtime/UI companions 100 passed; migration,
schema, generation-state, and persistence companions 126 passed. Each emitted
only the inherited Requests dependency warning. Seven restored mutants were
killed: adapter-shape fallback, global-id collision removal, detached queue ack
success, evidence omission, pre-callback effect completion, success cleanup
removal, and tombstone-cap removal. Scoped Ruff, formatter, privacy/source scans,
and `git diff --check` passed. No full repository sweep was run.
