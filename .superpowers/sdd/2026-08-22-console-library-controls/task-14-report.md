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

## Review fix round 2

Every durable preparation now stages its conversation, USER, and assistant IDs
once under the preparation RLock and reuses those exact owners on persistence
Retry. Real controller tests fail at conversation creation, policy persistence,
checkpoint insertion, and the actual outer SQLite COMMIT across Never/Automatic
manual/queued sends; each failure returns the same owner to
`PAUSED(PERSISTENCE)` with Retry/Cancel, then commits one USER, one empty
assistant, one checkpoint, and enters the provider once. Fingerprint construction
failure follows the same pause contract. A concurrent exact caller may receive an
explicit in-flight refusal, but cannot remove or pause the caller that owns the
commit; once that owner completes, the exact acceptance returns its cached commit.

One mandatory body-free `ConsoleDurableAcceptanceFingerprint` now carries a
SHA-256 digest of canonical preparation/attempt/session and effect ownership,
staged conversation/title/message identity, workspace/policy/conversation
arguments, USER body hash, parent, canonical attachments, origin/queue identity,
frozen authority, credential-free destination, reconstruction truth, and bounded
contribution fingerprints. Frozen dataclasses are canonical by typed fields;
other contributions must expose `durable_acceptance_fingerprint()` and
noncanonical or oversized values fail before SQLite. Every commit/effect cache API
requires and validates the exact fingerprint. Only the digest and owner IDs reach
the bounded tombstone; draft, attachment bytes, evidence, prefill, provider
messages, and contribution bodies are not retained.

Queue claims bind their exact `session_id + entry_id + preparation_id` before a
queued preparation is published. Return-to-pending clears the binding, reclaim
rebinds it, accepted settlement validates it, and the exact tombstone alone makes
detached replay idempotent. A forged preparation cannot settle the claim or a
different entry/session; later entries remain paused after accepted settlement.

The round-2 suite began RED at 19 failed/1 passed before production changes. A
later concurrency self-review ratchet independently failed on stolen in-flight
ownership before its fix. Fresh restored verification was:

- prior Task-14 files: 46 passed;
- round-2 file: 21 passed;
- exact Task-13 affected command: 628 passed;
- runtime/queue/UI companion command: 100 passed;
- exact DB/migration/state command: 126 passed.

The exact 126-test command was
`../../.venv/bin/python -m pytest -q --tb=short --show-capture=no`
`Tests/DB/test_chachanotes_console_library_policy_migration.py`
`Tests/ChaChaNotesDB/test_migration_atomicity.py`
`Tests/Chat/test_assistant_generation_state.py`
`Tests/Chat/test_chat_persistence_service.py`. Every gate emitted only the
inherited Requests dependency warning; no full repository sweep was run.

Four independently applied and restored round-2 mutants were killed: generating
a fresh assistant ID per Retry failed all four real Retry cases; omitting the USER
body hash failed the content-forgery ratchet; allowing a `None` effect fingerprint
failed the mandatory-API ratchet; and omitting claim/preparation validation let a
forged queue owner settle and failed the exact-binding ratchet. Each target below
was invoked with `../../.venv/bin/python -m pytest -q --tb=short --show-capture=no`;
the exact targets/results were:

- `Tests/Chat/test_console_durable_turn_fix_round2.py::test_real_persistence_retry_reuses_exact_staged_message_owners`
  — 4 failed;
- `Tests/Chat/test_console_durable_turn_fix_round2.py::test_cached_commit_rejects_each_material_acceptance_mutation[content]`
  — 1 failed;
- `Tests/Chat/test_console_durable_turn_fix_round2.py::test_every_postcommit_cache_api_requires_exact_non_none_fingerprint`
  — 1 failed;
- `Tests/Chat/test_console_durable_turn_fix_round2.py::test_claim_binding_rejects_forged_ack_then_correct_owner_settles`
  — 1 failed.

Scoped Ruff passed
for every affected file, formatter checks passed for the new test and changed
controller/queue files, changed store/contribution ranges were reviewed against
the repository's pre-existing whole-file format drift, and `git diff --check` plus
body-retention/source inspection passed.

ADR required: no. ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`. This round
tightens ADR-079's existing ownership/authentication boundaries. TASK-19900.3
remains In Progress with all 22 criteria unchecked. Task 15 still owns restart
hydration/recovery actions, queue restart reconciliation, terminal checkpoint
settlement/deletion, and Discard; the live terminal seam therefore still leaves
the durable checkpoint honestly `dispatch_started`.

## Review fix round 3

The full acceptance digest previously preceded the in-flight marker. Because a
generic contribution fingerprint callback is arbitrary user/runtime code, caller
1 could block there while caller 2 computed a different body digest, installed
the first marker, and committed the competing acceptance. Round 3 replaces that
post-digest fingerprint sentinel with a frozen structured reservation installed
under the preparation RLock before canonicalization.

The reservation has a unique caller token and owner thread plus the immutable
preparation/attempt/session, staged conversation/USER/assistant, origin, and queue
owner tuple. A foreign caller validates that tuple and fails immediately without
calling the canonicalizer, changing preparation state, touching SQLite, or
clearing the marker. The owner releases the lock for canonicalization and DB I/O;
on reacquire it must prove the exact reservation object and unchanged staged
identity/message owners before atomically installing the full fingerprint. Only
that reservation owner may clear the marker and pause after canonicalization or
DB failure. DB failure retains the full fingerprint and staged IDs for Retry;
completed-cache authentication computes the digest without installing a second
reservation. The reservation retains identifiers only, never body/evidence/
prefill/attachment/provider content.

Clean-head Task-14 baseline was 67 passed. The complete round-3 RED collected
three cases and produced 2 failed/1 control passed: both deterministic caller
orders let the contender return while caller 1 was blocked inside fingerprint
construction; canonicalizer failure already cleaned up for Retry. After the fix,
round 3 passed 3 and combined Task 14 passed 70. Fresh adjacent gates passed 628
exact Task 13, 100 queue/lifetime, and 126 DB/migration/state tests. Every run had
only the inherited Requests dependency warning; no full repository sweep ran.

The required mutant moved reservation installation from before to after digest
construction. Running
`../../.venv/bin/python -m pytest -q --tb=short --show-capture=no`
`Tests/Chat/test_console_durable_turn_fix_round3.py::test_pre_fingerprint_reservation_first_caller_owns_body`
failed both caller-order cases because the contender returned instead of raising
foreign in-flight. The mutant was restored and the same round-3 file passed 3.

Scoped Ruff passed for the changed production and test files, the new round-3
test is formatter-clean, the changed Store range matches Ruff formatting while
the documented pre-existing whole-file drift remains, and `git diff --check`
passed. Source/privacy review confirms the reservation contains only its opaque
token, thread identity, and durable owner identifiers; no content-bearing value.

ADR required: no. ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`. This is a
bounded correction to ADR-079's existing app-lifetime durable acceptance owner.
TASK-19900.3 remains In Progress with all 22 criteria unchecked. Task 15 still
owns restart recovery, queue reconciliation, actions/UI, terminal settlement,
checkpoint deletion, and Discard; the transitional live checkpoint remains
honestly `dispatch_started`.
