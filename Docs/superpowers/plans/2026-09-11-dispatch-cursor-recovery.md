# Dispatch cursor recovery

Goal: preserve the exact unresolved durable send when navigation writes a stale
conversation cursor, and recover already-stranded valid checkpoints on restore.

ADR required: no new ADR. This is a repair of the existing matching-checkpoint
recovery contract in ADR-079 (Console Library conversation authority). No schema,
provider authority, checkpoint payload, or automatic replay policy changes.

## Investigation

The incident had a single accepted checkpoint with matching user/assistant versions,
but the local conversation cursor selected the earlier greeting. The database passed
its integrity check. The checkpoint predates the provided startup log. The error was
`checkpoint_not_active_path`, not evidence that JSON or the database was corrupt.

`insert_with_messages` commits the user, assistant, checkpoint, and assistant cursor
together. `set_conversation_active_cursor` previously overwrote that cursor without
checking outstanding dispatch ownership. Store navigation writes through this API.
A regression reproduces the bad state through the production writer. The available
log starts after the original send and cannot identify the historical caller or the
reason the accepted send did not reach its dispatch marker.

## Implementation

- Guard cursor writes within an immediate transaction: every pending checkpoint must
  remain on the requested path. Rejected selection keeps the runtime cursor intact;
  before-first rewind restores its old runtime cursor on ownership refusal.
- Reconcile an off-path checkpoint only after full payload/version/role validation,
  exact assistant-to-user parent validation, complete nondeleted same-conversation
  ancestry, cycle rejection, and competing-owner checks. Re-read under an immediate
  transaction before repairing only the local cursor. Keep ambiguous records
  quarantined; do not delete records or infer authority from an empty message.
- Restore the committed cursor into the session instead of its stale pre-reconcile
  snapshot. Existing Retry/Retry-anyway/Discard semantics and reconstruction checks
  remain in force. Nothing automatically invokes a provider.
- Record the bounded quarantine error code, without checkpoint payloads or content.

## Verification

`Tests/Chat/test_console_dispatch_cursor_recovery.py` covers cursor rejection,
before-first rewind, old-state restore, accepted/dispatch-started Discard without
provider replay, version/parent/payload corruption, cyclic/competing ownership,
revalidation between read/write passes, and failed repair rollback.

The original code failed the writer, restore, and runtime-selection regressions.
The repaired code also restored the affected database snapshot to its original
pending turn. That checkpoint's reconstructability flags correctly disabled Retry;
Discard remained available. The live profile was not modified.

The existing targeted suites have separately baseline-confirmed failures: seven
Canvas manifest-integrity failures, two outdated migration-version assertions, one
postcommit test's provider-start expectation, and one queue callback type expectation.
Do not weaken those unrelated checks to make this fix appear fully green.

## Separate startup findings

The generic worker-state logger warns for unregistered handlers on every state
transition, including success. These messages do not establish worker failure.
Buddy installation independently fails on Windows because artwork publication
requires POSIX filesystem guards (`persona_visual_publication_denied` reproduced
against an isolated snapshot). Its library does not write dispatch checkpoints or
conversation cursors. Collections lifecycle-lock creation and bundled tiktoken hash
errors are separate failures; neither explains the pre-existing cursor mismatch.


## PR review follow-up

All six Qodo findings are addressed. Branch creation, ancestor editing, and
subtree deletion now check durable dispatch ownership under the same immediate
transaction as their writes, before runtime publication. Sibling failure rolls
back its runtime node; voice recovery selection returns False on cursor refusal.
Post-commit Sync v2 projection stays outside the branch transaction. Deleted or
missing conversations retain the documented False cursor result.

Recovery integration tests use a closing in-memory SQLite fixture, fixed complete
SQL statements, and transaction-managed deliberate corruption. Isolated helper
tests cover read-only/write decisions, malformed ancestry and competing owners
without initializing the application schema or store.

Diagnostic statement review found only the intended bounded quarantine error code,
the constant pending-dispatch warning, and revised wording of the existing cursor
exception diagnostic. No new content, secret, path, URL, or sink destination was
introduced. Regenerate the diagnostic inventory with this review recorded.


Review verification: 27 new recovery tests pass. Across the targeted dispatch,
branch-tree, edit/resend, regeneration and voice files, 197 tests pass and seven
fail; all seven also fail on unchanged dev at fb74902e2b. These are the two
previously recorded dispatch/queue expectations plus five edit/regenerate tests
with outdated fake persistence, substitution signatures, and selection expectations.
New tests pass Ruff check and formatting; the modified production files have the
same 262 pre-existing Ruff findings as dev, with no added findings.
