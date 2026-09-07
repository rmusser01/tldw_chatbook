# TASK-19872 Settings Backup Load Ordering Design

## Summary

Settings reads the Advanced Config backup in an exclusive thread worker. Textual
cancels the prior worker when a second load is dispatched, but cancellation
cannot stop the already-running thread or prevent its completion callback from
landing. The existing arrival guard compares the editor with the text captured
at dispatch. When two identical loads are in flight, the first callback can
write the backup preview and the second can then mistake that application write
for user typing, falsely reporting that unsaved edits were kept.

Use a monotonic request token to make only the newest dispatched load eligible
to update the editor or result line. Keep the existing dispatch-text comparison
on that newest result so real typing remains protected.

## Decision

Add one integer `_advanced_backup_load_token` to `SettingsScreen`.

On each **Load Backup** press:

1. Increment the token.
2. Capture the editor text and current token.
3. Pass both values through `_advanced_load_backup_worker` to
   `_apply_advanced_backup_preview_result`.

The completion callback first compares its captured token with the current
token. A mismatch means a newer load was dispatched, so the callback returns
without touching the editor, validation state, or result line. A matching
callback applies the existing dispatch-text guard: if the editor differs from
the text captured by the newest request, the user edited while that request was
loading, so the backup remains unapplied and the truthful preservation message
is shown.

This follows the existing monotonic-token and session-token patterns already
used by Settings manual sync and image-generation probes.

## Ordering Semantics

### Two loads overlap

The second press advances the token. Regardless of which thread finishes first,
the first request is stale and cannot paint. The second request is the only one
that can apply the preview or report an error. For identical backup content it
reports the ordinary successful "loaded backup preview" result selected by the
owner.

### First load finishes before the second press

The first request is current when it finishes, so it applies normally. The
second press captures that already-applied preview as its dispatch text, and its
completion also applies normally. Both operations are truthful successes.

### The user types while the newest load runs

The current token still matches, but the editor no longer matches the newest
request's dispatch text. The existing protection refuses the write and reports
that unsaved edits were kept.

Typing that already exists when the button is pressed is intentionally the
dispatch baseline: pressing **Load Backup** authorizes replacing the current
editor contents. Only changes made after that press are protected.

### The backup changes between reads

Only the newest request may paint, so an older callback cannot overwrite a
newer request with older backup contents. The design does not rely on backup
contents being identical.

## Alternatives Rejected

- **Accept when the editor equals the returned backup text.** Minimal for the
  reported identical-content sequence, but unsafe for out-of-order requests
  that read different backup contents and incomplete for three or more loads.
- **Track the last application-written text.** Distinguishes some application
  writes from typing, but does not establish which request owns the surface and
  can still let an old callback overwrite a newer result.
- **Disable or deduplicate rapid presses.** Hides the triggering gesture without
  fixing the thread-worker completion ordering contract. It also weakens the
  task's requirement that the guard distinguish application activity from real
  edits.

## Error Handling

Errors from stale requests are ignored because the user has already asked for a
newer result. The newest request keeps the existing decode, path, and missing
backup error behavior. Widget disappearance continues to use the existing
`QueryError` handling.

## Verification

Implementation begins with deterministic mounted regression tests against the
unmodified branch. Each overlapping worker gets its own `started` and `release`
handshake and a distinct result/backup payload. The test waits until worker 1
has entered its gate before dispatching worker 2, then waits until worker 2 has
entered before releasing either. A wrapper around the real arrival callback
records that each callback returned, so the next release never depends on a
fixed sleep.

The overlap matrix covers:

1. Older callback returns before the newer callback.
2. Newer callback returns before the older callback.
3. A stale error returns after a newer successful preview.

Every case asserts that the newest request owns the editor text, result line,
and validation state. Before the fix, the distinct payloads make current `dev`
either leave the older preview in place or replace the truthful success result
with the false preservation/error result. After the fix, stale callbacks have
no observable effect.

A serial case covers the first load completing before the second dispatch; both
must report ordinary success. The existing delayed-load typing test remains and
must continue proving that an edit made after dispatch survives with the
preservation message.

Mutation evidence removes or inverts the token mismatch return; the distinct
old/new overlap cases must then fail, proving latest-request ownership rather
than merely identical-content idempotence. Separately removing the existing
dispatch-text guard must fail the genuine-typing test. Focused verification
covers the new tests, the existing Advanced Config backup tests, Ruff on the
modified Python files, and both diff checks. The full repository suite is not
required unless separately requested.

## Documentation and Task Evidence

Update the Advanced Config user-guide row and its verification stamp to mention
that repeated loads are latest-request-wins and no longer manufacture an
unsaved-edit warning. Record the reproduced interleaving, RED failure, mutation
result, focused gate, and final decision in TASK-19872 implementation notes.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this is a localized completion-order correctness fix within the existing
Settings worker, UI ownership, and thread-worker boundaries. It introduces no
new persistence, service contract, security policy, dependency, or long-lived
application structure.
