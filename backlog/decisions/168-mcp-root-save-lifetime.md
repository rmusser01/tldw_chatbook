# ADR-168: Ordered app-owned MCP root saves

Status: Accepted
Date: 2026-09-18
Task: TASK-32791

## Context

Tools mode starts exclusive UI workers around threaded configuration writes.
Cancelling an observer leaves its admitted thread running. Independent mounted
review held save A, completed newer save B, then released A: disk contained A
while the UI claimed B. Ordinary child refresh also replaced unsaved root drafts.
The root controls incorrectly promise Console confinement, although the actual
consumers are standalone local MCP serving and the operator Hub test/provider
path. ADR-082/102 continue to govern Console scratch and admitted folders.

## Decision

A narrow application-owned coordinator serializes explicitly admitted root saves
in submission order. Admission captures configuration path, submission-time path
context, raw input and a draft identity/revision synchronously. Tasks are retained
independently of screen workers; observers await them through a shield. Duplicate
pending submissions of the same draft share their operation. The owner retains
the latest progress/outcome and the last committed root needed to explain a
cache-refresh warning. These are bounded session state, not a durable job ledger.
No screen references or exception bodies are retained.

Validation uses the existing path validator off-thread. Persistence uses the
existing atomic config mutation and its locked precondition to refuse a queued
write after configuration selection changes. Runtime tool authority and schemas
are unchanged. A successful replacement with failed cache publication remains a
saved-to-file warning; a successful no-op cannot falsely clear that warning.
The committed root overrides stale caches only while its captured configuration
publication generation and captured file revision remain current. Capture that generation under the existing
write lock; a successful publication increments it once. A later independent
publication supersedes the override and prevents an old receipt from restoring an
outdated value. Capture device/inode/mtime/size under the locked precondition
and after replacement; bootstrap reload of an external TOML edit does not advance
the runtime generation, so that file revision also fences receipt applicability.
Receipts only apply to their captured configuration path.

Tools owns its unsaved draft for its mounted lifetime. Ordinary refresh updates a
clean field but preserves dirty fields and local receipts. Completion canonicalizes
only the exact originating draft revision, including protection against A→B→A
edits. A recreated view projects current persisted state and the app-owned receipt;
failed-receipt copy does not claim the discarded ephemeral draft was retained.
A separate root receipt prevents master-toggle feedback from erasing its outcome.
Catalog refresh occurs after publication of the persistence result; presentation
failure cannot turn a committed write into a reported save failure.

Shutdown closes root-save admission before awaiting any owned cleanup, prevents
lazy owner construction after that fence, and drains admitted saves before config
and recovery teardown. The existing process exit watchdog remains authoritative;
there is no new timeout, retry policy, or guarantee after hard termination.

## Alternatives

- Keep exclusive screen workers: cancellation cannot recall their threads and loses
  receipts across screen recreation.
- Only disable Save during a write: prevents a useful newer explicit choice and
  does not fix screen lifetime or shutdown ownership.
- Reuse Tool Profile operations: couples configuration paths and validation to
  unrelated import/export/removal result types.
- Add persistent jobs or a general settings framework: unnecessary for this one
  existing staged field.

Existing ADR-033 governs honest save models; ADR-150/161 govern presentation.
The [design](../../Docs/superpowers/specs/2026-09-18-mcp-root-settings.md) and
[plan](../../Docs/superpowers/plans/2026-09-18-mcp-root-settings.md) define evidence.
