---
id: TASK-19007
title: Execute lasting sync through a durable recovery journal
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:45'
updated_date: '2026-08-21 05:32'
labels:
  - notes
  - sync
  - recovery
dependencies:
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Admit recovery capacity before destructive work and execute guarded local note and filesystem operations through resumable durable journal states with verified outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local note and managed-membership reads and writes pass only through `NotesScopeService`; the executor never opens ChaChaNotes or File Notes authority directly.
- [x] #2 Recovery capacity and durable intent are admitted before destructive work, and pending, unresolved, or Undo-eligible recovery cannot be evicted.
- [x] #3 Each operation revalidates observations, advances a durable stage after each authority mutation, verifies both outcomes, updates binding ownership, and completes last.
- [x] #4 Interruption resumes only against matching observations; stale or partial outcomes become explicit attention with bounded resume, restore, or disconnect choices.
- [x] #5 Capacity failure, cancellation, and injected failure after every stage produce no blind replay, false atomicity, or hidden mutation.
- [x] #6 Logs and public diagnostics exclude content, paths, hashes, recovery bytes, credentials, and raw exception text.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED NotesScopeService authority and per-stage journal fault/cancellation tests using real temporary private databases.
2. Add the minimum service wrappers and private journal/recovery operations with capacity admitted atomically before mutation.
3. Implement deterministic resumable execution with observation revalidation, durable stages, verified outcomes, binding/membership updates, and explicit attention on stale/partial state.
4. Benchmark representative recovery payloads, choose one documented bounded capacity, and prove privacy/cancellation behavior.
5. Run the focused store/service/executor gates, benchmark, static checks, independent review, and task hygiene.

ADR required: no new ADR
ADR path: backlog/decisions/055-library-destructive-action-reversibility-rule.md; backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-055/059/073 already define reversible destructive work, service authority, durable stage order, recovery admission, privacy, and stale-claim fencing; this task implements those existing decisions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a local-only Notes authority adapter and service wrappers for optimistic reads, creates, replacements, tombstones, and owner-isolated managed-folder membership reconciliation. The executor never opens Notes storage directly; the server path remains fail-closed until its claim contract exists.
- Added atomic recovery-and-intent admission, durable stage transitions, incomplete-operation reconstruction, exact observation/binding/root/folder/direction fencing, and executable Resume, Restore, and Disconnect recovery for update, create, and guarded same-root move operations. Committed partial filesystem cleanup is identity-bound, capacity-reserved, private, and restart-actionable.
- Offloaded blocking authority work, shielded and joined an admitted mutation through its durable boundary before re-delivering cancellation, and kept public results, reprs, logs, and errors free of content, paths, hashes, recovery bytes, credentials, and raw exception text. TASK-19009 must hold the TASK-19006 root lease across each complete executor call; TASK-19007 deliberately does not import the later runtime composition.
- Benchmarked a 10 MiB replacement recovery at 10,485,785 bytes and sixteen real guarded 10 MiB moves at 167,792,448 bytes. Applying the documented 1.5x headroom yields 251,688,672 bytes, below the single 268,435,456-byte (256 MiB) default. Capacity tests prove refusal before mutation and protect pending, attention, and retained recovery.
- Verification: implementation commit `98aa5db443068f8503f737b6360a2c91e6259a3d`; exact task gate 282 passed with one pre-existing dependency warning; broader Notes gate 398 passed; benchmark, Ruff check/format, and diff checks passed. Two independent final reviews reported no Critical, Important, or Minor findings and Ready.
- ADR check: no new ADR. The implementation follows ADR-055, ADR-059, and ADR-073; no ownership, conflict, privacy, or recovery policy was introduced beyond those decisions.
<!-- SECTION:NOTES:END -->
