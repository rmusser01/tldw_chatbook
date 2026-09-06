---
id: TASK-31809
title: Restore terminal exchange flush and temporary chat completion
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 02:47'
labels:
  - tests
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The paired terminal-generation persistence path bypasses the existing exchange
sidecar flush and treats an intentionally unsaved temporary chat as a failed
durable write. The remaining exchange file also contains two outdated privacy
repository fixtures. Restore the established terminal behavior without relaxing
durable generation validation or privacy fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Complete, stopped and failed ordinary assistant generations retain attached exchange captures durably without an extra message-version bump.
- [x] #2 Temporary terminal messages retain content and captures in memory without creating durable conversations, messages or exchanges.
- [x] #3 Best-effort serialization failure, compression reuse, abandoned captures and deferred-terminal behavior retain their existing assertions; privacy fixtures match the current repository contract.
- [x] #4 Real SQLite regressions fail before repair and pass after it; complete affected files, static checks and independent review qualify the repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the complete exchange file and add real SQLite terminal tests covering complete/stopped/failed across durable and temporary sessions, durable reopen and exact message version. The baseline is 11 failed/13 passed before edits.
2. Repair the common terminal persistence owner: temporary sessions need no durable projection, successful paired-generation writes must perform the existing best-effort exchange flush, and fallback paths must not acquire a duplicate flush. Keep settlement, failure and canonical version rules intact.
3. Correct the two obsolete repository fixtures to supply current privacy fields and the privacy-write seam, retaining explicit Safe/pending assertions. Run complete exchanges, store, terminal citation and paired-generation regression selections; inspect diagnostic delta, lint/format and obtain independent review. Update the draft checkpoint.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Restore existing local-only legacy capture and temporary-session behavior at the existing terminal owner; no new persistence, capture authority or schema contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the existing best-effort exchange flush after successful paired terminal
generation writes. Temporary sessions settle without attempting a durable
projection; durable failure and canonical version guards remain unchanged.
Updated the two privacy repository fixtures to the current contract, testing
both privacy-write and detail-write failures without weakening Safe/pending
assertions. No new ADR is required; the existing ADR-097 boundary is unchanged.

Six real SQLite regressions cover complete/stopped/failed durable and temporary
sessions, reopened durable rows, exactly one message-version increment, zero
temporary rows and zero remaining connection handles. After correcting fixture
setup, the pre-fix exchange run was 17 failed/13 passed; all 31 exchange tests
pass after repair. Eight complete directly affected files pass 667 tests in
148.81 seconds, with two existing dependency warnings. Evidence:
`/private/tmp/tldw-store-exchanges-real-red.xml`,
`/private/tmp/tldw-store-exchanges-green.xml`, and
`/private/tmp/tldw-store-exchanges-final.xml`.

Ruff checks, test-file formatting, changed-store-range formatting and diff checks
pass. The store diagnostic inventory remains 81 calls with no statement changes.
Independent review found no blocking correctness issues. Deferred citation
callers retain their explicit idempotent sidecar flush (previously also repeated
on the ordinary fallback); removing that redundancy is outside this repair.
No full-suite, live UI or merge-readiness claim is made. Broader failures remain
tracked in the review checkpoint.
<!-- SECTION:NOTES:END -->
