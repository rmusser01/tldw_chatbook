---
id: TASK-31423
title: Chunking Lab - durable profile-local session checkpoints
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 23:12'
updated_date: '2026-09-05 00:55'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31422
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recover the latest durable experiment after reopening or crashing, including invalid drafts and completed A/B outputs, without coupling scratch state to template writes. Covers spec section 8 and AC 2, 5, 9, 11-13, 16, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: private storage, transactions, retention, and cross-instance conflict policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A dedicated versioned profile-local SQLite store uses existing private-path protections and atomically publishes results with their checkpoint references; current, previous, and active undo snapshots remain intact.
- [x] #2 Reopening restores exact sample, raw drafts, pending edits, A/B results, and view state; unfinished work becomes Interrupted with no automatic execution or source re-read.
- [x] #3 Serialized revision-aware autosaves target 300 ms debounce and a one-second maximum normal typing interval; Saved locally only reflects the latest committed revision and conflicts preserve the losing in-memory state.
- [x] #4 Crash injection, disk failures, incompatible schemas, concurrent writers, and delayed acknowledgments cannot overwrite valid recovery data, falsely report saved state, or resurrect a cleared epoch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements the approved dedicated recovery store, private ownership and serialized revision-aware writer. 1. Read Task 3 brief, task-3-context and current state/publication contracts. 2. Write failing private SQLite round-trip, atomic result publication, real crash and two-writer conflict tests. 3. Implement versioned checkpoint/blob storage and serialized autosave with honest status and retained references. 4. Run the targeted DB/autosave/private-owner suites, lint and format; review independently. 5. Record AC evidence and implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Current task-level status: Done after independent review. Earlier pending-review
statements below are historical; original failed census evidence remains unchanged.
The final correction verifies malformed known UI/draft fields trigger the existing
previous-checkpoint fallback on mounted reopen. No store/schema contract changed;
see [Chunking Lab verification](../../Docs/Chunking_Lab_Verification.md).

<!-- SECTION:NOTES:BEGIN -->
Implemented schema-1 private recovery in `DB/Chunking_Lab_DB.py`, a serialized off-loop writer in `Chunking/lab_autosave.py`, shared small-graph validation in `Chunking/lab_models.py`, and private owner `db.chunking_lab` with inventory row C54 and tests. This directly implements ADR-118 (`backlog/decisions/118-chunking-lab-local-execution-and-recovery.md`); no additional ADR or Media DB migration is needed.

Publication uses explicit BEGIN IMMEDIATE/COMMIT, WAL with synchronous FULL, epoch/generation CAS, and separate content-addressed immutable snapshots. GC runs after publication and preserves current, previous-valid, in-session undo, and reserved restore-undo references. Clear atomically removes every reference and leaves a new-epoch tombstone. Initial reads normalize unfinished runs without execution, explain fallback, and refuse malformed/newer storage. Captured blobs are detached and identity-cached, so small edits neither copy nor rewrite existing reports; application-level bounds on accumulated authoring history remain separate work.

`await AutosaveWriter.load()` is the initial off-thread seam. Load errors and CAS conflicts preserve memory and never grant empty overwrite authority. A fresh store/writer after closing the failed writer is the explicit load-retry path. Ordinary storage failures retry the latest submitted session through flush. The 300 ms trailing debounce has a one-second max-wait deadline; critical submits remain immediate, old acknowledgments cannot mark newer drafts saved, and Clear fences submissions before draining in-flight work.

Evidence includes real child-process kills synchronized immediately before/after actual SQLite COMMIT, real SQLITE_FULL rollback/retry, simultaneous SQL CAS, stale epochs after Clear, malformed/newer checkpoint refusal and fallback, immutable-cache ownership and failed-publication recovery, private DB/WAL/SHM modes, undo/restore-undo retention and GC, canceled flush awaits, and writer timing/status tests. Focused state/storage/writer/execution and new-owner checks: 94 passed with the known Requests and vendored datetime warnings. The combined private-SQLite inventory run has three unrelated census failures reproduced identically in a temporary archive of base `17d4996a1372706a61823567b136b406f1b3b6de`; affected legacy production files are unchanged. New/modified Lab code passes Ruff and formatting; the two legacy registry/inventory files retain exactly their base 44 lint diagnostics without new findings.

Final combined verification (DB, autosave, private SQLite, owner inventory, state, execution): 367 passed, 2 platform skips, 2 known warnings, and the same 3 baseline-only census failures. Self-review fixed cache entries surviving a rejected publication, fallback incorrectly retaining a corrupt predecessor, cache assumptions after another instance clears, malformed state masquerading as empty recovery, and typing delaying an already-immediate checkpoint. The private-path seam may fail closed if two processes race first-file creation; this is surfaced without bypassing protection. Status remains In Progress for independent review.
<!-- SECTION:NOTES:END -->
