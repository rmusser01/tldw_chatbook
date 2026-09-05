---
id: TASK-31641
title: Chunking Lab - recovery export restore and undo
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 23:12'
updated_date: '2026-09-05 01:31'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31640
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a usable recovery path when local persistence fails, including safe snapshot restoration and bounded undo of replacement. Covers spec section 8 and AC 13, 16, 24, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: private-data transfer and transactional replacement policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Versioned bounded JSON export from in-memory state preserves exact samples, invalid JSON, pending controls, full result snapshots, and integrity references without requiring a writable recovery database.
- [x] #2 Restore validates structure, sizes, versions, and reference digests without executing templates, reading embedded paths, or making network calls; malformed input leaves the active session intact.
- [x] #3 Explicit replacement preserves the displaced checkpoint atomically, changes epoch only after commit, drains or invalidates old writer requests, and exposes a transaction boundary for callers that have quiesced execution; failure keeps the old session and its retry authority.
- [x] #4 Undo restore remains available across view-only autosaves until content changes; Clear removes all recovery and undo references and cannot be reversed by late writes.
- [x] #5 Repeated edits and reruns retain only one application content-action undo plus reachable current, previous and undo-needed snapshots; native editor undo remains separate, view-only changes preserve undo, and exceeding active recovery bounds refuses the edit without losing the prior value.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements approved bounded private recovery transfer and atomic replace/undo. 1. Read Task 4 brief/context and current store/writer contracts. 2. Write failing export/import integrity and bounds tests plus atomic replacement/undo cases. 3. Implement structural recovery validation and writer-owned replace/undo with epoch changes only on commit. 4. Bound application content Undo and prune unreachable active snapshots while preserving current/previous evidence. 5. Run targeted recovery/storage/state/autosave tests and static checks, self-review and independent review, then record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Current task-level status: Done after independent review; earlier pending-review
statements below are historical. Final correction admission rejects malformed known
view/draft shapes before replacement while retaining unknown extensions and opaque
invalid input. The screen previews the bounded validated import and requires explicit
Replace current session; initial unreadable storage still permits read-only inspection.
Existing ADR-118 and transactional displaced-session protection remain in force.
Additional evidence: `.superpowers/sdd/2026-09-04-chunking-lab/final-fix-report.md`.

<!-- SECTION:NOTES:BEGIN -->
- Added `lab_recovery.py` for versioned UTF-8 transfer, digest and structural validation, active reachability, and matching byte/depth/count admission. Raw authoring text remains opaque; captured recipe hashes use their original authored/effective documents and runtime identity. No source-path reads or executable preflight occur on recovery.
- Added transactional replace/Undo restore to the existing private store and serialized autosave owner. A single commit preserves exact displaced in-memory content, a rebased previous fallback, and the new session. Failed publication rolls back new blobs/checkpoints and retains old retry authority. View-only saves retain the displaced checkpoint; content saves and Clear release it.
- Replaced growing application undo history with one prior content action and pruned obsolete sample/result map entries. Per-session nonserialized size measurements reuse retained immutable payloads and drop dead cache entries. Persisted `content_revision` (default 0 for older checkpoints) expires restore undo even when an edit and Undo coalesce back to identical content.
- Replacement retires the active manifest after unfinished members become Interrupted; original run epochs, batch IDs, recipe/runtime snapshots, and reports remain intact. Coordinator integration must quiesce execution and suspend edits before invoking replacement. AC3 was clarified with controller approval to express this headless transaction boundary; the process-stop requirement remains part of coordinator integration.
- ADR required: yes; direct implementation of accepted [ADR-118](../decisions/118-chunking-lab-local-execution-and-recovery.md), without a new decision or SQLite schema migration. Source files: recovery, state, models, autosave, and Chunking Lab DB; focused tests cover those interfaces.
- Verification: 144 targeted recovery/state/DB/autosave/execution tests passed, with only existing Requests and vendored `datetime.utcnow` warnings. Ruff check and formatter check passed on all nine scoped Python files; `git diff --check` passed. Independent review remains pending, so status remains In Progress.
- Review correction: centralized unfinished-member interruption in one recovery helper used by import, authority rebase, and DB load. A partial A/B regression covers all three boundaries, preserving completed output, captured provenance, and caller-specific manifest/authority semantics. The focused recovery/DB/autosave/state selection passed 117 tests with the existing Requests warning; independent re-review remains pending.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Formerly TASK-31424; moved to TASK-31641 during the user-approved
2026-09-05 pre-push bookkeeping correction. Upstream dev independently uses
31421–31424; the complete Lab chain moved together to preserve dependency
ordering without changing upstream tasks. Original creation dates, acceptance
and implementation history are retained. Historical commits and ignored review
artifacts retain the old IDs; current references use the new IDs. See
Docs/Chunking_Lab_Verification.md for the complete mapping and provenance.
