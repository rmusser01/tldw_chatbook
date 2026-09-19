---
id: TASK-32703
title: Verify Library conversation archive and restore journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 21:38'
updated_date: '2026-09-16 21:53'
labels:
  - ui
  - conversations
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the Archive and Restore verification gap after the Conversations filter and reader review, preserving saved identity, recoverability and usable controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancel, Archive, versioned Undo and Restore only act on the intended disposable conversation and preserve its messages.
- [x] #2 Active and Archived browsing retain the search query and expose usable recovery controls at compact and wide sizes in both themes.
- [x] #3 A fresh native process discovers archived state and restores the same saved conversation without changing Console context.
- [x] #4 Targeted checks, bounded native captures, persistence evidence and any demonstrated limitations are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: qualify the shipped archive and recovery contracts without changing data ownership, schemas or UX structure.

1. Trace existing Archive/Restore/Undo, query scope and retained reader behavior; review neighboring test evidence.
2. Exercise mounted Cancel, Archive, Undo, Archived and Restore only through real native services and private saved conversations at 80x24 and 170x48 in both themes.
3. Leave a known archived conversation, restart the app process, restore it from Archived and verify exact conversation/message identity and unchanged Console context.
4. Inspect one batched capture set, run targeted automated/static checks and review findings. Record any demonstrated defects before expanding repair scope.
5. Reconcile the feature-review ledger and earlier semaphore status references with the new evidence, then close the task if criteria pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified the shipped Archive/Restore journey without production changes. Real native controls in both themes at 80x24 and 170x48 preserve the query and target only Alpha; Cancel preserves its version, confirmed mutations record the incremented version, and Undo reverses Archive and Restore. A separate process finds Alpha archived at version 26 and restores it to version 27 with the same message IDs/bodies; Beta remains active at version 1. Console active session/list and the empty draft are unchanged within each process.

Both native processes return normally with exit 0 and independently absent PIDs. Ten private SQLite databases pass integrity checks; four fixture message IDs, owners and content hashes match; default-profile hashes are unchanged and logs have no ERROR/CRITICAL lines or traceback headers. Ten captures were rendered and inspected in one round with no actionable defect.

56 targeted checks pass in 21.64s. The native helper passes Ruff lint/format, evidence hash/XML/JSON/link checks pass, and independent review reports no actionable findings. No full suite was run. Native controls use explicit focus plus Enter and compact pane toggles; complete Tab traversal, nonempty drafts/attachments, Export and exact Resume are outside this evidence.

Files: Docs/superpowers/qa/2026-09-16-conversation-archive/ contains the guarded native helper, final results, lifecycle/persistence receipts, captures and verification limits. The Library workflow report records completion and next Export/Resume journeys. Historical Import semaphore notes in TASK-32701, TASK-32388 and Conversations QA now link the completed TASK-32700 qualification.

ADR required: no. Existing ADR-147 (archive/exact resume), ADR-086 (adaptive reader) and ADR-150 (design language) govern this verification-only work. No new architectural decision or general lesson. No implementation-plan deviation.
<!-- SECTION:NOTES:END -->
