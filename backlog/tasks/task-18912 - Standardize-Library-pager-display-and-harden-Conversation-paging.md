---
id: TASK-18912
title: Standardize Library pager display and harden Conversation paging
status: Done
assignee: []
created_date: '2026-08-15 02:44'
updated_date: '2026-08-16 00:37'
labels:
  - library
  - pagination
  - conversations
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every top-level Conversation reachable through a consistent 20-item Library pager while establishing the small pure display convention reused by later source tasks. Preserve full-source search, deterministic deep links, selection safety, and truthful recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation pages contain at most 20 records and expose exact range, total, Previous, Next, loading, disabled-reason, and retry presentation.
- [x] #2 Full-source Conversation search runs before paging, and deterministic stable ordering makes every matching record reachable.
- [x] #3 Off-page Conversation navigation opens the target's coherent rank-derived owning page without injecting an extra page-1 row.
- [x] #4 Conversation count and page rows come from one coherent read transaction and malformed page or locator envelopes fail closed inside the canvas.
- [x] #5 Conversation selection clears with visible notice on page or scope change, while focus and detail/back behavior follow the approved design.
- [x] #6 The shared code is limited to one pure pager-display calculation; Conversation retains request, state, worker, widget, and event ownership.
- [x] #7 Automated state, service, mounted Textual, geometry, race, privacy, mutation, and isolated live verification required by the approved design pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: TASK-18912 changes the Conversation page/locator service contract and establishes the pure shared display contract.

Detailed plan: Docs/superpowers/plans/2026-08-14-task-18912-library-conversation-pagination.md

1. Add the pure immutable pager-display calculation with exhaustive state tests.
2. Make Conversation count/rows coherent and add a bounded stable-ID owning-page locator.
3. Validate Conversation summaries and integrate the pure display without a generic controller/widget.
4. Harden requested/applied scope, retry, focus, selection, restore, races, unmount, clamping, and deep-link lifecycle.
5. Render and geometry-test the Conversation-specific pager.
6. Run inverse mutations, owner/full gates, isolated live verification, docs, reviews, and task closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the source-owned Conversation pagination contract from ADR-067. A
pure immutable pager-display function owns only range/page/disabled/retry copy;
Conversation storage and services retain coherent count/page reads and the
bounded stable-ID locator, while the Library screen/canvas retain request
generation, applied scope, workers, selection, focus, stale rows, and Retry.
Pages are limited to 20 records, filtering is source-wide before paging, and
malformed envelopes fail closed.

- Production files: `library_pager_state.py`, Conversation state/services and
  DB query code, `library_screen.py`, the Conversation canvas, and its layout
  CSS. Tests cover the pure display, DB/service coherence and locator, mounted
  lifecycle/races, multiselect safety, visibility, geometry, and navigation.
  User-facing behavior is documented in
  `Docs/User_Guide/library/media-and-conversations.md`.
- ADR: followed
  `backlog/decisions/067-library-top-level-pagination-contracts.md`; no new
  architectural decision was introduced.
- Inverse mutations: all 6/6 exact oracles failed and were restored green.
  Removing the generation fence applied `old` instead of `new`; restoring the
  prepend path landed on page 1 instead of page 2; splitting count/page reads
  produced a mixed snapshot; dropping missing/duplicate IDs produced the wrong
  cardinality; enabling a stale action changed selection; and allowing another
  clamp called offsets `[40, 20, 0]` instead of `[40, 20]`. File hashes and
  status were clean between mutations.
- Automated evidence: the post-rebase task-local suite passed 274 tests; the
  Conversation owner/mounted suite passed 75 tests twice (678 deselected each).
  The recovered-DOM synchronization regression passed 30/30 repetitions, and
  the related Retry/focus/page-failure matrix passed 60/60. After final review,
  the owner suite passed 76 tests with the added locator-reentry case. The full
  Library shell passed 625 tests; its only two failures were the Notes
  `create_discard` keyboard parametrizations, reproduced identically on the
  then-pinned pre-feature comparison base
  `2ff12ac50b0d7a73599f34e796ca9e933f40a4e8`. The final branch base is
  `e032b5b882f880eee2d6295f3c0be3806247ffaf`. Ruff and diff checks passed.
- Final review found that rail navigation could abandon a Conversation locator
  while leaving its loading/requested scope active. Navigation now revokes only
  that pending locator, restores the retained applied scope, and permits cold
  recovery. Mounted regressions prove one authoritative read for an
  uninitialized source and zero extra reads while preserving a warm page 2.
- Live/privacy evidence: fresh isolated profiles at true
  100x30 and 170x48 each passed all 8 checkpoints with 45 synthetic
  conversations: exact three-page ranges, fixed pager/row 20, source-wide
  oldest-page search and clear, selection clearing, stable-ID navigation,
  recoverable last-good rows/Retry, disabled reasons, and focus contracts.
  Each TUI PID had zero real-profile or foreign DB/config handles and zero TCP
  listeners. The real-profile before/after manifests were byte-identical
  (`ce899523b94e0dd3499e9f0f8c796900817cfb8972b0894bcff707d7d06f0c06`).
- Deviation: a repository-wide pytest run emitted broad failures outside the
  Conversation/Library-pagination files and lost its final summary when the
  terminal turn was interrupted after the last captured 95% progress. Per the
  user's explicit closeout direction, unrelated failures were not rerun or
  classified. Stale pre-fix node IDs remaining in pytest's cache do not exist
  at this HEAD. The authoritative Task-18912 suites above were rerun after that
  event and contain zero related failures.
<!-- SECTION:NOTES:END -->
