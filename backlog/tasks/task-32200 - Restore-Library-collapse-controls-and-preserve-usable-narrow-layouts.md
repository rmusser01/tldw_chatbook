---
id: TASK-32200
title: Restore Library collapse controls and preserve usable narrow layouts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:35'
updated_date: '2026-09-09 19:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the Library layout regression reported after PR #2550 and restore the intentionally wide collapse controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Library reader retains full-height five-cell collapse controls with working pointer and keyboard targets.
- [x] #2 The requested wider default columns fit without hiding a list or rail that can still fit beside the reader.
- [x] #3 Production-styled rendered checks cover narrow and wide layouts, populated readers, and collapse and restore interactions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the five-cell grip and narrow-view regressions with the production Library shell and CSS.
2. Restore five-cell collapse controls and make automatic preferred widths yield to available space before a still-usable pane disappears; retain custom-width behavior.
3. Verify populated and empty readers at narrow and wide sizes, including pointer/keyboard collapse and restore; review the correction.
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: repair the existing adaptive geometry and control-size contract; no new ownership or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the shared five-cell collapse controls on Media, Conversations, Skills and Collections. Automatic layouts surrender only the added five Library and ten Items cells before using the earlier collapse boundaries; wide layouts recover the requested increases. Explicit reopen bypasses resize hysteresis for its requested pane. Custom-width handling, reader floors and saved preferences remain intact.

Changed the four production width/profile files, added rendered click-and-keyboard regressions, updated affected geometry expectations, and clarified the guide and ADR-086. No new ADR required: this repairs the existing geometry and control-size contract.

Validation: 460 focused layout tests passed; 109 Media/Conversations/Skills interaction tests passed; Ruff and diff checks passed. Independent review found and verified the explicit-reopen edge fix. Production-CSS captures inspected at terminal widths100,124,160; Media shell width is100 in compact mode,120 at terminal124, and156 at terminal160.

Adjacent check: test_more_stays_compact_at_the_narrow_reader_width still fails because Move to trash is clipped. Reproduced unchanged on the original PR branch938a1d8b; it predates this repair and is not hidden by a changed expectation.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-32184, colliding with the older
"Library-Notes-adaptive-reader-items_width-mismatch-for-wide-terminals-now-reachable"
follow-up (created 2026-09-09 17:53, on `fix/library-notes-r-tests`), which
arrived first. This task was created 2026-09-09 19:35 (add commit db86a39b19
on `codex/library-layout-repair`, merged to dev as PR #2559). Per the owner
rule decided 2026-08-21 in TASK-19601 (**the older arrival by
`created_date` keeps the id regardless of status; the younger task
renumbers with a provenance note**), it renumbered to TASK-32200. The other
TASK-32184 holder is the older arrival and keeps the id.

Inbound references moved with it: the two "five-column restoration" notes in
`Docs/User_Guide/library/media-and-conversations.md` (lines ~151 and ~558),
which cite this repair. Both surviving references now read TASK-32200.
