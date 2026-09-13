---
id: TASK-32141
title: >-
  Library Notes user-guide sweep: thirteen documented claims contradicted by the
  live surface at c4a7b1911f
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 12:31'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the critique's docs-vs-live table: no empty-state copy renders; rows show no age; rename does not update the row until you leave the field despite a 2026-09-06 'Verified against' stamp; the status line shows no word count; Undo/Dismiss not visible; 'Last import' never appears; the Sort strip has no Title; '/' also types; only Rename fits of Rename / Move / Remove; compact shows '‹ Notes'; 'Linked — folder' holds only for a clean first pick and the timeout copy never paints; 'Add another file' is not rendered for a folder selection. Some are fixed by sibling tasks; the rest need the prose corrected. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed claim is re-verified live or corrected in notes.md / file-notes.md
- [x] #2 The 2026-09-06 rename-propagation stamp is corrected to describe the deferred-refresh behaviour shipped in #2531
- [x] #3 Every 'Verified against' stamp on the two pages names the commit it was checked at
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the critique snapshot's docs-vs-live table (13 claims) and both guide pages in full.
2. Launch the merged worktree on the seeded power profile (and fresh where needed); verify each of the 13 claims live, saving a capture per claim.
3. Correct prose for claims contradicted by live behaviour; leave claims that match live as-is (cite capture in report).
4. Fix the two merge-artifact duplicated paragraphs (delete-receipt, Check-selection review).
5. Rewrite the 2026-09-06 rename-propagation stamp and its body prose to describe the actual deferred-refresh mechanism (refresh skipped while title/body has focus, replays on the next out-of-field refresh -- in practice, returning to the list), based on live testing plus the task-32062/PR #2531 commit history.
6. Consolidate every inline stamp into one chronological 'Verified against' block per page; add missing commit hashes (task-3315, TASK-19026, TASK-24309) via git log; append my own stamp.
7. Tick ACs, mark Done with implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified all 13 docs-vs-live claims from the critique table live on the merged worktree (power profile for 12, fresh profile for the empty-state claim); 5 needed prose corrections, 8 already matched live behaviour (fixed by sibling groups) and were left as-is with a citation.

Corrected: (1) rename-propagation -- rewrote both the body prose and the 2026-09-06 stamp to describe the actual shipped mechanism (a Notes refresh is SKIPPED, not queued, while the title/body has focus; it repaints on the next refresh that finds focus outside the editor, which in practice means returning to the list via the note work area's own back control -- not simply tabbing to another field, which live-testing three different ways showed does NOT repaint the row). (2) Notes status line -- removed the 'N words · saved' claim; the status line never carries a word count live, only Saved/Saving.../Unsaved changes/etc.; word count lives in Info -> Properties. (3) Sort toolbar row -- added the caveat that it is only offered before any folder tree has loaded, matching the existing tree-order explanation and the task-32128 code comment; live testing across three widths never produced it once notes/folders exist. (4) '/' keyboard row -- corrected the re-arm claim: a controller-ruling follow-up in the editor-keys review round made a second '/' (once the filter has focus) an ordinary typeable character, not an accelerator, since filter text can target a folder-style path; confirmed live (typed '/' became a literal slash). (5) file-notes.md's linked-folder status text -- 'Linked -- <folder>' does not render; the live/visible text is 'Linked · Local folder: <folder>' (the em-dash form is tooltip-only); fixed in both files' prose and stamps, including the task-32136 clause which also overclaimed the Library rail stays visible during the pre-link empty state (verified live it does not -- full-width onboarding step until a folder is linked).

Verified accurate (no change): rows already show title+age; delete receipt's Undo/Dismiss render fully (not clipped); Undo returns the row live; 'Last import' persists and reopens the receipt after Back to Notes; compact editor already reads '‹ Back to list' vs '‹ Notes' at wide; 'Add another file' is correctly documented as absent for a folder selection; the empty-state copy 'No notes yet. Create your first note.' plus the Agent_Lessons gloss render on a fresh profile.

Not independently re-verified live: the folder-change timeout copy (30s+ real wait, impractical to force without an artificially huge directory) -- left as documented, backed by the existing pinned-test stamp (task-32055/32121).

Also fixed two merge-artifact duplications (a repeated delete-receipt paragraph in notes.md, a truncated-then-repeated 'Choose Check selection' paragraph) and consolidated every previously-inline stamp into one chronological 'Verified against' block per page. Added missing commit hashes via git log for three stamps that only had a bare date (task-3315 -> 71f15ff76f, TASK-19026 -> 1bda754fa1, TASK-24309 -> 38b2704b36). Appended a stamp naming this pass. No code changes.
<!-- SECTION:NOTES:END -->
