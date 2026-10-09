---
id: TASK-34424
title: Debounce personas conversation search and read-only index check
status: Done
created_date: 2026-10-07 02:42
updated_date: 2026-10-07 12:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F14: personas conversation search fires a full search cycle with a write transaction per keystroke and media filter debounce is 0.12s vs 0.2-0.3s elsewhere
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 5-char burst fires exactly one search,READY index performs zero write transactions,Media filter debounce 0.25s
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 12 (T12)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Three mechanical patches: (1) personas conversation search debounced at 0.2 s mirroring the sibling PERSONAS_SEARCH_DEBOUNCE_SECONDS idiom (timer re-arm, last-value-wins, unmount cancel, guard re-evaluated at fire time so stale pending timers no-op); (2) ensure_keyword_index gains a read-only READY probe before the write transaction — live-connection SELECT mirroring the write path's READY determination exactly (BUILDING precedence included), write path unchanged and still runs when stale, race note in-code (unconditional singleton UPDATE is value-neutral to skip); (3) dedicated MEDIA_FILTER_DEBOUNCE_SECONDS = 0.25 for the media filter handler, SELECTION_SETTLE_SECONDS 0.12 untouched for selection. Hermetic tests drive the real handler bodies (only set_timer/run_worker faked); the personas test fires all five stale timers to prove stop() semantics. Evidence: 10-keystroke burst 10 -> 1 dispatches; READY index 3 -> 0 immediate write transactions (non-READY still builds); media arms 0.25 / selection 0.12. Size-ratchet pin bumped 4768->4769 (exactly the one added import). Files: UI/Screens/personas_screen.py, DB/character_conversation_search.py, UI/Library_Modules/library_media_controller.py, Library/library_media_reader_state.py + 3 test files. Report: .superpowers/sdd/task-12-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
