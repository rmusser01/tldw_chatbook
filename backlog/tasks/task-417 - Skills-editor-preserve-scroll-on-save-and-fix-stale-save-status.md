---
id: TASK-417
title: Skills editor - preserve scroll on save and fix stale save status
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 16:30'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). Pressing Save at the bottom of the editor snaps the viewport back to the top so the 'Saved.' status line and the trust-state change render below the fold unseen. The 'Saved.' text then persists indefinitely - it still read 'Saved.' after a later bootstrap-trust action. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Save the viewport still shows the Save button and adjacent status line (scroll position preserved or scrolled to the status),Save status is cleared or replaced when a different action runs so it never reports a stale outcome,Save outcome is perceivable without scrolling,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scroll: create-save arms a one-shot _library_skill_scroll_pending; the snapshot-refresh recompose that used to snap the canvas to the top now passes scroll_to_actions to the canvas, whose on_mount scroll_visible()s the Save row back into view (existing-skill saves never recomposed, so they were already fine - live-verified during the review). Stale status: _mark_library_skill_dirty clears a lingering 'Saved.' alongside the dirty mark, and all four trust action handlers (setup/unlock/review/approve) clear it up front. Chasing the create-save regression this exposed a LATENT bug (since Task 4): the post-create snapshot recompose remounts the editor's Inputs/TextArea, whose spurious mount-time Changed events can land AFTER any call_after_refresh re-arm - the armed-flag dance cannot win that race, so a just-saved create editor was silently marked dirty (Back veto for no reason; my status-clear made it visible as vanished 'Saved.'). Fix: value-aware dirty-marking - _library_skill_text_fields_match_state compares live fields to editor state and skips pure mount echoes; toggle/cycle handlers (which mutate state before marking) pass force=True. 4 new tests watched fail first; suites: skills canvas 65 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
