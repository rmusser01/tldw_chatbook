---
id: TASK-501
title: >-
  Console branching: keep transcript selection across sibling swipe for
  back-to-back comparison
status: Done
assignee:
  - '@claude'
created_date: '2026-07-23'
updated_date: '2026-07-24'
labels:
  - console
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With Console branching (Phase A, PR #799), pressing `<` / `>` to navigate sibling branches moves the active leaf and re-syncs the transcript, which clears the selected message (pre-existing precedent from PR #359: `selected_message_id` is dropped whenever the message no longer occupies a transcript slot, shared with the "continue" action). Consequence: the action row disappears after each swipe, so comparing branches back-to-back requires re-selecting the message and re-opening the action row before each `<` / `>`. Since rapid branch comparison is the primary reason this feature exists, the swipe path should keep the selection anchored (e.g. re-select the new active node at the same turn position) so repeated `<` / `>` works without a re-click.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a `<` / `>` sibling swipe, the transcript keeps a selection at the swiped turn so the `<` / `>` action row stays available
- [x] #2 Repeated `<` / `>` presses navigate siblings without an intervening row re-click
- [x] #3 The "continue"/other-action selection-clear behavior is not regressed
- [x] #4 Verified in the live TUI
<!-- AC:END -->

## Implementation Plan

1. Hand the swiped-to sibling's id from the `variant-previous`/`variant-next`
   action handler to the transcript as a PENDING selection (not an eager
   `select_message`) so it is applied at ingest time, after the post-swipe
   view actually reaches the widget.
2. Hold the handoff on the SCREEN (`_pending_console_swipe_selection`) and
   transfer it onto whichever transcript instance receives the next push —
   a recompose can remount the transcript between the swipe and the push.
3. Apply the pending id inside `set_messages` BEFORE the stale-selection
   clear, so the swipe that removed the old selection lands directly on the
   swiped-to sibling.
4. Extend the swipe UI test to pin selection-follow + repeated no-re-click
   swipes; leave "continue"/other actions on the existing clear rule.

## Implementation Notes

- `_select_console_message_variant` now returns the landed sibling's native
  id (None on boundary no-ops); the action handler stores it in
  `_pending_console_swipe_selection` (screen-held, remount-proof) and the
  sync transfers it to `ConsoleTranscript.pending_selection_id`, which
  `set_messages` applies once the id is present in the ingested set —
  selecting eagerly would either miss the membership guard or be cleared by
  reconciliation against the stale message set.
- Only the `variant-previous`/`variant-next` branch of
  `handle_console_message_action` sets the handoff; "continue" and every
  other selection-clearing action keep the PR #359 clear-on-swap rule
  (AC #3), pinned by the untouched selection-contract suite.
- The pre-existing test tail was written for the old clear-on-swipe world
  (it re-selected a1 while the view sat on a1's path); with the new
  second-swipe coverage the view rests on a2, so the tail now swipes back
  to a1 via the followed selection and asserts the boundary disabled states
  there, ending on selection == a2 after the final swipe (was: None).
- Verified live in the real TUI (tmux, local llama_cpp @9099, scratch
  profile): regenerate to 3 siblings, `<`/`<`/`>` back-to-back with NO row
  re-clicks — counter walked (3/3)→(2/3)→(1/3)→(2/3) with the action row
  and guide persistent throughout.
- Files: `tldw_chatbook/UI/Screens/chat_screen.py`,
  `tldw_chatbook/Widgets/Console/console_transcript.py`,
  `Tests/UI/test_console_native_chat_flow.py`.
