---
id: TASK-16800
title: 'Turn file card: annotate/feedback loop and Review affordance'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-15'
updated_date: '2026-08-17 16:50'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-1972
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console turn file card (`ConsoleTurnFileCard`) lets a user expand any
changed file's diff inline, but it is read-only: there is no way to leave
feedback on a specific hunk, and the only path into the full Review screen
(turns, retention, guarded per-path revert) is the keyboard-only `v`
binding, undiscoverable from the card itself. This is V1.5 of the turn
file card design (`Docs/superpowers/specs/2026-08-15-console-turn-file-review-design.md`,
"Out of scope" section): the review screen and guarded revert already
exist via TASK-1972, so this task is scoped to two additions — an
annotate/feedback loop on expanded hunks, and a `Review` button on the
card that opens that same screen at the turn.

Feedback recorded here should be usable as context for the agent's next
reply, closing the loop between "the agent shows me a diff" and "I tell it
what to change" without leaving the transcript to type a follow-up
message by hand.

Two other V1.5 polish items were also trimmed from V1 and belong in this
same follow-up bucket: a header collapse/expand-all chevron (today each
row toggles independently, with no all-at-once control) and middle-elided
per-row paths (today a long path is shown in full rather than elided to
fit the row).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An expanded diff row exposes an action to attach a note to a specific hunk, without leaving the transcript
- [x] #2 A note attached to a hunk is durably recorded (survives session resume, like the rest of the card's source data) and is available to the agent as context on its next reply
- [x] #3 The card exposes a `Review` affordance that opens the existing Review screen scoped to that turn — equivalent to pressing `v`, reachable without the keyboard shortcut
- [x] #4 No control added to the card performs a destructive action; revert remains exclusively on the Review screen behind its existing confirm (the TASK-1845/TASK-1972 precedent)
- [x] #5 With `[console] turn_file_cards` set to `false`, the plain-text marker row and its `v` binding are unaffected by this feature (no regression to the kill-switch fallback)
<!-- AC:END -->

## Implementation Plan

Executed as a 7-task plan (`.superpowers/sdd/2026-08-17-console-turn-file-annotate/`,
spec `Docs/superpowers/specs/2026-08-17-console-turn-file-annotate-design.md`):

1. `change_notes` table + CRUD API in `AgentRunsDB` (audit version 8, this
   DB's own append-only convention).
2. Pure hunk segmentation (`split_unified_diff`/`DiffHunk`/`hunk_excerpt`)
   and the diff-feedback block/disclosure formatters in
   `console_display_state.py`.
3. Restructure the card's expand path from one flat `Static` per row to
   one block per hunk, with a cached `list[DiffHunk]` replacing the old
   joined-string cache.
4. Hunk-level note UI (`✎ note` → inline `Input` → save/cancel/delete) and
   the `on_key` reclaim fix for Enter/Escape inside a nested
   `ConsoleTranscript`.
5. Delivery: `ConsoleAgentBridge.run_reply` auto-attaches pending notes to
   the outbound copy of the next agent-path send, stamps exactly the
   attached ids delivered at run completion, and emits a disclosure row.
6. Disclosure resume re-derivation, anchored at the delivering run
   (`delivered_by_run_id`, audit version 9).
7. Affordances + docs + close-out (this task's own final wave): `Review`
   button, expand/collapse-all chevron, middle-elided paths, User Guide
   update, task close-out.

## Implementation Notes

**Approach.** Tasks 1-6 landed the persistence, segmentation, per-hunk
note UI, and delivery/disclosure/resume mechanics (see their own commits
for detail: `5bac23c74`, `82d21b626`, `4808b9455`, `53fcb8475`,
`121ec831f`, `a47fc325c`, `9efea3d8a`, `83e1a2494`, `9d0213b92`,
`608b3c56a`). Task 7 (this close-out) added the three remaining
affordances and closed the loop:

- **`Review` button.** `ConsoleTurnFileCard.ReviewRequested(Message)`
  carries the card's own `run_id`; a compact header button (`compact
  =True`, `active_effect_duration = 0`) posts it on press.
  `ChatScreen.handle_console_turn_file_card_review_requested` handles it
  by calling the screen's existing `_open_change_review(run_id)` opener —
  the same recipe the `v` binding and the run inspector's own "Review
  changes" button already use. **Finding:** `ChangeReviewScreen.
  __init__`'s `initial_run_id` parameter and `AgentRunsChangeReviewProvider
  .turn_for_run` already existed, landed with the ORIGINAL TASK-1972 work
  (commit `e1480f831`), predating this whole V1.5 series — so this task
  needed no change to `change_review_screen.py` at all, only to wire a
  new caller into the opener that was already there. Unknown/stale run
  ids already fell back to the latest turn (`_initialize_turns`), so
  AC#3's "equivalent to pressing `v`" holds for that edge case for free.
- **Expand/collapse-all.** A header toggle button reads live DOM state
  (`all(body.display for body in bodies)`) rather than tracking its own
  boolean — a user can still expand/collapse one row individually via its
  own button without the toggle drifting out of sync. Expand-all loads
  any uncached row's diff SERIALIZED inside one coroutine (sequential
  `await`s, never `asyncio.gather` or N `run_worker` calls), reusing the
  existing per-row `_hunk_cache`; a single row's provider/diff failure is
  caught and logged without aborting the rest. The per-hunk mounting
  logic (colored `Static` + action row + notes box + existing notes) was
  factored out of `on_button_pressed` into `_mount_hunk_blocks` so the
  single-row expand path and expand-all can never render a row
  differently.
- **Middle-elided paths.** `middle_elide_path(path, budget)` in
  `console_display_state.py`: keeps the first and last `/`-split
  components, replaces everything between with a single `…`, and returns
  the path unchanged when it already fits or has ≤2 components (nothing
  meaningful left to drop). The card computes each row's label budget
  from `self.size.width` at mount, on toggle, and in a new `on_resize`
  handler (`Resize` does not bubble, so this only fires for the card's
  own width changes); the row `Button`'s `tooltip` always carries the
  full, un-elided path.
- **AC#4 guard.** A new test asserts no button anywhere on a fully
  expanded card (header + every hunk's note/delete buttons) has a label
  or class matching `revert`/`undo` — proven RED by construction, since
  neither string appears anywhere in the card's source.
- **Kill switch (AC#5).** The existing byte-parity factory test was left
  untouched (still green); a new bridge-level test,
  `test_kill_switch_off_does_not_prevent_note_delivery`, forces
  `[console] turn_file_cards = false` via the same monkeypatch shape as
  the factory test and confirms a pending note still auto-attaches,
  stamps, and discloses on the next agent send — the delivery seam lives
  in `ConsoleAgentBridge.run_reply` and never reads that presentation
  switch.

**Decisions carried from earlier tasks, restated for this close-out:**

- **The delivering-run ruling (Task 5/6).** A note is stamped delivered
  only by the exact id list its OWN attach step captured for the run that
  actually carried it — never a blanket "all pending for the
  conversation" stamp. This closes a real race: annotating an older
  turn's card while a newer run is already in flight must not let that
  newer run's completion silently swallow the older card's just-created
  note. The disclosure row's resume re-derivation was anchored the same
  way (`delivered_by_run_id`, audit version 9, Task 6) so a fresh session
  re-derives the SAME disclosure content, anchored after the SAME run's
  marker, rather than after whichever run happens to be newest.
- **The Enter-key `on_key` fix (Task 4).** A `BINDINGS`-only approach for
  the note input's Enter/Escape is provably wrong once the card is
  mounted inside a real `ConsoleTranscript`: that ancestor's own raw
  `on_key` (`enter -> confirm_selection`, `escape -> clear_selection`)
  unconditionally stops the event before Textual's non-priority binding
  resolution ever sees it, so a real Enter keypress inside a focused note
  `Input` selected the transcript row instead of saving — silently, with
  nothing raised or logged. The card now defines its own raw `on_key`,
  closer to the input than `ConsoleTranscript`, and reclaims both keys
  there; this is pinned by a live-transcript regression test (Task 4) and
  remains the single source of truth for both keys in every host.

**Modified/added files (whole TASK-16800 V1.5 arc, Tasks 1-7):**

- `tldw_chatbook/DB/AgentRuns_DB.py` — `change_notes` table + CRUD (Task 1).
- `tldw_chatbook/Chat/console_display_state.py` — hunk segmentation,
  delivery block/disclosure formatters (Task 2), `middle_elide_path`
  (Task 7).
- `tldw_chatbook/Widgets/Console/console_turn_file_card.py` — per-hunk
  blocks + note UI (Tasks 3-4); `ReviewRequested`, expand/collapse-all,
  path elision + `on_resize`, `_mount_hunk_blocks`/`_read_hunks`
  refactor (Task 7).
- `tldw_chatbook/Chat/console_agent_bridge.py` — auto-attach, exact-id
  stamping, disclosure emission + resume re-derivation (Tasks 5-6).
- `tldw_chatbook/UI/Screens/change_review_screen.py` — `turn_for_run`
  run-scoped read (Task 3-era; `initial_run_id` itself predates this
  series, from TASK-1972).
- `tldw_chatbook/UI/Screens/chat_screen.py` — imports
  `ConsoleTurnFileCard`; `handle_console_turn_file_card_review_requested`
  handler (Task 7).
- `Docs/User_Guide/console/agent-runs-and-tools.md` — Change review
  section rewritten (annotate flow, Review button, expand-all, elision,
  kill-switch note) + stamp (Task 7).
- Tests: `Tests/Chat/test_change_notes_db.py`,
  `Tests/Chat/test_console_agent_bridge.py`,
  `Tests/Chat/test_console_diff_hunks.py`,
  `Tests/Chat/test_console_diff_feedback_delivery.py`,
  `Tests/UI/test_console_turn_file_card.py`,
  `Tests/UI/test_console_turn_file_card_notes.py`,
  `Tests/UI/test_console_native_transcript.py`,
  `Tests/UI/test_change_review_screen.py` (across all 7 tasks; Task 7 added
  the Review-button/expand-all/AC#4-guard/tooltip/elision cases, the
  `initial_run_id` open/fallback cases, the pure `middle_elide_path` cases,
  and the kill-switch delivery case).

**Verification.** `Tests/UI/test_console_turn_file_card.py`,
`Tests/UI/test_console_turn_file_card_notes.py`,
`Tests/UI/test_console_turn_file_card_factory.py`,
`Tests/UI/test_change_review_screen.py`,
`Tests/Chat/test_console_diff_hunks.py`, and
`Tests/Chat/test_console_diff_feedback_delivery.py` — 91 passed. Full
`Tests/UI/` collection swept clean (13,122 tests, no import errors) after
the `chat_screen.py` import addition. No live-tmux walkthrough of this
task's own affordances (a real end-to-end scenario needs a real agent run
against a real git root); the app was smoke-launched on a scratch profile
and confirmed to start and render without a crash after these changes.
