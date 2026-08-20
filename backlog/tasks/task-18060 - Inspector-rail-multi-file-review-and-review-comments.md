---
id: TASK-18060
title: Inspector-rail multi-file review and review comments
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18'
updated_date: '2026-08-20 15:22'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-16800
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Arc A of the V2 turn-file-card design, split out of TASK-16801 (owner
ruling 2026-08-18: tackle the two V2 halves individually; this half
first). Today a user reviews changes one card/turn at a time, and the
Review screen is reachable per turn only; nothing shows the
conversation's changed files across all turns, and review feedback is
limited to the card's hunk notes.

This task adds (per the code-grounded spec,
`Docs/superpowers/specs/2026-08-18-console-review-rail-design.md`):
a "Changed files" section in the existing Inspector rail listing the
conversation's cross-turn latest state per file (cached-summary pattern —
never a DB/git read on the rail's sync tick), click-through to the
existing Review screen focused on that file, and plannotator-style
commenting in the Review screen — a comment on a specific diff line or on
the whole file — anchored to the immutable snapshot diffs and delivered
to the agent through the same TASK-16800 auto-attach loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Inspector rail shows a Changed-files section listing the conversation's changed files across ALL turns (latest state per file: status, ±counts, note badge), capped with an honest overflow tail, and rendering nothing when the conversation has no recorded changes
- [x] #2 The section's data is never computed on the rail's sync tick: an unchanged conversation state performs no recompute (verified by test), and a new turn's changes appear via one off-thread refresh
- [x] #3 Selecting a listed file opens the existing Review screen on that file's turn with that file focused (constructor-state plumbing; no post-push race)
- [x] #4 In the Review screen, the user can attach a comment to a specific diff line (cursor over the rendered diff) and to the whole file, without leaving the screen; comments are validated and persisted like TASK-16800 notes
- [x] #5 Line and file comments are delivered to the agent through the existing auto-attach loop with kind-aware block and disclosure rendering (byte-identical live vs resume), stamped by exact id, and surviving session resume
- [x] #6 The Review screen displays the focused file's existing feedback (hunk notes, file comments, line comments) with pending-vs-sent state; pending comments can be deleted, delivered ones cannot
- [x] #7 Revert behavior, the single-file diff render cap, and the `[console] turn_file_cards` kill-switch behavior are all unchanged (no regression to their existing pinned tests)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented as the 9-task arc in `backlog/plans/2026-08-18-console-review-rail.md`,
against the code-grounded spec at
`Docs/superpowers/specs/2026-08-18-console-review-rail-design.md`. Tasks
1-8 shipped the feature; this close-out (Task 9) verified every doc claim
against the shipped code, wrote the user-guide section, and ran the full
targeted sweep.

**Approach.** Cross-turn aggregation is a pure function
(`conversation_file_summary` in `Chat/console_display_state.py`) fed by a
new provider method (`AgentRunsChangeReviewProvider.conversation_changed_
files`) that does the git subprocess work per snapshot row; the rail
widget (`Widgets/Console/console_changed_files_section.py`) is pure
presentation over precomputed `ConsoleChangedFilesState`. The Review
screen (`UI/Screens/change_review_screen.py`) gained a diff-line cursor,
`c`/`C` comment creation, and a notes strip; `change_notes` gained
`anchor_kind`/`diff_line_index`/`diff_line_text` (schema audit v11 in
`DB/AgentRuns_DB.py`) so a comment can anchor to a specific diff line or
to the whole file, in addition to the existing hunk anchor.

**Delivering decisions:**
- **Cached-summary pattern** (rail): the section never queries the DB or
  runs git on the Inspector's 0.2s sync tick — `ChatScreen` holds a
  `_console_changed_files_summary` cache plus a per-row git memo, and a
  guard tuple (`conversation_id`, newest `change_review_run_id` seen in
  the session's in-memory messages — an O(messages) scan, no DB read)
  decides when to dispatch ONE off-thread (`asyncio.to_thread`) recompute.
  Every note-mutation path (card save/delete, the Review screen's
  dismissal callback) resets the guard so a stale `✎ N` badge can't
  linger. This is the exact precedent the dictionary/world-book rail
  sections already established, not a new pattern.
- **Snapshot-aware selection**: a rail row click passes both
  `initial_path` and `initial_snapshot_id` into the Review screen's
  constructor (never a post-push method call — the opener's own
  `call_after_refresh`-before-compose race is documented at
  `change_review_screen.py:414-427`). `select_file` prefers the leaf
  whose owning row id equals `initial_snapshot_id`, falling back to
  first-path-match for legacy (no-snapshot) callers — this disambiguates
  two windows of the same run covering the same path.
- **Delivering-run precedent**: line and file comments write through the
  same `add_change_note` call the card's hunk notes already use (now with
  `anchor_kind`/`diff_line_index`/`diff_line_text` keywords, defaulted so
  every existing caller stays byte-compatible), and ride the identical
  auto-attach/stamp/disclosure loop in `Chat/console_agent_bridge.py` — no
  new delivery mechanics, only the two shared formatters
  (`render_diff_feedback_block`, `format_diff_feedback_disclosure`)
  learned to render a `file` or `diff_line` kind alongside the existing
  `hunk` shape.
- **No-wrap root cause**: the diff pane's cursor assumes one rendered row
  per logical line. Textual 8.x converts a `rich.text.Text` into its own
  `Content` type at render time, which ignores `Text(no_wrap=True)` and
  instead reads the `text-wrap` CSS property — so the actual fix is
  `.change-review-diff-body { text-wrap: nowrap; width: auto; }` in
  `_change_review.tcss`, with `#change-review-diff`'s `overflow-x: auto`
  making a long line horizontally scrollable instead of silently
  desyncing the cursor's target row.
- **Inline markers**: a `diff_line` note appends a dim `● comment` marker
  to the END of its own diff line (never a new line), computed once per
  file-focus/note-mutation from the same `_notes_for_leaf` read the notes
  strip uses, and merely consulted (no I/O) inside `_render_diff` so a
  cursor move never costs a query.

**Modified/added files** (tasks 1-8; see `git diff --stat f00acbd8b HEAD`):
`tldw_chatbook/Chat/console_display_state.py`,
`tldw_chatbook/DB/AgentRuns_DB.py`,
`tldw_chatbook/UI/Console_Modules/right_rail.py`,
`tldw_chatbook/UI/Screens/change_review_screen.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/Widgets/Console/__init__.py`,
`tldw_chatbook/Widgets/Console/console_changed_files_section.py` (new),
`tldw_chatbook/Widgets/Console/console_turn_file_card.py`,
`tldw_chatbook/css/components/_change_review.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss`, plus new/expanded tests in
`Tests/Chat/test_change_notes_db.py`,
`Tests/Chat/test_console_conversation_files.py`,
`Tests/Chat/test_console_diff_feedback_delivery.py`,
`Tests/Chat/test_console_diff_hunks.py`,
`Tests/UI/test_change_review_screen.py`,
`Tests/UI/test_console_changed_files_section.py`,
`Tests/UI/test_console_changed_files_wiring.py`, and
`Tests/UI/test_console_turn_file_card_notes.py`. Task 9 (this close-out)
additionally updated `Docs/User_Guide/console/agent-runs-and-tools.md`.

**Verification.** The targeted 11-suite sweep — `Tests/Chat/test_change_
notes_db.py`, `Tests/Chat/test_console_conversation_files.py`,
`Tests/Chat/test_console_diff_hunks.py`, `Tests/Chat/test_console_diff_
feedback_delivery.py`, `Tests/UI/test_change_review_screen.py`,
`Tests/UI/test_console_changed_files_section.py`, `Tests/UI/test_console_
changed_files_wiring.py`, `Tests/UI/test_console_turn_file_card_notes.py`,
`Tests/UI/test_console_turn_file_card.py`, `Tests/UI/test_console_turn_
file_card_factory.py`, and `Tests/Chat/test_console_agent_bridge.py` —
414 passed, 0 failed (160.88s). Every AC above was checked against a
specific passing test (see the sweep) rather than ticked on trust, and
every doc claim in the user-guide update was checked against the shipped
source it describes before being written (byte-exact strings verified
against the widget/screen source, not recalled).
<!-- SECTION:NOTES:END -->
