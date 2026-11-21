---
id: TASK-42003
title: "Restore the User Guide section on reviewing or undoing a turn's changed files"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:24
labels:
- follow-up
- docs
- console
priority: low
references:
- Docs/User_Guide/console/agent-runs-and-tools.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). Squash 6a6242ce3a (2026-09-12) removed "Reviewing or undoing a turn's changed files" (the turn file card's **Undo All**) from `agent-runs-and-tools.md`, and it never came back; see `backlog/docs/lessons-backlog-hygiene.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The section is restored, with every sentence checked against current code and cited in the notes
- [x] #2 The retired Inspector "Changed files" section and its `[console] changed_files_section` settings bullet, which the same squash put back in this section's place, are removed
<!-- AC:END -->

## Implementation Plan

1. Recover the section from `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md` (it was added by f9a06ff625 in place of the retired rail Changed-files section).
2. Check each sentence against `console_turn_file_card.py`, `chat_screen.py`'s Undo All handler, `review_selection.py`'s `prepare_console_turn_undo`/`apply_console_turn_undo` and `ChangeRevertConfirmModal`; rewrite what no longer holds.
3. Put it back where the squash took it from, which is where the squash re-added the retired "rail's Changed-files section"; drop that stale section and its `[console] changed_files_section` settings bullet (no code reads that key), and fix the intro sentences that say the card has no undo control.
4. Run the doc tests that read this page; preflight before the final commit.

## Implementation Notes

- **What the squash did.** f9a06ff625 (TASK-22305) replaced the Inspector's "The rail's Changed-files section" with "Reviewing or undoing a turn's changed files". Squash 6a6242ce3a reversed that swap: it dropped the Undo All section and put the retired rail section (and its `[console] changed_files_section` settings bullet) back. Neither the rail widget nor the key exists today (`grep -rn changed_files_section tldw_chatbook` finds nothing), so the section goes back in its old slot and the rail section and bullet are removed (AC #2, added before the edit). The squash also removed three opt-in/privacy paragraphs at the top of "Change review"; that is a separate loss, listed as a proposed follow-up, not restored here.
- **Intro fixes.** "The card never mutates anything directly; there is no undo or revert control on it" and "Revert-all ... live only on that screen" were false (they predate the card's Undo All). The header now lists **Undo All** beside **Review**, and the Review screen's `u`/`U` are named. Review rev1 asked for one clause saying the card needs Change Review on for the workspace: turn snapshots cover only consent-admitted roots (`capture_change_review_admission`, `Chat/console_turn_context.py:40`, via the consent service's `admit_turn`), and the per-workspace control is in Settings ▸ Workspaces (`_render_workspace_change_review`, `UI/Screens/settings_screen.py:21944`). The `[console] turn_file_cards` off-switch list now names Undo All too (the plain marker row has no buttons).
- **Sentence-by-sentence sources** (all at base d85e90bc9e):
  - Card header buttons "Undo All" and "Review", Undo All built `disabled=True`: `Widgets/Console/console_turn_file_card.py:345` (compose). Enabled only after the rows mount: `_load_rows`, `:467` (`self._undo_available = True`).
  - "Undoing…" / "Undone" / back to "Undo All": `_sync_undo_button`, `console_turn_file_card.py:570-587`. "Undoing…" starts at the press, before the confirmation (`begin_undo_all` `:549-560`, called from `on_button_pressed`), and ends through `finish_undo_all` on cancel, refusal or completion. An already-reverted turn loads as Undone: `_all_entries_reverted` `:589`.
  - Same confirmation as Review, "Undo all N files from this turn?", Revert/Cancel, "⚠ Changed since this turn (reverting overwrites that work)": `UI/Screens/chat_screen.py:20050` and `UI/Screens/change_review_screen.py:3231` both build `ChangeRevertConfirmModal` (`change_review_screen.py:3566`, warning `:3590`, buttons `:3596`).
  - Cancel leaves files untouched and re-enables the button: `_confirmed(False)` -> `_finish_console_turn_undo(success=False)`, `chat_screen.py:20109`; the revert only runs in `run_console_turn_undo` after confirmation.
  - Partial failure notice names the first five, button back to Undo All: `_publish_console_turn_undo`, `chat_screen.py:20056` (`failed[:5]` at `:20092`), success requires every outcome ok and settled.
  - Refusals: active run (`prepare_console_turn_undo`, `UI/Console_Modules/review_selection.py:406`, rechecked at apply `:466`); a tracking error on any one of the turn's roots, a partial failure, since a turn whose every root failed gets no card (`:410-414`); files changed while the confirmation was open (`apply_console_turn_undo`, `:484`, "Open Review and try again.").
  - Two windows on one workspace: `ConsoleTurnUndoAmbiguousError` (`review_selection.py:399`) makes the screen notify "... Opening Review to undo safely." and open Review (`chat_screen.py:19996`); Review's own `U` then refuses with `MULTI_WINDOW_UNDO_REFUSAL` ("Revert files individually from one labeled window at a time.", `change_review_screen.py:166`, `action_undo_all` `:3198`). The turn-plus-sub-agent window example is the card's own comment in `_load_rows`.
  - Several workspaces in one step, labels are the folder name or the full root when names collide: `root_labels` / `edited_labels`, `review_selection.py:429`, `:448`.
  - Review screen keys `u` Revert file, `U` Undo all: `change_review_screen.py:588-589`.
  - Behaviour pinned by `Tests/UI/test_console_turn_undo_all.py` (confirm/cancel/undo on a mounted card, duplicate-root routing to Review, partial failure and retry).
- **Left out from the old text:** nothing; every old claim still holds. Added two refusals and the partial-failure notice the old text did not mention.
- **Checks:** `Tests/Docs` + `Tests/MCP/test_mcp_documentation_contract.py` through the wrapper, `-n 4`: 119 passed. No test pins the removed strings (`grep -rn "changed_files_section\|rails-changed-files" Tests scripts`: nothing).
- File: `Docs/User_Guide/console/agent-runs-and-tools.md`.
- **Review rev1 fixes:** "Undoing…" timing, the partial tracking failure, the off-switch list and the Change Review qualifier, above. The refusals paragraph now says "folders" (the code works per bound root and labels by folder name), so it does not clash with the intro's Settings ▸ Workspaces. Rechecked with `Tests/Docs` + `Tests/MCP/test_mcp_documentation_contract.py`.
