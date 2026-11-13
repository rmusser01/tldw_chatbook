---
id: TASK-40296
title: "A Media item open when the user left Library is lost on return, because the first compose clears the restored selection before it is fetched"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 02:28'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- media
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/app_navigation.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified both nodes as one product defect and left them red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** A Media item that was open when the user left Library is not reopened when they come back. Its detail is never fetched, and a deleted item never shows its recovery state.

**Cause, from TASK-39152's notes.** `restore_state` (`UI/Screens/library_screen.py:10664`; `app_navigation.py` calls it) sets the selected id. The first `compose_content` then clears it through `_adopt_library_media_highlight` (`UI/Library_Modules/library_media_controller.py:3224`) before `on_mount` holds the highlight.

**Nodes.**
- `Tests/UI/test_library_shell.py::test_library_shell_restored_media_viewer_fetches_detail_on_mount`: the restored selection is `None`.
- `::test_library_shell_restored_media_viewer_with_deleted_item_shows_recovery`: "#library-media-viewer-error never mounted within 30.0s".

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Returning to Library with a Media item open reopens that item's viewer and fetches its detail
- [x] #2 Returning with an open item that was deleted meanwhile shows the viewer's recovery state
- [x] #3 Both nodes pass 3 of 3 at `-n 4`, shown red before the fix, and their lines leave `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. Re-check at base 26b2e677dd: run both nodes, confirm red and the cause (compose before on_mount).
2. Begin the restored Reader selection (and its highlight hold) in `restore_state`, where the selection is set, so the first compose's highlight adoption keeps it; leave the detail fetch in `on_mount`.
3. Run both nodes 3x at `-n 4`, the restore covering tests, the size ratchet; drop the two deselect lines.

## Implementation Notes

- Premise still held at base 26b2e677dd (2026-11-13, load 45-52): `fetches_detail_on_mount` failed `assert None is not None` on `_media_state.detail`, `with_deleted_item_shows_recovery` failed "#library-media-viewer-error never mounted within 30.0s". Cause confirmed: Textual composes before `on_mount`; the media branch of `compose_content` calls `_adopt_library_media_highlight` on a list with no page, and with no Reader session begun yet (`on_mount` began it) nothing held the highlight, so the selection was written to "" and `on_mount`'s `if selected_media_id` skipped the fetch. The compose-time adoption came from TASK-34000.18 (b273392327); the hold was only taken in `on_mount` (a29bfe8258).
- Fix: `restore_state` now resolves the Reader identity and calls `begin_selection(..., immediate=True)` plus `highlight_held_by_reader = True` for a restored viewer, the same pair the by-link door (`_open_library_item_by_id`'s media branch) already sets before compose. `on_mount` keeps only the fetch. The old `pending_request is None` guard was dropped: `restore_state` only runs on a freshly constructed screen (`app_navigation.py`), and a newer restore should supersede anyway.
- Evidence: both nodes red at base (above), green 3 of 3 at `-n 4` after the fix at load 52-70 (twice: before and after a comment compaction), together with `test_library_shell_restore_state_degrades_editor_view_without_matching_id`. No new test: the two existing nodes are the red-to-green proof. Both lines left `scripts/library_shell_push_deselect.txt`.
- Files: `tldw_chatbook/UI/Screens/library_screen.py` (net -3 lines), `scripts/library_shell_push_deselect.txt`, `Docs/User_Guide/library/media-and-conversations.md` (one sentence: leaving Library and coming back reopens the open item, or its Retry state).

