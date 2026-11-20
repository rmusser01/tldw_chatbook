---
id: TASK-42006
title: "The Media page names its bulk actions by labels the canvas does not show"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:36
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/library/media-and-conversations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). `media-and-conversations.md` calls Media's bulk actions "Export selected", "Delete selected" and "Review selected" in several places, but the canvas labels are "Export", "Review", "Analyze" and "Delete" (`library_media_canvas.py` ~898-939). (The `settings.md` category map's missing rows are TASK-35016's AC#12.) Conversations' own bulk action really is labelled "Export selected" (`library_conversations_canvas.py` ~246), so only Media's wording is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every mention of Media's bulk actions uses the canvas labels (Export, Review, Analyze, Delete), checked against the code; Conversations' "Export selected" is left as it is
<!-- AC:END -->

## Implementation Plan

1. List every mention of Media's bulk actions in `Docs/User_Guide/library/media-and-conversations.md`.
2. Check the labels in `library_media_canvas.py` (`_select_mode_bulk_buttons`, `_analyze_selected_button`, `_delete_selected_button`) and Conversations' "Export selected" in `library_conversations_canvas.py`.
3. Rename Media's to Export, Review, Analyze and Delete; leave Conversations' "Export selected".
4. Run the doc tests that read this page; preflight before the final commit.

## Implementation Notes

- **Labels in code** (base d85e90bc9e, `Widgets/Library/library_media_canvas.py`): Export `:898` and Review `:906` (`_select_mode_bulk_buttons`, on the Clear row `:1439`), Analyze `:925` (`_analyze_selected_button`, its own row `:1500`), Delete `:939` (`_delete_selected_button`, its own danger row `:1519`, below Analyze). The bulk-delete confirmation copy "Delete N selected items? ..." (`:1383`) is unchanged and still quoted. Conversations' "Export selected": `Widgets/Library/library_conversations_canvas.py:246`.
- **Renamed in `media-and-conversations.md`:** the shared-grammar strip (now names Conversations' "Export selected" and Media's Export/Review/Analyze/Delete), the disabled-marker bullet ("○ Delete selected" was Media-only and wrong; the column-hold example now says "Export"), the "Export…" / select-mode export bullet, **Media's "Delete"** (was "Delete selected", and "next to Export selected" is now "a row of its own"), the Reader table's "as it does for the bulk Delete", **Review** in review sets (was "Review selected"), the export common task (both spellings), the delete common task step 2, and the "Select ▸ Delete" aside. Conversations' "Export selected" mentions are unchanged. Headings that are not labels ("Export selected items as a bundle", "Delete selected media items") stay.
- **One mention on another page:** `import-and-export.md` said "Export selected" is the select-mode export wherever a list offers it; it now says Media spells it "Export". This is within AC #1's "every mention"; no other User Guide page names Media's bulk actions (`grep -rn "Delete selected\|Review selected\|Analyze selected" Docs/User_Guide`: only Prompts' own "Delete selected", which is Prompts' real label, `library_prompts_canvas.py:904`).
- **Checks:** `Tests/Docs` + `Tests/UI/test_library_crit9_media_reader.py` (it reads this page) through the wrapper, `-n 4`: 47 passed, 6 failed. The 6 are `RecoveryRequired: raw_source_selection_changed` (the local trap) in async UI tests that never read the guide; the same file at base content gives the same 6 failed, 3 passed. Its two guide-reading tests pass.
- Files: `Docs/User_Guide/library/media-and-conversations.md`, `Docs/User_Guide/library/import-and-export.md`.
