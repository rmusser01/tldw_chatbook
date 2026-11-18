---
id: TASK-41147
title: "User Guide body corrections found while removing the stamps"
status: Done
assignee: ['@claude']
created_date: 2026-11-15 19:35
updated_date: 2026-11-16 21:18
labels:
- follow-up
- docs
- user-guide
priority: low
references:
- Docs/User_Guide/library.md
- Docs/User_Guide/library/search-and-rag.md
- Docs/User_Guide/library/import-and-export.md
- Docs/User_Guide/library/skills.md
- Docs/User_Guide/library/media-and-conversations.md
- Docs/User_Guide/library/notes.md
- Docs/User_Guide/library/file-notes.md
- Docs/User_Guide/console/agent-runs-and-tools.md
- Docs/User_Guide/meetings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 15 of the 2026-10-23 burn-down, from TASK-41101's read of every stamp (review r15doc-rev1 authorized filing it). While checking each removed stamp against its page, the helpers found body sentences that were already wrong before the stamps came off: copy that names a label the code no longer shows, a callout contradicted by a later fix, a duplicated line, two numbered steps with the same number, and limitation notes phrased as verification stamps. Each correction is checked against the current code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 library.md's Get started section no longer says a pre-existing config.toml keeps an empty profile off Get started; it says an empty profile with no stored Library choice settles on Get started (task-32349)
- [x] #2 search-and-rag.md Common tasks step 6 quotes the mode toggle as "mode: ✓ Search ⇄ RAG Answer"
- [x] #3 import-and-export.md quotes the retry suffix as " · retry N", describes the worker-start failure the way the row now shows it (no stale task-32054 pointer), and its export scope example reads "128 media items"
- [x] #4 skills.md names the toggle "User can invoke"
- [x] #5 media-and-conversations.md's Conversations section gives one account of the reader's status line (by title) and message headings (sender and age), with no contradicting "rough edge" paragraph
- [x] #6 notes.md has no duplicated line in the Import once picker paragraph, one step 3 under "Set up lasting folder sync", and one consistent account of Info's footer
- [x] #7 agent-runs-and-tools.md's pointer to the headless-wake close-out report is a working link to Docs/superpowers/plans/2026-08-14-headless-wake-closeout-report.md
- [x] #8 file-notes.md's "Linking a folder without a mouse" step 4 names the "Select folder" button
- [x] #9 meetings.md's sentences about "this page's verification host" or "live session" read as plain limitations
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. For each item, read the page passage and the code that produces the behaviour or copy it describes.
2. Correct only that passage, in the page's own style; leave unrelated history prose alone.
3. Run Tests/Docs and preflight; commit separately from the TASK-41101 fix round.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each correction is checked against the code, and each touches only the passage named.

1. **library.md.** Removed the "You will probably never see this" callout. It said a profile whose config pre-dated the run never reaches Get started, and that task-32059 tracked it. task-32349 (Done) changed that: `library_screen._apply_library_onboarding_evidence` demotes an unstored EXPANDED to UNKNOWN when all sources settle EMPTY, and `aggregate_library_lifecycle` resolves that to STARTER. `coerce_library_lifecycle(raw=None, is_new_profile=False)` still opens EXPANDED as the no-flash default. "Existing profiles without this preference open the full Library" now says exactly that: the full rail while sources are checked, then Get started if every source is empty, the full Library if any holds content.
2. **search-and-rag.md** step 6. The label is "mode: ✓ Search ⇄ RAG Answer", flipping to "mode: Search ⇄ ✓ RAG Answer" (`library_search_rag_panel` → `library_toggle_label("mode", ("Search", "RAG Answer"), …)`).
3. **import-and-export.md.**
   - Step 7 now reads " · retry 1", then " · retry 2" (`library_ingest_state._retry_suffix`, `f" · retry {job.retry_count}"`); the "counts attempts, not retries" aside is gone.
   - The Quirks bullet now describes the macOS semaphore failure as the row shows it: "The import worker couldn't start on this machine (system resource limit) · Restart the app, then Retry" (`map_ingest_failure`). **Show details** holds the raw text (`can_show_details = bool(job.error_detail) or bool(job.error)`). The "tracked as task-32054" pointer is gone.
   - The scope example reads "128 media items" (`library_export_scope._count_phrase(..., "media item")`).
4. **skills.md.** "You can invoke" became "User can invoke" in the bullet and the layout sketch (`skill_user_invocable_label`).
5. **media-and-conversations.md.**
   - The reader bullets now say `Loaded <title> · N of M messages · complete.`, or "this conversation" when there is no title, with messages headed `user · 27m` (`library_conversation_reader`: the status line is built by title per task-32067, `_message_copy` uses `format_console_relative_age`).
   - The "rough edge… raw UUID… ISO timestamp" paragraph is removed. Its parenthetical named "Open in Console", which is now "Resume conversation" in the controls table.
6. **notes.md.**
   - Dropped the repeated "clicked if you clicked one, otherwise the folder being browsed." line.
   - Dropped the superseded first step 3 (the "File name" field of the files-or-one-folder dialog). The second step 3 already records that task-32611 replaced it with the folder-only picker.
   - "Info's footer is fixed instead — 'enter run action'" became "Info names its focused control the same way (see below)". In `library_screen`, the context tier replaces the generic enter chip with `_library_focus_enter_label()` (task-32607), which the paragraph after it already described.
7. **agent-runs-and-tools.md.** The bare "see the headless-wake close-out report" is now a working relative link to `Docs/superpowers/plans/2026-08-14-headless-wake-closeout-report.md`. That report's invariant table (":30", "Deliveries stay serialized app-wide (one `_delivering` per runtime)") backs the sentence, and task-15860's notes name it. The first pass removed the pointer on a false premise. My search was a case-sensitive `grep -rn "close-out report\|closeout report" Docs`, which missed the report's title, "Close-out report — headless wake", and its file name, `…-closeout-report.md`. Review r15doc-rev2 found the file, and the pointer was repaired rather than dropped (lesson added to `lessons-backlog-hygiene.md`).
8. **file-notes.md** step 4. "**Select folder** links it" (`base_dialog.FOLDER_CONFIRM_LABEL = "Select folder"`).
9. **meetings.md.** I judged the three sentences stamp-like: they described "this page's verification host" and "this page's live session", not the product. They now read as plain limitations:
   - live speaker labels are covered by pilot tests only and have not yet been tried in a real recording;
   - call mode is not yet confirmed end to end, since only room mode has been, and call mode also needs the System Audio Recording grant (the self-test recipe stays);
   - Pause/Resume, remembered device choices and level meters are covered by pilot tests only.

Docs-only, so there is no red/green pair. Checks: grep found no test or script pinning any replaced sentence (the "attempt 2" hits in `Tests/UI/test_library_shell.py` and `Tests/Chat/test_console_turn_preparation.py` are product copy, not these pages). `Tests/Docs` and preflight were run before the commit; the counts are in the round report.

Files: `Docs/User_Guide/library.md`, `library/search-and-rag.md`, `library/import-and-export.md`, `library/skills.md`, `library/media-and-conversations.md`, `library/notes.md`, `library/file-notes.md`, `console/agent-runs-and-tools.md`, `meetings.md`; `backlog/docs/lessons-backlog-hygiene.md` (fix round 2).
<!-- SECTION:NOTES:END -->
