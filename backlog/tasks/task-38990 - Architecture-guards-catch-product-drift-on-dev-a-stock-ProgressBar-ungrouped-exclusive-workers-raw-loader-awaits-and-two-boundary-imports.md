---
id: TASK-38990
title: "Architecture guards catch product drift on dev: a stock ProgressBar, ungrouped exclusive workers, raw loader awaits and two boundary imports"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- ui
- architecture
priority: medium
references:
- tldw_chatbook/UI/Screens/meetings_screen.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/watchlists_collections_screen.py
- tldw_chatbook/Actor_Packs/contracts.py
- tldw_chatbook/Persona_Visual/builtin_pixel_migu.py
- Tests/Architecture
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. A serial `Tests/Architecture` run (1044 s, load 7-10) failed five guards, each on real product drift. No PR lane runs these guard files, which is why the drift reached dev.

The five guards and what they caught:
- **Progress guard:** a stock `ProgressBar` at `tldw_chatbook/UI/Screens/meetings_screen.py:272-273`, dating from 9fe7f34a14 (2026-09-04).
- **Ungrouped exclusive workers,** 7 sites:
  - `chat_screen.py:7824`
  - `watchlists_collections_screen.py:1647`
  - `SmartContentTree.py:192`
  - `ChatbookImportWizard.py:662`
  - `console_goal_setup_modal.py:287`
  - `workspace_change_review.py:121`
  - `workspace_create_modal.py:548`
- **Raw loader awaits,** 3 sites:
  - `schedules_workbench._request_tasks_refresh._load`
  - `watchlists_collections_screen._follow_coordinated_briefing`
  - `_request_briefings_refresh.refresh`
- **Actor packs boundary:** `tldw_chatbook/Actor_Packs/contracts.py` imports `tldw_api.character_persona_schemas`.
- **Persona visual boundary:** `tldw_chatbook/Persona_Visual/builtin_pixel_migu.py` imports `Character_Chat.builtin_pixel_migu`.

None of the branches due to merge before this one touches these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The meetings screen uses the repo's pausable progress bar instead of a stock ProgressBar
- [ ] #2 Every listed exclusive worker names a group, and each listed refresh dispatches through its loader-group helper
- [ ] #3 Actor_Packs and Persona_Visual no longer import the forbidden packages, or each boundary records a reviewed exception with its reason
- [ ] #4 The five guard nodes pass on dev (red before: 5 of 5 at 29592ecc9b), and each still fails when its drift is reintroduced (negative control)
<!-- AC:END -->
