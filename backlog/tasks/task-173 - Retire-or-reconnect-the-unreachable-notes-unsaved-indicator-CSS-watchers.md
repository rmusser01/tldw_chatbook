---
id: TASK-173
title: Retire or reconnect the unreachable notes unsaved-indicator CSS/watchers
status: Done
assignee: []
created_date: '2026-07-11 23:53'
labels:
  - follow-up
  - notes
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The .unsaved-indicator/.has-unsaved/.auto-saving/.saved CSS family plus app.py's watch_notes_unsaved_changes/watch_notes_auto_save_status watchers are practically unreachable (the target widget is never composed since the standalone Notes screen was retired) but still grepped-live in app.py, so the F2/quick-wins CSS sweeps kept them. Adjudicate: either remove the dead watchers + CSS together, or reconnect them to the Library notes editor's real save-state indicator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unsaved-indicator CSS + watchers are either removed together or reconnected to a live widget,No dead grep-live references remain
<!-- AC:END -->

## Implementation Notes

Adjudicated **remove** (not reconnect): the `#notes-unsaved-indicator` widget is never composed by any live screen (the standalone Notes screen that used it was already retired), the `notes_unsaved_changes`/`notes_auto_save_status` reactives are never mutated by any live path, and the Library notes editor already has its own save/conflict/auto-sync UI — there was nothing to reconnect to.

Deleted:
- 2 reactives in `tldw_chatbook/app.py`: `notes_unsaved_changes` and `notes_auto_save_status` (left `notes_sort_by`, `notes_sort_ascending`, `notes_preview_mode`, `notes_auto_save_enabled`, `notes_auto_save_timer`, `notes_last_save_time` untouched — those are live).
- 2 watcher methods in `app.py`: `watch_notes_unsaved_changes` and `watch_notes_auto_save_status` (~44 lines).
- The `.unsaved-indicator` / `.has-unsaved` / `.auto-saving` / `.saved` CSS block (plus its explanatory comment) in `tldw_chatbook/css/features/_notes.tcss`.
- Rebuilt `tldw_chatbook/css/tldw_cli_modular.tcss` via `build_css.py`; diff confined to the removed CSS block + the regenerated `Generated:` timestamp (verified via `git diff | grep -E "^\+[^+]" | grep -v "Generated:"` returning nothing).

Added `Tests/UI/test_notes_unsaved_indicator_removed.py`: a grep-guard test parametrized over the six dead identifiers (fails if any reappear in shipped `tldw_chatbook/` source) plus a boot-smoke test confirming the app still mounts with the CSS rules gone. Guard test was RED before the deletions (all 6 identifiers found) and GREEN after (7 passed). Import smoke (`import tldw_chatbook.app`) also confirmed clean.
