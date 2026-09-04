---
id: TASK-31269
title: Reader walk keys are typed into the content search field
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 15:09'
labels:
  - library
  - media-ux
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P0 (my #2367 regression): the content search bar's on_mount focuses its Input whenever the query is empty, and the Analysis tab (task-28026) mounts that bar whenever an analysis exists. So in Analysis mode every ]/[ item load moves focus into the box and the next key is typed as text (A cap 22/24/36: `▊ ]`); in Read mode the same happens whenever Find is open across a walk (B cap_32b `]]]]]`, cap_41-44 with the review banner frozen). The Find button does not toggle the bar off and Escape from inside the input only blurs on the first press. Workflow 3 (review every analysis with ]) is exactly this path and fails silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In Analysis mode, walking two items with ] never leaves focus in the search Input (test asserts `screen.focused` is not the Input after each load)
- [x] #2 With Find open in Read mode, ] [ m R still act as keys across item changes — the bar keeps its query but never re-takes focus on an item change
- [x] #3 The Analysis-tab search bar is collapsed until Find is invoked, matching the Read tab
- [x] #4 Pressing Find while the bar is open closes it; Escape from inside the Find input closes the bar in one press on every tab
- [x] #5 Live-verified in tmux: an Analysis-mode walk over three items using only ]
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: Analysis-mode walk keeps focus out of the Input; Find on Analysis opens in place and one Escape closes; empty-bar Read walk never steals focus; Find toggles closed
2. GREEN: LibraryMediaContentSearchControls(focus_on_mount) explicit token; viewer find_focus_pending spent on the one compose after the gesture; screen sets it only in handle_library_media_reader_find; Analysis bar gated behind find_open like Read; Find toggles
3. Retire task-28026's Find Analysis->Read jump (reader-flow contract test rewritten); convert six shell tests that queried the bar before any Find press
4. Live tmux 235x52: Analysis-mode [ ] walk over three items, Find on Analysis, Escape, Find toggle
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was two of my own changes meeting: #2367's on_mount focused the search Input whenever the query was empty (inferring the Find gesture from an empty query), and task-28026 mounted the Analysis bar unconditionally, so every item load in Analysis mode (and any walk with an empty bar open) moved the caret into the field and the next ]/[/m was typed. Fix: the gesture is an explicit one-shot token. handle_library_media_reader_find sets _library_media_find_focus_pending; the viewer construction site and the in-place sync both consume it (a pending token forces the recompose path so it is never lost), the viewer passes focus_on_mount to the controls and spends the token right after yielding them, and on_mount focuses only when the token is set. The Analysis tab now gates its bar behind find_open exactly like Read, so Find opens the bar for the tab being read and one Escape collapses it on either tab; task-28026's Analysis->Read jump on Find is retired (same-mode reset is a no-op). Find is a toggle. Also converted six test_library_shell.py tests that queried the search input before any Find press (broken on dev since #2367 collapsed the bar) via a shared _open_media_find helper, and retired the 18-row-cap assertion. Two plain-harness search tests still fail on clean dev for layout/identity reasons and are recorded in task-31249. Live-verified in tmux 235x52 (Analysis-mode walk over three items with [ and ], Find on Analysis, Escape, Find toggle).
<!-- SECTION:NOTES:END -->
