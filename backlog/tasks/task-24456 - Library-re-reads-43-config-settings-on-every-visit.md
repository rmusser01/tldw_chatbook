---
id: TASK-24456
title: Library re-reads 43 config settings on every visit
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - library
  - config
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`UI/Screens/library_screen.py::_load_library_ingest_options_from_config` runs from `on_mount`
and performs 43 `get_cli_setting` reads. The app constructs a NEW `LibraryScreen` on every
visit -- verified live, three visits produced three distinct instance ids -- so those 43 reads
repeat on every navigation to the Library to produce an identical answer.

CORRECTION TO THE ORIGINAL FILING: this was first written up as "Library ingest options are
re-read from config on unrelated screen switches", derived from a 33.5-`get_cli_setting`-
per-switch figure averaged over a Library+Console PAIR. Measured per destination that is wrong:
Console switches read 18-21 settings and none of them come from `library_screen`. The average
smeared one screen's mount cost across both switches. The waste is real, but it is a per-visit
remount cost, not cross-screen leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeat visits to the Library screen do not re-read ingest options from configuration
- [x] #2 The options are still refreshed when the underlying configuration changes
- [x] #3 A screen with no running app reads fresh, so a caller that swaps the settings source sees its own values
- [x] #4 Library ingest behaviour is unchanged for a user who edits ingest options and returns to the screen
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
The 43 reads are memoised on `current_config_generation()` -- a new lightweight accessor in
`config.py` that returns the counter every config mutation bumps, without the locks and deep
copy that `get_runtime_config_snapshot` takes.

The cache lives on the running `App`, not at module scope. That was not the first attempt: a
module-global cache passed in isolation and then failed
`Tests/UI/test_library_screen.py::test_load_ingest_options_from_config` inside the full suite,
because it outlived the app and served one test's stubbed config to the next. App-scoping fixes
that structurally -- an unmounted screen has no app, so it reads fresh, which is also the
correct production behaviour for any caller that has swapped the settings source.

Measured: first Library visit 49 `get_cli_setting` calls (unavoidable), repeat visits 46 -> 2.

Modified: `tldw_chatbook/config.py` (new `current_config_generation`),
`tldw_chatbook/UI/Screens/library_screen.py`.
<!-- SECTION:NOTES:END -->
