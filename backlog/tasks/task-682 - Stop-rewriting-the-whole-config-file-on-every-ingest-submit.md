---
id: TASK-682
title: Stop rewriting the whole config file on every ingest submit
status: Done
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 04:09'
labels:
  - ingest
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each ingest submission saves its options one key at a time, and every save re-reads and re-parses the entire config file and invalidates the global settings cache. A single submission triggers several full reload cycles, which is wasteful for one file and grows with every option and every submission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Submitting an ingest writes its options in a single batched save
- [x] #2 The global settings cache is invalidated at most once per submission
- [x] #3 Saved option values are unchanged from the previous behaviour
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each submission looped save_setting_to_cli_config once per option key, and every call re-read and re-parsed the entire config file and invalidated the global settings cache -- about six full reload cycles for one submitted file, growing with every option and every submission.

save_settings_to_cli_config already existed and does exactly one read/write and one cache reload for a whole mapping, so the submit path now builds the per-group mapping and calls it once. Saved values are unchanged; the existing persist-options test was rewritten to assert a single batched write rather than a stream of individual ones.

Changed: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_screen.py
<!-- SECTION:NOTES:END -->
