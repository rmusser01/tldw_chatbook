---
id: TASK-32583
title: 'Tests: the dev-red Notes pins wave 4 measured but did not own'
status: To Do
assignee: []
created_date: '2026-09-14 22:47'
labels:
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 compared every branch against a detached origin/dev worktree by FAILED-name SET, which surfaced a set of reds that are dev's, not any group's, and that nobody owns. Recorded here with what measured them rather than absorbed as 'pre-existing' — the wave's own census red (test_state_object_declares_the_censused_field_count, 105 vs 106) was found precisely because a group refused to absorb one. The set: two test_screen_footer_hints.py pins expecting footer chips the tiers deliberately dropped (group 6); a load-sensitive pair in test_library_notes_reader.py that is red on both trees (group 6); five Notes geometry params that are red on dev identically (group 8); and the Folder-files flaky pair, 4 failures in 9 paired runs on dev (group 8, and see the #file-notes-path-label rider for its likely cause). Added by the wave's landing pass (task-32558), same rule, different family: Tests/Docs/test_console_library_controls_docs.py fails on README.md twice — test_console_library_control_docs_state_the_contract[README.md-required_text0] (missing 'Manual Search Library', 'Auto: Never / Automatic', 'Assistant: Blocked / Allowed', 'Direct / RAG') and test_console_library_control_docs_have_valid_local_links[README.md] (missing local link 'mailto:contact@rmusser.net', which is a mailto the link checker should not be resolving as a local path at all). Measured on a detached origin/dev worktree at fd30614dcd: 2 failed, 19 passed — an identical FAILED-name set to the branch's own run, so it is dev's, not the docs sweep's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each named red is reproduced on a clean detached origin/dev worktree and classified: stale expectation, real defect, or load-sensitive
- [ ] #2 Stale expectations are re-pinned at the shipped strings, not loosened
- [ ] #3 Real defects get their own task; load-sensitive ones are either stabilised or recorded with a measured flake ratio
- [ ] #4 Tests/ is green on dev for these names, or each survivor has an owning task id
<!-- AC:END -->
