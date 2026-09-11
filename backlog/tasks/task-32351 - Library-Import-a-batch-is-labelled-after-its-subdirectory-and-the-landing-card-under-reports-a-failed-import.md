---
id: TASK-32351
title: >-
  Library Import: a batch is labelled after its subdirectory, and the landing
  card under-reports a failed import
status: Done
assignee: []
created_date: '2026-09-11 06:16'
updated_date: '2026-09-11 07:21'
labels:
  - library
  - import
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Importing <profile>/inbox produced a queue group named 'nested — 6 files' (the subdirectory) though 5 of the 6 files were in inbox/ itself (B D1 caps 06/10); after 4 failed and 2 skipped the landing's Needs-attention card reads only 'An import needs review.' (B D2 cap 14). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A folder import's group is named after the folder the user imported
- [x] #2 The landing card states the outcome counts ('4 failed, 2 skipped') not a neutral 'needs review'
- [x] #3 Both pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the folder-batch name and the failure tally (Tests/Library/test_library_ingest_state.py, Tests/UI/test_library_crit10_onboarding.py).
2. Name a batch after the common root of its members' parents in build_ingest_queue_groups._flush, falling back to the first member's parent when commonpath refuses.
3. Tally live FAILED/SKIPPED jobs in _library_landing_attention_action and state the counts.
4. Update the two 'An import needs review.' pins and the import docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: `build_ingest_queue_groups._flush` named a batch after `members[0].source_path.parent.name`. For a recursive folder scan the first member is whichever subdirectory the walk yielded first, so a six-file import of `inbox/` read 'nested — 6 files' after its one nested file. The folder the user chose is the common root of every member, which the members already carry -- `os.path.commonpath` over the members' parents, falling back to the old expression when it refuses (mixed absolute/relative, URL sources). No new field, no schema change.

Note the plan's proposed test did not fail before the fix: it listed the nested file LAST, and the first member's parent was already 'inbox'. The pin puts the nested file first, which is the order that produced the bug, and reproduces the exact critique string ('nested — 6 files · active · 6 queued').

AC#2: `_library_landing_attention_action` returned the neutral 'An import needs review.' on the first live FAILED job -- neutral where the queue itself was exact. It now tallies live FAILED and SKIPPED jobs from the same registry snapshot it already walked: 'Last import: 4 files failed, 2 skipped.' (singular 'file' at one). The two existing copy pins in test_library_shell.py were updated to the new sentence.

Live-verified on an empty profile importing its own inbox/ (6 files, one nested): the queue group reads 'inbox — 6 files · now · 2 skipped · 4 failed' and the landing card reads 'Last import: 4 files failed, 2 skipped.' with Review. Captures 05-06 under scratchpad crit10/wave/onboarding-import/caps. The import SUCCESS path was not exercisable on this host -- POSIX semaphores are exhausted, so every local parse worker fails at pool start; that is what produced the 4 failures, and it is the fixture the critique itself ran under.

Files: tldw_chatbook/Library/library_ingest_state.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/Library/test_library_ingest_state.py, Tests/UI/test_library_crit10_onboarding.py (new), Tests/UI/test_library_shell.py, Docs/User_Guide/library/import-and-export.md
<!-- SECTION:NOTES:END -->
