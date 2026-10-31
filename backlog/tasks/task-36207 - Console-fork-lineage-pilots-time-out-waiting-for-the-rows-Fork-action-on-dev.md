---
id: TASK-36207
title: Console fork-lineage pilots time out waiting for the row's Fork action on dev
status: To Do
assignee: []
created_date: '2026-10-22 18:33'
labels:
  - console
  - tests
  - follow-up
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_fork_fresh_lineage_flow.py (not in the UI PR gate census, so no PR lane runs it) is mostly red locally on dev 7700c63a68 (2026-10-22, macOS, plain command, host load 19-45).
- **Run alone**, test_fresh_saved_pair_opens_the_fork_dialog_from_both_rows, test_refused_fork_is_not_advertised_and_names_the_blocking_row and test_quoted_row_text_is_painted_literally_not_as_markup all fail, on dev and on TASK-35660's branch alike: 'Timed out waiting for #console-message-action-fork-<id>' in the file's own _select wait. The shared provider-selection helper had already returned and the send had settled.
- **In a whole-file run**, two of those three were red on each tree (a different pair each time), and test_help_then_send_then_fork_opens_the_fork_dialog passed on both.
- The visible text at the timeout shows the Console with no Fork action for the selected row. It also shows 'Could not load local character chats · Retry', which may point at the local test profile rather than the product.

Found while sampling the shared helper's callers for TASK-35660's rebase. Not caused by that branch: the same tests fail the same way on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four fork-lineage pilots passes under the plain pytest command on dev, alone and in a whole-file run, or the cause is shown to be local-only (for example the test profile) and recorded
- [ ] #2 If the selected row's Fork action is genuinely missing after a fresh send, that product defect is fixed or filed separately with its evidence
<!-- AC:END -->
