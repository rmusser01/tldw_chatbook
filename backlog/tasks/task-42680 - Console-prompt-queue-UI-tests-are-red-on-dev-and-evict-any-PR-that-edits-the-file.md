---
id: TASK-42680
title: >-
  Console: the prompt-queue UI tests are red on dev and evict any PR that
  edits the file
status: To Do
assignee: []
created_date: '2026-11-28 09:00'
labels:
  - console
  - tests
  - ci
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_prompt_queue.py` has tests that fail on dev. The light CI tier runs every test file a PR changes, so any PR that edits this file inherits those failures, and the merge queue evicts it.

PR #3769 (TASK-33621.30) edited one docstring in the file and was evicted after its required check failed twice (run 46042456647, attempts 2 and 3, job "Dev smoke and affected contracts"). The docstring edit was reverted from that PR rather than inheriting the reds.

Failing in CI:
- `test_dirty_queue_edit_vetoes_navigation_and_preserves_text`
- `test_wired_queue_admission_freezes_view_source_filter[busy|edit|race]`
- `test_wired_queue_rejects_wrong_owner_before_draft_or_queue_mutation[busy]`

Locally on 2026-11-24, the whole file showed 8 failed / 64 passed on dev 75bac413f5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing test's cause is identified as a test defect or a product defect, with evidence
- [ ] #2 Test defects are fixed so the file passes whole on dev
- [ ] #3 Product defects are fixed or filed with their own task
<!-- AC:END -->
