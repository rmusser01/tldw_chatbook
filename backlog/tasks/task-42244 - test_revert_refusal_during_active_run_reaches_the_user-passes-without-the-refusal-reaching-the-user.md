---
id: TASK-42244
title: "test_revert_refusal_during_active_run_reaches_the_user passes without the refusal reaching the user"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- change-review
priority: low
references:
- Tests/UI/test_change_review_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs2 stream (round 19). `Tests/UI/test_change_review_screen.py:589` (`test_revert_refusal_during_active_run_reaches_the_user`) sets `provider.run_active = lambda: True`, opens the review, presses `u`, clicks `#change-revert-yes` and then asserts only `(root / "edit.txt").read_text() == "after\n"`. That file content is also unchanged if the click lands before the modal is laid out, if the confirm modal never opens, or if the refusal is swallowed. Nothing asserts that a refusal notice or message is shown, so the test's name is not what it checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test asserts the text the user sees for the refusal (a notification or the on-screen copy), not only that the file is unchanged
- [ ] #2 The new assertion is shown red under a temporary source edit that swallows the refusal, then green
- [ ] #3 The test waits for the confirm modal to be laid out before the click, so it cannot pass on a missed click
<!-- AC:END -->
