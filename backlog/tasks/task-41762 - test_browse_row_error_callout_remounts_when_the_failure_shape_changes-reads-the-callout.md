---
id: TASK-41762
title: "test_browse_row_error_callout_remounts_when_the_failure_shape_changes reads the callout's region before layout"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
- flaky
priority: low
references:
- Tests/UI/test_library_shell.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_shell.py::test_browse_row_error_callout_remounts_when_the_failure_shape_changes` (`:17093`) covers the Notes load-failure callout and has been on dev since 2026-09-07. It waits for DOM state, then reads the callout's painted text before a layout pass. When it fails, the callout's `node.region` is 0x0.

Found while verifying the media chain's last rebase (rb5, 2026-11-18): dev 45b1949ca3 as base, the chain tip as head.

| Run | Load | Base red | Head red |
|---|---|---|---|
| the rb5 A/B itself | - | 0 of 1 | 1 of 1 |
| 3 serial rounds | - | 0 of 3 | 2 of 3 |
| 10 paired single-id rounds | 19-22 | 0 of 10 | 1 of 10 |
| 6 concurrent runs per side, 5 rounds | 20-28 | 2 of 30 | 3 of 30 |

Overall the base failed 2 of 44 runs and the head 7 of 44 (p about 0.16), with the same assertion on both sides. The chain touches no callout or source-failure code, so this is a race in a dev test. At dev 908e9840aa (2026-11-18) the test passed 4 of 4 alone at load about 11.

The file runs whole on every push to dev (`derived-artifacts.yml:1851`). The sibling test, `test_browse_row_error_callout_keeps_the_right_tint_across_a_retry`, is recorded as a load flake in TASK-39152 and TASK-40713.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test reads the callout's painted text only after the callout has a laid-out, non-zero region
- [ ] #2 It passes 10 of 10 at -n 4 under load, with the load recorded
- [ ] #3 The sibling tint test is checked for the same shape, and fixed if it has it
<!-- AC:END -->
