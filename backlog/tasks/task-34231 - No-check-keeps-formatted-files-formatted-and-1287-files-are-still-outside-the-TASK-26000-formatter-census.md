---
id: TASK-34231
title: >-
  No check keeps formatted files formatted, and 1,287 files are still outside
  the TASK-26000 formatter census
status: To Do
assignee: []
created_date: '2026-10-04 00:07'
labels:
  - ci
  - formatter
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the TASK-26000 series (PR #2993), measured with the pinned Ruff 0.15.22: 6,410 files are format-clean and 1,287 still need formatting (2,983 before the series). Nothing in preflight or CI runs the formatter, so a formatted file stops being formatted the next time it is edited; several files the series owned were already unformatted again by the time the PR was reviewed.

Reformatting is not free of side effects here. It changes line counts and line shapes, and three kinds of check depend on those: size-ratchet rows pinned to exact line counts, tests that match source text by literal, and generated artifacts that hash or key on formatting. PR #2993 broke one of each without any statement changing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A check in preflight and in the required CI job fails when a file that is format-clean today stops being format-clean at the pinned Ruff version
- [ ] #2 The check follows the same no-install rule as the other derived-artifact checks, or the pinned formatter is provisioned in a way that rule explicitly allows
- [ ] #3 Every remaining unformatted file is either formatted or listed as an exception with a reason
- [ ] #4 Size-ratchet rows, source-text pins and generated artifacts affected by a reflow are updated in the same change, each with evidence that no statement changed
<!-- AC:END -->
