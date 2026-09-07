---
id: TASK-23145
title: Reader test doubles lack the scroller seam production now reads
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - library
priority: medium
dependencies: []
---

## Description

6 tests in `Tests/UI/test_library_media_reader_flow.py` fail with
`AttributeError: 'types.SimpleNamespace' object has no attribute 'scroller'`. Production is correct
and deliberately explicit: the reader resolves the scroll offset through the virtualized scroller
rather than the body. The doubles were never updated when that seam moved.

## Acceptance Criteria

- [ ] All 6 tests pass with body doubles exposing the scroller seam production actually reads
- [ ] The tests assert the offset is read **through** the scroller, so a future seam change fails
  loudly instead of a double silently satisfying the old shape

## Evidence

Production: `tldw_chatbook/UI/Screens/library_screen.py:37003` `content = body.scroller` (also
`:37121`, `:37391`). Doubles return `SimpleNamespace(scroll_x=0, scroll_y=17)` with no `.scroller`
at `Tests/UI/test_library_media_reader_flow.py:755`, `:794`, `:1097`.

Introduced by `f08e942295` (authored 2026-08-26) "fix(library): resolve the reader scroller
explicitly and scroll to matches exactly (TASK-22500)", landed on dev 2026-08-28 via merge
`b5eaa9cf64` (PR #2129). That commit updated `Tests/UI/test_library_shell.py` for the same seam but
not this file.
