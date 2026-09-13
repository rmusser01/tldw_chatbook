---
id: TASK-32097
title: >-
  Critique #8 gap-fix Qodo tail: Info-tab interior half-flag + complementary
  test coverage
status: To Do
assignee: []
created_date: '2026-09-08 22:22'
labels:
  - library
  - media
  - tests
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three Qodo notes on the merged critique-#8 gap-fix PR #2527, tracked as a rider (low priority, arc complete). (a, correctness, narrow) `_display_keywords_text` (library_media_viewer_state.py) drops a lone regional indicator only when it is the TRAILING run of a comma-separated keyword part; a lone indicator INTERIOR to a keyword (e.g. 'a<RI>b') still distorts the Info 'Keywords:' line. This is a pathological input (a real flag is an adjacent PAIR; a lone interior indicator does not arise from normal keywords), which is why it was not fixed in task-32087. (b, testability) the Info-line flag guard has a state-level unit test but no MOUNTED-Reader integration test. (c, testability) the 0-result-filter reader-repaint branch (task-32086, library_screen.py ~15865) has a mounted-app integration test but no isolated unit test of the mount guard / focus callback / early return.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Info 'Keywords:' line never paints a lone/odd regional indicator regardless of its position within a keyword (interior as well as trailing), or a note records why interior is out of scope
- [ ] #2 A mounted-Reader integration test asserts the sanitized Keywords line as rendered (not only the generated state)
- [ ] #3 An isolated unit test covers the 0-result-filter repaint branch (mounted and unmounted cases, refresh invocation, focus scheduling, early return)
<!-- AC:END -->
