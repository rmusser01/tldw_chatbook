---
id: TASK-3022
title: 'Ambient Library test debt: fixture bypass and two standing failures'
status: To Do
assignee: []
created_date: '2026-08-07 12:20'
labels:
  - tests
  - library
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ambient test debt on dev, repeatedly re-confirmed (A/B against clean HEAD) by every task of the
`fix/library-uat-p1s` arc — none of it caused by that branch:

1. **Fixture-bypass cluster (~7 failures)**: several test fixtures construct `LibraryScreen`
   bypassing `__init__`, so `_library_ingest_preflight_generation` is never set and the tests fail
   on attribute access. Traced by git archaeology to `6fdde2e68` (2026-08-02, task-2011 generation
   stamp) — the fixtures predate the attribute. Fix the fixtures (construct properly or set the
   attribute), not the production code.
2. `Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story`
   fails on unmodified dev — adjacent to task-2520 (landing footer keyboard story) and task-2860
   (F6 hint stripped by `_RESERVED_GLOBAL_KEYS` in `AppFooterStatus.py`); fixing those may fix
   this. Coordinate rather than patch the assertion.
3. `test_shared_form_and_native_inputs_use_thin_non_semantic_focus` (`_forms.tcss`-adjacent) fails
   on unmodified dev.

A green run of the Library-adjacent suites currently requires knowing which failures are ambient;
that knowledge should live in fixed tests, not session notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The fixture-bypass cluster is fixed at the fixture level; the ~7 affected tests pass on dev
- [ ] #2 The two named standing failures either pass or are traced to their owning open tasks with a note in this task
- [ ] #3 Targeted Library suites run green on dev with no known-ambient exclusion list
<!-- AC:END -->
