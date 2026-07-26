---
id: TASK-698
title: Ingest form does not reset to defaults on Library re-entry
status: To Do
assignee: []
created_date: '2026-07-26 05:36'
labels:
  - ingest
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-entering the Library ingest canvas via a nav deep-link is supposed to clear any stale form state, but the form no longer equals a fresh one, so something from the previous visit survives. A user returning to the screen can find it carrying values they thought they had left behind. Pre-existing on dev, found while regression-testing the 684.2 registry work: Tests/UI/test_library_shell.py::test_library_shell_ingest_nav_context_deeplink_reentry_resets_stale_form fails identically at 05ebe2ab7, before the ingest UAT batch merged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-entering the ingest canvas leaves no state from the previous visit
- [ ] #2 The existing deep-link re-entry test passes
- [ ] #3 Whichever field was surviving is identified, and it is clear whether keeping it was intended
<!-- AC:END -->
