---
id: TASK-41641
title: "The app-lookup census misses names reached through conditional expressions, enclosing loops and global or nonlocal bindings"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- architecture
- tests
priority: low
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
dependencies:
- TASK-41318
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41318 taught the census to follow literal names through dicts, module tuples, nested loops and same-module returns. Its docstring lists the sources it still does not follow:
- names chosen by a conditional expression (`settings_screen.py` has one);
- a loop variable bound by an enclosing loop;
- names bound through `global` or `nonlocal`.

A lookup reached that way is not counted, so it can grow without the ratchet noticing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census counts literal names reached through conditional expressions, enclosing loops and global/nonlocal bindings
- [ ] #2 Pins are re-measured with dated notes
- [ ] #3 A test per new source shape asserts it is counted
<!-- AC:END -->
