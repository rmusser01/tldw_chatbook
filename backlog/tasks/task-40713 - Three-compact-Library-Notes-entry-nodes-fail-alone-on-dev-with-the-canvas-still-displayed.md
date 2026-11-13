---
id: TASK-40713
title: "Three compact Library Notes entry nodes fail alone on dev with the canvas still displayed"
status: To Do
assignee: []
created_date: 2026-11-13 05:15
labels:
- follow-up
- library
- tests
priority: medium
references:
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
- .github/workflows/derived-artifacts.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down, from the TASK-39152 stream. Checked on dev 26b2e677dd, 2026-11-12.

Three TASK-39153 watch items in `Tests/UI/test_library_shell.py` now fail when run alone, 3 of 3 runs at load 59-70. Each fails with `assert True is False` on `#library-canvas`'s `display` after entering compact Notes. In round 13 each passed alone at least once.
- `test_library_note_wide_deep_link_back_clears_future_compact_intent` (~:32527)
- `test_library_note_unsafe_session_outranks_rail_focus_on_compact_entry` (~:33366)
- `test_library_note_load_failure_outranks_rail_focus_on_compact_entry` (~:33421)

None of them is in `scripts/library_shell_push_deselect.txt`. So if they are red, the `library-shell-dev` push job is red on every dev push, and real regressions in that job go unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say, with evidence, whether each node is red from a product defect or from a test race, each red shown first with the same command
- [ ] #2 A product defect is fixed and its node passes; a test race is fixed by waiting on a condition; each node passes 5 of 5 alone and in its file at -n 4
- [ ] #3 If a fix is out of scope, the node is deselected in the push job with its owning task named, and the push job's pins still pass
<!-- AC:END -->
