---
id: TASK-38984
title: "Library Notes trust-dialog path test is red on dev, because the centred dialog message pads the folded path lines"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- library
- file-notes
- tests
priority: low
references:
- Tests/UI/test_library_notes_w5_session_git_layout.py
- tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
- tldw_chatbook/css/features/_media.tcss
dependencies:
- TASK-32615
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 9.5).

**The red.** `Tests/UI/test_library_notes_w5_session_git_layout.py::test_the_trust_dialog_never_breaks_the_path_it_asks_consent_for` fails at all three sizes. Its expected path hard-codes `/Users/macbook-dev/Documents/qa/...`.

**Not an isolation problem.** `SessionGitTrustDialog` (`tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`) only formats the string, through `_repository_path_for_display` and `fold_path_lines`. The test never reads or writes that path.

**Cause.**
- The bundled `.dialog-message { text-align: center; }` (`tldw_chatbook/css/features/_media.tcss:706-709`) centres the message.
- The painted path lines therefore start with padding: `'           /Users/macbook-dev/Documents/qa/'` and `'      library-notes-and-vaults/crit4/power/vault'`.
- The test strips only trailing space (`line.rstrip()`), so the joined string contains the leading spaces and the comparison fails.
- The fold itself is correct: every break falls at a separator, and every path component is whole.
- Which commit made the dialog centred was not bisected.

**Ownership and lanes.** TASK-32615, which wrote the test, is still In Progress (last updated 2026-09-15) with no active owner. No branch due to merge before this one changes the file, and no lane runs it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trust dialog paints the complete repository path at all three critique sizes, each non-final line ending at a separator
- [ ] #2 The test passes 3 of 3 runs at each size and still fails on a fold that splits a path component (negative control: the unfolded path)
<!-- AC:END -->
