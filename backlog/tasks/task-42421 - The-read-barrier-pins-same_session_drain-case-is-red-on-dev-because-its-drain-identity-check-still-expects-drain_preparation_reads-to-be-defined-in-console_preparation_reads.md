---
id: TASK-42421
title: "The read-barrier pin's same_session_drain case is red on dev because its drain identity check still expects drain_preparation_reads to be defined in console_preparation_reads"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:40
labels:
- follow-up
- console
- tests
- dev-red
priority: low
references:
- Tests/UI/test_console_tab_publication_read_barrier.py
- tldw_chatbook/Chat/console_preparation_reads.py
- tldw_chatbook/Backup_Recovery/runtime_producer_lifetime.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** `Tests/UI/test_console_tab_publication_read_barrier.py::test_actual_close_publishes_real_tabs_before_stock_read_release[same_session_drain]` fails every time, here and on dev's source (TASK-42342's notes, line 84). The `[after_close]` and `[coalesced]` cases pass. The file is in no PR step and not in the UI census, so nothing reports it.

**Cause: test drift.** The case enters `_original_close_read_drain` (test line 775), which first checks that the drain it wraps is defined where it imports it from: `assert function.__globals__ is module.__dict__` (line 583), with `module` = `Chat/console_preparation_reads`. Since 064d5cb014 (2026-11-09, TASK-39147), `drain_preparation_reads` is defined in `Backup_Recovery/runtime_producer_lifetime.py:315`, and `Chat/console_preparation_reads.py:18` only re-exports it, so that assert fails. Everything after it is a consequence: the case's own cleanup then reports `released original held reader never returned; {"entered": false, ...}` while handling that first failure (pytest's chained "During handling of the above exception" traceback shows both). Re-run on 2026-11-22: same chain, `test_console_tab_publication_read_barrier.py:583: AssertionError` first.

TASK-37040 (open) is about this test taking the session lock before the screen's pump is idle; that is a different, unrelated concern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The drain identity check pins the module that defines `drain_preparation_reads` (or pins the re-export from `console_preparation_reads` explicitly), so it holds on the current code and still fails if the drain is replaced
- [ ] #2 All three cases of the test pass, alone and in a `-n 4` run of the file
- [ ] #3 The file (or this test) runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
