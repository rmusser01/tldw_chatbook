---
id: TASK-42008
title: "The query-synthesis Hypothesis test has a real falsifying example"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:36
labels:
- follow-up
- tests
- dreams
priority: low
references:
- Tests/Dreams/test_query_synthesis.py
- tldw_chatbook/Dreams
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41709 (round 17), red at base. `test_no_wrapping_lets_a_mark_through_or_loses_its_goal` fails for `'Kyoto: **"[g1]"Wednesday 13" tour dates 2026"**'`: the query becomes `'Kyoto Wednesday 13" tour dates 2026"'` where `'Kyoto "Wednesday 13" tour dates 2026'` is expected. The file is on no PR or dev-push lane; the release `test.yml` sweep runs it. The input comes from the generator's `front-glued` path with the `"{}"` wrapper, which keeps `exact=True`, yet the result is textually the `front-glued-wrapped` `"[g1]"` shape the same generator excludes as unreadable (~873-890), so the defect may be the generator's exactness rule rather than `_query_row`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named in the notes (generator exactness or product)
- [x] #2 Either `_query_row` returns `'Kyoto "Wednesday 13" tour dates 2026'` for this input, or the generator marks this shape inexact
- [x] #3 The input is pinned with `@example`, red before the fix
- [x] #4 The file runs on a PR lane, with its time recorded. Added 2026-11-20 (review r18ln-rev1, coordinator decision).
<!-- AC:END -->

## Implementation Plan

1. Replay the falsifying input through `_query_row` and the generator to see which reading each side takes.
2. Name the cause: compare the input with the generator's front-glued and front-glued-wrapped rules and with `_opens_the_next_phrase`'s documented reading of the same quoted shape.
3. Pin the input with `@example` and show it red at base.
4. Fix the side that is wrong, then run the file's property tests with extra examples.

## Implementation Notes

Cause: the product (`_opens_the_next_phrase`), not the generator. The input is a quoted line (`"[g1]"Wednesday 13" tour dates 2026"`, generator path front-glued + `"{}"`) inside bold behind a subject label. The cleaner already reads that quoted shape the way the generator expects when the quote ends the line (round 6 G-N7 / round 8 J1: `"[g1]"Wednesday 13" tour dates 2026"` and `Kyoto: "[g1]"..."` both give the phrase), but its "the line is itself quoted" check was `text.endswith(quote)` on the raw line, so a closer of wrapping opened before the quote (`**`, `__`, `_`, a backtick, a curly quote) hid the line's end and the quote after the mark was taken as the mark's closer. The generator's front-glued-wrapped exclusion is a different shape (an unquoted line whose glued `"[g1]"` meets a query quote), so its exactness rule stands.

Fix: `_opens_the_next_phrase` takes the text in front of the quote and drops the closers that text's wrapping owes (`_unclosed_wrappers`, the loop `_without_marks` already uses for the body) before checking the line's end.

Red/green: the `@example` failed at base (`AssertionError: ('Kyoto: **"[g1]"Wednesday 13" tour dates 2026"**', 'Kyoto Wednesday 13" tour dates 2026"')`); after the fix the file is 311 passed at -n 4. A scratch sweep of 60,000 generator draws comparing the old and new helper found 22 lines that changed, every one now equal to the generator's expected query (11 of them exact cases the old code got wrong, e.g. `Kyoto: _"[g1]"Wednesday 13" tour dates 2026"_`), and none exact that turned wrong.

Files: `tldw_chatbook/Dreams/query_synthesis.py`, `Tests/Dreams/test_query_synthesis.py`.

Rev1 (2026-11-20, AC#4): the file needs hypothesis, which only the `console-p0-gate` install line has among the PR jobs (TASK-42028), so it joins that job's sandboxed step (the file is sandboxed and reads no config). That profile took 4-6 min of its 35 over 17 recent PR runs; the admission-sensitive profile (26-33 min) was left alone. Time with the step's flags (`-p no:xdist --timeout=180 --tb=short`): 311 passed in 3.8 s at load 18. CI pins: the sandboxed step's inherited target list in `Tests/CI/test_response_rules_critical_gate.py`; the job's install-line note names the new reason for hypothesis. The whole step locally, serially: 541 passed in 304 s (load 17-21). Tests/CI at -n 4: 1811 passed, 2 skipped. Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_response_rules_critical_gate.py`.
