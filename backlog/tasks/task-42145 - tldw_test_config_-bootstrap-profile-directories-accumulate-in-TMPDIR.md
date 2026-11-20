---
id: TASK-42145
title: "tldw_test_config_* bootstrap profile directories accumulate in TMPDIR"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- disk
priority: low
references:
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the skillsrec stream (round 18), which counted about 430 of them. `Tests/conftest.py` creates the bootstrap profile root with `tempfile.mkdtemp(prefix="tldw_test_config_")` (line 56) when it inherits none, and removes it in `pytest_sessionfinish` (lines 1324-1325), only in the owning process and never in an xdist worker.

On 2026-11-20 at 19:58 UTC `$TMPDIR` held 437 of these directories, about 200 MB in all, the oldest from 2026-11-08. Measured at d85e90bc9e through the test wrapper: a passing serial run, a failing run, a `-n 2` run, `--collect-only` and a private-profile test each left none; `pytest --version` left one, because it loads `Tests/conftest.py` (`testpaths = ["Tests"]`) and never reaches `pytest_sessionfinish`. Runs killed or interrupted mid-session are the other likely source; that was not measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The sources of left-behind directories are named, including measurements for `pytest --version` or `--help` and for a run killed mid-session
- [ ] #2 Each source that can be closed leaves no directory, shown by a before-and-after count
- [ ] #3 Any source left open is named in the notes with the reason
<!-- AC:END -->
