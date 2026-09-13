---
id: TASK-31949
title: Library - the source-snapshot timeout branch leaves no log record
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:34'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR G ruled no new logger in that module, so the Library source-snapshot deadline branch returns silently: a report of 'the snapshot never arrived' cannot be separated from a fetch error in the logs, and the branch is diagnosable only from the UI. Reusing the existing warning call with a timeout marker respects the no-new-logger ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source-snapshot timeout leaves a log record naming the deadline and the source
- [x] #2 No new logger object is introduced
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm on the merge-base whether the deadline branch already logs through the one shared helper with a deadline marker. 2. If so, close by evidence; otherwise route it through `_log_source_snapshot_failure` with the marker.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by evidence, no code. PR G's final review already routed the `except TimeoutError` branch of the source-snapshot load through the one shared `_log_source_snapshot_failure(f" (deadline: waited {…:g} s)")` helper, whose message names the source and the deadline, and pinned it (`test_source_snapshot_timeout_logs_one_warning_with_a_deadline_marker` asserts exactly one warning carrying `waited 0.05 s`). Verified at the merge-base 3dbe1448d: the log line and the pin both present; diagnostic inventory unchanged (no new call site).
<!-- SECTION:NOTES:END -->
