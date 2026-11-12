---
id: TASK-40316
title: "test_console_variant_stream.py runs on no PR lane, though it pins that a failed regenerate keeps the prior good answer"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- lanes
priority: low
references:
- Tests/Chat/test_console_variant_stream.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** `test_regenerate_empty_stream_retains_failed_sibling_and_restores_anchor` (`Tests/Chat/test_console_variant_stream.py:366`, #2164, 473e7c9298) guards that a failed regenerate keeps the prior good answer in provider context, a TASK-571 regression class. Nothing under `.github/` or `scripts/` runs the file; `test.yml` runs on pushes to main only. The file has 14 tests and no profile marker; TASK-33621.22 (2026-10) recorded 4 reds in it, and its state today is unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file passes under its plain command at `-n 4`, each red first classified with its cause
- [ ] #2 The file runs on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->
