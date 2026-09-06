---
id: TASK-31803
title: >-
  Daily Brief report rows and preview show raw microsecond-precision UTC ISO
  timestamps
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 15:12'
labels:
  - bug
  - ux
  - artifacts
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Report list rows and the preview header render timestamps like '2026-09-05T23:10:2...' raw. Format for humans (local time, minute precision) in list and preview surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Brief timestamps render in a human-readable local format in the list and preview.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: _label() and the preview header render row['created_at'] raw (aware microsecond ISO from the demo/accept write path, e.g. 2026-09-05T23:10:20.123456+00:00).\n2. Add format_report_timestamp() in daily_reports_view (local zone, minute precision; handles naive DB-default CURRENT_TIMESTAMP and aware ISO; None/blank -> placeholder).\n3. Use it in _label and in ArtifactsScreen._report_preview_renderable header.\n4. RED unit tests for the formatter + label; the UI preview shares the same path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live: _label produced 'Daily Brief — 2026-09-05T23:10:20.123456+00:00'. The demo/accept path writes created_at as datetime.now(timezone.utc).isoformat() (aware, microsecond, T-separated); the DB default CURRENT_TIMESTAMP writes a naive 'YYYY-MM-DD HH:MM:SS'.

Fix (TASK-31803): added format_report_timestamp() to Subscriptions/daily_reports_view.py -- parses both shapes via datetime.fromisoformat (naive treated as UTC), renders local-zone minute precision ('YYYY-MM-DD HH:MM ZZZ'), matching the Watchlists screen's _local_schedule_time; None/blank -> 'unknown time', unparseable -> raw text. Wired into _label (list rows) and ArtifactsScreen._report_preview_renderable (preview header).

Tests (Tests/Subscriptions/test_daily_reports_view.py): format_report_timestamp aware/naive/blank cases + label-uses-formatted-stamp (minute-precision regex, RED->GREEN).

Files: tldw_chatbook/Subscriptions/daily_reports_view.py, tldw_chatbook/UI/Screens/artifacts_screen.py, Tests/Subscriptions/test_daily_reports_view.py, Docs/User_Guide/artifacts.md.

PR #2460 Qodo review follow-up (#3): added test_report_preview_header_uses_formatted_timestamp asserting the preview renderable formats the timestamp (guards against a silent regression in the preview header path, separate from the list-label test).
<!-- SECTION:NOTES:END -->
