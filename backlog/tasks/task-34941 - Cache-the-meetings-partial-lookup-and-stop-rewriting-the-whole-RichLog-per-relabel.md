---
id: TASK-34941
title: Cache-the-meetings-partial-lookup-and-stop-rewriting-the-whole-RichLog-per-relabel
status: Done
assignee: []
created_date: 2026-10-13 14:59
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Meetings _apply_event does query_one(#meetings-partial) on every ASR partial — the highest-frequency event stream in the app (meetings_screen.py:782) — and every speaker relabel clears and rewrites the ENTIRE transcript RichLog from rendered_lines (:838-848), O(all lines) per event, acknowledged in-code as cheap at meeting sizes but growing over a long meeting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Partial widget reference cached (no query_one per ASR partial)
- [x] Relabel updates only affected lines or whole-log rewrite is bounded with a measured justification
<!-- AC:END -->
