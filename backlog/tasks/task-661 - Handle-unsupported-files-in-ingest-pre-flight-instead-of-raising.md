---
id: TASK-661
title: Handle unsupported files in ingest pre-flight instead of raising
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-flight analysis crashes for any unsupported file, and for any folder containing even one, replacing the whole summary with a raw Python error string. The user loses the file count, size and per-type breakdown, and the missing-tooling guardrail can no longer fire. Common everyday files such as images, subtitles and JSON are enough to trigger it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Analyzing a single unsupported file returns a result instead of raising
- [ ] #2 Analyzing a folder of supported files plus one unsupported file returns the full summary for the supported ones
- [ ] #3 Unsupported files are reported to the user in their own summary line, not as an error string
- [ ] #4 Tooling warnings for the supported groups in a mixed folder still reach the guardrail
- [ ] #5 No raw exception text is shown in the pre-flight area for this case
<!-- AC:END -->
