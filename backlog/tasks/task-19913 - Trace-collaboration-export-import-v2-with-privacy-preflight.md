---
id: TASK-19913
title: Trace collaboration export import v2 with privacy preflight
status: To Do
assignee: []
created_date: '2026-08-22 18:31'
labels: []
dependencies:
  - TASK-19907
  - TASK-19910
  - TASK-19911
references:
  - Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19913-trace-v2-collaboration.md
  - backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make sharing a first-class, privacy-safe Trace workflow with a versioned causal event bundle, export preflight, and read-only collaborative import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export supports safe summary, redacted diagnostic, and explicit full-trace modes with a preflight inventory of sensitive, redacted, omitted, and truncated fields
- [ ] #2 Trace format v2 preserves event identity, order, lineage, timing, schema version, redaction provenance, and missing-data reasons
- [ ] #3 Imported bundles are visibly labeled read-only shared traces and never write conversation or trace data to local persistence
- [ ] #4 Readers reject unsupported versions and integrity failures with actionable errors while retaining v1 import compatibility
- [ ] #5 Round-trip, privacy, tamper, malformed input, and zero-database-write tests pass
<!-- AC:END -->
