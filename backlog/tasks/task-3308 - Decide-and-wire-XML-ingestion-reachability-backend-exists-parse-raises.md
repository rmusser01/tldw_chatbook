---
id: TASK-3308
title: >-
  Decide and wire XML ingestion reachability — backend exists, parse path raises "not yet implemented"
status: To Do
assignee: []
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - parity
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-07 parity audit: `Local_Ingestion/XML_Ingestion.py` exists, but `.xml` is not in `detect_file_type` and `parse_local_file_for_ingest` raises "XML file processing is not yet implemented". Either wire XML through (extension → group → parse) or record the deferral and keep the unsupported classification honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An owner decision (ship or defer) is recorded; if ship, `.xml` ingests end-to-end; if defer, the raise is unreachable from the queue (preflight classifies `.xml` unsupported with honest copy)
<!-- AC:END -->
