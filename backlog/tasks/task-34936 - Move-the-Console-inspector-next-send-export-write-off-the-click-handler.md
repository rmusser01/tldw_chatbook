---
id: TASK-34936
title: Move-the-Console-inspector-next-send-export-write-off-the-click-handler
status: To Do
assignee: []
created_date: '2026-10-13 14:56'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The inspector next-send-save handler does path.mkdir + path.write_text of the export-scrubbed next-send JSON — potentially hundreds of KB — synchronously on click (console_conversation_inspector.py:1955-1956, plus a sha256 fingerprint on failure); artifacts_dreams_modal.py:617 has the same shape for its markdown export.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 File writes run via to_thread or a worker with handler only orchestrating and reporting status,Both the inspector export and the dreams modal export covered
<!-- AC:END -->
