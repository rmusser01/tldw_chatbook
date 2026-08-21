---
id: TASK-19502
title: Decouple Change Review finalization from Console turn completion
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - performance
  - concurrency
dependencies:
  - TASK-19501
priority: critical
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure a completed assistant response releases the Console turn and prompt queue without waiting for filesystem snapshot finalization, while preserving per-root ordering, survivor-window attribution, durable review publication, and safe shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The bridge schedules end-of-turn review work and returns the terminal run outcome without awaiting filesystem or Git finalization
- [ ] #2 One bounded app-owned coordinator orders baseline-to-end windows FIFO per canonical root across conversations without holding UI, store, or database objects in filesystem workers
- [ ] #3 Multi-root admission is all-or-nothing and shared-root survivor windows retain correct successor ordering
- [ ] #4 Review markers derive idempotently from durable assistant anchors and durable change rows for both result-first and anchor-first races and after remount
- [ ] #5 A deterministic mounted three-turn test holds review finalization and proves the third turn starts after the second assistant is durable
- [ ] #6 Shutdown stops admission, bounded-drains pure workers, rejects late-generation results, disposes the coordinator, and only then closes AgentRunsDB
- [ ] #7 Failed and cancelled turns still request review finalization and bounded failures persist honest tracking-error results
<!-- AC:END -->
