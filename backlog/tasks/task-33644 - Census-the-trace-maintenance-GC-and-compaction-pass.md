---
id: TASK-33644
title: Census the trace-maintenance GC and compaction pass
status: To Do
assignee: []
created_date: '2026-09-30 18:12'
labels:
  - perf
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2888's storage-unit census measures legacy trace maintenance per 1 Hz tick through run_batch only. The production loop (console_runtime) also runs a physical pass after a logically complete batch -- graph-epoch read, garbage collection and compaction through owned database calls -- at most once per GC interval, and PERF-10 (TASK-33269) wakes parked maintenance for it every interval. Qodo noted on #2888 that storage admissions or helper spawns added to that pass cannot fail any ratchet today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census drives one deterministic eligible GC interval through the production maintenance loop (after PERF-10's parking) and records its config admissions, storage admissions, helper spawns and os.open calls
- [ ] #2 Those counts are pinned as ceilings per GC pass, and an added admission in the pass fails the ratchet (negative control recorded)
<!-- AC:END -->
