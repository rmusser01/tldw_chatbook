---
id: TASK-13216
title: >-
  LocalToolProvider todo_write is last-write-wins under concurrent fleet
  children
status: To Do
assignee: []
created_date: '2026-08-10 20:48'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR2a Task 8's provider thread-safety audit found LocalToolProvider safe for concurrent invoke() (read-only specs dict, stamps already lock-guarded, handlers are pure per-call functions) with one narrow exception: the todo_write handler does an unguarded 'store[:] = items' on the session's shared todos list, which is the SAME LocalToolProvider instance's todo_store across a parent run and every fleet child it spawns (PR2a Task 6). Two concurrent todo_write calls race last-write-wins -- memory-safe (no corruption, no crash) but one caller's update can be silently discarded. Not locked as part of Task 8 because it is a single-tool, single-field hazard, not a proof that the whole provider is unsafe, and locking the entire provider would throttle every unrelated local tool (fs_read, fs_grep, web_fetch, ...) fleet-wide to protect one list splice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either todo_write's read-modify-write is made safe under concurrent calls (e.g. its own small lock, or a compare-and-swap/merge policy), or the last-write-wins behavior is deliberately documented as accepted and out of scope
- [ ] #2 If fixed, a concurrency test proves two overlapping todo_write calls from different runs cannot silently lose one caller's update
<!-- AC:END -->
