---
id: TASK-13216
title: >-
  LocalToolProvider todo_write is last-write-wins under concurrent fleet
  children
status: In Progress
assignee: []
created_date: '2026-08-10 20:48'
updated_date: '2026-08-11 13:41'
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
- [ ] #1 `todo_write` is replaced by `todo_create`, `todo_update`, `todo_get`, and `todo_list` operations over stable session-local task IDs shared by the parent and fleet children
- [ ] #2 Concurrent creates and jointly valid updates to different task IDs preserve every successful caller's change without exceeding the live-task cap
- [ ] #3 `todo_update` and deletion require an expected task version; a stale caller receives an explicit conflict and cannot overwrite the winning update
- [ ] #4 Concurrent task operations preserve the one-`in_progress` invariant, keep public IDs and versions within the portable JSON exact-integer domain `1..2**53-1`, return complete defensive results within the provider cap, and produce ordered transcript snapshots
- [ ] #5 Deterministic concurrency tests cover create, different-task update, same-task conflict, capacity, callback reentrancy, and parent/fleet shared-state behavior
- [ ] #6 Valid task records and the next-ID high-water mark survive ordinary in-process Console navigation without becoming durable across application restarts
<!-- AC:END -->

## Design References

- Design: `Docs/superpowers/specs/2026-08-11-local-todo-task-api-design.md`
- ADR required: yes
- ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
- Reason: ADR-032 already owns the local-tool provider and todo permission boundary; this task adds an item-oriented concurrency addendum rather than a competing ADR.

## Implementation Plan

1. Amend ADR-032 and the existing local-agent-tool designs before production changes so the stable-ID/CAS contract is the governing boundary.
2. Add a stdlib-only `SessionTodoStore` with strict validation, stable IDs and versions bounded to `1..2**53-1`, a private `2**53` exhausted-ID sentinel, defensive navigation snapshots, fixed atomic numeric exhaustion, compare-and-swap mutation, and the two-lock callback-ordering protocol.
3. Replace conditional `todo_write` registration with strict `todo_create`, `todo_update`, `todo_get`, and byte-aware `todo_list` provider handlers and schemas that enforce the same portable ID/version/cursor ceiling and return complete bounded JSON.
4. Make `ConsoleChatSession` own the store, wire it into provider reconstruction, and preserve its pure-data records/high-water counter through in-process screen navigation only.
5. Harden transcript rendering, migrate real find/load/permission and parent/fleet tests, and pin external MCP/Hub absence when no Console store exists.
6. Run focused/reachability/full tests, static/security gates, mutation probes, independent review, and only then complete the task record.

Detailed plan: `Docs/superpowers/plans/2026-08-11-local-todo-task-api.md`

ADR required: yes

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: ADR-032 already owns the local-tool provider and todo permission boundary; this task adds an item-oriented concurrency addendum rather than a competing ADR.
