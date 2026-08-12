---
id: TASK-13216
title: >-
  LocalToolProvider todo_write is last-write-wins under concurrent fleet
  children
status: Done
assignee: []
created_date: '2026-08-10 20:48'
updated_date: '2026-08-12 18:49'
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
- [x] #1 `todo_write` is replaced by `todo_create`, `todo_update`, `todo_get`, and `todo_list` operations over stable session-local task IDs shared by the parent and fleet children
- [x] #2 Concurrent creates and jointly valid updates to different task IDs preserve every successful caller's change without exceeding the live-task cap
- [x] #3 `todo_update` and deletion require an expected task version; a stale caller receives an explicit conflict and cannot overwrite the winning update
- [x] #4 Concurrent task operations preserve the one-`in_progress` invariant, keep public IDs and versions within the portable JSON exact-integer domain `1..2**53-1`, return complete defensive results within the provider cap, and produce ordered transcript snapshots
- [x] #5 Deterministic concurrency tests cover create, different-task update, same-task conflict, capacity, callback reentrancy, and parent/fleet shared-state behavior
- [x] #6 Valid task records and the next-ID high-water mark survive ordinary in-process Console navigation without becoming durable across application restarts
- [x] #7 The synthetic workspace provider exclusively owns `__local__` / `local:__local__`: external MCP profile save, import, load, and Hub projection reject or ignore that exact reserved identity while all other currently valid profile IDs retain their behavior, and current MCP user documentation identifies `todo_create`, `todo_update`, `todo_get`, and `todo_list` as Console-session-only and absent from the Hub
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-032 and the existing local-agent-tool designs before production changes so the stable-ID/CAS contract is the governing boundary.
2. Add a stdlib-only `SessionTodoStore` with strict validation, stable IDs and versions bounded to `1..2**53-1`, a private `2**53` exhausted-ID sentinel, defensive navigation snapshots, fixed atomic numeric exhaustion, compare-and-swap mutation, and the two-lock callback-ordering protocol.
3. Replace conditional `todo_write` registration with strict `todo_create`, `todo_update`, `todo_get`, and byte-aware `todo_list` provider handlers and schemas that enforce the same portable ID/version/cursor ceiling and return complete bounded JSON.
4. Make `ConsoleChatSession` own the store, wire it into provider reconstruction, and preserve its pure-data records/high-water counter through in-process screen navigation only.
5. Harden transcript rendering, migrate real find/load/permission and parent/fleet tests, pin external MCP/Hub absence when no Console store exists, and reserve ADR-032's synthetic `local:__local__` identity against user-controlled external MCP profile IDs at save, load, and catalog-projection boundaries.
6. Correct the current MCP user guide so the Hub inventory does not imply session task tools are present outside the Console.
7. Run focused/reachability/full tests, static/security gates, mutation probes, independent review, and only then complete the task record.

Detailed plan: `Docs/superpowers/plans/2026-08-11-local-todo-task-api.md`

ADR required: yes

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: ADR-032 already owns the local-tool provider and todo permission boundary; this task adds an item-oriented concurrency addendum rather than a competing ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a stable Console-session task API backed by SessionTodoStore: strict bounded IDs and versions, compare-and-swap updates/deletes, one-in-progress enforcement, complete UTF-8 byte-aware pages, defensive snapshots, and four replacement tools with native-provider schema projections. Console sessions and fleet children share the same store; in-process navigation restores records and the next-ID high-water mark without durable persistence; transcript markers are ordered, bounded, sanitized, and omit protocol IDs/versions. The two-lock design serializes mutations and callback order while releasing the state lock before callbacks; callback mutations fail fast. External MCP and Hub surfaces remain task-free without a Console store, and normalized reserved __local__ identity is rejected or filtered at save, import, load, runtime, and raw projection boundaries. ADR-032 and current documentation were amended. Verification: post-material-rebase focused matrix 1,283 passed with two dependency warnings; final zero-overlap rebase smoke 436 passed with one warning; bridge 190 passed; installed-artifact contract passed. All 16 required mutation categories produced RED and were restored. Ruff lint, overlap formatting, Bandit on changed overlap, compileall, and diff checks passed. Broad formatter, mypy, and Bandit retain only independently compared upstream findings. The mandatory pre-final-rebase full suite completed non-green at 235 failed, 39,418 passed, 191 skipped, 4 xfailed, 128 errors; clean-base lastfailed replay reproduced every persistent failure, 15 apparent branch-only nodes passed standalone, and two cache keys were retired tests. Independent final code review returned Ready with no findings. A final zero-overlap dev replay required targeted smoke only.
<!-- SECTION:NOTES:END -->

## Design References

- Design: `Docs/superpowers/specs/2026-08-11-local-todo-task-api-design.md`
- ADR required: yes
- ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
- Reason: ADR-032 already owns the local-tool provider and todo permission boundary; this task adds an item-oriented concurrency addendum rather than a competing ADR.
