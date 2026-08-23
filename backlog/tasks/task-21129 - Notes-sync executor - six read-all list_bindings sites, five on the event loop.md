---
id: TASK-21129
title: >-
  Notes-sync executor - six read-all list_bindings sites, five on the event loop
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notes-sync
  - database
priority: medium
dependencies:
  - TASK-21101
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21129).

`Notes/notes_sync_executor.py:1144,1256,1761,1978,2260,2682` each read EVERY binding for the
root (notes_device_state_store.py:889-897 - no LIMIT; no index on root_id found), and five of
the six are invoked without to_thread from async methods (unlike notes_sync_runtime.py:1363
which wraps correctly) - ~3K full owner-set reads per sync batch, each also paying TASK-21101's
per-op connect + census until that lands.

## Acceptance Criteria

- [ ] The read-all sites use indexed predicates (starting with `_require_new_candidate_owner`); an index on the binding root/owner columns exists
- [ ] The five loop-side call sites route through to_thread
- [ ] Sync outcomes unchanged - existing executor tests green
