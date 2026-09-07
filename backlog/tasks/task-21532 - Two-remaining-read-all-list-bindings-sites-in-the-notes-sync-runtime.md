---
id: TASK-21532
title: >-
  Two remaining read-all list_bindings sites in the notes-sync runtime
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - notes-sync
  - database
priority: low
---

## Description

TASK-21129 replaced the executor's read-all `list_bindings` calls with narrow store
projections. Two sites in the runtime still hydrate every binding of a root to keep one field.
Both are already `asyncio.to_thread`-wrapped, so this is latency and allocation only -- it does
not block the event loop. Worth doing only when a root gets large or someone is next in the file.

## Acceptance Criteria

- [x] `notes_sync_runtime.py:2564` uses a narrow projection for CANDIDATE `binding_id`s instead of hydrating full binding records
- [x] `notes_sync_runtime.py:509`'s state check is served without materialising every binding, or the task records why the full read is required there
- [x] The projections reuse the store methods TASK-21129 added rather than adding parallel query shapes
- [x] Measured before/after allocation counts are recorded; if the win is not measurable at a realistic root size, the task is closed rather than shipped

## Evidence (verified first-hand on dev 022b67fc7, 2026-08-23)

`notes_sync_runtime.py:509`:
```python
bindings = await asyncio.to_thread(self._store.list_bindings, root.root_id)
if any(
    binding.state
```

`notes_sync_runtime.py:2564` keeps only `binding.binding_id` for CANDIDATE bindings from a full
hydration -- exactly the shape TASK-21129 measured, where **88% of the read cost was
`_binding_from_row` building a dataclass, nested profile and enum per row**, not the query.

Surfaced by the TASK-21129 implementer as explicitly out of its scope.

## Implementation Plan

1. Verify both sites on the base (dev `7f38cb6ef`) rather than trusting the filed line numbers.
2. Read what each site's `bindings` tuple is actually consumed for downstream.
3. Measure the candidate-set read against a narrow projection at 100/500/1,000 bindings --
   wall time with tracemalloc OFF, peak allocation with it ON, arms interleaved.
4. Ship the projection only if the delta measures; otherwise close.
5. Pin the projection with an equivalence test against the retired expression and a runtime
   census that fails if the read-all comes back or the projection moves onto the event loop.

## Implementation Notes

**Shipped for `:2564`, declined with a recorded reason for `:509`.**

`:2564` (`NotesSyncRuntimeOwner._activate_root`) now calls a new
`NotesDeviceStateStore.candidate_binding_ids(root_id)` -- one covering-index read of
`binding_id` -- instead of hydrating every binding of the root to keep one field. Measured
on the real store, arms interleaved, at three root sizes:

| bindings | read-all | projection | delta |
|---:|---|---|---|
| 100 | 1.413 ms / 110,490 B peak | 0.034 ms / 13,228 B | -1.379 ms / -97,262 B |
| 500 | 7.595 ms / 531,538 B peak | 0.168 ms / 60,676 B | -7.427 ms / -470,862 B |
| 1,000 | 15.257 ms / 1,058,854 B peak | 0.295 ms / 120,992 B | -14.961 ms / -937,862 B |

(The 1,000-binding read reproduces TASK-21129's 15 ms exactly. Timings are medians of 11
interleaved samples with tracemalloc off; peaks are medians of 5 `tracemalloc.reset_peak()`
runs -- measuring both at once inflates the allocation-heavy arm to ~92 ms and would have
overstated the win six-fold.)

`:509` (`_ProductionRuntimeAdapter.observe_root`) keeps its read-all, and the code now says
why: the tuple looks like a one-field read at the `binding.state` check, but the *same*
tuple is the input to every loop after it, and between them they consume all thirteen
hydrated columns (`binding_id`, `note_id`, `normalized_relative_path`,
`stable_identity_digest`, `content_digest`, `note_scope_id`, `note_version`,
`serialization`, `state`). A projection there would have to be followed by the full read
anyway.

The AC asked the projections to reuse TASK-21129's store methods. They cannot:
`active_binding_note_ids` projects `note_id` for the `active` state and
`has_binding_for_note_or_path` is a `LIMIT 1` predicate, so neither can answer "which
bindings of this root are candidates". `candidate_binding_ids` is therefore a new method,
written in the same idiom and served by the same
`idx_notes_sync_bindings_root(root_id, state, binding_id)` covering index (verified with
`EXPLAIN QUERY PLAN`: `SEARCH ... USING COVERING INDEX`, no temporary B-tree sort). SQL
`ORDER BY binding_id` under BINARY collation is the same order as the Python `sorted()` it
replaces.

Modified: `tldw_chatbook/Notes/notes_device_state_store.py`,
`tldw_chatbook/Notes/notes_sync_runtime.py`,
`Tests/Notes/test_notes_device_state_store.py`, `Tests/Notes/test_notes_sync_runtime.py`.

### Mutation results (every new test proven to discriminate)

| mutant | `..._matches_the_full_scan_it_replaces` | `..._orders_by_binding_id_and_never_leaks_a_root` | `..._projects_candidate_ids_without_a_read_all` |
|---|---|---|---|
| state filter dropped | FAIL | pass (all seeded bindings are candidates) | n/a |
| root filter dropped | FAIL | FAIL | n/a |
| `CANDIDATE` -> `ACTIVE` | FAIL | FAIL | n/a |
| call site reverted to the read-all | n/a | n/a | FAIL |
| projection taken off `asyncio.to_thread` | n/a | n/a | FAIL |

Not caught by anything, and deliberately so: dropping `ORDER BY binding_id` leaves the
result unchanged, because the covering index already yields rows in `binding_id` order.
The clause is kept as the contract rather than as an optimization.

### Quit / error walk

`_activate_root`'s `except Exception` block is unchanged and still covers the projection:
a store failure there lands in the same rollback path as before (folder rollback, then
`record_root_activation_recovery`), because `asyncio.to_thread` re-raises in the awaiting
coroutine exactly as the previous `to_thread(list_bindings)` did. `owner.shutdown()` after
the new test is clean. The projection opens and closes its own connection through the same
`transaction()` context manager as `list_bindings`, so a cancellation during shutdown
cannot leak one.

### Test counts

`Tests/Notes/test_notes_sync_runtime.py`, `test_notes_device_state_store.py`,
`test_notes_sync_executor.py`, `test_notes_sync_cutover.py`: **284 passed, 1 failed**.
The one red -- `test_library_screen_has_no_legacy_timer_worker_or_mutating_handler`
(`_library_notes_..._sync_timer:3729` in `library_screen.py`, a file this task does not
touch) -- fails identically on pristine dev `7f38cb6ef` (**281 passed, 1 failed**, the
same nodeid), A/B'd in a separate worktree at that SHA.
