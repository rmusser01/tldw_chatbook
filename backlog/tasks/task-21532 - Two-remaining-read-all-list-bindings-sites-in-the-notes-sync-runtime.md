---
id: TASK-21532
title: >-
  Two remaining read-all list_bindings sites in the notes-sync runtime
status: To Do
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

- [ ] `notes_sync_runtime.py:2564` uses a narrow projection for CANDIDATE `binding_id`s instead of hydrating full binding records
- [ ] `notes_sync_runtime.py:509`'s state check is served without materialising every binding, or the task records why the full read is required there
- [ ] The projections reuse the store methods TASK-21129 added rather than adding parallel query shapes
- [ ] Measured before/after allocation counts are recorded; if the win is not measurable at a realistic root size, the task is closed rather than shipped

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
