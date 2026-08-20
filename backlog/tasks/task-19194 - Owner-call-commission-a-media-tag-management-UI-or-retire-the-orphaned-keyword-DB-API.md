---
id: TASK-19194
title: 'Owner call: commission a media tag-management UI or retire the orphaned keyword DB API'
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - owner-decision
  - dead-code
  - media-db
dependencies:
  - TASK-19046
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-TASK-19046 (which retired the never-mounted CollectionsTagWindow and its
dead keyword event loop), the media DB's keyword-management surface has no
reachable UI. Verified at dev `7877defba`:

- `DB/Client_Media_DB_v2.py::rename_keyword` (:4988) — zero production
  callers, zero test references
- `DB/Client_Media_DB_v2.py::merge_keywords` (:5097) — zero production
  callers, zero test references
- `DB/Client_Media_DB_v2.py::soft_delete_keyword` (:4865) — lost its only
  caller with CollectionsTagWindow (same-named methods in Notes/Prompts DBs
  are separate APIs and unaffected); DB-level tests exist in
  `Tests/Media_DB/test_media_db_v2.py`
- `DB/Client_Media_DB_v2.py::get_keyword_usage_stats` (:5230) — HAS a live
  consumer (`Chat/scope_picker_listers.py:503`) and stays either way

CollectionsTagWindow was the only conceivable media-keyword management UI and
had been unreachable since 2025-08-02, so nothing user-visible was lost — the
question is forward-looking: should users be able to rename/merge/delete media
keywords at all? Owner decision between (a) commissioning a reachable
tag-management surface that consumes this API, or (b) retiring the three
orphaned methods and their DB-level tests. Per the standing ruling, prefer the
durable outcome: a deliberate product decision either way, not an indefinite
zombie API. If retiring: `Client_Media_DB_v2.py` is diagnostic-bearing (354
inventory calls), so the deletion must include the diagnostic-inventory
hand-edit, and the security-coverage-map check applies before removing the
DB-level tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The owner has decided between commissioning a reachable media tag-management UI and retiring the orphaned DB API surface, and the decision is recorded in this task.
- [ ] #2 If commissioning: a follow-on task (or tasks) exists specifying the surface, and this task records what it must consume (`rename_keyword`, `merge_keywords`, `soft_delete_keyword`).
- [ ] #3 If retiring: the three orphaned methods and their now-purposeless DB-level tests are removed, `get_keyword_usage_stats` and its live consumer are untouched, the security-coverage-map check is recorded, and the diagnostic inventory row for `Client_Media_DB_v2.py` is hand-edited in the same PR.
- [ ] #4 The caller census above is re-verified against dev at implementation time before acting on it.
<!-- AC:END -->
