---
id: TASK-15600
title: Reconcile post-82b5950 diagnostic inventory drift
status: To Do
assignee: []
created_date: '2026-08-11 17:55'
labels:
  - testing
  - baseline
  - security
dependencies:
  - TASK-15103
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15103 froze and reconciled the 19-owner incident at exact `origin/dev`
`82b595049d97836482c118cfeb4d31df537a86a1`. During its close-out rebase
attempt, dev had already advanced ~144 commits to
`97a7d4de4dccd37345ad4aaf901208d7db723eb0`, and the regenerated manifest
check surfaced NEW unreviewed drift in that range, deliberately NOT blessed
by TASK-15103's regeneration:

- 11 changed owner rows: `DB/ChaChaNotes_DB.py` 323→322,
  `DB/Client_Media_DB_v2.py` (digest), `DB/RAG_Indexing_DB.py` 5→6,
  `Event_Handlers/note_ingest_events.py` 28→32,
  `Library/library_rag_state.py` NEW owner 0→1,
  `RAG_Search/ingestion_indexing.py` (digest),
  `RAG_Search/simplified/rag_service.py` 57→66,
  `UI/Screens/chat_screen.py` 158→157,
  `UI/Screens/watchlists_collections_screen.py` 78→79,
  `Widgets/Console/console_transcript.py` 7→8, `app.py` (digest).
- Summary totals 488→489 owner files, TASK-494 calls 6,990→7,005.
- The persistent-sink topology CHANGED (still 6 files, different content) —
  this alone requires explicit review; TASK-15103's boundary pinned an
  unchanged topology.

Three of the touched rows are TASK-15103 owners, so its `reviewed_final`
evidence is exact only at its frozen base. TASK-15103 was landed at that
base rather than silently absorbing this drift. On a dev checkout past
`82b5950`, `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`
and `test_task_15103_reviewed_final_state_is_ledger_exact` hold this
residual drift red until it is reviewed — that is the boundary working as
designed, not a defect in those tests. Review each delta under ADR-029 and
regenerate the manifest with only reviewed changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every changed/new diagnostic in `82b595049..97a7d4de4` (and any further advance at review time) is reviewed under ADR-029, with unsafe captures repaired before being blessed
- [ ] #2 The persistent-sink topology change is explicitly reviewed and documented, not absorbed by regeneration
- [ ] #3 The production inventory manifest is regenerated with only reviewed changes and the full architecture gate passes at the new base
- [ ] #4 TASK-15103's reviewed-final evidence is refreshed (or superseded by a new ledger) for the three re-touched owners
<!-- AC:END -->
