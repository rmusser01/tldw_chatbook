---
id: TASK-15481
title: Retire dead schedulers and dead DB modules
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit — modules that look alive (and cost every future audit/reader time) but are unreachable in production: `Notes/auto_sync_manager.py` (1 s wake-up loop + watchdog, never instantiated; `app.py:9767-9769` stops a field never assigned), `Notes/sync_service.py:436` auto-sync loop (would run sync on the loop; `create_profile` has no production caller), `app.py:9755-9760` stops `self._subscription_scheduler`, also never assigned, `DB/Mindmap_DB.py` (no callers; calls a nonexistent `self.get_connection()` — would AttributeError at `:122/:129`), `DB/search_history_db.py`, `DB/Research_DB.py`, `DB/Writing_DB.py`, `DB/Sync_Client.py` (`ClientSyncEngine` never constructed), and `Widgets/prompt_selector.py` (no non-test importers; its on_mount would issue up to 501 sequential sync queries on the loop if ever wired).

Per the owner's long-term-stability preference: delete (with git-log provenance recorded) or explicitly quarantine each — leaving loaded-gun code that a future contributor wires up IS the instability. Verify each is still dead at implementation time (lessons-backlog-hygiene: verify a reported state still exists before acting on it). Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed module is removed (provenance in notes) or explicitly quarantined with a test asserting non-construction
- [ ] #2 app.py no longer stops fields that are never assigned
- [ ] #3 Full targeted suite green; no remaining runtime references (grep evidence)
<!-- AC:END -->
