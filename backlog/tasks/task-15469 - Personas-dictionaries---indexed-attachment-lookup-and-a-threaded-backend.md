---
id: TASK-15469
title: Personas dictionaries: indexed attachment lookup and a threaded backend
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - personas
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: clicking one dictionary row (`UI/Screens/personas_screen.py:3498-3546` via `_handle_entity_selected:3315`) runs 4+ synchronous queries on the event loop, including `list_dictionary_conversations` (`Character_Chat/local_chat_dictionary_service.py:868-901`): `SELECT id, title, metadata FROM conversations WHERE deleted = 0 AND metadata LIKE '%active_dictionaries%'` — a leading-wildcard LIKE that scans the entire conversations table and JSON-parses matches. The dictionary record is loaded twice per click (`get_dictionary:371` and `get_statistics:760` are each a full load), and `list_dictionaries(include_usage=True)` is N+1 via the same scan per dictionary. With thousands of conversations this is 50-500 ms per click on slow hardware.

Fix direction: thread the local backend in the scope service; replace the LIKE scan with an indexed lookup (JSON1 query with an expression index, or a proper attachment table — a ChaChaNotes schema change means bumping `_CURRENT_SCHEMA_VERSION` with a migration); load the record once per selection and derive statistics from it. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A dictionary row click issues no full-table scan (query-plan or timing evidence) and at most 2 queries, none on the event loop
- [ ] #2 Statistics and attachment lists return identical values (tests)
- [ ] #3 Click latency before/after with a 1,000+-conversation DB recorded
<!-- AC:END -->
