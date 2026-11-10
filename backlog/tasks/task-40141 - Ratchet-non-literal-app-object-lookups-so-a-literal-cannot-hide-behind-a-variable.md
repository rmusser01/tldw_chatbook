---
id: TASK-40141
title: Ratchet non-literal app-object lookups so a literal cannot hide behind a variable
status: To Do
created_date: 2026-11-10 20:10
labels:
- architecture
- tests
- follow-up
dependencies: []
priority: low
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3490#issuecomment-6777451323
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
- Tests/Architecture/app_private_attrs.txt
- tldw_chatbook/UI/Screens/settings_screen.py
updated_date: 2026-11-10 20:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36942.4 ratchets string-named attribute lookups on the app object and fails any new underscore-prefixed name that is not in a frozen allowance. It counts only lookups whose name is a string literal; lookups whose name comes from a variable, a constant or an expression are left out of both the count and the private-name rule, a gap its docstring records (45 such lookups at the time).

That gap lets a change look like progress while moving a lookup out of view. Dev commit eced473b58 (TASK-39133) moved one literal lookup of `_vllm_connection_owner` in `settings_screen.py` behind a parameter, shared with the llama.cpp owner. The counted UI lookups fell from 1262 to 1261 with no check pointing at the cause; both owner names stay visible to the private-name rule only because other modules still look them up literally. The pin was not lowered, so the PR Fast Lane went red for every PR on dev (PR #3490 lowers the pin).

The blind spot is already real: seven private app attributes are reached only through module constants, are not in the allowance, and so are invisible to the private-name rule today (`_console_fleet_unseen_revision`, `_console_pending_interrupts`, `_console_reaction_preview_coordinator`, `_library_last_export_directory`, `_rag_service_generation`, `_rag_service_stamp`, `_tldw_library_ingest_options_cache`). Measured on dev ae9f9e8d21 with the census's own receiver rule: 47 non-literal lookups (UI 24, Event_Handlers 9, Chat 6, Backup_Recovery 3, `<root>` 2, RAG_Search 2, Scheduling 1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new non-literal lookup on an app-like receiver fails an architecture test that names the file and says how to reach the service explicitly instead.
- [ ] #2 Rewriting a counted literal lookup as a non-literal one fails, rather than reading as a reduction of the literal count.
- [ ] #3 Every non-literal lookup that exists when the guard lands is recorded where a reviewer can see it, and that record can only shrink; a lookup that goes away fails until it is removed from the record.
- [ ] #4 Each private app attribute reached only non-literally is either recorded together with the lookup that reaches it or reached explicitly; the frozen private-name allowance does not grow.
- [ ] #5 The guard runs in the same required lane as the existing census.
- [ ] #6 A negative control shows the guard fails on a planted non-literal lookup.
<!-- AC:END -->
