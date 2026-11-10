---
id: TASK-40141
title: Ratchet non-literal app-object lookups so a literal cannot hide behind a variable
status: To Do
created_date: 2026-11-10 20:10
labels:
- architecture
- tests
priority: low
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3490#issuecomment-6777451323
updated_date: 2026-11-10 20:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36942.4 ratchets string lookups on the app object (`getattr(app, "<name>")` / `hasattr(app, "<name>")`) and fails any new underscore-prefixed name. It counts only literal names; lookups whose name is a variable are left out of both the count and the private-name rule, a gap its docstring records (45 at the time).

That gap lets a change look like progress while hiding a lookup. Dev commit eced473b58 (TASK-39133) replaced the literal-name lookup of `_vllm_connection_owner` in `settings_screen.py` with one through a parameter, now used for both the vLLM and llama.cpp owners. The counted UI lookups fell from 1262 to 1261 and two private app attributes left the private-name rule's view, with no check pointing at the cause; the pin was not lowered, so the PR Fast Lane went red for every PR on dev (PR #3490 lowers the pin).

Non-literal lookups need their own guard so that moving a literal behind a variable is a visible, reviewed change rather than a silent count drop. Measured on dev ae9f9e8d21 with the census's own receiver rule: 47 non-literal lookups (UI 24, Event_Handlers 9, Chat 6, Backup_Recovery 3, `<root>` 2, RAG_Search 2, Scheduling 1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A new non-literal `getattr`/`hasattr` lookup on an app-like receiver fails an architecture test, which names the file and line and says how to reach the service explicitly instead.
- [ ] Rewriting a counted literal lookup as a non-literal one fails, rather than reading as a reduction of the literal count.
- [ ] Every non-literal lookup that exists when the guard lands is recorded where a reviewer can see it, and that record can only shrink; a lookup that goes away fails until it is removed from the record.
- [ ] Private app attributes reached through a non-literal lookup today (at least `_vllm_connection_owner` and `_llamacpp_connection_owner` via `settings_screen.py`) are either listed against the frozen private-name allowance with the reason, or reached explicitly.
- [ ] The guard runs in the same required lane as the existing census and has a negative control showing it fails on a planted non-literal lookup.
<!-- AC:END -->
