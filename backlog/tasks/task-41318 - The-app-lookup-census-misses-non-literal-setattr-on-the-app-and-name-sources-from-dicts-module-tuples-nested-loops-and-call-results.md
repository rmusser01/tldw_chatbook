---
id: TASK-41318
title: "The app-lookup census misses non-literal setattr on the app, and name sources from dicts, module tuples, nested loops and call results"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- architecture
- tests
priority: medium
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_nonliteral_lookups.tsv
- tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py
- https://github.com/rmusser01/tldw_chatbook/pull/3538
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. These are the known ceilings of TASK-40141 (PR #3538), recorded in its `ponytail:` comment. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code and running a throwaway AST scan; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-40141 made the census follow literal names that reach the app object through a variable, a helper parameter or a literal loop. Two gaps remain. A string lookup or app mutation hidden in one of these places is not counted. So an "explicit service" migration can move a literal there and read as a reduction, which is the dodge TASK-40141 set out to close.

**Evidence.**
- **`setattr` is not scanned at all.** The scan matches only `getattr`/`hasattr` (`Tests/Architecture/test_app_string_lookup_census.py:160-171`). An AST scan for `setattr`/`delattr` on the census receivers (`_RECEIVERS`, `:45`) with a non-literal name finds 15 sites. Six are in `Event_Handlers/LLM_Management_Events/server_lifecycle.py` (`:99`, `:255`, `:283`, `:311`, `:382` with `process_attr`; `:586` with `attribute`). Others: `Chat/console_runtime.py:6915`, `Chat/console_fleet_attention.py:69`, `UI/Screens/chat_screen.py:15816` and `:15844`, `UI/Screens/library_screen.py:681`, `UI/Navigation/main_navigation.py:110`, `RAG_Search/semantic_availability.py:243`, `UI/Library_Modules/library_file_export.py:253`, `UI/Console_Modules/reaction_preview.py:59`.
- **Name sources the census does not follow** (their rows read `-`; 31 of the 47 rows in `Tests/Architecture/app_nonliteral_lookups.tsv`):
  - dict lookups: `_SOURCE_DB_ATTR.get(source_type)` (`Scheduling/automation_health.py:73`), `service_attributes[owner]` (`UI/Screens/library_screen.py:22320`);
  - module-level tuples: `LAB_SERVER_SOURCES` (`UI/Lab_Modules/lab_server_status.py:38`, `:92`);
  - nested loops: the `declarations` loop in `Backup_Recovery/runtime_maintenance.py:457`;
  - call results: `process_attr = _validate_provider(provider)` (`server_lifecycle.py:91`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census counts or records non-literal `setattr`/`delattr` on app receivers, with a negative control that plants one and fails
- [ ] #2 A literal reached through a dict lookup, a module-level tuple, a nested loop or a function's literal return value is counted, each with a negative control; any source left unfollowed is listed in the test's docstring with its reason
- [ ] #3 The pins and the non-literal record are re-measured, and the census passes on dev
<!-- AC:END -->

## Implementation Plan

ADR required: no. This extends an existing architecture test (TASK-36942.4, TASK-40141) without changing product code or its rules; only what the census can see grows.

1. Reproduce: negative controls that pass on the unmodified census -- a planted non-literal `setattr`/`delattr` on the app is not recorded, and a literal moved into a dict (`.get`/`[k]`), a module-level tuple, a nested loop or a function's literal return value drops out of the count and the private-name rule.
2. Record non-literal `setattr`/`delattr` on the census receivers in `app_nonliteral_lookups.tsv` (recorded, not counted: a literal `setattr` is an attribute assignment, not a string lookup).
3. Replace the loop-only follower with one that follows a name through local and module-level assignments, enclosing loops (nested loops and comprehensions), dict lookups and same-module function returns. A name that is itself an UPPER_CASE constant stays the record's business, as today.
4. List the sources left unfollowed, with reasons, in the test docstring.
5. Re-measure the pins and the record; record the numbers with a dated note.
