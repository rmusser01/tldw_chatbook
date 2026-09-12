---
id: TASK-22224
title: 'Add isolation_level=None to held-connection stores, including the template'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 22:50'
labels:
  - database
  - hardening
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22224).

Latent but load-bearing (mechanism verified empirically this review on Python 3.12/SQLite
3.49.1): without `isolation_level = None`, any bare DML on a held connection opens an
implicit DEFERRED transaction; `TransactionContextManager` then BORROWS it
(`ChaChaNotes_DB.py:18969-18978`) and `transaction(immediate=True)` silently degrades —
defeating the task-21100 hardening (12 IMMEDIATE sites) the moment one bare DML lands.
`ChaChaNotes_DB.py:3121-3141` never sets it; nor do `DB/Library_Ingest_Jobs_DB.py:48-60`
(the store TEMPLATE others copy), `DB/Evals_DB.py:180-203`,
`Widgets/Tamagotchi/tamagotchi_storage.py`, `Sync_Interop/sync_state_repository.py`,
`Kanban_Interop/local_kanban_db.py`, `Scheduling/db/scheduled_tasks_db.py`,
`Sync_Interop/notes_mirror.py`. `Notifications/event_state_repository.py:224` is the one
store that gets it right. Currently zero firing call sites were found — this is a loaded
gun, filed as hardening under the stability-over-quick-wins ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All held-connection stores set `isolation_level = None` at open (or the exception is documented at the site)
- [x] #2 A test reproduces the degradation mechanism (bare DML then `transaction(immediate=True)` must still BEGIN IMMEDIATE) and pins it repo-wide or at the template
- [x] #3 The template file (`Library_Ingest_Jobs_DB.py`) carries the rule in its docstring so copies inherit it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-census held-connection stores (review pin vs tip): two explicit-BEGIN-manager stores to flip (Library_Ingest_Jobs_DB [template], ChaChaNotes_DB), three flip-safe stores with single-statement/per-op writes (notes_mirror, tamagotchi_storage, local_kanban_db), three implicit-transaction-reliant stores taking the AC's documented-exception hatch (Evals_DB, sync_state_repository, scheduled_tasks_db). Shared-seam change in private_sqlite rejected: 47 callers; Subscriptions_DB.py:1106-1129 documents explicit reliance on default isolation.
2. Red-first: parametrized guard in Tests/DB/test_held_connections.py -- bare DML then the store's explicit-BEGIN transaction must still emit BEGIN [IMMEDIATE] (trace-callback asserted), plus autocommit/no-open-transaction and isolation_level-is-None pins per flipped store.
3. Implement: isolation_level=None at each flipped opener; template module docstring carries the rule; ChaChaNotes vacuum() stops restoring legacy isolation; Library_Ingest v2->v3 version stamp becomes a single-statement UPDATE (executescript ends the manager's txn in both modes -- verified empirically on 3.12.11/SQLite 3.49.1); documented exceptions at the three implicit-reliant openers.
4. Per-store commit()/rollback()/write-site census recorded in Implementation Notes (atomicity proof).
5. Targeted suites + new guard + --collect-only sweep (tee'd), ./scripts/preflight.sh, mutation test (drop the setting from one store -> its parametrization reds).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Flipped the provable stores to true autocommit and took the AC's documented-exception hatch for the implicit-transaction-reliant ones, after re-censusing every store from the review pin against tip. Shared-seam change in connect_private_sqlite REJECTED: 47 call sites, and Subscriptions_DB (TASK-1362 comment) explicitly relies on the default isolation level; with-conn stores would silently lose multi-statement write atomicity.

FLIPPED (isolation_level=None at open, write paths censused per store):
- DB/Library_Ingest_Jobs_DB.py (THE TEMPLATE): module docstring now carries the held-connection rule; transaction() docstring names itself sole transaction owner; v2->v3 version stamp became a single-statement UPDATE (executescript ends the manager's explicit txn and commits as it goes in BOTH isolation modes -- verified empirically on 3.12.11/SQLite 3.49.1 -- so a DELETE+INSERT stamp pair could be crash-split under autocommit); two now-no-op commit()s removed with comments. Census: all multi-statement writes (upsert_retry, migration stamp pairs v1v2/v3v4/v4v5/v5v6/v6v7) run inside transaction(); delete_job is single-statement.
- DB/ChaChaNotes_DB.py: setting in _get_thread_connection; vacuum() no longer toggles isolation (the old restore-to-"" flipped the held connection back to legacy mode after any vacuum); borrow-path comment updated. Census: all DML manager-owned; commit sites = manager + execute_query/execute_many commit branches with ZERO commit=True/script=True callers repo-wide; every direct get_connection() site in the file is a read; external consumers swept repo-wide -- Chat_Dictionary_Lib (3 single-DML+commit spans, DDL-only trigger repair), Notes_Library note-link schema (DDL-only script), Persona_Visual publication + visual_identity (caller-explicit BEGIN IMMEDIATE -- the borrow path's remaining legitimate client), Actor_Packs (in_transaction guards only), Persona_Visual repository (execute_query DML strictly inside transaction(immediate=True)).
- Sync_Interop/notes_mirror.py, Widgets/Tamagotchi/tamagotchi_storage.py: held conns whose writes are all single statements (with-conn commit becomes a harmless no-op).
- Kanban_Interop/local_kanban_db.py: per-op conns with an explicit-BEGIN transaction() helper; initialize_schema parity verified (script self-commits both modes; the two meta upserts self-commit; _ensure_schema retries and the fts_available reader degrades gracefully); one no-op commit removed.

DOCUMENTED EXCEPTIONS at the opener with conversion guidance (AC hatch): DB/Evals_DB.py (multi-statement with-conn writes incl. a deliberately NESTED with-conn pair explicit BEGIN cannot express), Sync_Interop/sync_state_repository.py (bare with-conn transaction(); multi-statement commit spans; only :memory: holds), Scheduling/db/scheduled_tasks_db.py (NOT held -- per-op conns closed per call, mechanism cannot fire), plus the re-census found three more held stores the review did not list: DB/Prompts_DB.py, DB/Client_Media_DB_v2.py, DB/Subscriptions_DB.py -- each needs its own ChaChaNotes-style census before flipping (follow-up work).

TESTS: Tests/DB/test_held_connections.py section 6 -- parametrized guards over 5 stores (isolation pin / bare-DML-leaves-no-open-txn / bare-DML-then-transaction-still-BEGINs via trace callback, + close-survival pins for template and notes_mirror). Red-first: all 15 failed pre-fix in the two predicted modes (template/kanban raised "cannot start a transaction within a transaction"; chachanotes silently borrowed -- trace showed no BEGIN). Mutation: removing the setting from kanban reds exactly its 3 parametrizations; from the template, exactly its 4. Ported test_local_authority_accessor_borrows_caller_owned_sqlite_transaction to arm with explicit BEGIN (subject unchanged).

VERIFIED: guards 47/47; ingest/kanban/mirror/tamagotchi suites 199; Tests/DB+ChaChaNotesDB 1820; Evals-DB/Scheduling/Sync_Interop 687; citation files 340; --collect-only 59459 collected (28 errors, all missing optional deps: numpy/playwright/mlx/audio -- none intersect this diff); preflight all green. Pre-existing reds PROVEN inherited by A/B against base 76f130138 production files (identical 17 failures) and by legacy-isolation revert on the chachanotes line (identical failure): the Console-send family (test_console_local_citation_boundary 60+F plus a hang, test_console_agent_*, test_console_command_suggestions, test_console_context_compaction, test_console_diff_feedback_delivery, test_console_durable_turn_fix_round1, test_console_headless_wake_invariants, test_child_run_scope_ordering), test_chat_mocked_apis (mocks leak to the network guard, red standalone), test_image_data_integrity (numpy absent).
<!-- SECTION:NOTES:END -->
