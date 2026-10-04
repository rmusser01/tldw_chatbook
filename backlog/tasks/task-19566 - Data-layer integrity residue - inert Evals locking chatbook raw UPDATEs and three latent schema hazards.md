---
id: TASK-19566
title: >-
  Data-layer integrity residue — inert Evals locking, chatbook import raw
  UPDATEs, and three latent schema hazards
status: Done
assignee:
  - rmusser01
created_date: '2026-08-21 20:16'
updated_date: '2026-10-02 08:15'
labels:
  - db
  - data-integrity
  - tech-debt
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F6, F8, F9, F11, F12**. Grouped as one triage batch: each is a real
integrity defect, none is an active user-facing incident, and each carries an
honest reachability qualifier that must survive into the fix. All re-verified
at this branch base.

**F8 — Evals optimistic locking is inert (CONFIRMED).** Five tables in
`DB/Evals_DB.py` declare a `version` column (35 mentions of `version` in the
file), but **`expected_version` appears zero times** and **no `UPDATE` carries
`AND version = ?`**. It has the shape of concurrency control and provides
none — which is worse than having none, because reviewers will read it as
protection.

**F9 — chatbook import writes raw UPDATEs that bypass ChaChaNotes versioning
(CONFIRMED).** `Chatbooks/chatbook_importer.py:650` (`UPDATE messages SET
variant_of = ?…`) and `:676` (`UPDATE conversations SET
active_leaf_message_id = ?…`): no version bump, no `client_id`, no
`last_modified`. Unlike Media, ChaChaNotes has **no guard trigger** to catch
this, so the rows silently desynchronise from the sync log. Sharpest detail:
`deleted = ?` is set **from untrusted archive data** on an existing row — an
imported chatbook can mark a user's existing conversation deleted.

**F6 — two Media UPDATEs are guaranteed `IntegrityError` (CONFIRMED, but
UNREACHABLE).** The Media `BEFORE UPDATE` guard trigger requires `version + 1`;
`Chunking/chunking_interop_library.py:449` and `:471` (and
`DB/Client_Media_DB_v2.py:8261`) do not supply it, so they cannot succeed.
**Honest reachability, carried through from the lane: the calling widget has no
production importer, and `mark_media_as_processed` has no callers.** Frame this
as **wire-or-retire**, not "fix the bug": decide whether this code is meant to
be reachable. If it is, fix it and wire it; if it is not, delete it. Do not
repair dead code and leave it dead.

**F11 — two databases run with `foreign_keys` OFF (CONFIRMED, LATENT).**
`DB/Library_Ingest_Jobs_DB.py` and `DB/RAG_Indexing_DB.py` never enable the
pragma. Latent today because neither declares any foreign keys — but the next
schema change silently gets no enforcement.

**F12 — a cascade that would take the whole chat history (CONFIRMED schema, NO
LIVE TRIGGER).** `character_cards → conversations → messages` with
`ON DELETE CASCADE` (`ChaChaNotes_DB.py:470, 525`) and foreign keys ON. One
hard `DELETE` of a character card would remove the user's entire chat history
for it. **No such hard delete exists outside `Tests/` today** — the app soft-
deletes. This is a landmine for a future change, not a present bug; the value
here is the guard, not a schema change.

## Acceptance Criteria

- [x] Evals optimistic locking either works — `expected_version` supplied and
      `AND version = ?` on the UPDATEs, with a conflict test — or the `version`
      columns and their appearance of protection are removed
- [x] The chatbook importer goes through the versioned write path: version
      bump, `client_id`, `last_modified` set on every row it touches
- [x] `deleted` is never set on an existing row from archive-supplied data; an
      imported chatbook cannot mark a user's existing conversation deleted
- [x] A guard (trigger or test) fails if a ChaChaNotes UPDATE bypasses
      versioning, matching the protection Media already has
- [x] The two Media UPDATEs are resolved as **wire-or-retire** with the
      decision recorded — repaired *and* reachable, or deleted
- [x] `Library_Ingest_Jobs_DB` and `RAG_Indexing_DB` enable `foreign_keys`, or
      a comment records why they deliberately do not
- [x] A test fails if a hard `DELETE` of a character card becomes reachable
      from production code, so the cascade cannot be armed unnoticed

## Implementation Plan

1. F11 (done, see notes): enable the per-connection `foreign_keys` pragma on `Library_Ingest_Jobs_DB` and `RAG_Indexing_DB`, with pin tests.
2. F12 (done, see notes): an architecture census test that fails if a hard `DELETE FROM character_cards` appears in production code.
3. F8 (remaining): Evals optimistic locking — make it real (`expected_version` + `AND version = ?` + conflict test) or remove the inert columns; decision recorded.
4. F9 (remaining): chatbook importer goes through the versioned write path; `deleted` never set on existing rows from archive data; versioning-bypass guard (trigger or test) matching Media's protection.
5. F6 (remaining): the two unreachable Media UPDATEs resolved as wire-or-retire with the decision recorded.

## Implementation Notes (interim — F11 and F12 landed 2026-10-01)

**F11 — foreign_keys pragma enabled on both latent DBs.** `Library_Ingest_Jobs_DB` (connection-creation block) and `RAG_Indexing_DB` (`_configure_connection`, the single place connections are configured) now execute `PRAGMA foreign_keys = ON` per connection, matching the established repo idiom (AgentRuns_DB, Client_Media_DB_v2). Inert today by construction — neither schema declares any FK — so enabling changes no behavior; the value is that the NEXT schema change declaring one is enforced instead of silently inert. Pin tests: `Tests/DB/test_library_ingest_jobs_db.py::test_connections_enable_foreign_key_enforcement` and `Tests/DB/test_rag_indexing_db.py::TestForeignKeyEnforcement` assert pragma state 1 on live connections.

**F12 — cascade landmine guarded by census.** New `Tests/Architecture/test_character_card_hard_delete_census.py`: a whitespace-tolerant `DELETE FROM character_cards` census over all production sources (`tldw_chatbook/**/*.py`) with an explicit, currently-empty allowlist. Zero production sites today (verified by grep at implementation time), so the test pins zero — adding a hard delete anywhere in production turns it red by file:line with a message pointing at the soft-delete path. The cascade schema itself is deliberately untouched (the landmine is the reachability, not the DDL).

**Verification.** `python -m pytest Tests/DB/test_library_ingest_jobs_db.py Tests/DB/test_rag_indexing_db.py Tests/Architecture/test_character_card_hard_delete_census.py -q` — 50 passed (includes the three new tests).

**Remaining for this task:** F8, F9, F6 per the plan above — not started; reserved for the next session on this branch.

ADR required: no (so far) — pragma enablement and a census guard implement existing policy; revisit if F8/F9's wire-or-retire decisions change a storage contract.

## Implementation Notes (update 2026-10-01, second session — F6 and F9 landed; F8 remaining)

**F6 — both unreachable Media UPDATEs RETIRED (decision: retire).** `Client_Media_DB_v2.mark_media_as_processed` (module-level helper) and `ChunkingInteropLibrary.set_document_config`/`clear_document_config` (the two `UPDATE Media SET chunking_config ...` writers) had ZERO callers anywhere in production or tests (verified by repo-wide grep before deletion). Per this task's own framing — "do not repair dead code and leave it dead" — all three were deleted rather than fixed. Verification: `Tests/UI/test_chunking_lab_screen.py` + `Tests/RAG_Admin/test_chunking_lab_service.py` run identically before and after the deletion (13 failed / 5 passed / 26 errors both sides — pre-existing red mass, zero delta). Observed residue, deliberately NOT touched: `get_document_config` (the read side of the same dead seam) is also callerless but was outside the task's named scope.

**F9 — chatbook importer graph patches are now versioned writes.** The two raw UPDATEs in `Chatbooks/chatbook_importer.py` (variant metadata on `messages`, `active_leaf_message_id` on `conversations`) now carry `version = version + 1, last_modified = ?, client_id = ?`, matching the repo's established versioned-write idiom (`ChaChaNotes_DB.py` attachment-append path). New pin test `test_graph_import_patching_is_a_versioned_write` (in `Tests/Chatbooks/test_chatbook_thinking_round_trip.py`) drives the real export->import round trip and asserts version==2, last_modified, and client_id on every graph-patched message row and the conversation row. That suite also gained a `keep_bootstrap_profile` enrollment (admission signature, established precedent) and is now **21/21 passed**; `test_import_transactions.py` shows an identical 3-failed A/B before/after this change (pre-existing). On the `deleted`-from-archive clause: at current dev both UPDATEs bind only ids created inside the import transaction itself (`add_message`/`add_conversation` inserts), so an archive cannot mark a pre-existing user row deleted — held by construction and pinned by the round-trip test's WHERE shapes; the older hazard this task described no longer exists in this code path. Guard scope, recorded honestly: the versioning-bypass guard is test-shaped and scoped to the importer seam, not a repo-wide trigger like Media's — a ChaChaNotes-wide BEFORE UPDATE trigger would currently fail many legitimate writers (e.g. the feedback UPDATE at `ChaChaNotes_DB.py` ~13101 bumps no version); a writer-compliance sweep is the prerequisite and is noted as follow-up material.

**F8 — REMAINING (the only open AC).** Evals optimistic locking is confirmed inert at this base (six `version INTEGER NOT NULL DEFAULT 1` columns; `expected_version` appears zero times; no UPDATE carries `AND version = ?`). The update surfaces are `update_task`, `update_dataset`, `update_run_status`, `update_run`, `update_ab_test_status`, `update_ab_test_results` (`DB/Evals_DB.py`). Recommendation for the next session: for a single-user local benchmark store with no caller able to supply an expected version today, REMOVE the inert columns (with an Evals schema migration) unless a multi-writer surface is found — implementing real locking means threading `expected_version` through every caller of six methods, and optional-but-unused parameters would preserve exactly the false-appearance-of-protection problem this finding names.

**Verification commands (this session).** `pytest Tests/Chatbooks/test_chatbook_thinking_round_trip.py` -> 21 passed; `pytest Tests/UI/test_chunking_lab_screen.py Tests/RAG_Admin/test_chunking_lab_service.py` -> identical pre/post-deletion A/B; `pytest Tests/DB/test_library_ingest_jobs_db.py Tests/DB/test_rag_indexing_db.py Tests/Architecture/test_character_card_hard_delete_census.py` -> 50 passed.

## Implementation Notes (update 2026-10-02, third session — F8 landed; task complete)

**F8 — the inert `version` columns were REMOVED (decision: remove, not lock).** Caller map first, as the plan required, across all six update surfaces:

- `update_task`: `Evaluations_Interop/local_evaluations_service.py`, `Evaluations_Interop/evaluation_scope_service.py` (via service), `UI/Evals/snippet_editor.py`, and the `Evals/word_bench|skill_eval|character_probe` storage helpers — every one a single-UI-thread edit flow (bench/snippet editors, scope service).
- `update_dataset`: same family plus `character_probe/storage.py` — UI-thread only, no second writer of a given row.
- `update_run_status`: `Evals/eval_orchestrator.py` and `Evals/word_bench/runner.py` are **asyncio coroutines on the app loop** (verified: `asyncio.current_task()` machinery, no thread spawning), the UI cancel paths (`UI/Evals/sample_bench.py`, `UI/Screens/evals_screen.py`) and `Event_Handlers/eval_db_operations.py` (which hops through `run_in_executor`).
- `update_run`: evals_screen, the three storage helpers, `Event_Handlers/eval_db_operations.py`.
- `update_ab_test_status` / `update_ab_test_results`: `Evals/ab_testing.py` only, inside its own runner.

A mechanical multi-writer surface does exist on the run-status methods (a runner coroutine and a UI cancel can both write one `eval_runs` row), but **optimistic locking is not applicable to it**: no caller on any of the six surfaces reads a version before writing (`expected_version` appeared zero times repo-wide), and the overlapping writes are lifecycle state transitions where a version guard would reject legitimate terminal writes (a runner's "completed" landing after a UI "cancelled"); cancel-vs-complete ordering is already resolved by the runner's own cancellation checks. The two methods that bumped `version` (`update_task`/`update_dataset`) have no second writer at all, and four of the six surfaces never bumped it — it was not even a coherent change counter. Locking that no caller can participate in is the false-appearance problem renamed, so the columns are gone, per the second AC branch.

**What shipped.** `DB/Evals_DB.py`: `SCHEMA_VERSION` 5→6; the `version` column removed from the five `_create_schema` tables (the historical v1→v2 migration DDL deliberately keeps it — a v1 DB still creates `ab_tests` with the column and the v6 step drops it in the same pass); a v6 `_migrate_schema` step doing `ALTER TABLE ... DROP COLUMN version` per carrying table (guarded by `PRAGMA table_info` for idempotency, table names from the module literal `_VERSION_COLUMN_TABLES` through `validate_identifier`, mirroring `_PROBE_ANNOTATION_CASCADE_TABLES`); the `version = version + 1` bumps removed from `update_task`/`update_dataset`. No INSERT ever named the column and no SELECT names it — `SELECT *` rows simply stop carrying the key (verified: `get_task`/`get_dataset`/`search_tasks` dicts have no `"version"`; `Evaluations_Interop/evaluation_normalizers.py`'s `_safe_int(data.get("version"))` already tolerates absence, and no Evals consumer reads that key). `Evals/recovery.py`: `_VERSIONS = (5, 6)`, a v6 exact-SQL declaration captured from the real v6 constructor, and `_MIGRATIONS` 5→6 (five `DROP COLUMN` statements + `PRAGMA user_version = 6`) — the migration steps are REQUIRED, not optional: the restore-validate path demands exactly one declared step per version once v5 stopped being `max(policy.versions)`. Verified empirically (SQLite 3.50.4): `DROP COLUMN` edits the stored CREATE text in place, so a **migrated** v5→v6 database's schema catalog is byte-identical to a **fresh** v6 database's — a single v6 declaration validates both; `_Adapter().validate()` returns `()` for real fresh-v6 and migrated files alike. Residual risk recorded in that file's header: if a future SQLite build canonicalises the rewritten text instead, migrated DBs would fail validation fail-safe (`unsupported_schema`, never silent) and that build's text becomes a second v6 alternative entry (`SchemaPolicy` docstring: repeated version entries are exact alternatives).

**Tests.** New `Tests/Evals/test_evals_db_v5_to_v6_migration.py` (mirrors the v3→v4 suite's hand-built-legacy-DDL pattern): fixture sanity (v5 really carries the columns, seeded version values non-default 7/3/5/4), the migration itself (columns gone from all five tables, every seeded row's payload columns intact, FTS still finds the seeded task through the surviving triggers, post-migration `update_task` works), idempotent reopen, and fresh-schema-has-no-version-column. `Tests/Evals/test_evals_db.py`: `test_concurrent_modification` REMOVED — it manually bumped the dropped column and asserted the update still succeed ("optimistic locking is not implemented yet"), i.e. it pinned the exact false appearance this finding removes; a comment marks the spot. `Tests/Evaluations_Interop/test_local_evaluations_service.py`: `assert updated["version"] == 2` → `assert "version" not in updated`. `Tests/Evals/word_bench/test_storage.py`: the deliberate schema pin `test_schema_version_is_five` → `test_schema_version_is_six` with the F8 rationale appended to task-1691's (that pin exists precisely to force conscious acknowledgment of bumps — this is it working as designed).

**Verification (exact commands and results).** `pytest Tests/Evals/test_evals_db_v5_to_v6_migration.py Tests/Evals/test_evals_db.py Tests/Evaluations_Interop/test_local_evaluations_service.py Tests/Evals/word_bench/test_storage.py Tests/Evals/test_evals_db_v3_to_v4_migration.py -q` → **117 passed**. Full-dir A/B (`pytest Tests/Evals/ -q`, baseline produced via `git checkout HEAD -- <paths>` swaps — never `git stash`, shared stack): HEAD → 60 failed / 815 passed / 12 skipped / 10 errors; with F8 → failure list **byte-identical** (`diff` of the FAILED/ERROR lists: empty). `pytest Tests/Backup_Recovery/test_domain_owners.py -q` → 25 failed / 63 passed at BOTH HEAD and with F8, FAILED lists diff-identical (all research/writing parametrizations; zero evals-variant failures either side). `pytest Tests/Backup_Recovery/test_eval_retained_definitions.py -q` → 1 failed (`test_complete_rebackup_retains_eval_after_source_and_candidate_removal`) identically at HEAD and with F8 — pre-existing. No config-admission `RecoveryRequired` signature appeared in any of these suites, so no `keep_bootstrap_profile` enrollment was needed this session. Static check: `python -m compileall` clean on all touched files (no ruff/flake8 config or install exists in this venv).

ADR required: no — mechanical debt removal within the existing Evals schema contract. The store's enforced behavior is unchanged (no API, data-ownership, or concurrency-model change — the concurrency model was and remains "none enforced", only the misleading appearance is gone), the migration follows the established `SCHEMA_VERSION`/`_migrate_schema` pattern with repo precedent (no prior Evals schema bump v2–v5 has an ADR), and the recovery schema contract is extended additively (v5 declarations retained; v6 added) rather than altered. The removal-vs-locking decision and its caller-map evidence live in these notes and in `_VERSION_COLUMN_TABLES`' rationale comment. (Numbering provenance noted: 206–208 are held by this session's unpushed branches, 209/211+ were available and would have been used had an ADR been judged required.)

## Implementation Notes (landing rebase, 2026-10-03)

Rebased onto `dev` after #2975/#2993. One change: the two versioned graph-patch writes in the chatbook importer (F9) stamped `last_modified` with `datetime.now(timezone.utc).isoformat()`, which ADR-173's timestamp-writer check (newer than this branch's base) refuses. They use `utc_now_iso()`, the helper the importer already imported. The diagnostic inventory is regenerated; its drift was the logging calls that went with the retired writers (F6) and the removed `notes_au` self-heal.

