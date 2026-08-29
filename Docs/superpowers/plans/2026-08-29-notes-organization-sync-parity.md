# Notes Organization Sync Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task with review checkpoints.

**Goal:** Make Chatbook a conforming, recoverable consumer of the server's complete six-domain Notes organization Sync-v2 group without exporting filesystem ownership or changing the existing `notes.note` wire contract.

**Architecture:** Keep portable identity and mutation intent in the owning ChaChaNotes database. A strict contract module validates server-compatible payloads and deterministic link IDs; a Notes repository applies resources and links transactionally; the existing local-first service enrolls the six domains as one group and idempotently copies immutable Notes intents into the separate general SyncState outbox. Legacy adoption remains an explicit review step, and folder projection distinguishes an explicit tombstone from descendants hidden by the local tree UI.

**Tech Stack:** Python 3.11, SQLite/FTS5, Pydantic-compatible Sync-v2 envelope models, `httpx`, `pytest`.

---

## Scope and prerequisites

- Implements `TASK-24307` only.
- Before the first code edit, set TASK-24307 to `In Progress` and add an `## Implementation Plan` section to its task file linking this document and ADR-102; keep it out of `Done` until every DoD item is evidenced.
- Read the approved spec, ADR-102, ADR-059, ADR-073, and the testing/live-verification lessons before execution.
- Recheck `tldw_server` `origin/dev` before coding and integration. Reviewed baseline: `1ad2f1e5b30c49ea75396e4b713496b73e875fec` (2026-08-28).
- Normative server sources: `tldw_Server_API/app/core/Sync/v2/notes_organization.py`, `models.py`, `notes_organization_bootstrap.py`, and `tests/Sync/test_sync_v2_notes_organization_identity.py`.
- Do not add a seventh Notes domain, another folder contract, a dependency, or portable filesystem metadata.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/102-portable-notes-organization-and-agent-lessons.md`

Reason: persistent identity, schema/migrations, synchronization ownership, conflict policy, and the client/server contract change. ADR-102 records the approved boundary and amends ADR-059/ADR-073.

## Task 1: Pin the six-domain contract and identity vectors

**Files:**

- Create: `tldw_chatbook/Sync_Interop/notes_organization.py`
- Create: `Tests/Sync_Interop/test_notes_organization_contract.py`

- [ ] **Write failing contract tests.** Pin the ordered group `notes.keyword`, `notes.keyword_link`, `notes.keyword_collection`, `notes.keyword_collection_link`, `notes.folder`, `notes.folder_link`; strict unknown-field rejection; name bounds; canonical lowercase UUIDv4 resource IDs; link/resource tombstone shapes; and these exact server vectors:

```text
notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0
notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199
notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670
```

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_contract.py` must fail because the module is absent.

- [ ] **Implement the minimal pure contract.** Port only public server validators and identity functions. Canonical link JSON is:

```python
canonical = json.dumps(
    {"domain": domain, "members": list(members), "schema_version": 1},
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
```

Use standard-library `json`, `hashlib`, and `uuid`; expose stable local validation reason codes; do not import server internals.

- [ ] **Run green:** `pytest -q Tests/Sync_Interop/test_notes_organization_contract.py`.

- [ ] **Commit:**

```bash
git add tldw_chatbook/Sync_Interop/notes_organization.py Tests/Sync_Interop/test_notes_organization_contract.py
git commit -m "feat(sync): pin Notes organization contract"
```

## Task 2: Add stable identities and durable Notes state (schema v50)

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v49_to_v50_notes_organization_sync.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `Tests/DB/test_chachanotes_notes_organization_migration.py`

- [ ] **Write a genuine v49 reopen test.** Build through the repository's real historical schema path, insert active and deleted keywords/collections/folders/memberships, then reopen. Assert every resource receives a stable unique canonical UUIDv4 `sync_id`, local PKs remain unchanged, second reopen is stable, and failure rolls schema/version back together. Never stamp a partial synthetic schema as v49.

- [ ] **Run red:** `pytest -q Tests/DB/test_chachanotes_notes_organization_migration.py` must fail at version 49/missing columns.

- [ ] **Add v50 DDL.** Add nullable `sync_id` to `keywords`, `keyword_collections`, and `note_folders`; backfill in Python within the migration transaction; then add unique indexes. Migration and fresh schema use equivalent executable constraints. The intent table is:

```sql
CREATE TABLE notes_organization_sync_intents(
  intent_id TEXT PRIMARY KEY,
  server_profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  domain TEXT NOT NULL CHECK(domain IN (
    'notes.keyword', 'notes.keyword_link',
    'notes.keyword_collection', 'notes.keyword_collection_link',
    'notes.folder', 'notes.folder_link'
  )),
  object_id TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('upsert', 'tombstone')),
  schema_version INTEGER NOT NULL CHECK(schema_version = 1),
  encryption_policy TEXT NOT NULL CHECK(encryption_policy = 'server_trusted_v1'),
  payload_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL CHECK(
    length(payload_hash) = 64 AND payload_hash = lower(payload_hash)
    AND payload_hash NOT GLOB '*[^0-9a-f]*'
  ),
  base_object_revision INTEGER,
  base_object_hash TEXT,
  dependency_refs_json TEXT NOT NULL DEFAULT '[]',
  source_version INTEGER NOT NULL CHECK(source_version >= 1),
  created_at TEXT NOT NULL,
  outbox_client_envelope_id TEXT,
  copied_at TEXT,
  acknowledged_at TEXT,
  CHECK((base_object_revision IS NULL) = (base_object_hash IS NULL)),
  CHECK(outbox_client_envelope_id IS NULL OR outbox_client_envelope_id = intent_id),
  UNIQUE(server_profile_id, dataset_id, domain, object_id, source_version, operation)
);
CREATE INDEX idx_notes_organization_intents_pending
  ON notes_organization_sync_intents(server_profile_id, dataset_id, created_at, intent_id)
  WHERE acknowledged_at IS NULL;
```

Also add constrained tables for organization heads; profile/bootstrap checkpoints (`server state`, pull cursor, inventory phase, last inventory key); adoption reviews; and `note_folder_sync_suppressions` with a pair primary key and note foreign key. JSON is canonical compact JSON validated before insert. `intent_id == client_envelope_id` is the permanent cross-database correlation. This feature retains acknowledged Notes intents; it does not invent a compaction policy.

- [ ] **Wire migration.** Add `_migrate_from_v49_to_v50`, register step `49`, update fresh-schema DDL, and set `_CURRENT_SCHEMA_VERSION = 50`. Generate UUIDv4 in deterministic table/PK iteration; do not derive identity from mutable names/local IDs.

- [ ] **Run green:**

```bash
pytest -q Tests/DB/test_chachanotes_notes_organization_migration.py
pytest -q Tests/DB/test_chachanotes_note_folders_migration.py
pytest -q Tests/DB/
```

`Tests/DB/` is the schema-bump migration sweep, not the full repository suite.

- [ ] **Commit:**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v49_to_v50_notes_organization_sync.sql Tests/DB/test_chachanotes_notes_organization_migration.py
git commit -m "feat(notes): add portable organization identity state"
```

## Task 3: Implement one cursor-aware organization repository

**Files:**

- Create: `tldw_chatbook/Notes/notes_organization_repository.py`
- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `tldw_chatbook/Notes/note_folder_models.py`
- Create: `Tests/Notes/test_notes_organization_repository.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`

- [ ] **Write failing tests** using real in-memory `CharactersRAGDB`: lookup by sync ID, server-compatible casefold collision keys, hierarchy cycles, parent-first materialization, deterministic link verification, note/conversation keyword links, collection links, folder suppressions, replay idempotency, and exact restore intent. Pin `.`, `..`, slash/backslash rejection, the relative 500-character path limit, and Unicode pairs such as full-width `Ａ` versus ASCII `A` that local NFKC merges but portable casefold does not; an unrepresentable local collision remains a review candidate rather than being merged.

- [ ] **Run red:** `pytest -q Tests/Notes/test_notes_organization_repository.py`.

- [ ] **Implement `NotesOrganizationRepository`.** It accepts `CharactersRAGDB` and cursor-aware methods such as:

```python
def apply_envelope(self, cursor, *, dataset_id, domain, object_id,
                   operation, payload, object_revision, object_hash,
                   server_cursor, restore_intent=False) -> ApplyResult: ...
def record_intent(self, cursor, *, profile, dataset, domain, object_id,
                  operation, payload, source_version) -> str: ...
```

Keep wire validation in `Sync_Interop.notes_organization` and SQL/projection here. Add a separate portable `casefold()` key/path resolver; do not reuse `note_folder_models.normalize_folder_name`, whose NFKC behavior remains the local UI collision rule. Portable paths are relative server-valid segment sequences; local `/...` paths remain derived UI data. Effective folder membership is `(active manual UNION active managed) MINUS suppressions`.

- [ ] **Separate explicit and derived folder deletion.** Extend `FolderMutationResult` with `explicit_folder_id`. Existing UI subtree hiding/restoration stays local; only the explicit ID emits a portable tombstone/upsert. Descendant canonical heads stay unchanged.

- [ ] **Run green:**

```bash
pytest -q Tests/Notes/test_notes_organization_repository.py
pytest -q Tests/Notes/test_note_folder_repository.py Tests/Notes/test_notes_scope_service_folders.py
```

- [ ] **Commit:**

```bash
git add tldw_chatbook/Notes/notes_organization_repository.py tldw_chatbook/Notes/note_folder_repository.py tldw_chatbook/Notes/note_folder_models.py Tests/Notes/test_notes_organization_repository.py Tests/Notes/test_note_folder_repository.py
git commit -m "feat(notes): materialize portable organization"
```

## Task 4: Build a resumable adopted legacy inventory

**Files:**

- Modify: `tldw_chatbook/Notes/notes_organization_repository.py`
- Create: `tldw_chatbook/Sync_Interop/notes_organization_inventory.py`
- Create: `Tests/Sync_Interop/test_notes_organization_legacy_inventory.py`

- [ ] **Write failing inventory tests** with active and soft-deleted resources, active relationships, dormant relationships preserved beneath a deleted resource, persisted soft-deleted keyword/folder/collection link rows or sync-log history, `keep_local` and `merge` adoption decisions, missing note/conversation dependencies, and a crash after each inventory phase/object. Mere absence with no persisted relationship evidence must not invent a tombstone; an existing soft-deleted link row/history is evidence and must be reconstructed.

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_legacy_inventory.py`.

- [ ] **Build only after bootstrap pull and adoption resolution.** Inventory adopted local state in deterministic phases: resource upserts parent-before-child; active/dormant link upserts after referenced organization and note/conversation identities exist; then tombstones for every resource or link with persisted soft-delete/history evidence. Deleted resources and evidenced deleted links therefore produce upsert-then-tombstone history; link tombstones retain the required identity payload. Compute every link ID from the final adopted resource sync IDs and verify it against the normative hash function.

- [ ] **Persist immutable intents and checkpoints.** Commit each bounded phase with `inventory_phase` and `last_inventory_key` in the Notes database. Retry resumes at the next uncommitted key and reuses the same intent IDs; it never rebuilds from a newer snapshot or restarts initialization merely because the process stopped. Dependencies that are not enrolled stay local and reviewable.

- [ ] **Run green:** `pytest -q Tests/Sync_Interop/test_notes_organization_legacy_inventory.py Tests/Notes/test_notes_organization_repository.py`.

- [ ] **Commit:** stage the three files and commit `feat(sync): inventory legacy Notes organization`.

## Task 5: Apply all six incoming domains transactionally

**Files:**

- Create: `tldw_chatbook/Sync_Interop/domain_adapters/notes_organization.py`
- Modify: `tldw_chatbook/Sync_Interop/domain_adapters/__init__.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/restore_service.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Sync_Interop/test_notes_organization_adapters.py`
- Modify: `Tests/Sync_Interop/test_envelope_applier.py`
- Modify: `Tests/Sync_Interop/test_local_first_sync_service.py`

- [ ] **Write failing adapter tests** for all six domains through `SyncEnvelopeApplier`: stale/duplicate heads, invalid identities/payloads, dependencies, hierarchy conflicts, tombstone restore, link restore without `restore_intent`, and rollback when head persistence fails.

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_adapters.py`.

- [ ] **Register one adapter family.** Map all six domains to one `NotesOrganizationSyncAdapter`; validate then delegate one transaction to the repository. Inject it explicitly through `SyncEnvelopeApplier(..., notes_organization_repository=...)`, local-first, and restore paths. Do not overload the unrelated generic `local_store` seam.

- [ ] **Wire production.** Construct the repository from the canonical ChaChaNotes DB in `app.py`; do not leave real operation dependent on the optional `sync_v2_local_store` test seam.

- [ ] **Run green:**

```bash
pytest -q Tests/Sync_Interop/test_notes_organization_adapters.py
pytest -q Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_local_first_sync_service.py
```

- [ ] **Commit:** stage only the listed files and commit `feat(sync): apply Notes organization domains`.

## Task 6: Capture immutable intents and copy them idempotently

**Files:**

- Create: `tldw_chatbook/Sync_Interop/notes_organization_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/notes_outbox_producer.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Sync_Interop/test_notes_organization_intent_dispatch.py`
- Create: `Tests/Sync_Interop/test_notes_organization_app_wiring.py`
- Modify: `Tests/Sync_Interop/test_sync_state_repository.py`
- Modify: `Tests/Notes/test_notes_scope_service_folders.py`

- [ ] **Write deterministic crash tests with two real SQLite files.** Inject failure: after Notes commit/before outbox insert; after outbox insert/before `copied_at`; after server acknowledgement/before `acknowledged_at`. Restart and assert one outbox row per intent, no lost mutation, immutable old payload, and eventual acknowledgement. Also cover same-ID/different-envelope rejection, general-outbox compaction/reset, and re-copy after a missing general row without creating a new logical operation.

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_intent_dispatch.py`.

- [ ] **Implement idempotent drain.** Use `intent_id` as `client_envelope_id`. General outbox enqueue returns an identical existing row and rejects same-ID/different-payload. Drain reads the immutable intent and builds the exact `server_trusted_v1` clear organization envelope without re-reading newer resource state and without applying client-private payload encryption; existing transport and server at-rest protections remain authoritative. It insert-or-confirms the outbox row, marks `copied_at`, then reconciles acknowledgements. Never hold both DB transactions or claim cross-DB atomicity.

- [ ] **Wire the production graph after SyncState exists.** In `TldwCli._wire_server_parity_state_repositories` (or the immediately following parity-service composition block), construct the note producer, organization repository/service, and intent drain/finalizer from `self.chachanotes_db` plus `self.sync_state_repository`; then assign the producer to the already-created `self.notes_scope_service.sync_v2_notes_producer`. `Tests/Sync_Interop/test_notes_organization_app_wiring.py` must build this production-shaped graph and prove an ordinary mutation creates a Notes intent instead of silently returning through the current `None` seam.

- [ ] **Capture owner mutations behind readiness.** For a synchronized profile, folder create/rename/move/delete/restore, manual/effective memberships, keywords, and collections first require the complete group to be `ready`, then write mutation plus intent in one Notes transaction. `initializing`, `pulling`, `adoption_review`, and `failed` reject the general organization mutation without a partial local write. Permanently local-only profiles continue direct local behavior. The sole pre-ready write exception is the content-only pending Agent Lessons flow implemented by TASK-24308; this task must leave a narrow service hook for it without enabling general organization writes.

- [ ] **Run green:**

```bash
pytest -q Tests/Sync_Interop/test_notes_organization_intent_dispatch.py
pytest -q Tests/Sync_Interop/test_notes_outbox_producer.py Tests/Sync_Interop/test_sync_state_repository.py
pytest -q Tests/Sync_Interop/test_notes_organization_app_wiring.py
pytest -q Tests/Notes/test_notes_scope_service_folders.py
```

- [ ] **Commit:** stage only listed files and commit `feat(sync): journal Notes organization intents`.

## Task 7: Enroll, bootstrap, and adopt as one capability

**Files:**

- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/notes_organization_sync_service.py`
- Modify: `tldw_chatbook/Sync_Interop/conflict_review.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_profile_status_state.py`
- Modify: `tldw_chatbook/tldw_api/sync_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Create: `Tests/Sync_Interop/test_notes_organization_enrollment.py`
- Modify: `Tests/Sync_Interop/test_server_sync_service.py`
- Modify: `Tests/Sync_Interop/test_sync_v2_conflict_review.py`
- Modify: `Tests/tldw_api/test_sync_schemas_m1.py`
- Modify: `Tests/tldw_api/test_sync_client.py`

- [ ] **Write failing enrollment tests:** new/existing datasets, partial advertisement, incompatible encryption policy, interruption after every checkpoint, missing note/conversation domains, no same-name adoption, and merge/rename-local/keep-local decisions. Parse the server's dataset-level `notes_organization {state,captured_count,expected_count,error_code}` status instead of dropping it as an extra. Publication is forbidden before the authoritative server state and local state are both `ready`.

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_enrollment.py`.

- [ ] **Implement profile parsing and persisted transitions.** Add typed `SyncV2NotesOrganizationStatus(state, captured_count, expected_count, error_code)` as `SyncV2ProfileDatasetStatus.notes_organization`. Submit one existing `client.bootstrap_sync_v2_profile(SyncV2ProfileBootstrapRequest(...))` request containing the complete six-domain group, explicit note/conversation dependencies, and adapter version 1 for each organization domain. The bootstrap request has no encryption-policy field: verify the response dataset's `encryption_policy == "server_trusted_v1"` and reject another policy. Persist returned server state/counts/error, resume bounded status checks with `get_sync_v2_profile`, then pull bootstrap/history and enter adoption review. Local `ready` requires authoritative server `ready`, complete applied history, zero open reviews, completed legacy inventory, and dependencies. Retry resumes from the last durable server/local phase and cursor, reusing the bootstrap identity; start a new bootstrap only when the server explicitly reports the old one absent/incompatible.

- [ ] **Reuse conflict review.** Present content-free adoption rows with explicit actions. Bounded logical folder display names and relative portable paths are shown because they are required to distinguish same-visible-path identities. Resolution updates the Notes-owned review and resumes idempotently; never expose note bodies, ciphertext, physical filesystem paths/bindings, or secrets.

- [ ] **Run green:**

```bash
pytest -q Tests/Sync_Interop/test_notes_organization_enrollment.py
pytest -q Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_sync_v2_conflict_review.py
pytest -q Tests/tldw_api/test_sync_schemas_m1.py Tests/tldw_api/test_sync_client.py
```

- [ ] **Commit:** stage only listed files and commit `feat(sync): enroll Notes organization group`.

## Task 8: Prove two-device convergence and tombstone semantics

**Files:**

- Create: `Tests/Sync_Interop/test_notes_organization_two_device.py`
- Modify: `Tests/Sync_Interop/test_notes_organization_intent_dispatch.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`

- [ ] **Write the end-to-end local integration test.** With two real ChaChaNotes files and a fake transport: A creates all resource/link types; B pulls them under different local PKs; A deletes only a parent; only the parent tombstone is sent; descendants/links are dormant; restore reveals them; source-managed unlink emits a tombstone only when the effective union becomes absent.

- [ ] **Run red:** `pytest -q Tests/Sync_Interop/test_notes_organization_two_device.py`.

- [ ] **Fix only owning seams, then run green:**

```bash
pytest -q Tests/Sync_Interop/test_notes_organization_contract.py Tests/Sync_Interop/test_notes_organization_adapters.py Tests/Sync_Interop/test_notes_organization_intent_dispatch.py Tests/Sync_Interop/test_notes_organization_enrollment.py Tests/Sync_Interop/test_notes_organization_two_device.py
pytest -q Tests/Notes/test_notes_organization_repository.py Tests/Notes/test_note_folder_repository.py Tests/Notes/test_notes_scope_service_folders.py
```

- [ ] **Commit:** stage only these tests/minimal owning fixes and commit `test(sync): prove Notes organization convergence`.

## Task 9: Document, verify safely, and close TASK-24307

**Files:**

- Create: `Docs/Development/Sync-v2-client.md`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `backlog/tasks/task-24307 - Consume-server-Notes-organization-sync-group.md`
- Modify `backlog/docs/lessons-*.md` only if a real reusable incident occurred.

- [ ] **Document** the group, identity split, enrollment/adoption, dependencies, filesystem exclusion, recovery, suppression, and non-cascading tombstones. The new development page becomes the canonical client-runtime reference; link it from the Notes guide rather than duplicating its recovery details.

- [ ] **Run targeted verification:**

```bash
python -m compileall -q tldw_chatbook/Sync_Interop tldw_chatbook/Notes tldw_chatbook/tldw_api
pytest -q Tests/DB/ Tests/Notes/test_notes_organization_repository.py Tests/Notes/test_note_folder_repository.py Tests/Notes/test_notes_scope_service_folders.py Tests/Sync_Interop/test_notes_organization_contract.py Tests/Sync_Interop/test_notes_organization_legacy_inventory.py Tests/Sync_Interop/test_notes_organization_adapters.py Tests/Sync_Interop/test_notes_organization_intent_dispatch.py Tests/Sync_Interop/test_notes_organization_app_wiring.py Tests/Sync_Interop/test_notes_organization_enrollment.py Tests/Sync_Interop/test_notes_organization_two_device.py Tests/tldw_api/test_sync_schemas_m1.py Tests/tldw_api/test_sync_client.py
git diff --check
```

Do not run the full repository suite unless the user explicitly asks.

- [ ] **Use the schema-safe live gate.** Do not launch this schema-bumping branch while another active worktree still supports only v49; coordinate compatibility first. After coordination, create a task-specific root with `TLDW_VERIFY_ROOT=$(mktemp -d /private/tmp/tldw-notes-org-verify.XXXXXX)`, create `home`, `xdg-config`, `xdg-data`, `xdg-cache`, and `data` beneath it, and use `apply_patch` to create `$TLDW_VERIFY_ROOT/config.toml` containing `[paths] data_dir = "$TLDW_VERIFY_ROOT/data"` and `[model_catalog] auto_refresh_enabled = false`. Launch only the child process with `env TLDW_TEST_MODE=1 HOME="$TLDW_VERIFY_ROOT/home" XDG_CONFIG_HOME="$TLDW_VERIFY_ROOT/xdg-config" XDG_DATA_HOME="$TLDW_VERIFY_ROOT/xdg-data" XDG_CACHE_HOME="$TLDW_VERIFY_ROOT/xdg-cache" TLDW_CONFIG_PATH="$TLDW_VERIFY_ROOT/config.toml" .venv/bin/python -m tldw_chatbook.app`. Verify both effective config and database resolve under that root before connecting to the rechecked real server. Then enroll, mutate/restore organization, sync a second equally isolated profile, and inspect the product Notes tree plus conflict/retry status. Record evidence and remove the scratch root only after reviewing it; no pre-coordination app launch is allowed.

- [ ] **Self-review** every AC, durable owner, dispatcher, startup seam, migration, and envelope. Confirm no filesystem data and no partial group advertisement.

- [ ] **Close task hygiene:** check ACs, add concise notes plus ADR-102/test/live evidence, set TASK-24307 Done through the supported path, and record a lesson only if an incident earned one.

- [ ] **Commit docs/task closure**, then repeat the task/ADR collision sweep from `lessons-backlog-hygiene.md`; ADR-102 and 24307-series IDs remain provisional until that merge-time check.
