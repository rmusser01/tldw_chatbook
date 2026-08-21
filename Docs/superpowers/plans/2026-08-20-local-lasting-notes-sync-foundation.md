# Local Lasting Notes Sync Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the inert, local-only lasting Notes sync substrate required to replace the legacy single-root engine safely.

**Architecture:** One private `NotesDeviceStateStore` owns roots, bindings, journals, recovery, and migration state. Pure reconciliation consumes frozen observations. A low-level filesystem adapter preserves representation and identity; a `NotesScopeService` adapter owns note mutations. A per-root OS lease gates one application-owned runtime whose polling watcher emits hints only. All substrate remains inert until the separate atomic cutover task.

**Tech Stack:** Python 3.11, SQLite, asyncio, pathlib/os/stat/tempfile, existing `portalocker`, Textual application lifecycle, pytest with real temporary SQLite/filesystems.

---

## Governance, boundaries, and task graph

ADR required: no new ADR

ADR paths:

- `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
- `backlog/decisions/027-portable-database-note-session-coordinator.md`
- `backlog/decisions/029-local-private-data-boundary.md`
- `backlog/decisions/055-library-destructive-action-reversibility-rule.md`
- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: ADR-059/073 already decide private ownership, binding uniqueness, journaling, recovery, direction, representation, leases, migration, and server gating. A new ADR is required only if implementation adds a watcher dependency, changes deletion/conflict/backup policy, merges File Notes ownership, dual-writes legacy state, or invents a server/Sync-v2 contract.

Dependencies:

```text
TASK-19003
  -> TASK-19004 private store
       -> TASK-19005 filesystem + pure planning
       -> TASK-19006 root coordinator
       -> TASK-19008 legacy migration (also needs 19005)
TASK-19004 + TASK-19005
  -> TASK-19007 authority adapter + journal executor
TASK-19005 + TASK-19006 + TASK-19007 + TASK-19008
  -> TASK-19009 gated app runtime + watcher
```

Tasks 19006, 19007, and 19008 may run concurrently after their prerequisites because they own different files. TASK-19009 follows all four foundation pieces. Do not activate a lasting root, acquire a root lease, start watcher hints, reconcile, or remove legacy code in this plan; the runtime remains inert until TASK-19011 records the cutover marker.

## TASK-19004 — Add the private lasting-sync root registry

**Files:**

- Create: `tldw_chatbook/Notes/notes_device_state_schema.py`
- Create: `tldw_chatbook/Notes/notes_device_state_store.py`
- Create: `tldw_chatbook/Notes/notes_sync_models.py`
- Modify: `tldw_chatbook/Notes/note_import_receipts.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Create: `Tests/Notes/test_notes_device_state_store.py`
- Create: `Tests/Notes/test_notes_sync_models.py`
- Modify: `Tests/Notes/test_note_import_receipts.py`
- Regression: `Tests/Notes/test_note_import_executor.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `Tests/DB/test_private_sqlite_interop_owners.py`

- [x] Start TASK-19004 and write RED migration tests.

  Cover empty v0 -> current, a pinned historical v1 DDL fixture with seeded import receipts -> current, value-for-value receipt preservation, idempotent reopen, rollback after an injected migration failure, newer-version refusal, receipt/executor regression, and no access to the real profile path. Do not manufacture the historical fixture by stamping the evolving current bootstrap schema as v1. Expected initial failure: the shared owner/store modules do not exist.

- [x] Introduce the single private owner without changing receipt APIs.

  `NotesDeviceStateStore` owns `_connect(*, read_only=False, must_exist=False)`, `transaction(immediate=False)`, and `initialize()`. `notes_device_state_schema.py` owns the shipped v1 import-ledger DDL, `LATEST_NOTES_DEVICE_SCHEMA_VERSION`, and the v1-to-current lasting-sync migration. Preserve all existing receipt tables, columns, checks, foreign keys, unique constraints, and named indexes; a v1 reopen still repairs missing receipt indexes, while malformed or newer schemas fail closed. `NoteImportReceiptRepository(database_path)` delegates connection/migration/transactions while preserving every public method and TASK-19003's SQLite-enforced read-only lookup. Preserve TASK-19003's single-owner `{PRIVATE_FILE, READ_ONLY_URI}` policy, default `preserve_read_only_source_mode=False`, and ADR-029 permission hardening; moving ownership must not weaken it to a writable existence check or introduce a second/raw connection path.

  Keep only these lasting tables unless a test proves another is required:

  - `notes_sync_roots`
  - `notes_sync_bindings`
  - `notes_sync_operations`
  - `notes_sync_recovery`
  - `notes_sync_legacy_migrations`
  - `notes_sync_store_settings`

- [x] Define frozen validated domain models and transactional constraints.

  Add directions, root/binding/operation states, serialization profile, file identity, observations, actions, plans, and recovery admission. Reject malformed opaque IDs, absolute or parent-traversing relative paths, illegal transitions, active roots without logical folder ownership, duplicate active note scope, duplicate `(root_id, normalized_relative_path)`, and duplicate active stable identity.

- [x] Move inventory ownership and prove privacy.

  Point the existing `notes.sync_state`/C50 entry at `NotesDeviceStateStore._connect`, keep backup disabled, and update the exact private-owner inventory instead of adding another owner or row. The existing user-data-directory containment already protects this direct-child database, so do not add a redundant sensitive-path accessor. New lasting-sync public projections and diagnostics must exclude absolute paths, contents, hashes, recovery bytes, and exception messages; preserve the existing receipt records' public digest fields because the executor contract consumes them.

- [x] Run the task gate and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_device_state_store.py Tests/Notes/test_notes_sync_models.py Tests/Notes/test_note_import_receipts.py Tests/Notes/test_note_import_executor.py Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py
  git diff --check
  ```

  Commit: `feat(notes): add private lasting-sync registry`

## TASK-19005 — Plan mutation-free lasting Notes reconciliation

**Files:**

- Modify: `tldw_chatbook/Notes/sync_paths.py`
- Create: `tldw_chatbook/Notes/notes_sync_filesystem.py`
- Create: `tldw_chatbook/Notes/notes_sync_reconciler.py`
- Create: `Tests/Notes/test_notes_sync_filesystem.py`
- Create: `Tests/Notes/test_notes_sync_reconciler.py`
- Modify: `Tests/Notes/test_sync_containment.py`
- Create: `Helper_Scripts/Benchmarks/benchmark_notes_sync_reconciliation.py`

- [x] Write RED byte/identity/representation tests.

  Cover descriptor-verified reads, guarded atomic replacement, same-root move, root-symlink rejection, directory symlink traversal, hard links, aliases, invalid UTF-8, mixed newlines, BOM, LF/CRLF, final newline, supported mode preservation, identity change between observation and mutation, and Windows read-only admission.

- [x] Extend only the low-level pinned-root primitives.

  Add byte-safe observation/replacement/move operations to `PinnedSyncRoot`; retain `read_file()`/`write_text()` as legacy compatibility wrappers until cutover. `PosixNotesSyncFilesystem` composes those primitives and captures serialization/metadata/recovery. Reuse `NativeWindowsReadOnlyFilesystem` for safe Folder -> Library observation; keep Windows bidirectional writes unavailable until an equivalent guarded adapter exists.

- [x] Write the complete RED direction matrix before planner code.

  Table-drive no-change, one-sided create/update, out-of-direction change, both-side conflict, identity-proven file move, ambiguous move, note-implied filesystem move, every missing-side/deletion case, offline root, overlap, capability loss, duplicate authority, and stale observation. Assert move classification precedes missing-side logic and no case selects a global winner.

- [x] Implement `plan_reconciliation()` as a pure function.

  `notes_sync_reconciler.py` imports frozen models only. It does no SQLite, filesystem, clock, logging, Textual, or service I/O. Return safe actions, attention rows, skips, managed-placement effects, and a stable observation token. Repeated calls with the same inputs must be equal.

- [x] Measure before choosing deletion-burst and paging defaults.

  Add a deterministic representative-tree benchmark with no network or real profile access. Record plan time/memory at several file counts. Use the evidence to set the smallest bounded page size and deletion-burst grouping threshold that keeps review responsive; document the measured values in TASK-19005 Implementation Notes. Do not add a generic tuning framework.

- [x] Run the task gate and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_sync_containment.py Tests/Notes/test_notes_sync_filesystem.py Tests/Notes/test_notes_sync_reconciler.py
  ../../.venv/bin/python Helper_Scripts/Benchmarks/benchmark_notes_sync_reconciliation.py
  git diff --check
  ```

  Commit: `feat(notes): plan representation-safe reconciliation`

## TASK-19006 — Coordinate one active process per sync root

**Files:**

- Create: `tldw_chatbook/Notes/notes_sync_coordinator.py`
- Create: `Tests/Notes/test_notes_sync_coordinator.py`
- Create: `Tests/Notes/test_notes_sync_coordinator_process.py`

- [x] Write RED validation and two-process tests.

  Cover every ancestor/descendant overlap direction across active lasting roots, `app.file_notes_session_owner.current_binding()`, and `Utils.sensitive_paths.find_root_binding_conflict`; root symlinks/reparse points; missing/offline roots; passive admission; two processes racing one root; and forced process death.

- [x] Implement the minimal coordinator with existing `portalocker`.

  Add `NotesSyncRootCoordinator`, `RootLease`, `RootAdmission`, `validate_candidate_root`, `try_acquire`, `release`, and `close_admission`. Store lock files in an owner-private fixed runtime directory and name them with canonical-root digests so paths are not disclosed. The OS lock is authority; SQLite lease data is display/diagnostic state only.

- [x] Prove close-admission ordering.

  A held lease may finish the current mutation-or-durable-stage boundary. New admission fails immediately. Release occurs only after the settlement callback completes. Passive processes cannot start watchers, plans, or executors.

- [x] Run and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_sync_coordinator.py Tests/Notes/test_notes_sync_coordinator_process.py
  git diff --check
  ```

  Commit: `feat(notes): coordinate lasting-sync roots`

## TASK-19007 — Execute lasting sync through a durable recovery journal

**Files:**

- Modify: `tldw_chatbook/Notes/Notes_Library.py`
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Create: `tldw_chatbook/Notes/notes_sync_authority.py`
- Create: `tldw_chatbook/Notes/notes_sync_executor.py`
- Modify: `tldw_chatbook/Notes/notes_device_state_store.py`
- Modify: `tldw_chatbook/config.py`
- Create: `Tests/Notes/test_notes_sync_authority.py`
- Create: `Tests/Notes/test_notes_sync_executor.py`
- Modify: `Tests/Notes/test_notes_scope_service_folders.py`
- Create: `Helper_Scripts/Benchmarks/benchmark_notes_sync_recovery_capacity.py`

- [ ] Write RED service-boundary tests.

  Add bounded `get_note_summaries(user_id, note_ids)` and `get_note_sync_summaries(...)`, plus a `reconcile_note_folder_owner_memberships(...)` wrapper over the existing managed-membership repository. Assert one bounded query, optimistic version propagation, and owner-isolated membership changes. `LocalNotesSyncAuthority` may call only `NotesScopeService`.

- [ ] Write a fault-injection test for every durable stage.

  Pin the order: revalidate -> admit recovery/intent -> mutate first authority -> persist -> mutate counterpart -> persist -> update binding/membership -> verify -> complete. Inject failure/cancellation after each boundary, reopen the real temporary private DB, and assert matching observations resume deterministically while changed observations become attention.

- [ ] Add journal/recovery store operations and the executor.

  Implement `admit_operation`, `advance_operation`, `mark_needs_attention`, `complete_operation`, `list_incomplete_operations`, `load_recovery`, `expire_recovery`, and capacity accounting. Recovery admission and journal intent share one private transaction. Pending, unresolved, and Undo-eligible bytes cannot be evicted; normal completed recovery retention is 30 days.

- [ ] Establish a measured recovery capacity.

  Benchmark representative replacement and managed-move recovery payloads using temporary files. Record the largest representative operation and the safety factor in task notes, then expose one bounded `get_notes_sync_recovery_capacity_bytes()` setting with that documented default. Tests inject smaller capacities to prove rejection occurs before mutation. Do not add multiple quotas or adaptive policy.

- [ ] Prove privacy and cancellation semantics.

  Capture logs around every failure stage and assert no content, absolute path, hash, recovery bytes, credential, or raw exception text. Cancellation is delayed only through the current mutation-or-durable-stage boundary and then re-delivered.

- [ ] Run and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_sync_authority.py Tests/Notes/test_notes_sync_executor.py Tests/Notes/test_notes_scope_service_folders.py Tests/Notes/test_notes_device_state_store.py
  ../../.venv/bin/python Helper_Scripts/Benchmarks/benchmark_notes_sync_recovery_capacity.py
  git diff --check
  ```

  Commit: `feat(notes): execute lasting sync durably`

## TASK-19008 — Migrate legacy Notes sync into paused candidates

**Files:**

- Create: `tldw_chatbook/Notes/notes_sync_legacy.py`
- Create: `Tests/Notes/test_notes_sync_legacy_migration.py`

- [x] Write RED migration matrices.

  Cover multiple roots, config-only and row-only roots, missing roots, overlap, duplicate identity/path, out-of-root relative paths, unsafe rows, invalid values, repeated migration, and a crash between private writes. Assert every output root is paused and no mock note/file/folder/watcher mutation is called.

- [x] Implement snapshot, pure plan, and one private write.

  Read legacy `[notes]` keys, per-note disk metadata columns, and `sync_sessions` as historical evidence only. Create one paused candidate per distinct canonical safe root, recognizable candidate bindings, a bounded report, and a source fingerprint in `notes_sync_legacy_migrations`. Do not carry `newer_wins`, `disk_wins`, `db_wins`, or `auto_sync` into new policy.

- [x] Require a fresh check before activation.

  Candidate roots have no watcher/lease admission and cannot become active until current observations pass TASK-19005 planning and a user-approved activation in the UI/cutover tasks. Absence in a missing or retargeted root never becomes deletion.

- [x] Run and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_sync_legacy_migration.py Tests/Notes/test_notes_device_state_store.py
  git diff --check
  ```

  Commit: `feat(notes): migrate legacy sync as paused candidates`

## TASK-19009 — Build gated lasting-sync application runtime

**Files:**

- Create: `tldw_chatbook/Notes/notes_sync_watcher.py`
- Create: `tldw_chatbook/Notes/notes_sync_runtime.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Notes/test_notes_sync_watcher.py`
- Create: `Tests/Notes/test_notes_sync_runtime.py`
- Create: `Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py`

- [ ] Write RED hint-only watcher tests.

  `PollingNotesSyncWatcher` emits root IDs, coalesces duplicate hints, handles a missing root, and never imports or calls the planner/executor/filesystem. Use an injected clock/interval in tests; production uses one simple interval, not a scheduler framework.

- [ ] Write RED inert-startup, post-cutover, manual, and shutdown tests.

  Before cutover, test that construction/start publishes `Awaiting cutover` but acquires no root lease, starts no watcher, plans nothing, executes nothing, and rejects activation/manual apply. Seed a marker while the code-owned gate remains false and prove it is still inert. Only when both an injected code-owned admission and marker are true may tests exercise passive/offline/paused/attention gates, complete startup reconciliation, manual reviewed check token validation, safe automatic one-sided apply, stale-plan supersession, durable state publication, and cancellation-resistant shutdown.

- [ ] Implement one app-owned runtime facade.

  Add `NotesSyncRuntimeOwner` and `build_notes_sync_runtime_owner`. The builder receives one private code-owned `cutover_admitted` value; `app.py` passes `False` in this task and TASK-19011 removes/flips that temporary gate only after legacy production paths are deleted. This is not user configuration. UI-facing methods structurally satisfy TASK-19010's narrow port: `snapshot`, `check_root`, `apply_reviewed`, `request_sync_now`, `pause_root`, `resume_root`, `activate_root`, `retarget_root`, `disconnect_root`, and `shutdown`. Watcher callbacks call `schedule_hint(root_id)` only.

  Startup before cutover: initialize store -> run the TASK-19008 migrator idempotently -> require both code-owned admission and cutover marker -> publish inert state and return without root leases or watchers when either is absent.

  Startup/enable after cutover: verify marker -> classify incomplete journals under leases -> claim eligible active roots -> full reconcile -> start hints.

  Shutdown: close admission -> stop hints -> settle/journal current stage -> release leases -> close resources.

- [ ] Wire `TldwCli` lifecycle without Library ownership.

  Construct the runtime once after `notes_scope_service`; start it from app mount. Add `_shutdown_notes_sync_runtime()` and call it from `_shutdown_app_owned_lifecycles()` before generic teardown and before File Notes owner release. The Library screen must not construct, start, own, or directly orchestrate it. Starting the object before cutover is safe because its RED contract forbids every mutation-capable action until the marker exists.

- [ ] Run and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_sync_watcher.py Tests/Notes/test_notes_sync_runtime.py Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py Tests/App/test_app_lifecycle_events.py
  git diff --check
  ```

  Commit: `feat(notes): add gated lasting-sync runtime`

## Foundation integration gate

- [ ] Run all foundation suites together using only temporary databases and roots.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_device_state_store.py Tests/Notes/test_notes_sync_models.py Tests/Notes/test_sync_containment.py Tests/Notes/test_notes_sync_filesystem.py Tests/Notes/test_notes_sync_reconciler.py Tests/Notes/test_notes_sync_coordinator.py Tests/Notes/test_notes_sync_coordinator_process.py Tests/Notes/test_notes_sync_authority.py Tests/Notes/test_notes_sync_executor.py Tests/Notes/test_notes_sync_watcher.py Tests/Notes/test_notes_sync_runtime.py Tests/Notes/test_notes_sync_legacy_migration.py Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py
  git diff --check
  ```

- [ ] Confirm no code in this plan replaces the toolbar, deletes legacy modules, activates a migrated root, or enables server-backed sync. Close each Backlog task independently with exact evidence and ADR-059/073 notes.
