# Private SQLite Owner and Backup Lifecycle Implementation Plan

> **Current-dev reconciliation (2026-07-27):** The later eval-console
> retirement deleted `tldw_chatbook/Event_Handlers/eval_events.py`. Its
> historical parent-creator row and owner policy are therefore intentionally
> absent from the live registry; only `EvaluationOrchestrator` remains an eval
> database-path owner.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan one task at a
> time. Every production slice is red-green-refactor TDD and receives an
> independent review before the next slice begins.

**Goal:** Route every production SQLite connection, sidecar, backup, and
restore owner through one checked private-storage boundary so private databases
are created as `0600`, eligible existing artifacts are hardened before SQLite
opens them, and unsafe custom namespaces fail closed.

**Architecture:** A dependency-light `DB/private_sqlite.py` module owns the
checked owner registry, target classification, SQLite connection preparation,
sidecar preflight, read-only URI construction, and SQLite backup/copy
operations. It delegates lexical no-follow filesystem work to
`Utils/private_paths.py`. Production callers retain their existing row
factories, pragmas, connection lifetimes, and domain-specific exception
translation, but no longer resolve a selected path, create arbitrary parents,
or call SQLite directly. The default application data and backup directories
are explicitly secured as `0700`; a custom database parent must already exist
and pass the trusted, non-attacker-writable namespace check.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `os`, `stat`, `urllib.parse`,
`ast`, existing private-path primitives, pytest.

## Verified Baseline

- `rg -n 'sqlite3\.connect' tldw_chatbook --glob '*.py'` returns exactly 31
  direct connection sites in 18 production modules.
- Three production methods call `sqlite3.Connection.backup()` directly:
  ChaChaNotes, Media, and Prompts.
- `UI/Tools_Settings_Window.py` has six database `shutil.copy2()` sites: three
  bulk backups, one single backup, one pre-restore backup, and one restore.
- `Widgets/Tamagotchi/tamagotchi_storage.py` also copies a JSON adapter file.
  That is not a SQLite artifact and is an explicit, justified exclusion from
  this task.
- `Client_Media_DB_v2.create_incremental_backup()` and
  `create_automated_backup()` are no-op placeholders that create no artifact;
  inventory them as non-owner exclusions rather than pretending they are
  migrated backup paths.
- There are no production `aiosqlite.connect` owners.
- `Client_Media_DB_v2.check_database_integrity()` is the one existing
  read-only URI owner. The three browser-cookie clones are read-only consumers
  and will be moved into per-clone `0700` temporary directories before using
  the same supported URI mode.
- A local behavioral probe under `umask(000)` reproduced `0644` main, WAL,
  SHM, and rollback-journal files with raw SQLite creation. Exclusively
  pre-creating the main database as `0600` made all three SQLite-created
  sidecar forms `0600`; the implementation still keeps this as a regression
  test rather than relying on the probe.

## Global Constraints

- Preserve lexical selected paths until link and namespace validation
  completes. Do not call `Path.resolve()` in database target selection,
  constructors, same-target checks, connection preparation, backup, or restore.
- Keep `Utils/private_paths.py` stdlib-only and free of config, logging,
  database, Textual, or application imports.
- Keep `DB/private_sqlite.py` free of config, UI, and domain database imports.
- A writable file-backed database requires an existing trusted parent where
  unauthorized users cannot replace the database or pre-create SQLite
  sidecars. A shared sticky parent is insufficient for writable connections,
  even when the main file already exists.
- No SQLite file is opened directly inside a shared-sticky parent, including
  with a read-only URI: SQLite may create or write `-shm` during `mode=ro`.
  The seam always verifies the selected SQLite parent with
  `allow_shared_sticky=False`.
- New database parents are never created by the connection seam. Callers may
  create or harden only an explicitly application-owned default data or backup
  directory.
- Pre-create a missing writable database exclusively as `0600`; harden an
  eligible existing database and existing `-wal`, `-shm`, and `-journal`
  sidecars before calling SQLite.
- On Windows, successful file-backed preparation emits at most one
  `SQLitePrivacyUnverifiedWarning` per literal owner ID per process. Exact
  `:memory:` emits no privacy warning. This bounded warning is the observable
  `unverified_platform` contract until native ACL verification exists.
- Do not pre-create missing sidecars. The trusted writable parent and private
  main-file mode are the creation boundary; real WAL, SHM, and rollback journal
  modes must be verified under a permissive process umask.
- Reject final or intermediate symlinks, non-regular leaves, hard-linked
  private files, wrong-owner files, writable shared parents, unsupported
  writable SQLite URI input, and existing unsafe sidecars before raw SQLite is
  invoked.
- Support only the existing `:memory:` token and path-based read-only
  `mode=ro` URI contract. Do not accept arbitrary caller-supplied SQLite URI
  strings or a pass-through `uri=True` escape hatch.
- Percent-encode read-only file URIs so spaces, `?`, `#`, and non-ASCII path
  characters retain path identity.
- On Windows, retain the ADR-029 `unverified_platform` posture and current
  functional behavior. Do not claim ACL privacy.
- A failed new database or backup operation may retain an owner-only partial
  file. Do not name-unlink after a failure because the selected entry could
  have been replaced.
- Preserve current public exception/boolean contracts at domain call sites.
  The common seam raises structured failures; existing wrappers may translate
  them as they do other database errors.
- Do not open, move, delete, or inspect the contents of
  `openai-api-key.txt` or `moonshot-api-key.txt`.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already requires a checked inventory, pre-created private
SQLite files, private sidecars/backups, trusted custom namespaces, and
fail-closed database behavior. This task implements that accepted boundary
without changing it, so no new ADR is needed.

## File Map

- Create `tldw_chatbook/DB/private_sqlite.py`: owner registry, target
  classification, private connection seam, sidecar preflight, read-only URI
  construction, and centralized SQLite backup/copy operations.
- Modify `tldw_chatbook/Utils/private_paths.py`: add a read-only trusted
  directory verification primitive without changing custom directory modes.
- Create `Tests/DB/test_private_sqlite.py`: seam behavior, modes, sidecars,
  URIs, unsafe targets, and backup/copy behavior.
- Create `Tests/DB/test_private_sqlite_inventory.py`: checked inventory,
  literal owner IDs, raw-owner source guard, and behavioral registry matrix.
- Create `backlog/docs/sqlite-private-owner-inventory.md`: human-readable
  classification of all 31 connection sites, nine database backup/restore
  sites, every production DB-parent creator, and explicit exclusions.
- Modify `tldw_chatbook/config.py`: secure default data directories and retain
  lexical custom database paths.
- Create `Tests/test_database_path_privacy.py`: default/custom directory and
  database path-selection regressions.
- Modify `tldw_chatbook/Evals/eval_orchestrator.py`,
  `tldw_chatbook/Event_Handlers/eval_events.py`, `tldw_chatbook/app.py`,
  `tldw_chatbook/Notes/Notes_Library.py`, and the executable example in
  `tldw_chatbook/DB/Sync_Client.py`, plus
  `tldw_chatbook/runtime_policy/server_parity_state.py`: remove or secure
  DB-parent creation that occurs outside the direct connection owners.
- Modify all 18 currently direct connection-owner modules:
  `DB/base_db.py`, `DB/ChaChaNotes_DB.py`, `DB/Client_Media_DB_v2.py`,
  `DB/Evals_DB.py`, `DB/Library_Ingest_Jobs_DB.py`, `DB/Prompts_DB.py`,
  `DB/RAG_Indexing_DB.py`, `DB/search_history_db.py`,
  `Kanban_Interop/local_kanban_db.py`,
  `Notifications/client_notifications_db.py`,
  `Notifications/event_state_repository.py`,
  `Research_Interop/local_research_service.py`,
  `Sync_Interop/notes_mirror.py`,
  `Sync_Interop/sync_state_repository.py`, `UI/Tools_Settings_Window.py`,
  `Web_Scraping/cookie_scraping/cookie_cloner.py`,
  `Widgets/Tamagotchi/tamagotchi_storage.py`, and
  `Writing_Interop/local_writing_service.py`.
- Modify focused existing tests for each migrated owner only where the new
  trusted-parent contract intentionally changes setup or assertions.
- Modify
  `backlog/tasks/task-489 - Apply-private-storage-boundary-to-every-SQLite-owner-and-backup.md`
  for plan linkage, implementation notes, evidence, and completion.

---

### Task 1: Check in the complete owner and backup inventory

**Files:**

- Create: `tldw_chatbook/DB/private_sqlite.py`
- Create: `Tests/DB/test_private_sqlite_inventory.py`
- Create: `backlog/docs/sqlite-private-owner-inventory.md`

**Interfaces:**

- Produces `SQLiteTargetKind`, immutable `SQLiteOwnerPolicy`, and
  `SQLITE_OWNER_REGISTRY`.
- Does not yet expose a connection function.

- [ ] **Step 1: Write the inventory document before changing call sites**

  Add stable IDs `C01` through `C31` for every verified direct connection,
  `B01` through `B09` for every database backup/restore operation, and `P`
  rows for every production DB-parent creator, including creators outside the
  18 direct-owner modules. Each row records the production module, containing
  symbol, owner ID, classification, read/write intent, and migration
  disposition. Add an explicit exclusion section for the Tamagotchi JSON
  backup, the two no-op Media backup placeholders, and the absence of any
  `aiosqlite` owner.

- [ ] **Step 2: Write failing registry and inventory-consistency tests**

  The tests parse the Markdown tables and assert:

  - exactly 31 unique `C` rows and nine unique `B` rows;
  - every row names a registered literal owner ID;
  - every current production module containing `sqlite3.connect` is represented;
  - every checked DB-parent creator has a `P` row and disposition;
  - each owner declares only `private_file`, `memory`, or `read_only_uri`;
  - backup/restore owners explicitly opt into centralized backup behavior;
  - registry module paths exist and no policy has an empty reason.

- [ ] **Step 3: Run the tests and confirm the missing registry failure**

  Run:

  ```bash
  python3 -m pytest -q Tests/DB/test_private_sqlite_inventory.py
  ```

  Expected: collection fails because `tldw_chatbook.DB.private_sqlite` does not
  exist.

- [ ] **Step 4: Add the immutable registry**

  Define policies for the 31 connection sites and nine backup sites using
  stable owner IDs. Consolidation is allowed only when multiple legacy call
  sites are the same production owner, such as the six `SQLiteStorage`
  methods or the three bulk Settings backups. The inventory retains every
  legacy row even when several rows map to one policy.

- [ ] **Step 5: Re-run the inventory tests**

  Expected: all pass while the production call sites remain unchanged.

- [ ] **Step 6: Commit the checked baseline**

  ```bash
  git add tldw_chatbook/DB/private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    backlog/docs/sqlite-private-owner-inventory.md
  git commit -m "docs(security): inventory sqlite storage owners"
  ```

---

### Task 2: Build the private SQLite connection and sidecar seam

**Files:**

- Modify: `tldw_chatbook/Utils/private_paths.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Create: `Tests/DB/test_private_sqlite.py`
- Modify: `Tests/Utils/test_private_paths.py`

**Interfaces:**

- Produces
  `verify_trusted_directory(path, *, allow_shared_sticky) -> PrivatePathResult`.
- Produces
  `connect_private_sqlite(owner_id, database, *, read_only=False, **kwargs)`.
- Keeps the one raw `sqlite3.connect` call private to
  `DB/private_sqlite.py`.

- [ ] **Step 1: Add red trusted-directory tests**

  Cover an owned non-writable parent, an existing shared-sticky parent with
  `allow_shared_sticky=True`, rejection with `allow_shared_sticky=False`,
  intermediate and final directory symlinks, wrong-owner simulation,
  group/world-writable non-sticky parents, missing parents, no chmod of custom
  directories, descriptor closure, descriptor/entry identity replacement
  during the walk, a forced final-directory postcondition failure, and Windows
  `unverified_platform`.

- [ ] **Step 2: Add red connection classification tests**

  Cover `:memory:`, ordinary `str` and `Path` file targets, path-like objects,
  a read-only request, invalid owner IDs, owner/kind mismatches, NUL paths,
  caller-supplied `uri=True`, and arbitrary `file:` URI strings.

- [ ] **Step 3: Add red private file and pre-SQLite failure tests**

  Under POSIX, verify:

  - first creation is `0600` under `umask(000)`;
  - an existing `0644` database is `0600` when the raw connect stub runs;
  - an owner-owned `0400` database is pinned `O_RDONLY`, hardened to `0600`,
    reopened `O_RDWR`, and identity-revalidated before raw connect;
  - missing, symlinked, non-regular, multiply-linked, and wrong-owner targets
    fail before the raw connect stub;
  - writable targets in shared sticky or non-sticky writable parents fail
    before raw connect;
  - replacement of the target between no-follow classification and guarded
    open is rejected by identity/postcondition checks before raw connect;
  - a simulated raw connect failure retains only a `0600` residue;
  - no parent is silently created or chmodded.

- [ ] **Step 4: Add red sidecar preflight tests**

  Parameterize `-wal`, `-shm`, and `-journal`: eligible `0644` sidecars are
  hardened before raw connect; owner-owned `0400` sidecars follow the same
  read-pin, harden, writable-reopen, and identity-revalidation order; symlink,
  hardlink, non-regular, wrong-owner, or replacement/postcondition-failure
  sidecars block raw connect. Missing sidecars are not pre-created.

- [ ] **Step 5: Add red real SQLite mode tests**

  With `umask(000)` and a real database:

  - WAL mode creates private main, WAL, and SHM files while the connection is
    open;
  - DELETE journal mode creates a private rollback journal while a write
    transaction is open;
  - all artifacts are regular current-user-owned files with mode `0600`;
  - reopening hardens eligible historical sidecars before use.

- [ ] **Step 6: Add red read-only URI tests**

  Verify read-only access works for filenames containing spaces, `?`, `#`, and
  Unicode; writes fail with SQLite read-only errors; a missing target, an
  unsafe existing target, and every target directly inside a shared-sticky
  parent fail closed; the caller cannot append URI query parameters. This
  stricter namespace rule is required because SQLite may create or write a
  missing `-shm` file during a `mode=ro` WAL open.

  Add pure URI-builder fixtures for Windows drive-letter and UNC paths,
  including spaces, `?`, `#`, and Unicode. On Windows CI, run the functional
  read-only and `:memory:` cases: file-backed preparation reports
  `unverified_platform` through a warning deduplicated to one emission per
  literal owner ID per process, while exact memory remains filesystem-free and
  warning-free. On non-Windows CI, test the deterministic URI builder without
  pretending the POSIX host opened a Windows path.

- [ ] **Step 7: Run the focused red suite**

  ```bash
  python3 -m pytest -q \
    Tests/Utils/test_private_paths.py \
    Tests/DB/test_private_sqlite.py
  ```

  Expected: new tests fail for missing public directory verification and
  connection seam behavior.

- [ ] **Step 8: Implement trusted directory verification**

  Reuse descriptor traversal and the existing owner/write/sticky model.
  Verification must not create or chmod a custom directory. Return a bounded
  structured posture and close every descriptor on all paths.

- [ ] **Step 9: Implement target preparation and raw connection**

  Infer the target kind, enforce the owner policy, validate the writable parent,
  exclusively create or harden the main file, preflight existing sidecars, and
  only then call the module-private raw SQLite function. Build read-only URIs
  internally from a lexical path with correct percent encoding. Existing
  owner-only but read-only (`0400`) artifacts must be pinned with `O_RDONLY`,
  hardened, then reopened writable with identity revalidation where the caller
  needs write access. Do not open SQLite databases directly in shared-sticky
  parents.

- [ ] **Step 10: Run the focused suite and refactor**

  Expected: all private-path and SQLite seam tests pass. Remove duplication
  without weakening the explicit ordering assertions.

- [ ] **Step 11: Commit the seam**

  ```bash
  git add tldw_chatbook/Utils/private_paths.py \
    tldw_chatbook/DB/private_sqlite.py \
    Tests/Utils/test_private_paths.py \
    Tests/DB/test_private_sqlite.py
  git commit -m "feat(security): add private sqlite connection seam"
  ```

---

### Task 3: Secure default directories and preserve lexical DB selection

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Utils/path_validation.py`
- Modify: `tldw_chatbook/Evals/eval_orchestrator.py`
- Modify: `tldw_chatbook/Event_Handlers/eval_events.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Notes/Notes_Library.py`
- Modify: `tldw_chatbook/DB/Sync_Client.py`
- Modify: `tldw_chatbook/runtime_policy/server_parity_state.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `Tests/conftest.py`
- Create: `Tests/test_database_path_privacy.py`
- Modify: adjacent config path tests where their old canonical-path assertion
  conflicts with ADR-029.
- Modify/Create: focused eval orchestration, app startup, Notes service, and
  Sync Client example/source-contract tests.

**Interfaces:**

- `get_user_data_dir()` returns the verified lexical application-owned user
  directory.
- Existing `get_*_db_path()` public functions keep their names and `Path`
  return type.

- [ ] **Step 1: Add red default data-directory tests**

  Cover creation as `0700` under `umask(000)`, hardening an eligible existing
  default user directory, intermediate symlink rejection, and failure without
  reporting a usable directory when the default namespace is unsafe.

- [ ] **Step 2: Add red custom data-directory tests**

  A configured custom base must already exist and remain mode-unchanged. Its
  Chatbook-owned per-user child is created/hardened as `0700`. Reject a missing
  custom base, unsafe writable base, and symlinked component without creating
  anything elsewhere.

- [ ] **Step 3: Add red custom database path tests**

  For all eleven `get_*_db_path()` helpers, assert that a selected path through
  a symlink remains lexically selected and is not canonicalized. Retain NUL and
  existing explicit dangerous-input validation. Assert default paths remain
  direct children of the secured user data directory.

- [ ] **Step 4: Add red non-owner DB-parent tests**

  Cover the evaluation orchestrator and eval event singleton, Prompts app
  startup, NotesInteropService, the Sync Client executable example/source
  contract, and the server-parity event/sync repository builder. Defaults must
  use an explicitly secured application directory. Custom parents must already
  exist and remain mode-unchanged. No caller may recursively create an
  arbitrary selected DB parent with the process umask. Also pin removal of
  config-load directory side effects for the unconsumed `DATABASE_URL` and
  unreachable `USER_DB_BASE_DIR` compatibility branches; environment-selected
  server paths must not create directories merely because Chatbook loaded
  settings.

- [ ] **Step 5: Run and confirm failures**

  ```bash
  python3 -m pytest -q \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/test_database_path_privacy.py \
    Tests/Library/test_library_collections_config.py \
    Tests/Scheduling/test_scheduled_tasks_db.py \
    Tests/Evals/test_eval_orchestrator.py \
    Tests/Notes/test_notes_library_unit.py \
    Tests/Media_DB/test_sync_client.py \
    Tests/RuntimePolicy/test_server_parity_state.py
  ```

- [ ] **Step 6: Implement directory and lexical path policy**

  Use the private-path primitives already accepted by ADR-029. Do not chmod a
  configured custom base. Replace early `.resolve()` calls in DB path helpers
  with lexical normalization after existing input validation. Route the
  default eval path through the secured user-data helper, make the event
  singleton delegate path selection to the orchestrator, remove the redundant
  Prompts and unused Notes parent creation, and explicitly secure or fail
  closed in the Sync Client example. Preserve the server-parity builder's
  lexical `data_dir` and require its selected parent to use the same default or
  custom namespace policy. Remove the stale config-load
  `DATABASE_URL`/`USER_DB_BASE_DIR` directory-creation branches, while leaving
  any compatibility values themselves unchanged.

  The default application data base honors `XDG_DATA_HOME/tldw_cli` when
  `XDG_DATA_HOME` is set and otherwise retains the existing
  `~/.local/share/tldw_cli` fallback. The root test bootstrap sets a dedicated
  private `XDG_DATA_HOME` before project imports so collection cannot touch a
  developer's real data directory. Preserve the exact `:memory:` token at the
  eval-orchestrator boundary; broader URI normalization remains Task 4.

  Custom DB selection uses a non-resolving mode of the existing input
  validator so even an existing symlinked leaf is not probed or canonicalized
  before the private SQLite seam. Preserve the scheduled-tasks helper's
  historical validation order so a raw `~/...` value remains rejected; other
  helpers retain their previous expand-before-validation behavior.

  In the checked inventory, change P01-P03 and P23-P28 from `current` to
  `migrated` in the same commit. The inventory test must then prove every
  listed legacy creator-call anchor is absent.

- [ ] **Step 7: Re-run focused config/startup tests**

  Expected: all selected tests pass and no custom parent mode changes.

- [ ] **Step 8: Commit path policy**

  ```bash
  git add Docs/superpowers/plans/2026-07-23-private-sqlite-owner-lifecycle.md \
    tldw_chatbook/config.py \
    tldw_chatbook/Utils/path_validation.py \
    tldw_chatbook/Evals/eval_orchestrator.py \
    tldw_chatbook/Event_Handlers/eval_events.py \
    tldw_chatbook/app.py \
    tldw_chatbook/Notes/Notes_Library.py \
    tldw_chatbook/DB/Sync_Client.py \
    tldw_chatbook/runtime_policy/server_parity_state.py \
    backlog/docs/sqlite-private-owner-inventory.md \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/test_database_path_privacy.py \
    Tests/Library/test_library_collections_config.py \
    Tests/Scheduling/test_scheduled_tasks_db.py \
    Tests/Evals/test_eval_orchestrator.py \
    Tests/Notes/test_notes_library_unit.py \
    Tests/Media_DB/test_sync_client.py \
    Tests/RuntimePolicy/test_server_parity_state.py \
    Tests/conftest.py
  git commit -m "fix(security): secure database path selection"
  ```

---

### Task 4: Migrate core database connection owners

**Files:**

- Modify: `tldw_chatbook/DB/base_db.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/DB/Evals_DB.py`
- Modify: `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- Modify: `tldw_chatbook/DB/RAG_Indexing_DB.py`
- Modify: `tldw_chatbook/DB/search_history_db.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite.py`
- Modify: focused tests in `Tests/ChaChaNotesDB/`, `Tests/Media_DB/`,
  `Tests/Prompts_DB/`, `Tests/DB/`, and `Tests/Evals/`.
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`

- [ ] **Step 1: Add red owner-level privacy regressions**

  For every module in this slice, exercise a real first connection under
  `umask(000)` and an eligible existing `0644` database. Assert `0600` before
  schema work, correct row factory/pragmas, and unchanged in-memory behavior
  where supported.

- [ ] **Step 2: Add red unsafe-parent and lexical-symlink regressions**

  Each constructor must fail before SQLite for a missing arbitrary parent,
  unsafe writable parent, or symlinked target/parent. A safe existing custom
  parent remains mode-unchanged.

- [ ] **Step 3: Run the focused red tests**

  Use the smallest named tests first, then:

  ```bash
  python3 -m pytest -q \
    Tests/DB \
    Tests/Evals/test_evals_db.py \
    Tests/ChaChaNotesDB/test_chachanotes_db.py \
    Tests/Media_DB/test_media_db_v2.py \
    Tests/Prompts_DB/test_prompts_db_pytest.py
  ```

- [ ] **Step 4: Migrate constructors and connections**

  Remove early path resolution and arbitrary recursive parent creation. Route
  every connection through its literal registered owner ID. Preserve
  thread-local reuse, liveness probing, connection arguments, row factories,
  pragmas, schema initialization, and domain exception translation. Route the
  three ChaChaNotes/Media/Prompts backup-target connections through their
  registered private-file owner IDs in this task as well, while leaving their
  direct `.backup()` calls for Task 6. This ordering ensures the Task 5
  raw-connect guard can pass.

  Optional SQLite sidecars use a bounded four-attempt full-generation
  revalidation loop for legitimate concurrent rollback-journal churn. A
  missing current name is treated as an absent optional sidecar; a present
  eligible current-user regular single-link candidate restarts the complete
  stat/open/fstat/harden/writable-reopen/postcondition cycle. Unsafe
  replacements and exhausted churn fail closed. Main-database identity changes
  never retry. This implements ADR-029's unauthorized-other-UID privacy
  boundary and does not claim same-UID process isolation.

  ADR required: no. This is an internal lifecycle correction required to
  implement ADR-029 without changing its accepted threat model.

  In the checked inventory, change P06-P15 from `current` to `migrated` in the
  same commit. The inventory test must prove the removed constructor and
  backup-parent calls remain absent.

- [ ] **Step 5: Re-run the focused core suites**

  Expected: all selected tests pass with no direct `sqlite3.connect` remaining
  in these modules.

- [ ] **Step 6: Commit core owners**

  ```bash
  git add Docs/superpowers/plans/2026-07-23-private-sqlite-owner-lifecycle.md \
    tldw_chatbook/DB \
    backlog/docs/sqlite-private-owner-inventory.md \
    Tests/DB Tests/Evals Tests/ChaChaNotesDB Tests/Media_DB Tests/Prompts_DB
  git commit -m "fix(security): migrate core sqlite owners"
  ```

  Stage only files actually changed for this task; do not stage unrelated
  files under those directories.

---

### Task 5: Migrate interop, UI maintenance, cookie, and widget owners

**Files:**

- Modify: `tldw_chatbook/Kanban_Interop/local_kanban_db.py`
- Modify: `tldw_chatbook/Notifications/client_notifications_db.py`
- Modify: `tldw_chatbook/Notifications/event_state_repository.py`
- Modify: `tldw_chatbook/Research_Interop/local_research_service.py`
- Modify: `tldw_chatbook/Sync_Interop/notes_mirror.py`
- Modify: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Modify: `tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py`
- Modify: `tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py`
- Modify: `tldw_chatbook/Writing_Interop/local_writing_service.py`
- Modify/Create: focused owner tests in the corresponding `Tests/` directories.
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`

- [ ] **Step 1: Add red real-owner tests**

  Cover Kanban, local writing, local research path mode, NotesMirror memory and
  file modes, the three persistent-memory repositories, and SQLiteStorage CRUD.
  Tests use existing safe parents; separate tests prove missing/unsafe custom
  parents fail.

- [ ] **Step 2: Add red read-only caller tests**

  Assert Media integrity, Settings integrity/schema reads, and Chrome/Firefox/
  Edge cookie clones request the registered read-only contract. Use real
  read-only SQLite files where practical and stubs only for OS/browser
  decryption dependencies. Confirm cookie clones live inside a verified `0700`
  temporary directory, their database files remain `0600`, and they cannot be
  written through their connections. Do not place a SQLite file directly in
  the system shared-sticky temporary directory.

- [ ] **Step 3: Add red Settings vacuum tests**

  Vacuum is writable and must use the writable registered owner; integrity and
  schema version are read-only. Unsafe database paths notify failure without
  raw SQLite access.

- [ ] **Step 4: Migrate the remaining 18-module inventory**

  Consolidate repeated `SQLiteStorage` opens behind one private instance
  helper. Remove arbitrary parent creation. Keep persistent `:memory:`
  connection reuse exactly as before. Change cookie copies to internally built
  read-only URI connections inside per-clone private temporary directories.

  In the checked inventory, change P16-P19 from `current` to `migrated` in the
  same commit. Keep P20-P22 `current` until the backup slice removes those
  legacy directory creators.

- [ ] **Step 5: Add the raw connection source guard**

  Parse production Python ASTs and recognize both `import sqlite3 as ...` and
  `from sqlite3 import connect as ...`. Also reject
  `sqlite3.dbapi2.connect(...)`, direct `sqlite3.Connection(...)`
  construction, and simple rebound aliases such as
  `raw_connect = sqlite3.connect; raw_connect(...)`. Add negative source
  fixtures for every forbidden spelling. Dynamic `getattr`/runtime code
  generation is explicitly outside this repository source guard. Assert raw
  SQLite construction occurs only in `DB/private_sqlite.py`. Parse
  approved-seam calls and require a literal registered owner ID whose
  registered module matches the caller.

- [ ] **Step 6: Run focused owner and guard suites**

  ```bash
  python3 -m pytest -q \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/Kanban \
    Tests/Writing_Interop \
    Tests/Research \
    Tests/Research_Interop \
    Tests/Sync_Interop \
    Tests/Subscriptions/test_client_notifications_db.py \
    Tests/UI/test_tools_settings_window.py \
    Tests/Web_Scraping \
    Tests/Widgets
  ```

  If `Tests/Widgets` does not exist, run the new Tamagotchi test file directly.

- [ ] **Step 7: Commit remaining connection owners**

  ```bash
  git add tldw_chatbook/Kanban_Interop \
    tldw_chatbook/Notifications \
    tldw_chatbook/Research_Interop \
    tldw_chatbook/Sync_Interop \
    tldw_chatbook/UI/Tools_Settings_Window.py \
    tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py \
    tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py \
    tldw_chatbook/Writing_Interop \
    Tests/DB/test_private_sqlite_inventory.py \
    backlog/docs/sqlite-private-owner-inventory.md
  git commit -m "fix(security): migrate remaining sqlite owners"
  ```

  Add only the focused test files actually modified or created.

---

### Task 6: Centralize backup, bulk backup, and restore behavior

**Files:**

- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Modify: focused ChaChaNotes, Media, Prompts, and Settings tests.
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`

**Interfaces:**

- Produces
  `backup_connection_to_private(owner_id, source_connection, source_database,
  target)`.
- Produces
  `copy_private_sqlite(owner_id, source_path, target_path)` for read-only
  source to writable private destination, used by Settings backup and restore.
- Produces a restore-specific operation that enforces the live-connection
  quiescence contract and performs pre-restore backup plus restore without a
  raw file copy.

**Implementation-review note:** A WAL connection that has already queried can
block the required WAL-to-DELETE quiescence probe even when Python reports
`in_transaction == False`. The safe helper therefore supports live restore
only when SQLite can establish and retain exclusivity; otherwise it fails
promptly and reports that offline maintenance is required. Adding an
application-wide database maintenance gate or pre-initialization pending
restore belongs with the later application-state decomposition. For successful
restores, the helper retains the exclusive lock, performs the transactional
page backup, and then reasserts and verifies the destination's original
DELETE/WAL mode because a source backup may carry a different mode. A
post-copy mode or source-identity failure rolls the live database back from the
private pre-restore snapshot. If that recovery itself fails, callers receive a
distinct indeterminate-state error with the live and snapshot paths and an
explicit warning not to retry automatically.

- [ ] **Step 1: Add red centralized backup tests**

  Verify new and eligible existing backup targets become `0600` under
  `umask(000)`, data is transactionally copied through SQLite's backup API,
  unsafe/missing parents and symlink/hardlink targets fail before backup,
  same lexical source/target and same existing inode are rejected before
  opening the destination, and a backup exception retains only a private
  partial target. The helper receives the source's selected lexical database
  path (or the `:memory:` token), re-verifies file-backed source identity, and
  never guesses source selection from `PRAGMA database_list`.

- [ ] **Step 2: Add red source-copy and restore tests**

  Verify a file source is opened read-only through the seam, an existing
  destination remains `0600`, a missing destination is pre-created privately,
  WAL-visible committed data is included, an unsafe source or destination
  fails closed, and pre-restore backup plus restore both use SQLite backup
  semantics.

  Define and prove the live-connection contract: an idle connection may remain
  open only if it observes the restored state after its transaction boundary;
  an active reader or writer transaction makes restore fail promptly without a
  success notification or partial replacement. The restore-specific helper
  must serialize the pre-restore snapshot and restore against one guarded
  destination lifecycle. If SQLite cannot provide these guarantees on a
  supported journal mode, fail closed with an explicit close/retry or restart
  message rather than reporting success.

- [ ] **Step 3: Add red application backup-directory tests**

  Bulk and per-database Settings backup directories are application-owned
  `0700`. Database backup files and pre-restore backups are `0600`. Metadata
  files are contained inside a `0700` directory. A configured/custom database
  parent is never chmodded.

- [ ] **Step 4: Implement the backup helpers**

  Keep `.backup()` inside `DB/private_sqlite.py`, close only locally owned
  connections, preserve caller-owned source connections, re-verify the
  explicit lexical source path/identity, and use the same target
  preparation/sidecar path as ordinary writable connections.

- [ ] **Step 5: Migrate all nine backup/restore sites**

  Replace the three direct backup methods and six Settings `copy2` sites.
  Preserve the existing public boolean/notification behavior and metadata.
  Do not touch the Tamagotchi JSON backup or unrelated user-export copies.

  In the checked inventory, change P20-P22 from `current` to `migrated` in the
  same commit. P04-P05 remain `current` because their narrowly documented
  legacy exclusions are intentionally not migrated.

- [ ] **Step 6: Extend source guards**

  Assert `.backup()` occurs only in `DB/private_sqlite.py`. Assert the six
  inventoried Settings DB backup/restore symbols call the centralized helper
  and contain no `shutil.copy2` database operation. Keep explicit exclusions
  narrow enough that a later raw backup owner fails the test.

- [ ] **Step 7: Run focused backup suites**

  ```bash
  python3 -m pytest -q \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/ChaChaNotesDB \
    Tests/Media_DB \
    Tests/Prompts_DB \
    Tests/UI/test_tools_settings_window.py
  ```

- [ ] **Step 8: Commit backup lifecycle**

  ```bash
  git add tldw_chatbook/DB/private_sqlite.py \
    tldw_chatbook/DB/ChaChaNotes_DB.py \
    tldw_chatbook/DB/Client_Media_DB_v2.py \
    tldw_chatbook/DB/Prompts_DB.py \
    tldw_chatbook/UI/Tools_Settings_Window.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    backlog/docs/sqlite-private-owner-inventory.md
  git commit -m "fix(security): centralize private sqlite backups"
  ```

  Add only the focused existing tests actually changed.

---

### Task 7: Prove every classification and close TASK-489

**Files:**

- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `Tests/DB/test_private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify:
  `backlog/tasks/task-489 - Apply-private-storage-boundary-to-every-SQLite-owner-and-backup.md`
- Modify: this plan only if implementation review requires a documented
  deviation.

- [ ] **Step 1: Add the behavioral owner matrix**

  Parameterize every registry policy and every allowed target classification.
  For each owner, create a real safe target (or `:memory:`), invoke the common
  seam, run a query, and verify the declared semantics. Backup-enabled owners
  also execute a real SQLite backup/copy. Combined with the AST literal-owner
  guard, this proves both call-site classification and behavior.

- [ ] **Step 2: Reconcile the human inventory**

  Mark all 31 connection rows and nine backup rows migrated, name the final
  helper used, reconcile every DB-parent `P` row, and record the explicit
  non-SQLite exclusions. Re-run the parser test so missing, duplicated, stale,
  or unregistered rows fail.

- [ ] **Step 3: Run focused privacy verification**

  ```bash
  python3 -m pytest -q \
    Tests/Utils/test_private_paths.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/test_database_path_privacy.py
  ```

- [ ] **Step 4: Run the complete migrated-owner regression set**

  ```bash
  python3 -m pytest -q \
    Tests/DB \
    Tests/ChaChaNotesDB \
    Tests/Media_DB \
    Tests/Prompts_DB \
    Tests/Evals \
    Tests/Kanban \
    Tests/Writing_Interop \
    Tests/Research \
    Tests/Research_Interop \
    Tests/Sync_Interop \
    Tests/Subscriptions/test_client_notifications_db.py \
    Tests/Scheduling/test_scheduled_tasks_db.py \
    Tests/UI/test_tools_settings_window.py \
    Tests/Web_Scraping
  ```

- [ ] **Step 5: Run repository-wide tests**

  Run: `python3 -m pytest -q`

  Record exact pass/fail/skip totals. Investigate every failure; do not dismiss
  an owner-path failure as unrelated without reproducing it on the task base.

- [ ] **Step 6: Run static and source-boundary checks**

  ```bash
  python3 -m compileall -q tldw_chatbook
  python3 -m ruff check <all changed Python files>
  python3 -m ruff format --check <all changed Python files>
  git diff --check
  rg -n 'sqlite3\.connect|\.backup\(' tldw_chatbook --glob '*.py'
  ```

  Expected: the two raw SQLite operations appear only in
  `tldw_chatbook/DB/private_sqlite.py`; any documentation examples are outside
  the production Python source guard.

- [ ] **Step 7: Perform independent final security review**

  Review the complete TASK-489 commit range for namespace races, symlink/
  hardlink handling, sidecar coverage, URI encoding, backup consistency,
  Windows honesty, caller contract regressions, and inventory omissions.
  Resolve and re-verify every finding before completion.

- [ ] **Step 8: Complete Backlog hygiene**

  Check all seven acceptance criteria, add concise implementation notes with
  exact verification evidence and ADR-029 linkage, and set TASK-489 to Done
  only after all Definition-of-Done items are satisfied.

- [ ] **Step 9: Commit closeout documentation**

  ```bash
  git add backlog/docs/sqlite-private-owner-inventory.md \
    'backlog/tasks/task-489 - Apply-private-storage-boundary-to-every-SQLite-owner-and-backup.md' \
    Docs/superpowers/plans/2026-07-23-private-sqlite-owner-lifecycle.md
  git commit -m "docs(backlog): complete private sqlite lifecycle task"
  ```

## Completion Gate

TASK-489 is complete only when:

- the checked inventory still has all 31 connection rows and nine database
  backup/restore rows plus every checked DB-parent creator, all migrated or
  explicitly justified;
- every production raw connect and `.backup()` call is confined to the common
  seam;
- every production seam call uses a literal registered owner ID;
- all private file-backed owners create/harden the main database before SQLite;
- writable custom parents reject attacker-writable namespaces;
- memory and read-only URI behavior are preserved and tested;
- real WAL, SHM, rollback journal, backup, pre-restore backup, and restored
  target modes are verified;
- focused, owner-wide, repository-wide, compile, Ruff, diff, and source-guard
  checks pass;
- independent final review has no unresolved findings;
- ADR-029, inventory documentation, acceptance criteria, implementation notes,
  and Backlog status are current.
