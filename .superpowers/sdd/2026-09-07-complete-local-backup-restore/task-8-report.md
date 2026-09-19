# Task 8 implementation report

Status: implementation complete, pending independent review. TASK-31991 remains
In Progress with ACs unchecked as instructed. No push, merge, publication, user-data
access, dependency installation, or shared environment changes occurred.

Execution checkout: `/private/tmp/chatbook-backup-execution-hgp11i7t/repo`
Branch: `codex/complete-local-backup-recovery`
Base: `65c77f34139a5320047f0168d0c1727e9ab8afab`
Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (read-only)
All application evidence ran under `Tests/` isolation with external pytest/ruff
caches and temporary fixtures. No full test sweep was run.

## Implemented result

Twenty-one installed operational declarations preserve actual durable histories and
recoverable bytes. Ten physical SQLite owners use exact literal registered capture
policies; Notes binding semantics share the physically verified core payload. Ten
raw owners use bounded checked copying and validation. All declare
`activation_required=True`, and relocation preserves historical identity/path/state
bytes without runtime constructors, migrations, filesystem intent replay or network
activation. No fresh live identity allocation is claimed: tasks17/20/21 own staged
restore and durable activation; task25 is only the minimal launcher.

The final qualification table in
`backlog/docs/backup-recovery-owner-inventory.md` explicitly supersedes the exact
task8 producer rows and names each source selector, schema, dependencies and live
claims. The original broad unsupported table remains historical staging context;
RAG remains unqualified until task19, other generic files until9, credentials until14.
Unknown existing durable state continues to block completeness.

| Factory module | Installed declarations / literal SQLite validation symbols |
| --- | --- |
| DB/recovery_operations | db.workspaces `_WorkspacesAdapter`; db.agent_runs `_AgentRunsAdapter`; db.subscriptions `_SubscriptionsAdapter` |
| Scheduling/recovery | db.scheduled_tasks `_ScheduledTasksAdapter` |
| Notifications/recovery | notifications.client `_NotificationsAdapter`; runtime.event_state `_EventsAdapter` |
| Sync_Interop/recovery | runtime.sync_state `_SyncAdapter` |
| Notes/recovery | notes.file_notes `_FileNotesAdapter`; notes.sync_state `_ReceiptsAdapter`; notes.sync_bindings `_SyncBindings` |
| Kanban_Interop/recovery | kanban.local `_KanbanAdapter` |
| MCP/recovery | mcp.local, mcp.targets, mcp.context, mcp.permissions, mcp.history |
| Workspaces/recovery | workspaces.change_tracking |
| Agents/recovery | agents.history |
| Subscriptions/recovery | subscriptions.assets |
| runtime_policy/recovery | runtime.source_state |
| Widgets/Tamagotchi/recovery | tamagotchi.config |

SQLite versions reflect actual constructors: Workspace2, AgentRuns12,
Subscriptions1, Scheduling3, Notifications1, EventState1, SyncState4, FileNotes
unstamped (installed SELECT 0 validation dispatch only), import receipts PRAGMA1,
Kanban metadata key1. MAX(schema_version.version) is used only by installed owners
that use that metadata. Every whole SQL catalog, foreign keys and quick_check must
match; no archival SQL, synthetic stamp or constructor replay is used.

Subscriptions has two actual complete layouts at version1: plain SubscriptionsDB
and the hybrid created by SiteConfigManager's real CharactersRAGDB construction.
The latter retains its actual embedded core42 stamp in db_schema_version. Capture
preserves real site definitions and subscription rows. An altered embedded stamp,
unknown schema additions, union-only fragments or reordered schema catalogs refuse.
The two alternative catalogs never become a union of accepted subsets.

The actual File Notes store is `file_notes.sqlite`, including FTS internals,
raw current/deleted bytes, protected paths and pre-edit revisions. ADR-021's intended
file_notes.db/notes_recovery.db split is not installed; no future filenames or fake
missing-required rows were added. The separate NoteImportReceiptRepository contains
actual import receipts and pending folder/payload/membership effects, not invented
lasting-sync root/journal tables. Legacy sync fields and manual/managed memberships
remain in the core database under notes.sync_bindings. Managed owner IDs and active
flags are preserved verbatim. External disk roots are opt-in and never accessed
from imported DB strings during capture/inspection.

Kanban get_storage_status and other local service calls use the independent
`tldw_chatbook_kanban.db`, not WorkspaceDB. Tamagotchi arbitrary-path SQLiteStorage
and JSONStorage have no installed construction sites; BaseTamagotchi uses memory by
default. The deterministic optional ConfigFileStorage JSON path and exact timestamp
backup pattern are captured if present, including corrupt opaque recovery bytes.
Unknown siblings block. The exact construction-site guard also covers NotesMirror,
which has no installed wiring and defaults to memory. Future wiring invalidates
that unused classification; no arbitrary pet database filename was fabricated.

## Interfaces and approved refinements

- Each owner-local `recovery_adapters() -> tuple[OwnerAdapter, ...]` returns frozen
  declarations. DB/recovery_operations aggregates only SQLite policies.
- `_validate_sqlite(..., *, version_query="PRAGMA user_version")` preserves the old
  default. SchemaPolicy.schema_sql now documents multiple exact ordered complete
  alternatives at the same installed version; matching is ANY whole catalog.
  The archive validator in task13 must preserve that meaning.
- `Backup_Recovery/file_inventory.inventory_tree(root: Path, *, owner: str,
  external: bool) -> tuple[StorageItem, ...]` is the approved brought-forward task9
  seam. It enumerates one exact owner root with pinned no-follow traversal, explicit
  parent topology, link/hardlink/special/limit refusal and bounds (100k entries,
  1TiB total, 256GiB/member, 1024 UTF-8 bytes/path, depth64). It does not qualify
  external metadata or copy directories. Unavailable ancestors fail closed.
- `Backup_Recovery/recovery_files` shares immutable raw declaration behavior,
  checked bounded JSON-object validation and opaque byte integrity. `_tree_member_id`
  derives exact profile payload dependencies from installed source-root/member IDs
  without probing historical paths during candidate validation.
- `_check_recovery_file(owner_id, candidate, *, max_bytes, cancel=None) -> None`
  consumes bounded chunks without collecting/exposing bytes or native handles.
  It shares `_consume_recovery_file` with the existing bounded byte-returning
  `_read_recovery_file`. Positive retirement of pinned parents/file FDs is unchanged.
  Opaque validation is byte integrity, not semantic permission/Git/audio authority.
- Raw permissions are an exact installed ID dictionary, never a prefix exemption.
  A caller still requires fixed application authority, intact selected bindings,
  selected namespaces plus bootstrap.unbound, exact source scope/private staging.
  Default SQLite factories and native retirement remain mandatory. Ordinary calls
  always use admission; an installed owner ID alone grants no filesystem access.
- Aggregate inventory now admits only explicit same-owner/same-profile immediate
  parent dependencies for tree topology. Missing/forged/cross-owner edges and
  independent overlapping roots still refuse. Core→notes.file_notes and
  core→notes.sync_bindings now use final exact IDs; other unqualified refs remain
  unresolved. The original physical identity/shared-group proof remains required.
- Complete briefing_audio rows declare exact profile assets and staged validation
  requires those exact peer IDs. Missing audio refuses aggregate completeness;
  another profile cannot supply an identically named payload. Unqualified historical
  paths outside the installed selected audio root fail closed, never disappear.

These are the controller's explicit bounded scope rulings (owner-local additions,
lazy packages, raw checked-reader reuse, exact tree seam, actual installed stores,
metadata variants and Notes dependency correction), recorded in task-8-brief.md.
No new public runtime activation API, generic path allowlist, Git execution during
capture, sidecar ledger or directory-copy implementation was added.

## Writer, filesystem and process evidence

MCP LocalMCPStore.save, ConfiguredServerTargetStore.save_targets,
UnifiedMCPContextStore.save, MCPPermissionStore.save/_backup_corrupt_file,
MCPExecutionLog.append/read_recent (including generation migration) now hold ordinary
admission around mutation. Tests retain actual stores/permission history unchanged
when maintenance refuses the real writers.

RunLogWriter.bind, append and _write_bytes cover directory/segment/manifests and
flush/fsync mutation. Existing best-effort close remains best-effort: its admitted
_write_bytes refuses mutation and close logs the refusal. ShadowRepo snapshot,
force_add, restore_paths, ensure_initialized and _run, plus prune_change_history,
hold ordinary admission through all owned-byte mutations and subprocess completion.
A real Git wrapper writes a handshake and waits; maintenance times out while the
actual child lives, then succeeds after release/exit. No external workspace bytes
are changed by capture or relocation.

JSONStorage construction, save/delete, writes, backups and cleanup retain ordinary
admission for their full mutation lifetime, covering ConfigFileStorage inheritance.
Final self-review caught the constructor's parent mkdir occurring before its
admitted write. A new behavioral RED proved a new parent appeared during maintenance
although the subsequent write refused; the constructor now admits before mkdir.

Briefing audio's actual atomic write was already admitted by private_paths; its
best-effort cleanup unlink is now also admitted through the deletion lifetime.
The recovery test calls the real cleanup and proves preserved bytes during
maintenance and successful cleanup after exit. Optional synthesis is not invoked.

Fixture capture explicitly calls gc.collect after closing EventStateRepository and
SyncStateRepository. Their `_get_connection` per-call native SQLite cycles otherwise
finalize later and checkpoint source WAL into the main file during fixture authority
setup. Their .close methods currently close only retained memory connections.
This deliberately drained fixture establishes snapshot/source-byte preservation,
NOT coordinated runtime maintenance drain. Task10 must close those actual file
handles before capture; with-connection commit/rollback is not native retirement.
The incident is recorded in lessons-testing-evidence.md. Writer exclusion evidence
is separate from this fixture-only setup.

## Test commands and outcomes

All commands ran from the execution checkout. The following exact executable prefix
was used for every pytest invocation:

`PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`

### Meaningful RED evidence

- `Tests/Backup_Recovery/test_operational_owners.py -q -o cache_dir=/private/tmp/task8-pytest-cache`
  initial importable regression: **4 failed**, including the specified nonempty
  activation-required adapter assertion and fresh-process declaration imports
  bootstrapping configuration. `/private/tmp/task8-red.log`.
- Same file, selected actual MCP mutation tests: **4 failed, 67 deselected**,
  each `DID NOT RAISE RuntimeError` while the real writer ran during maintenance.
  `/private/tmp/task8-mcp-red.log`. After mutation admission those same owner writes
  pass and preserve the source bytes.
- Actual tree aggregate: **1 failed, 90 deselected**, because included parent/child
  entries were incorrectly rejected as overlapping_owner_roots.
  `/private/tmp/task8-tree-red.log`; explicit topology repair now passes.
- Actual SubscriptionsDB+SiteConfigManager hybrid: **1 failed, 91 deselected**, exact
  installed schema rejected before same-version whole-catalog alternatives.
  `/private/tmp/task8-hybrid-red.log`; actual definition capture and altered-stamp
  refusal now pass.
- Fresh-process package REDs: Subscriptions/Kanban **2 failed, 4 passed**;
  Tamagotchi **1 failed** from eager config/runtime imports. No missing optional
  dependency was counted as the required behavioral RED. Lazy exports then pass
  exact identity/default behavior and available/unavailable optional-group checks.
- `Tests/Backup_Recovery/test_operational_owners.py -q -k pet_constructor -o cache_dir=/private/tmp/task8-final-cache`
  **1 failed, 115 deselected**: the parent directory existed after maintenance
  refused the constructor's file write. `/private/tmp/task8-pet-constructor-red.log`.

### GREEN and targeted guard evidence

1. `Tests/Backup_Recovery/test_operational_owners.py -q -o cache_dir=/private/tmp/task8-final-cache`
   **115 passed, 1 warning in 22.83s**, `/private/tmp/task8-operational-green.log`.
   Covers actual ten SQLite constructors and history dumps, FTS/raw File Notes
   revisions, managed memberships, pending receipts, activation flags, invalid
   authority/source/destination/cancellation/schema, opaque actual-byte limits,
   exact audio peers, shared physical cohort, tree topology, installed imports,
   optional-group semantics, real writer exclusion and Git process lifetime.
2. After the last constructor fix:
   `Tests/Backup_Recovery/test_operational_owners.py Tests/Widgets/test_tamagotchi.py -q -k 'pet or tamagotchi' -o cache_dir=/private/tmp/task8-final-cache`
   **38 passed, 110 deselected, 1 warning in 6.45s**,
   `/private/tmp/task8-pet-constructor-green.log`. Includes the additional 116th
   operational case plus real pet baseline behavior. No unrelated suite repeated.
3. Prescribed guards:
   `Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py Tests/Architecture/test_backup_owner_inventory.py -q -o cache_dir=/private/tmp/task8-guards-cache`
   Initial result **359 passed, 2 failed, 1 skipped** in 63.19s. Both failures were
   stale exact owner/copy counts, not runtime failures. Existing source-census,
   literal module-owned policy, default/native owner matrices and interop guards
   passed. `/private/tmp/task8-guards.log`.
   Corrective run:
   `Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py -q -k 'every_backup_enabled_owner or copy_backup or backup_and_restore_rows_explicitly' -o cache_dir=/private/tmp/task8-guards-cache`
   **23 passed, 317 deselected**, including each of the ten new literal actual
   transactional copy policies. `/private/tmp/task8-guards-fixed.log`.
4. Shared helper/core regression:
   `Tests/Backup_Recovery/test_domain_owners.py Tests/Backup_Recovery/test_core_owners.py -q -o cache_dir=/private/tmp/task8-regression-cache`
   **142 passed, 1 failed** in 53.93s; only old notes.file_notes:unresolved expectation
   failed after approved resolution. Domain raw reader/native parent-close fault,
   resource retirement, actual cross-process exclusion and SQLite defaults passed.
   `/private/tmp/task8-shared-regression.log`.
   Corrective run:
   `Tests/Backup_Recovery/test_core_owners.py -q -k 'discovery_custom_paths or local_cross_store' Tests/Architecture/test_backup_owner_inventory.py -o cache_dir=/private/tmp/task8-regression-cache`
   **2 passed, 71 deselected**. The -k expression selected the core corrections,
   not architecture tests. `/private/tmp/task8-core-fixed.log`.
5. Actual aggregate inventory/census recheck:
   `Tests/Backup_Recovery/test_operational_owners.py Tests/Backup_Recovery/test_inventory.py Tests/Architecture/test_backup_owner_inventory.py -q -o cache_dir=/private/tmp/task8-final-cache`
   **152 passed, 1 failed**; all 43 inventory/architecture cases passed. The sole
   added hybrid test attempted staged validation after leaving its namespace scope;
   admission correctly refused it. The fixture now validates under that scope.
   `/private/tmp/task8-final.log`.
   `Tests/Backup_Recovery/test_operational_owners.py -q -k 'site_config_manager or dormant or run_log_real' -o cache_dir=/private/tmp/task8-final-cache`
   **3 passed, 107 deselected**, `/private/tmp/task8-last-fixed.log`, followed by
   the 115-case all-operational GREEN in item1.
6. Actual changed writer regression:
   `Tests/Workspaces/test_change_tracking.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_survivor_lifetime.py Tests/MCP/test_permission_store.py Tests/MCP/test_execution_log.py Tests/MCP/test_local_store.py Tests/MCP/test_server_target_store.py Tests/MCP/test_unified_context_store.py Tests/Widgets/test_tamagotchi.py Tests/Subscriptions/test_briefing_audio_pipeline.py -q -o cache_dir=/private/tmp/task8-writers-cache`
   **222 passed, 1 skipped, 1 warning in 30.52s**,
   `/private/tmp/task8-writers.log`. The optional audio pipeline module cannot import
   pydub in the read-only shared environment; it is not used as recovery evidence.
   Required real cleanup/DB/audio-file tests are in the operational suite and pass.

The other skip was the existing Windows functional-posture test on this macOS host.
No platform or optional-engine release capability is inferred from either skip.
Warnings are the existing requests urllib3/chardet compatibility warning and the
architecture census parsing three existing invalid escape strings. No required new
operational test skipped. Earlier fixture-only failures (wrong SourceAuthority enum
assumption, missing Kanban client_id, delayed native SQLite cycles, and best-effort
RunLogWriter.close expectation) were corrected and are covered by the later GREEN.

### Static checks and self-review

`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select E9,F63,F7,F82 --cache-dir /private/tmp/task8-ruff-cache` ran on every changed/new Python file:
**All checks passed!** New modules and the operational test were formatted and
`ruff format --check` returned **15 files already formatted**. New-module unused
imports were removed (26 F401 fixes only); no shared environment or unrelated large
module reformat occurred. The exact path list is `/private/tmp/task8-final-python-files.txt`;
commands/output `/private/tmp/task8-final-lint.log`. Final constructor lint is
recorded below before commit. `git diff --check` passed.

Self-review covered every mutation diff, fixed owner IDs/selectors, metadata query
variants and complete catalogs, resource retirement reuse, owner/default imports,
profile dependency and original shared identity guards, census deltas and planned
activation boundaries. The final constructor gap was caught and tested RED/GREEN.
No unresolved failing test remains. The report's qualifications deliberately stop
short of live drain, staged activation, archive/container budgets, secret handling,
external/model metadata and a user-facing complete-backup capability.

## Governance and scope

ADR required: yes. Reuses
`backlog/decisions/126-complete-local-backup-and-recovery.md`,
`021-file-backed-notes-disk-authority-and-recovery.md`,
`059-notes-folder-import-and-device-local-sync-ownership.md`, and
`060-notes-sync-round-trip-and-interoperability-constraints.md` (not the unrelated
060 prompt-mutation ADR). Ordinary selective bundle exclusions are unchanged.
Controller-approved source-backed corrections are documented above and in the
owner inventory. Backlog Design references remain intact. Implementation status is
reviewable, not marked Done; AC checkboxes remain unchecked for independent review.

Final constructor/static verification: running
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B /private/tmp/task8-final-lint.py`
after the constructor fix returned `All checks passed!`, `15 files left unchanged`,
`All checks passed!`, `15 files already formatted`; `git diff --check` was clean.
The script executes new-module F401 checking, new-module formatting, changed/new
Python E9/F63/F7/F82 checks and new-module format --check using the named ruff binary
and external cache. Output: `/private/tmp/task8-final-constructor-lint.log`.
