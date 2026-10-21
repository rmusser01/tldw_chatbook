# K1 implementation report

Status: implemented, targeted verification complete with two independently reproduced inherited census failures; ready for parent independent review. Commit: `8bdb4be36df66222dc5f13ed6257ff86eb539b14` (`feat(workspaces): add folder-bound project index storage`).

Source: `33fa199eca7fd44ba1dd991b5086a87786bb6e66`; branch `codex/workspace-project-keyword-index`; managed worktree `C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui`. ADR240 and the reconciled task-1 brief govern this foundation. No scanner/service/tool/UI/semantic implementation, profile configuration, dependency installation, push, or original-checkout edits. Root-owned Backlog/plan/spec/ADR/execution artifacts were not staged.

## Implemented ownership and schema

`Workspaces/project_index/models.py` supplies immutable typed key/scope/policy/authority/status, raw capture/page, prepared chunk/page, job claim, candidate/hit/result, live-reader and retirement-verifier contracts. The complete key is SHA256-namespaced, including data authority and root identity. Keys persist only their namespace hash in the cache; paths inside it are relative. Models reject absolute/escaping paths and mismatching capture identity/hash/UTF-8 size. Recovery imports do not load DB constructors, registry runtime, chunking, or RAG.

`DB/Workspace_Project_Index_DB.py` implements private schema v1 with control epochs/policy/manifest revision, durable active claim, directory frontier, entries, files, chunks/FTS, duplicate page receipts and cleanup receipts. Fresh state is Off with no worker creation. Metadata, text, chunks/FTS, directory completion, page receipt, and manifest revision publish together. An injected FTS failure proves rollback of all parts. Old scope/job/authority revisions fail closed; another instance cannot publish an owner's claim. Literal FTS tokens, requesting-run exclusions, and the current processing policy filter before ranking/LIMIT; candidates contain metadata only, never cached excerpt text.

Workspace schema advances from 8 to 9 through its canonical inline migration runner. Binding authority witnesses are durable non-cascading tombstones, with insert/update/delete/workspace authority/archive triggers in the same transaction as the mutation. Removal/recreation/moving cannot reuse a revision. Migration failures roll back schema/version. Recovery retains the exact frozen v8 catalog and adds the exact frozen v9 successor, accepts (8, 9); the fresh v9 catalog matches byte for byte.

Owner is exactly `db.workspace_project_index`, private-file only. Kernel state initializes before schema work. Private admission and activation precede open; held native connections are registered under the exact optional repository class. Both finite-worker and maintenance closer maps include this owner. Failed initialization closes the native handle. Tests exercise actual SQLite closure, admitted private-profile maintenance, activation denial, detached namespace/sidecar/receipt discovery, and capture refusal. `Workspaces/project_index/recovery.py` declares rebuildable bytes intentionally excluded, not an archive payload.

Required documentation is `backlog/docs/sqlite-private-owner-inventory.md`, row C98. Its inventory range test includes C98. No unrelated owner census corrections were made.

## Exact seams for K2/K3/K4

```python
index_store_path(data_root: Path, key: ProjectIndexKey) -> Path
prepare_scan_page(page: ScanPage) -> PreparedScanPage
WorkspaceProjectIndexDB(path: Path)
load_status(key: ProjectIndexKey) -> ProjectIndexStatus
configure(key: ProjectIndexKey, policy: ProjectIndexPolicy, *, enabled: bool,
          expected_scope: ProjectIndexScope) -> ProjectIndexScope
claim_job(scope: ProjectIndexScope, *, owner_token: str, authority_revision: int,
          expected_job_revision: int) -> ProjectIndexJobClaim
pending_directories(scope: ProjectIndexScope, *, limit: int = 256) -> tuple[str, ...]
read_claim(scope: ProjectIndexScope) -> ProjectIndexJobClaim | None
replace_retired_job(expected_claim: ProjectIndexJobClaim, *, owner_token: str,
                    authority_revision: int,
                    retirement_verifier: IndexRetirementVerifier) -> ProjectIndexJobClaim
retire_job(scope: ProjectIndexScope, job_id: str, *, owner_token: str,
           expected_job_revision: int, state: str) -> bool
commit_scan_page(scope: ProjectIndexScope, job_id: str, page: PreparedScanPage, *,
                 expected_job_revision: int, authority_revision: int) -> bool
lexical_candidates(authority: ProjectIndexSearchAuthority, query: str, *,
                   limit: int) -> tuple[IndexCandidate, ...]
revoke(key: ProjectIndexKey, *, clear: bool,
       expected_scope: ProjectIndexScope | None = None) -> ProjectIndexScope
clear_revoked(scope: ProjectIndexScope) -> bool
close() -> None
```

`LocalWorkspaceRegistryService.project_index_authority_adapters(current_data_authority: Callable[[], str])` returns `(IndexAuthorityReader, IndexAuthorityFence)`. Reader accepts an exact `ProjectIndexKey` and returns authority or None; fence is the matching context manager. The code-owned callback executes under `binding_authority_fence(binding_id)`, a `transaction(immediate=True)` registry fence. It must match the key's data authority, workspace, local binding, canonical locator, and ancestor identity. Current mandatory and per-binding exclusions accompany the durable witness. `binding_authority_revision(binding_id) -> int` includes tombstones.

K3 owns actual issuance of the current-data-authority string from its admitted private profile/store identity plus restore/activation generation, including switch/restore qualification. No existing installed issuance API or constant production callback is assumed. Tests inject explicit IDs. No model input or global active-workspace inference is permitted.

Lock order is **registry fence, then cache transaction**. Call `prepare_scan_page` before either fence. It uses ChunkingService explicitly at 400 words / 100 overlap, computes validated character/line offsets and page digest, and produces offset-only prepared chunks without another queued text copy. Commit rejects raw ScanPage and has no implicit chunking fallback. Prepared DTOs are internal code-owned values, not a wire/model input format. K2 supplies tuple-valued bounded ScanPage records; K3 queues only bounded prepared records. No fence-detection mechanism was introduced.

`IndexRetirementVerifier = Callable[[ProjectIndexJobClaim], bool]`. K3 must observe native process/contained-worker retirement **outside all registry/cache locks**, then supply a bounded code-owned verifier for already-established evidence tied to the exact old scope/job ID/owner token/job revision/authority revision. It must do no wait, join or I/O. K1 requires the return value to be exactly True, then rechecks the complete old claim and scope transactionally. No elapsed-time lease, boolean API, or platform process subsystem was added. Tests use real two-connection retirement, refuse unproven takeover, reject a stale proof after replacement, and use a real child process to prove active-claim exclusion. Tests do not substitute for K3's lifecycle observation. `retire_job` likewise requires prior physical retirement; dropping a claim is not process retirement.

Durable frontier obligations: a newly claimed scan queues root `""`; discovery of a child directory records it incomplete. Publishing a complete directory page and all its entries/captures is one transaction. `pending_directories` returns only incomplete paths, at most 256. Exact orphan replacement preserves the frontier; completed directories do not reappear merely because of restart. A new ordinary claim restarts root discovery. Duplicate delivery within a job is idempotent by page digest. The scanner/service must publish every page, mark completion only after enumeration is complete, recapture/revalidate before publication, and set Ready only after the durable frontier is empty and the worker retired. K1 does not infer traversal completion from a returned page or implement a scanner scheduler/deletion reconciliation.

Clear protocol: K3 calls `revoke(clear=False, expected_scope=...)` inside its short registry fence, exits it, then calls `clear_revoked(returned_scope)`. Revoke commits Off and an advanced epoch before cleanup begins. Cleanup deletes derived rows only while the exact revoked scope remains Off; a concurrent Enable makes it return False. Cleanup receipts make retry harmless; control/epoch/job revision history remains. `revoke(clear=True)` is a convenience only when no enclosing registry fence is held. A two-connection regression observes Off with old rows still present at cleanup entry and proves delayed cleanup cannot erase a new Enable. No vacuum or whole-file deletion resets history.

K4 receives metadata candidates after current policy and requesting-run exclusion filtering. It must bind `ProjectIndexLiveReader = Callable[[IndexCandidate], FileCapture | None]` on `ProjectIndexSearchAuthority` after effective search/fs_read review, perform same-read live capture, revalidate root/revision/hash/identity, and only then build excerpts. The authority record includes exact scope/root/chain/exclusions/authority_revision, guard, optional existing executor, and nonserialized live-reader callback. K1 supplies no live-reading/tool/UI behavior.

Bounds: 256 page entries/captures, 1 MiB UTF-8 file capture, 4 MiB queued capture text, 250,000 discovered entries, 1 GiB physical cache budget. SQLite max_page_count reserves space for both database and rollback journal plus overhead; native SQLITE_FULL rolls back publication and records Partial. Cache accounting inspects only the DB and fixed `-wal`/`-shm`/`-journal` sidecars (bounded stats, no recursive scan). v1 creates no vector/staging files. K3 must extend accounting before introducing other namespace files. Clear retains the minimal control database and SQLite free pages; logical deletion/secure_delete is not a file-compaction promise.

## TDD and verification evidence

All pytest commands used worktree `.venv/Scripts/python.exe` (Windows Python 3.12.10) and task-private TEMP/TMP/TMPDIR at `.superpowers/sdd/2026-10-20-workspace-project-keyword-index/native-temp`; no guard or warning suppression. Logs below are in this report's directory.

Intended RED evidence:
- `python -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py -q`: `k1-red.log`, missing production module/contract collection errors.
- Focused v9 migration/witness tests: `k1-witness-red.log`, 2 failed (schema still 8 and missing authority witness).
- Orphan/fence/activation boundary tests: `k1-boundary-red.log`, 3 failed for missing/refused seams.
- Frontier/CAS regression: `k1-frontier-red.log`, 1 failed.
- Native-full regression: `k1-capacity-red.log`, 1 failed with actual SQLITE_FULL before Partial handling.
- `python -m pytest Tests/DB/test_workspace_project_index_db.py::test_revoke_is_visible_before_cleanup_and_old_cleanup_cannot_erase_enable Tests/DB/test_workspace_project_index_db.py::test_narrowed_processing_policy_filters_old_candidates -q`: `k1-cleanup-red.log`, 2 failed, 1 warning, 5.15s (missing cleanup seam and old candidate still returned).

The complete required targeted covering selection was run once:
```powershell
& .venv/Scripts/python.exe -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py Tests/DB/test_workspace_db.py Tests/Workspaces/test_workspace_registry_service.py -q
```
`k1-covering.log`: **4 failed, 183 passed, 10 skipped, 1 warning in 422.43s**. Ten skips are existing Windows/POSIX mode/trust expectations. Slow inventory scans accounted for buffered output delay; the run was allowed to finish. Two K1-owned failures were corrected: C98 required the exact `Migrated via connect_private_sqlite.` disposition punctuation, and the synthetic v7 fixture needed to remove v9 objects before testing v8 rollback. The other two failures are inherited, independently reproduced below. No claim that the entire covering selection passed is made.

Intermediate focused self-review run (`k1-self-review-check.log`): **20 passed, 1 warning, 35.47s**, including native capacity, offset prepared chunks, v8 rollback correction and operational Workspace recovery qualification.

Final amended-node selection after cleanup/policy fixes:
```powershell
& .venv/Scripts/python.exe -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/DB/test_private_sqlite_inventory.py::test_connection_and_backup_rows_record_completed_helper_migrations Tests/DB/test_workspace_db.py::test_v8_explicit_none_migration_rolls_back_when_version_write_fails Tests/DB/test_workspace_db.py::test_v9_authority_witness_migration_rolls_back Tests/Workspaces/test_workspace_registry_service.py::test_project_index_authority_witness_survives_remove_recreate_and_move Tests/Workspaces/test_workspace_registry_service.py::test_index_authority_fence_checks_exact_scope_and_blocks_other_process -q
```
`k1-final-focused.log`: **25 passed, 1 warning in 46.74s**, exit 0. Inherited pytest-asyncio unset loop-scope and Pydantic Field metadata deprecation remain unsuppressed. This reruns every new test and the amended failed nodes without repeating unrelated passing baseline selections.

## Exact inherited-failure reproduction

A task-private isolated export at `baseline-33fa199` was produced from **exact commit `33fa199eca7fd44ba1dd991b5086a87786bb6e66`**, using `git archive` for `tldw_chatbook`, `Tests/DB/test_private_sqlite_inventory.py`, `backlog/docs/sqlite-private-owner-inventory.md`, and `pyproject.toml`. Export retained tracked Python sources plus the original inventory/config; no shared checkout/stash changes. `run_baseline.py` prepended that export to sys.path and ran only the two static census node IDs under the export root/config, using the same worktree Python dependencies. This runner intentionally needs no app runtime fixture or modified K1 production files.

Command:
```powershell
& .venv/Scripts/python.exe .superpowers/sdd/2026-10-20-workspace-project-keyword-index/baseline-33fa199/run_baseline.py
```
Runner's pytest arguments:
```text
<baseline>/Tests/DB/test_private_sqlite_inventory.py::test_raw_connection_census_is_qualified_and_transition_aware
<baseline>/Tests/DB/test_private_sqlite_inventory.py::test_backup_inventory_matches_current_sqlite_and_settings_operations
-q --rootdir <baseline> -c <baseline>/pyproject.toml
```
`k1-baseline-census.log`: **2 failed in 148.23s**, exactly matching the covering run:
- Raw census extra: `('tldw_chatbook/app_service_wiring', '_notes_sync_state_db_has_configured_roots'): 1`.
- Backup census extra: `('tldw_chatbook/DB/recovery_dreams_guardian', '_DreamsAdapter.capture', 'copy_private_sqlite'): 1`.

Both are untouched inherited owners. Parent explicitly directed preserving them rather than broadening K1.

## Static checks and self-review

Command (shared installed Ruff, no installs):
```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/ruff.exe check --no-cache tldw_chatbook/DB/Workspace_Project_Index_DB.py tldw_chatbook/Workspaces/project_index Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/Workspaces/project_index_support.py
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/ruff.exe format --check --no-cache tldw_chatbook/DB/Workspace_Project_Index_DB.py tldw_chatbook/Workspaces/project_index Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/Workspaces/project_index_support.py
git diff --check
```
Final output: `All checks passed!`; `7 files already formatted`; diff check exit 0. Existing modified modules were checked against exact-base content with Ruff: `k1-lint-comparison.json` shows **no new findings**; inherited totals are private_sqlite 6, recovery_operations 4, registry_service 15, all other touched existing Python files 0. Full scoped lint is therefore not globally clean; no rule exclusions or unrelated cleanup was introduced. New blocks were formatted without mass-reformatting existing modules.

Self-review checked lock order, exact-token CAS, atomic FTS rollback, migration catalog compatibility, import inertness, both actual connection retirement paths, cleanup epoch retention, bounds and no cached excerpt projection. It found and corrected native capacity failure handling, prepared-text duplication, cleanup under revoke transaction, and retained candidates bypassing a narrowed policy. Root independent review remains next. No full suite or real user-folder indexing was run.

Commit completed with 17 exact owned paths (1913 insertions, 6 deletions). Post-commit status retains only root-owned plan/QA work; no K1 product/test changes remain unstaged.
