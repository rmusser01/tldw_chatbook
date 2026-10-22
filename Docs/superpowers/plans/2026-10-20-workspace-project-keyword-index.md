# Optional Workspace Keyword Index Implementation Plan

Current governance: [ADR-241](../../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md). Renumbered from project ADR-240 during 2026-10-22 dev integration; historical rulings and delivery notes below retain their original source-era labels.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide opt-in, folder-bound keyword discovery over large project trees with bounded resource use and live file authority.

**Architecture:** A project-index service owns per-binding SQLite/FTS caches, a durable scan frontier, and a contained scanner job. A read-only local-tool spec returns bounded, live-validated results. Settings manages indexing and Console only projects its status; Library ingestion and retrieval remain independent.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, stdlib SQLite FTS5/os.scandir, SQLiteStoreKernel, existing pinned-root/process-containment primitives, existing ChunkingService, pytest/Ruff. No new dependency or embedding requirement.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), optional-index sections and acceptance items 2-3 and 5-12. Deliver after [workspace guidance](2026-10-20-workspace-project-guidance.md).

**Reference:** [Codex CLI source findings](../specs/2026-10-20-codex-cli-project-context-reference.md), especially path-search sessions, cancellation, and instruction-cache revocation. This is a durable text index; do not copy Codex's in-memory fuzzy path catalog.

**ADR required:** yes, existing accepted decision.
**ADR path:** `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`.
**Reason:** Implements its derived storage, worker, file-tool, disclosure, and revocation boundaries. Link ADR-101/102/126/174 in the implementing Backlog tasks.

## Global Constraints

- Off by default; explicit enablement per local folder binding. SSH indexing is unavailable in this delivery.
- Preserve saved Library controls/scopes; create no Media/Notes records and no global Library search entries.
- Pages <=256 discovered entries; <=2 simultaneous file reads; <=4 MiB of extracted text queued for the writer.
- Index UTF-8 text files <=1 MiB; skip binaries, archives, generated/dependency/VCS content, symlinks, junctions, and reparse targets.
- Mandatory sensitive paths and exact binding exclusions apply before descent/read/ranking/disclosure.
- Default processing exclusions: `node_modules`, `.venv`, `venv`, `__pycache__`, `dist`, `build`, `.git`.
- <=250,000 eligible discovered entries and <=1 GiB total cache-owned storage per binding, including staging, temporary, and later vector data. Limit exhaustion is explicit partial coverage.
- Search <=10 results and <=16 KiB serialized output; smaller existing run budgets win.
- Revoke access and publication before cleanup; never deliver stale cached text or restore authority from a matching path string.
- Serialize authority-sensitive publication/disclosure with registry mutations across processes; app-local callbacks are invalidation hints only.
- Require expected scope/job state before management effects, and the requesting run's effective exclusion floor before ranking.
- Refresh and restart Resume are explicit. No filesystem watcher, automatic whole-tree prompt injection, or scan per send.
- Register storage/activation/maintenance ownership before production opens; no sync/export/ordinary-backup payload for the cache.
- Use only targeted tests unless the user separately requests a full sweep.
- At execution, create/read each atomic Backlog task, set In Progress, then add its task-file Implementation Plan and ADR check. K1-K5 are plan-local labels.

## Review Focus

- Root replacement, case aliases, and junctions can reuse a pathname: K1/K2 test filesystem identity and canonical path rejection.
- Unreadable subtrees and directory mutation can look like deletions: K2/K3 test explicit incomplete-directory receipts and non-destructive reconciliation.
- A blocked `fs_read` can coexist with Allowed index search: K4 tests the narrowing permission floor and Ask review.
- Exclusion/detach can occur after the last authority read in a different process: K1/K3/K4 hold the actual registry/publication/disclosure barrier and revoke first.
- A restored profile or delayed status callback can resurrect old readiness: K1/K5 test restoration invalidation and view/job epoch fencing.
- A delayed Enable can execute after Clear: K3/K5 fence the action before effects, not only its returned view.

## File structure

| Files to create | Responsibility |
| --- | --- |
| `tldw_chatbook/Workspaces/project_index/__init__.py` | Public service/models exports only; no eager runtime construction. |
| `tldw_chatbook/Workspaces/project_index/models.py` | Immutable keys, policies, authority snapshots, jobs, hits, status and search results. |
| `tldw_chatbook/DB/Workspace_Project_Index_DB.py` | Versioned FTS/manifest/frontier store through SQLiteStoreKernel. |
| `tldw_chatbook/Workspaces/project_index/recovery.py` | Exact rebuildable-cache declaration and activation/lifecycle integration. |
| `tldw_chatbook/Tools/project_index_scan_protocol.py`, `project_index_scan_worker.py` | Closed stdlib-only worker wire and root-pinned scanner. |
| `tldw_chatbook/Workspaces/project_index/scanner.py` | Containment, bounded transport/backpressure, and owned scanner retirement. |
| `tldw_chatbook/Workspaces/project_index/service.py` | Jobs, CAS publication, query candidates, live validation, revocation and cleanup. |
| `tldw_chatbook/Workspaces/project_index/runtime.py` | One app-owned service per data authority and shutdown integration. |
| `tldw_chatbook/Workspaces/project_index/lifetime.py` | Stable inherited native binding lease and exact crash-restart retirement evidence. |
| `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py` | Existing-token management card with no lifecycle ownership. |
| `Tests/Workspaces/project_index_support.py` | Real temporary-store/binding fixtures shared by the new test modules. |

Existing seams modified by their owning tasks: `Workspaces/recovery.py`,
`DB/private_sqlite.py`, `Backup_Recovery/participants.py`,
`Agents/local_tool_provider.py`, `Chat/console_agent_bridge.py`,
`Tools/workspace_file_roots.py`, `UI/Screens/settings_screen.py`,
`Widgets/Console/console_workspace_files_modal.py`, and `app_lifecycle.py`.
Keep the existing one-shot agent tool-worker protocol unchanged.

## K1 Versioned cache and authority contracts

**Files:** Create `models.py`, `DB/Workspace_Project_Index_DB.py`, and project-index `recovery.py`; extend `Workspaces/recovery.py`, `DB/private_sqlite.py`, `Backup_Recovery/participants.py`, `DB/Workspace_DB.py`, `DB/recovery_operations.py`, and `Workspaces/registry_service.py`. Add the next Workspace DB migration through its canonical runner. Create `Tests/DB/test_workspace_project_index_db.py`, `Tests/Backup_Recovery/test_workspace_project_index_owner.py`, and `Tests/Workspaces/project_index_support.py`; extend the existing Workspace DB/registry migration tests.

**Interfaces:** Define frozen records with no generic owner names accepted from user data:

```python
ProjectIndexKey(data_authority_id: str, workspace_id: str, binding_id: str,
                locator_fingerprint: str, root_identity_fingerprint: str)
ProjectIndexPolicy(include_prefixes: tuple[str, ...], text_suffixes: tuple[str, ...])
ProjectIndexScope(key: ProjectIndexKey, epoch: int, processing_revision: int)
ProjectIndexAuthority(key: ProjectIndexKey, root: LocalRoot, chain: DirectoryChain,
                      exclusions: tuple[SensitiveExclusion, ...], authority_revision: int)
ProjectIndexStatus(scope: ProjectIndexScope, state: str, discovered: int,
                   indexed: int, cache_bytes: int, last_scan_at: str | None,
                   job_id: str | None, job_revision: int,
                   manifest_revision: int)
```

`state` is validated against Off/Building/Partial/Ready/Paused/Interrupted/Error.
Use existing LocalRoot, DirectoryChain and SensitiveExclusion types. Define
`IndexAuthorityReader = Callable[[ProjectIndexKey], ProjectIndexAuthority | None]`;
the reader checks that exact workspace/binding, not the global active selection.
Also supply `IndexAuthorityFence = Callable[[ProjectIndexKey], ContextManager[ProjectIndexAuthority | None]]`
from the registry-backed runtime adapter. A plain reader is insufficient at a
publication or disclosure boundary.
The registry owns a durable monotonically increasing per-binding authority
revision, advanced in the same transaction as an authority-affecting mutation.
Retain a tombstone across deletion/recreation so the witness cannot repeat.
The existing process-local `mutation_generation` is not this witness. A scoped
registry read fence uses its existing SQLite write-serialization boundary;
lock order is registry fence, then project store transaction, never the reverse.
Hold it only for bounded admission/publication/disclosure acceptance, not across
scanning, model/network calls, worker joins or cleanup. Registry mutators do not
synchronously acquire a project-store lock or call back into the service.
K1 also defines `ScanEntry(relative_path: str, kind: str, identity: str,
mtime_ns: int, size_bytes: int)`, `FileCapture(relative_path: str, identity: str,
mtime_ns: int, size_bytes: int, content_sha256: str, text: str)`, and
`ScanPage(directory: str, directory_identity: str, entries: tuple[ScanEntry, ...],
captures: tuple[FileCapture, ...], directory_complete: bool, authority_revision: int)`.
`kind` accepts file/directory only. K1 defines the K4 `IndexCandidate`,
`ProjectIndexHit` and `ProjectSearchResult` records as part of this shared
contract; K4 supplies their query behavior. K2 consumes these page records rather
than defining a competing transport/store type.

Store these control/cache files under the active private data authority at
`project_index/<opaque binding namespace>/<complete-key namespace>/workspace_project_index.db`.
The stable workspace/binding accounting parent owns authority-scoped
`<sha256(data_authority_id)>.lifetime.lock` files and includes obsolete authority
generation namespaces in its bounded quota. Counting old bytes grants no authority.
Earlier flat unreleased scaffolding is not inferred or migrated.
The namespace is a fingerprint of the complete key, never a raw project name.

`WorkspaceProjectIndexDB(path: Path)` owns schema v1. Methods:
`load_status(key) -> ProjectIndexStatus`,
`configure(key, policy, *, enabled: bool, expected_scope: ProjectIndexScope) -> ProjectIndexScope`,
`prepare_scan_page(page: ScanPage) -> PreparedScanPage`,
`commit_scan_page(scope, job_id: str, page: PreparedScanPage, *, expected_job_revision: int, authority_revision: int) -> bool`,
`lexical_candidates(authority: ProjectIndexSearchAuthority, query: str, *, limit: int) -> tuple[IndexCandidate, ...]`,
`revoke(key, *, clear: bool, expected_scope: ProjectIndexScope | None = None) -> ProjectIndexScope`, `clear_revoked(scope: ProjectIndexScope) -> bool`, and `close() -> None`.
Use typed `ProjectIndexKey`/`ProjectIndexScope` on every method. Before the page
transaction, transform captures to chunks using ChunkingService's default
400-word/100-word-overlap policy and retain character-to-line mappings; metadata,
text, chunks and directory completion still publish in one transaction.
Each manifest/chunk membership change advances `manifest_revision` in that same
transaction. Reserve an active job with a durable owner/fencing token and job
revision before worker admission; page commits check that claim as well as scope.
Another process cannot start a competing writer. Resume/takeover must establish
old-owner retirement and replace the claim transactionally; elapsed time alone
does not establish retirement. K1 exposes exact-claim transactional replacement,
gated by code-issued retirement evidence prepared outside all fences; K3 owns
native process/contained-worker observation and the service orchestration.
The registry adapter accepts a required code-owned current-data-authority
callback. K3 issues that identity from its actual admitted private profile/store
and restore/activation generation, qualifying profile switch and restore; no
constant production identity or global active-workspace fallback is permitted.

- [x] **Write store/owner tests first.** The support module supplies `make_index_key(tmp_path) -> ProjectIndexKey` and `open_index_store(tmp_path) -> WorkspaceProjectIndexDB`. Pin `assert store.load_status(key).state == "Off"`, a committed chunk discoverable by FTS, `assert not store.commit_scan_page(old_scope, job_id, page, expected_job_revision=old_job_revision, authority_revision=page.authority_revision)` after revoke, and rollback of manifest/text/FTS/frontier/revision together. Test duplicate page delivery, hostile SQL-looking queries, case/path normalization, key mismatch, corrupt schema, and restore generation invalidation. Exercise Workspace migration rollback, monotonic authority witnesses across remove/recreate, and two connections/processes competing for a job claim. Owner tests assert cache declarations are intentionally excluded, admitted, and retired under maintenance rather than unknown files.
- [x] **Run:** `python -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py -q`. Expect missing production contracts or the intended assertions to fail.
- [x] **Implement contracts and schema v1.** Tables cover singleton identity/settings/epoch/manifest revision, active-job claims, directory frontier/completion, files, chunks/FTS, and cleanup receipts. Manifest/text/FTS/frontier updates share a parameterized transaction. Add the authority revision/tombstone and scoped serialization to the existing Workspace owner, advancing its schema through the canonical migration path. Keep raw locators out of persisted keys/logs. A namespace contains only one binding, making per-binding physical-byte accounting and cleanup possible. Register the exact private-SQLite owner and Workspaces recovery declaration before opening production handles. Initialize `_kernel_sqlite_owner`, `db_path_str`, `is_memory_db`, `_init_store_kernel()` and `db_path` before schema work; register the exact repository and finite-worker closer. Reuse installed admission/activation and finite owner retirement; do not declare an archive payload for rebuildable data.
- [x] **Verify the new files plus owner inventories:** `python -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py Tests/DB/test_workspace_db.py Tests/Workspaces/test_workspace_registry_service.py -q`. Expect PASS, including failed-open/maintenance cleanup and authority-revision migration. Run scoped Ruff and `git diff --check`.
- [x] **Commit:** `git commit -m "feat(workspaces): add folder-bound project index storage"` with exact task paths and complete Backlog evidence/ADR notes.

## K2 Bounded scanner with durable directory receipts

**Files:** Create `Tools/project_index_scan_protocol.py`, `Tools/project_index_scan_worker.py`, and `Workspaces/project_index/scanner.py`; extend `models.py`. Create `Tests/Tools/test_project_index_scan_worker.py`, `Tests/Workspaces/test_project_index_scanner.py`, and `Tests/Performance/run_project_index_profile.py`.

**Interfaces:** `run_project_scan_worker(stdin: BinaryIO, stdout: BinaryIO) -> int`
accepts one job/root initialization and then bounded directory requests/acks.
The closed protocol has version 1 and Started/Entries/File/DirectoryDone/Error
frames, each <=2 MiB. File text is split into <=64 KiB source-text parts with a
part ordinal and final flag before JSON encoding, so escaping cannot overflow a
frame for an otherwise eligible 1 MiB file. Root locators and file bodies travel
only on pipes.
Parent `ProjectScanner.scan(authority: ProjectIndexAuthority, scope: ProjectIndexScope,
job_id: str, frontier: Iterator[str], emit: Callable[[ScanPage], bool],
stop: threading.Event, *, admit_unit: Callable[[str], ProjectIndexAuthority | None], policy: ProjectIndexPolicy) -> ScanRunResult`
retains and retires one process tree. The parent calls `admit_unit(directory)`
before every bounded directory/page request, including continuation pages. K3
supplies this callback using its current authority fence and the job's narrowing
floor; None stops admission. The acknowledged revision is carried on each page.
K2 defines `ScanRunResult(status: str, directories_completed: int,
discovered: int, issues: tuple[str, ...])` in the shared models; status accepts
complete/partial/paused/interrupted/error.
`ScanPage` carries at most 256 metadata entries, bounded text captures, the
directory identity, and whether enumeration actually completed. False from
`emit` stops admission/publication. Root identity is pinned for the job; authority
is not indefinitely frozen. Before each bounded directory request/read admission,
the parent checks the current registry witness and sends that unit's revised
exclusion set/revision, or stops the worker. The worker acknowledges the exact
admission revision before new descent/reads. Previously admitted reads may finish,
but their results must pass the later publication/disclosure fence. Exclusion
removals never widen a running job's admission floor; a fresh operation is needed.
Authority is never inferred from a path.

- [x] **Write real-worker tests.** `test_wide_directory_is_not_listing_truncated` places a marker beyond 200 and 10,000 entries. Assert `all(len(page.entries) <= 256 for page in pages)`, `max_queued_text_bytes <= 4 * 1024 * 1024`, and `max_active_reads <= 2`. Hold a directory acknowledgement, add an exclusion in another process, then prove the next unit cannot read that subtree; also prove removal does not widen the old job. Cover changed root identity, junction/symlink escape, invalid frame size, excluded subtree, invalid UTF-8, 1 MiB+1 file, unreadable directory, and depth that would overflow recursive traversal. An eligible 1 MiB file made of JSON-escaped characters must round-trip in bounded parts. A Unicode filename containing newlines must remain a single data field. Use OS-supported fixtures; report platform skips explicitly.
- [x] **Run:** `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q`. Expect missing scanner behavior to fail.
- [x] **Implement the fixed scanner/transport.** Use os.scandir and persisted frontier work, bounded pipe frames/acks, the existing root pin and ExecutorProcessTree. Do not import ML/Textual/database code into the worker. Enforce sensitive/exclusion checks before descent; apply the spec's processing filters without adding a gitignore dependency. Emit incomplete-directory receipts on I/O failure/mutation. Restart re-enumerates an interrupted directory idempotently; no iterator offset certifies completion. Add fixtures that count actual worker exit and inherited handle retirement after cancel.
- [x] **Verify:** Run the two new test files and existing `Tests/Tools/test_workspace_root_pin.py`. Run `python Tests/Performance/run_project_index_profile.py --files 20000 --directories 20000 --wide-files 20000` against generated temporary corpora; it reports counts, peak RSS, queued bytes, throughput, cancellation latency and corpus/machine metadata. This command is a new harness defined by this task, not an existing benchmark claim. Results must prove the bounds; no tree-sized path list is allowed even when model output is only ten hits.
- [x] **Commit:** `git commit -m "feat(workspaces): scan project trees in bounded resumable pages"`; record actual Windows/POSIX qualification and any unsupported platform cases in task notes.

## K3 Job service incremental publication and revocation

**Files:** Create `service.py`, `runtime.py`, and the stdlib native `lifetime.py`; extend scanner/new-index protocol/worker only for inherited lifetime handoff, `models.py`, the store, `Workspaces/registry_service.py` (`save_runtime_binding`, `remove_runtime_binding`, `set_folder_binding_access`, `add_binding_exclusion`, `remove_binding_exclusion` notifications), and `tldw_chatbook/app_lifecycle.py::_shutdown_app_owned_lifecycles`. Create `Tests/Workspaces/test_project_index_service.py` and `Tests/Workspaces/test_project_index_recovery.py`.

**Interfaces:** `ProjectIndexService(store_factory, scanner, authority_reader: IndexAuthorityReader, authority_fence: IndexAuthorityFence)`;
`enable(key, policy, *, expected_scope: ProjectIndexScope) -> ProjectIndexStatus`,
`refresh(key, *, expected_scope: ProjectIndexScope, expected_job_revision: int) -> str` (job ID),
`pause(job_id: str, *, expected_scope: ProjectIndexScope, expected_job_revision: int) -> ProjectIndexStatus`,
`resume(job_id: str, *, expected_scope: ProjectIndexScope, expected_job_revision: int) -> str`,
`disable(key, *, expected_scope: ProjectIndexScope) -> ProjectIndexStatus`,
`clear(key, *, expected_scope: ProjectIndexScope) -> ProjectIndexStatus`,
`status(key) -> ProjectIndexStatus`, `revoke(key) -> None`, and
`async aclose() -> None`. Application methods run off the UI thread.
`expected_scope` is compared transactionally before any effect, including Enable
from Off; effective enablement/policy transitions advance its epoch or processing
revision. Pause/Resume/Refresh also compare the exact active job revision/state.
Internal `revoke` is code-owned invalidation, never an unchecked user action.
`get_project_index_service(app) -> ProjectIndexService` is lazy after data-authority
admission; `async close_project_index_service(app) -> None` is called by app
shutdown, independently of whether a Console view exists.

- [x] **Write lifecycle tests with real stores and worker barriers.** With two processes, hold a batch after its authority read but before fenced publication, detach/exclude/retarget/clear in the other process, then release. Assert `published is False`, no ready state returns, and competing active-job claims cannot create two writers. Test delayed Enable/Disable/Clear in both orderings and stale Pause/Resume against a replacement job. Test unchanged files, rename/delete, interrupted resume without duplicates, incomplete subtree preservation, narrowing processing policy versus widening+Refresh, repeated cancellation, cleanup failure, and two sessions sharing one index without retargeting. Physical worker exit and held-connection retirement must be observed separately from coroutine completion.
- [x] **Run:** `python -m pytest Tests/Workspaces/test_project_index_service.py Tests/Workspaces/test_project_index_recovery.py -q`. Expect the intended missing service/publication assertions to fail.
- [x] **Implement the single lifecycle owner.** One durable claimed writer owns transactions and status publication; no UI component owns a job. Under K1's registry-then-store fence, compare live authority revision/exclusions, scope, active-job token and job state, then commit the bounded batch. Registry notifications invalidate local views after commit; they are not the cross-process correctness mechanism. Reconcile a changed durable registry witness before any new processing/query admission, even when no callback fired. Content hashes establish capture identity; stat hints only avoid work. Reuse K1's chunk publication path and manifest revision. Only completed directories retire unseen paths. Enforce physical cache bytes including SQLite sidecars/temp headroom before publication; quota overflow pauses with Partial coverage. Disable/Clear revoke before terminating workers or deleting cache data. Clear removes enablement/consent and reclaims validated v2 derived payload across old root/authority generations with native lifetime proof, retained epoch/claim/lease state and journal/temp-inclusive headroom. Fresh caches use incremental auto-vacuum configured before table creation; no forced VACUUM migration or DB replacement. Insufficient headroom/unproven legacy remains Off with a truthful outcome. Startup leaves interrupted jobs inert until Resume; restore/authority changes invalidate readiness.
- [x] **Verify:** Run the two new test files plus K1/K2's targeted files. Observe the app lifecycle with no Console instantiated, a held writer, and a second profile. Confirm there is no background reopen/rebuild after Clear. Run scoped Ruff and `git diff --check`.
- [x] **Commit:** `git commit -m "feat(workspaces): own project index refresh recovery and revocation"`; update the Backlog AC/notes after actual cancellation and maintenance evidence exists.

## K4 Permission bounded index search through local tools

**Files:** Optimize query-local eligibility in `DB/Workspace_Project_Index_DB.py` with focused DB equivalence tests (measured p95 misses; preserve all semantics/admission). Extend `service.py`, `models.py`, `Agents/local_tool_provider.py`, `Chat/console_agent_bridge.py`, `Chat/console_chat_controller.py`, `Chat/console_interrupt_rounds.py` (exact per-call approval provenance), and `Tools/workspace_file_roots.py`; add a read-only captured local-chain accessor in `Tools/workspace_tool_executor.py` with focused root-pin coverage. Create `Tests/Agents/test_project_index_tool.py` and extend `Tests/Chat/test_console_workspace_folder_note_contract.py`.

**Interfaces:** Consume K1's `IndexCandidate(relative_path: str, file_identity: str,
content_sha256: str, line_start: int, line_end: int, rank: int)`,
`ProjectIndexHit(relative_path: str, line_start: int, line_end: int, excerpt: str,
origin: str)`, and `ProjectSearchResult(status: str, hits: tuple[ProjectIndexHit, ...],
coverage: ProjectIndexStatus | None, refresh_required: bool)`. Query status accepts
ready/partial/unavailable; origin is `workspace_file`.
Tool coverage counters use the same admitted path/exclusion predicate as results;
never copy unrestricted Settings counters into an older, narrower run. Use None
for unavailable scoped coverage rather than expose aggregate excluded counts.
`ProjectIndexService.search(authority: ProjectIndexSearchAuthority, query: str, *, limit: int = 10) -> ProjectSearchResult`
filters candidates then validates live bytes via the admitted pinned read executor.
`ProjectIndexSearchAuthority` is code-issued per call by the local provider from
its `RunAdmittedWorkspaceRoot`: it carries the exact scope/root, requesting run's
frozen and accumulated exclusion floor, and its admitted executor/guard. It is
not serialized or accepted as tool input. Union that floor with current live
exclusions before ranking, and recheck the durable registry witness at final
disclosure acceptance under K1's short fence. The shared service must not derive
run authority from the global active workspace or only the live registry.
The provider receives optional `project_index_service` and an immutable per-run
scope map. Add `fs_search_index` to the read/path spec sets, never the mutating
set. Arguments: existing alias contract, nonblank query <=2048 characters,
strict integer limit 1..10 (booleans rejected). Keyword mode is the only query
implementation in this delivery.

- [x] **Write integrated provider tests.** Use real LocalToolProvider invocation and real store/worker reads. Assert `len(result.hits) <= 10`, serialized bytes <=16384, correct relative paths/lines, and no fabricated `[S#]` labels. `test_indexed_instruction_text_stays_tool_evidence` places instruction-like text in a file and asserts it creates no system/developer guidance or implicit tool grant. Test `fs_read` Blocked with index Allowed yields no delivery; fs_read Ask still creates review. Remove an exclusion, refresh from a new operation, then search from an older provider: an advanced authority epoch returns unavailable without disclosure. Separately code-issue a compatible current scope retaining the older run floor and prove that floor filters names/counts before ranking; never silently upgrade the old provider scope. Add cross-process late exclusion/detach/file-rewrite barriers, a missing file with stored matching text, traversal-shaped aliases, zero compatible entries, and more unauthorized candidates than top-k. Assert no revoked names, counts, or snippets are revealed and that a live stale read marks Refresh required rather than returning cached text. It never starts a scan; Refresh remains explicit.
- [x] **Run:** `python -m pytest Tests/Agents/test_project_index_tool.py -q`; expect the missing spec/guard assertions to fail before wiring.
- [x] **Implement service search and provider composition.** Apply eligible binding/generation/path filters before ranking, then live identity/hash checks before disclosure. Search at most 100 candidates per call to fill at most ten hits; report Partial if stale candidates exhaust this budget. Construct excerpts from validated live text. Apply both search review and the effective fs_read narrowing floor without persisting a new grant. Preserve schema-hash invalidation and existing root path-target/project-instruction preparation. Register/advertise no index tool for Off, incompatible, SSH, or unreadable roots. Add the capability note only after final tool filtering; no global active-workspace lookup.
- [x] **Verify:** `python -m pytest Tests/Agents/test_project_index_tool.py Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Chat/test_console_library_runtime_policy.py -q`. Profile cached lexical candidate lookup on the K2 corpus: p95 <=500 ms on the recorded reference machine; measure live validation separately. Prove ordinary file tools and all Library policy values still work unchanged.
- [x] **Commit:** `git commit -m "feat(tools): search authorized project index candidates"` with task-local files and evidence.

## K5 Settings management Console status and release qualification

**Files:** Add the narrow admitted immutable `service.policy(scope)` read projection and focused stale-scope test, without new storage/DTOs. Create `Widgets/Settings_Widgets/project_index_binding_card.py`; integrate it through `UI/Screens/settings_screen.py::_render_workspace_folder_bindings`. Extend `Widgets/Console/console_workspace_files_modal.py` with status and a Settings navigation action. A narrow `UI/Console_Modules/workspace.py` snapshot/navigation handoff is allowed if the modal cannot consume the existing captured binding safely. Change source `_console.tcss` and rebuild the bundle only if mounted layout evidence requires it. Create `Tests/UI/test_workspace_project_index_settings.py`, `Tests/UI/test_console_project_index_status.py`; update the K2 performance harness, `Docs/Architecture/README.md`, `console-file-authority.md`, and `rag.md`; create `Docs/Architecture/project-indexing.md` and update the Console user guide.

**Interfaces:** `ProjectIndexBindingCard(binding_id: str, *, workspace_id: str, ...)` carries exact selected-binding input and projects immutable
`ProjectIndexStatus`; an action message carries the exact binding ID, expected
scope, job ID/revision where applicable, and requested action. Enable/Refresh/Pause/Resume/Disable/Clear and inclusion
controls call K3 via workers. The service rejects stale actions before effects;
view fencing alone is insufficient. Apply a completion only if the view's binding,
job, and epoch still match. Widget construction/compose and Console status
projection perform no filesystem/database/network reads.

- [x] **Write mounted UI regressions.** Off/Building/Partial/Ready/Paused/Interrupted/Error display the correct controls, last scan and discovered counts. Hold Enable before its worker begins, complete Clear, release Enable, and assert no new enablement/job/read; also test an old Clear against a later Enable. Assert a late callback from binding A cannot update B, blocked/unavailable status stays truthful, SSH explains unsupported indexing, and Settings is reachable from Console. Exercise keyboard focus and readable error copy; do not add conflicting global/terminal keybindings.
- [x] **Run:** `python -m pytest Tests/UI/test_workspace_project_index_settings.py Tests/UI/test_console_project_index_status.py -q`. Expect absent controls/dispatch to fail. Resolve the current workspace row renderer by its existing `registry.list_folder_bindings` use before editing; put management behavior in the new card rather than enlarging the screen's business logic.
- [x] **Implement the token-backed controls and background projections.** Reuse existing settings/status/action patterns and the existing Settings navigation contract. Show approximate storage, last scan activity, partial coverage and actionable limit/error state. Use discovered counts until totals are known. Semantic controls are absent until the separate semantic delivery. Rebuild source CSS only if a source stylesheet actually changed.
- [x] **Qualify the release in an isolated app/profile.** Enable, build, query a deeply placed marker, pause/resume, restart+Resume, refresh modified/deleted files, detach during a held query, and Clear. Hold a real production query operation and prove its admitted storage pin delays maintenance/restore replacement until actual native read retirement; isolated encrypted restore/rollback must invalidate old scope/key. Run the final service/store/tool pipeline on at least20,000 text files across20,000 nested/sibling folders and a separate20,000-file single directory (literal spec AC6); retain baseline/peak-extra-RSS, whole-binding disk/sidecars, throughput, candidate p50/p95, live-validation and physical retirement/cancellation receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/`. Prove missing embeddings and blocked sockets do not affect keyword operation. Run all new project-index test files plus `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_component_pattern_governance.py`, and `Tests/UI/test_css_bundle_sync_guard.py` only if relevant styles changed. Scope Ruff; `git diff --check` must pass.
- [x] **Commit:** `git commit -m "feat(settings): manage optional workspace keyword indexes"`. Finish task notes, measured limits, documentation, self-review and ADR linkage before Done.

## Delivery gate and next deliverable

Keyword indexing ships when the management journey, live-result permission and
revocation races, physical worker/DB retirement, and both scale corpora are
qualified. Embeddings/Chroma must not be imported or required by this path.
Opt-in semantic work begins only under its separately reviewed plan.
The user selected Subagent-driven execution. The
[pre-implementation review](../qa/2026-10-20-workspace-project-context/plan-review.md)
records the durable authority/job fence, refreshed scanner admission, per-run
search floor, and stale-action CAS corrections included here.

## Execution clarifications from current-source preflight (2026-10-21)

The approved behavior and ADR-240 remain unchanged. K1 also owns the exact Workspace recovery catalog successor in `DB/recovery_operations.py`, preserving frozen v8 and qualifying the next canonical schema. K4 wires the actual provider constructor in `console_chat_controller.py`, preparing service/scopes off the UI thread.

K1 defines code-only `ProjectIndexLiveReader = Callable[[IndexCandidate], FileCapture | None]` on `ProjectIndexSearchAuthority`. K2 implements `ProjectScanner.capture_file(authority: ProjectIndexAuthority, relative_path: str, *, stop: threading.Event) -> FileCapture | None` through one bounded private Capture request in the new index protocol. One worker capture helper supplies identity, SHA256 and strict UTF-8 from the same descriptor/read for both scans and live validation. It checks ancestors/reparse/exclusions, no-follow opening where supported, pre/open/post identity and the 1MiB+1 read ceiling. No frontier enumeration or durable writer claim is used for query capture.

K4 issues the adapter only after search plus effective fs_read review, retaining that run's exact admitted root/chain/guard, frozen/accumulated exclusion floor, cancellation/time/output budgets and current narrow restrictions. Short registry admission/disclosure fences never span worker I/O. The ordinary one-shot WorkspaceToolExecutor protocol/result contract and remote bundle are unchanged. Pure recovery imports must not load ChunkingService/runtime; parent chunking uses explicit 400-word/100-overlap arguments. Registry fencing uses explicit `transaction(immediate=True)` without nested kernel transactions. Clear retains advanced durable CAS history even while Off.

Existing Workspace DB and registry baseline: 90 passed, 1 inherited warning in 162.31s. Keyword implementation uses a separate branch from guidance43054ffa6f and its own private test TEMP; no full sweep, saved Library migration or semantic work is authorized.

K1 execution clarification: prepare raw ScanPage into an immutable PreparedScanPage outside both registry/cache fences, with chunks and line mappings ready. Publication accepts prepared data only and rejects raw pages; it performs no lazy/fallback chunking. This resolves the original raw-page annotation against the binding short-fence rule. K2 still emits ScanPage; K3 prepares before entering its registry fence, then commits the prepared page. The exact private-SQLite inventory documentation row is also K1-owned coverage.

## K5 native registry connection retirement extension

Ruling: Extend K5 narrowly to retire newly opened registry thread-local connections at both real writer and supervisor boundaries using existing operation_owned_connection, injected by runtime as a finite connection-ownership context (not authority/admission caching). Actual held-query recovery diagnostic identifies db.workspaces leases owned by stopped project-index-writer and project-index-owner threads after Capture retirement, blocking maintenance. Cost is one service/runtime callback and deterministic normal/error/cleanup-failure drain regressions. Native-retired flags, lifetime/read permits and operation pins must not be released before required native handles have positively closed; cleanup failure remains unproven and retains ownership. No foreign-thread close, SQL transaction across worker I/O, global storage contract change or retained admission interval is authorized. Existing ADR240/126/228 applies; update task plan before source work.
Allowed source extension: Workspaces/project_index/service.py and runtime.py plus focused service/recovery tests. Use the existing operation_owned_connection helper to close only newly opened handles on their owning thread. Preserve borrowed handles and all per-call admission/activation/current authority/job CAS. Arrange cleanup before retirement success and pin/permit release; a close failure must keep admission and report unproven retirement. Capture the existing actual stopped-thread failure as RED, then prove post-writer and post-query maintenance drains and actual encrypted rollback/old-key refusal. Performance batching/retained leases remain unapproved.

## K5 bounded multi-directory unit amendment (before source work)

Ruling: Amend the index-only scan unit to allow multiple directories under one freshly admitted bounded read/publication unit, rather than add retained admission intervals or retain per-directory preflight. Idle12-file writer27.558s versus actual scanner0.083s and220 storage acquisitions show per-directory control cost makes the literal20k shape impractical; publication-only batching leaves that preflight cost. Cost is a closed index-protocol v2, immutable aggregate unit DTOs, cache v3 migration/durable frontier marker and fresh boundary/replay/resource qualification. Each unit has TOTAL<=256 discovered entries, <=256 directory attempts including empty directories and <=256 raw iterator inspections, <=256 pending paths, existing combined4MiB text/scratch ceiling and2MiB frames, <=2 shared active reads; parent authority/exclusions/exact Building claim admission precedes its first descent/read and current final registry->store fence commits all receipts atomically. A change after admission can affect only this bounded in-flight unit and cannot publish without the final current fence. No job-wide authority freeze, generic admission cache, SQL transaction across scanning, ordinary executor protocol change, Library mutation or semantic work. This explicitly supersedes K2 plan per-directory/page cadence; spec/ADR previously said bounded directory/read unit without defining multiple directories.

Approved minimal implementation: ScanUnit/PreparedScanUnit with ordered independent directory identity/complete/more receipts and one admitted revision; retain existing ScanPage preparation/publication as bounded singleton adapters. Closed index-only v2 wire adds Unit/receipts/UnitDone with aggregate sender AND receiver accounting; ordinary WorkspaceToolExecutor and Capture permission semantics stay unchanged. Keep one current directory iterator, explicit freshly admitted continuation, iterative bounded pending paths that may include newly discovered children, and existing no-follow/root/descriptor/exclusion checks before every descent/read. Request/receipt metadata and escaped payloads must fit existing frame/metadata bounds. Unstarted seeds remain durable eligible work. No unbounded empty-directory or filtered-entry loop.

All chunks/line mappings and ordered whole-unit digest are prepared before registry/cache fences. In one activated store transaction under the current registry fence, check exact key/scope/epoch/active owner/job revision/state/current authority/exclusions/quota before ANY unit effect; insert entries before applying directory receipts, retain identity-compatible child completion, record seen membership before complete-directory pruning, increment manifest once, and make digest replay a no-op. Incomplete receipts never prune; rejected final admission or capacity rolls back the whole unit. Interrupted enumeration restarts safely; iterator offsets never establish coverage.

Cache schema v3 adds directories.attempt_revision INTEGER NOT NULL DEFAULT 0 and index(complete,attempt_revision,path). Terminal complete/blocked/incomplete receipts write current claim.job_revision; more=True continues explicitly. Eligible frontier reads complete=0 AND attempt_revision<claim.job_revision ORDER BY attempt_revision,path LIMIT256, using the index and no visited set/lexical cursor. Existing job revision advances on explicit new run/Resume, retrying incomplete work once while preserving completed directories. New children default0 even when their path sorts behind previous seeds. Keep pending_directories all-incomplete semantics for truthful final Partial. v2 migration preserves controls/claim/frontier; v1 unproven quarantine remains. Any source-discovered contradiction must be reported before changing this contract.

Allowed source/test expansion: Workspaces/project_index/models.py, scanner.py, service.py; Tools/project_index_scan_protocol.py and project_index_scan_worker.py; DB/Workspace_Project_Index_DB.py; their existing focused protocol/scanner/store/service/recovery tests and performance harness. No global storage-admission/participant API or dependency changes. Runtime remains allowed for the separate finite registry-handle cleanup only.

TDD/qualification: aggregate entry/raw-inspection/empty-attempt and frame/source bounds on both ends, empty/deep/wide shapes, bounded pending state, parent-child ordering/prune/digest replay/atomic capacity rollback, continuation/death/Resume, children inserted behind prior frontier keysets, efficient query plan, current exclusions/remote Pause before NEXT unit and revocation before publication, unchanged ordinary executor and shared read budget. First run the SAME12-file passive production phase probe after this change; retain earlier32.594s pipeline/27.558s writer as baseline and record call counts/source. Do not start literal full-corpus rerun until root checks the measured result. Existing source-qualified tests need only focused amended coverage plus final required task selection, not repeated unchanged Library suites. ADR required: yes, amend existing ADR240 with cache migration and private protocol boundary; no new ADR duplicate.

## K5 actual query/preparation connection ownership extension

Ruling: Extend K5 finite registry ownership to the entire actual fs_search_index invocation before provider preflight/guard/exclusion callbacks, controller scope preparation before its first guard, and direct service/key entrypoints. Source confirms the recovery fixture wrapper concealed a production registry-handle leak. Reuse registry_connection_scope() exposing only the existing runtime-injected operation_owned_connection context, tiny index-only provider wrapper, controller whole-preparation wrapper and nested service/key ownership. Cost is narrow provider/controller integration and real guard/exclusion/close barrier regressions. No new token broker, threadlocal authority state or admission cache. Index query pin/read slot/done spans actual Capture and index-store retirement; an outer registry connection independently retains its own counted db.workspaces native lease until positive close BEFORE the provider entry returns. Maintenance remains blocked across both owners. Registry close failure must not return successful evidence and retains the actual counted connection. Writer/supervisor stricter cleanup-before-retirement remains unchanged. Staged recovery may exit the proven-retired query process before actual parent encrypted rollback and fresh admitted old-key refusal; label those stages, not simultaneous or same-live-service recovery.
Allowed additional paths: Agents/local_tool_provider.py (index-only invoke_detailed wrapper, existing body moves mechanically to _invoke_detailed); Chat/console_chat_controller.py (entire _prepare_project_index_scopes finite owner before first guard); focused Tests/Agents/test_project_index_tool.py and existing controller composition tests, plus authorized service/runtime/recovery tests. Expose ProjectIndexService.registry_connection_scope() as finite ownership only. service.search context starts before first guard, _store/_start management and runtime key helper also close their newly opened registry handles; nested contexts preserve borrowed ownership. Real production guard/exclusion callback tests must remove the test-only query wrapper and prove maintenance held after inner query pin retirement but before actual outer native registry close, successful drain after close, and counted connection retained on close failure/no successful evidence. Do not alter ordinary tools/permissions/wire/Library controls. Actual encrypted rollback uses valid pre-replacement config and fresh selected child; no post-publication rewrite/identity stand-in.

## K5 strict finite connection retirement proof

Ruling: Use an index-specific strict runtime connection-owner adapter around existing operation_owned_connection rather than change generic close/helper semantics. Source SQLiteStoreKernel.close swallows native failure/denied close, so helper exit alone cannot prove retirement. Snapshot borrowed/current handle without opening it using the same existing _core_cached_connection ownership pattern; after helper exit require positive native SQLite closure and registered-core retirement of the exact newly owned/replacement handle. Live/unknown outcome raises ProjectIndexRetirementUnproven, retains actual counted ownership and prevents successful evidence; borrowed caller handles remain untouched. Cost is a small deliberately local ownership/proof adapter and actual swallowed/denied-close regressions. No private lease mutation, foreign close, new admission API, generic helper option or authority caching. Reuse this strict scope at new K5 outer boundaries so a plain outer owner does not conceal the failure.
The existing index store close_native pattern uses the native sqlite3.Connection.in_transaction descriptor as a no-I/O closed-handle witness. Existing _core_cached_connection also records completed registered native retirement; inspect the exact captured handle, not merely an erased thread-local slot. Runtime may expose a narrow code-owned project-index registry scope helper for controller/UI/key callers without constructing a service before their first guard. It is connection ownership only, no authority acquisition/cache. Use the same strict adapter for writer/supervisor, actual complete index invocation, direct service/key and newly added K5 UI/preparation owners. Existing generic DB close/default operation_owned_connection behavior stays unchanged. Tests inject actual captured native close failure swallowed by kernel, denied close, borrowed active connection preservation, complete invocation barrier/drain and no successful evidence on failed close. This supersedes any assumption that plain helper exit proves positive closure.

Batch migration exact-path refinement (before file creation): tldw_chatbook/DB/migrations/workspace_project_index_003.sql is allowed for the approved file-backed v2->v3 cache migration. Store catalog verification must validate the exact eligibility index as well as canonical tables. Preserve a validated v2 cleanup_only catalog path so Clear can reclaim old namespaces without migrating/allocating an index first; v1 unproven refusal remains. No generic recovery-owner schema/catalog change is authorized without a concrete source requirement.

## K5 migration-only binding capacity/lifetime guard

Ruling: Extend the runtime factory and cache constructor with a migration-only native ownership/quota context for v2->v3. Source confirms existing-path construction precedes writer lifetime and current configuration budgets only one DB; CREATEINDEX can grow materially near the whole-binding cap. Detect legacy version on an admitted native connection outside SQL transaction, acquire exact stable BindingLifetime nonblocking, count all binding namespaces/sidecars and reserve DELETE-journal plus existing1MiB headroom, refuse if current page_count already exceeds allowed max, then recheck schema and migrate atomically under that cap. Cost is one narrow injected constructor context and scaled near-capacity/refusal/rollback/serialization coverage. Hold capability through commit/rollback and failed-constructor native closure; unproven closure retains counted ownership/capability, refuses affected work and reports unresolved retirement. Never release on elapsed/PID proof. Currentv3 status opens skip migration lease, and validatedv2 cleanup_only remains nonmigrating. No generic recovery/storage owner change; production factory supplies exact binding context, standalone direct-store fixtures retain a bounded local default rather than infer binding authority.
Workspaces/project_index/runtime.py is additionally allowed for this exact factory migration callback, beyond finite connection cleanup. Constructor may accept an optional code-owned migration context; production factory always supplies the exact current binding accounting/lifetime guard. No registry/cache authority fence spans native lease acquisition, whole-binding disk accounting, worker I/O or joins. Initial version read occurs without a SQL transaction; transaction rechecks after capability acquisition. v3 ordinary status/search construction must remain usable during an active build. Failed admission/allocation/SQLITE_FULL leaves the validated v2 cache intact and Clear-able, no partial v3 catalog. Prove actual scaled near-capacity migration refusal/atomic rollback, other-namespace bytes and journal bound, concurrent native lease busy, successful migration preserving claim/frontier/control, and cleanup-only v2 without index allocation. Existing v1 unproven quarantine remains. If positive failed-constructor close/ownership retention cannot use a narrow existing seam, report concrete gap before inventing an ownership framework.

## K5 existing filename/path keyword requirement

Ruling: Include filename/path-only keyword matches in K5 cache v3 while that schema is being implemented. The accepted spec explicitly requires FTS over paths and text/filename support, but source indexes chunk text only and files without chunks are undiscoverable. Use a path column in the same FTS, index each chunk path, and one zero-length path-search chunk for an eligible file with no text chunks (including empty/whitespace-only cases). Cost is a v3 transactional SQL rebuild/backfill and focused query/migration/live-validation tests plus final latency qualification; no separate search service, new mode/permission/result contract, retained Library data or semantic work. Existing candidate filtering before ranking/counts and live identity/hash Capture remain mandatory; path text never bypasses disclosure gates. Source timing from K4 text-only cache is historical, final v3 timings must qualify actual path+text workload.
Allowed within existing store/models/003 migration/scoped DB/provider tests. v3 FTS may use explicit path,text columns; preserve current query argument validation, escaping/tokenization semantics, per-run filtering-before-rank/LIMIT, <=10/16KiB output, live identity/hash and unchanged ordinary file tools. Native Capture already supports empty UTF8; line1/zero-length excerpt must remain honest, with no cached-text fallback. Preparation guarantees one zero-length chunk only when a valid captured file yields no ordinary chunks; SQL migration/backfill covers existing such files without Python whole-cache materialization. Migration remains under authorized native lease/whole-binding headroom/transaction/rollback, and v2 cleanup_only retains its old catalog. Test unique filename/path term absent from body, eligible empty/whitespace-only file, exclusions before rank/count, deleted/changed live source omission, singleton/unit publication and successful/failed v2->v3 migration. Preserve previous failed timings/receipts; measure final path+text schema on retained actual corpus. No new UI or query mode is needed.

Migration failure retention refinement: Ruling: Retain failed-constructor migration store/capability pairs in one runtime-factory-owned map keyed by affected binding, then refuse further construction for that binding and make close report unresolved after normal service drain. Source has no prior failed-constructor owner. Cost is one local failure-retention map (at most one pair per affected binding) and actual native-close-failure/refusal/close-report coverage; no retry broker, foreign close, lease mutation or authority restoration. Actual counted connection and native binding capability remain held until positive ownership retirement or process exit; repeated requests cannot accumulate pairs for the same binding.

## K5 management recovery and responsive close

Ruling: Add immutable management_snapshot(key)->(existing status, existing policy, requires_upgrade bool) over validated nonmigrating current-cache open, and use it for Settings/Console projections and Clear completion. Actual public Clear otherwise attempts v2 migration first, and UI normal status failure hides the expected scope/Clear action under pressure. Cost is one read projection/store-bound status helper and mounted old-cache-under-pressure journey, no new DTO/storage/status enum. Legacy copy states upgrade required, offers exact-scope Clear and explicit Enable/Refresh, and never claims query-ready coverage or auto-polls a legacy Building job. Public Clear uses the same code-owned cleanup open before mutation. Existing normal status/search/explicit upgrade consumers remain current-schema paths; visiting management surfaces never silently upgrades.

Ruling: Move service aclose stop/snapshot lock acquisition into asyncio.to_thread INSIDE its existing shielded drain, and do the analogous finite composition snapshot within shielded runtime close. Source holds these locks across factory/admission, now potentially migration work, so synchronous acquisition can freeze Textual shutdown. Cost is small async boundary/cancellation/heartbeat tests, no changed dispatch lock order, new lifecycle owner or early cancellation/orphan return. Both runtime and service close finish actual drain/proofs before surfacing caller cancellation.

Allowed existing paths: service/runtime; new card/Console modal; focused new UI tests and service/recovery tests. factory.open_cleanup is code-owned validated nonmigrating open for existing namespaces; fresh namespace behavior stays latest schema. management_snapshot reuses bound admitted status/orphan handling and exact-scope policy, returning only immutable values plus validated in-memory upgrade flag. UI displays upgrade required with actionable copy, no Ready claim, retained exact scope/job data, Clear plus explicit Enable from Off or Refresh when enabled. No automatic legacy Building poll. Test actual mounted current-v2 cache under insufficient migration capacity -> management load -> Clear without DDL/index allocation, scopes/policy preserved, after-Clear state truthful. Normal schema-unavailable/unknown caches still refuse; never infer authority from path. Service/runtime close shielding includes initial off-thread lock snapshot, actual thread/native drain and factory unresolved assertion; held-migration heartbeat and repeated cancellation must not finish/orphan early.

## K5 processing-choice fidelity and default coverage

Ruling: Default the new indexing controls to all eligible UTF8 text and faithfully round-trip empty include/suffix tuples as all. Source UI arbitrarily defaults to only.txt/.md/.py/.rst, substitutes DEFAULT whenever either saved field is empty, and rejects blanks although the policy contract defines blanks as all. Cost is concise input/placeholder copy, removal of that reset/rejection and focused mounted policy/manifest coverage; opt-in builds may process more eligible files under existing generated/sensitive/file-size/entry/disk limits. Explicit narrowing remains available and saved Library choices remain untouched. No new checkbox/mode or permission is needed.

Use existing Input semantics: blank folders = all folders, blank suffixes = all eligible UTF8 text. Preserve exact saved tuples in projection (including src-only with all suffixes, all prefixes with .py-only, and fully empty all/all). A literal root inclusion can display '.' without replacing an empty tuple with it implicitly. Validate nonempty values through ProjectIndexPolicy; do not reject valid blanks or replace a policy because only one field is empty. Defaults impose no arbitrary four-extension filter. Existing mandatory sensitive/generated/binary/no-follow/1MiB gates apply. Show clear token-backed labels/placeholders, no new controls/schema; test mounted round-trip and actual valid extensionless/manifest (.toml/.json or equivalent) inclusion under the default, while explicit suffix narrowing still works. Library controls/scopes remain exactly unchanged.

Processing producer alignment: Ruling: Align scanner empty-prefix/empty-suffix processing with the accepted empty=all contract and existing SQL filtering; source worker._allowed currently treats both as exclude-all. Cost is two local predicate corrections and real default/all/partly-empty plus mandatory-exclusion tests. Archives/generated/sensitive/no-follow/UTF8/size gates remain; this does not introduce a new mode or broaden agent permission. Required before default-all UI can be qualified.


## K5 review scheduling refinement

Ruling: Begin the single independent K5 task review on an exact frozen source commit while the remaining targeted qualification runs, rather than serially wait through the long regression run. Source implementation/formatting is complete and both literal corpora are retired; review is read-only and no second reviewer seat is added. Cost is explicit pending-evidence qualification and a possible scoped rereview if verification later changes source. Reviewer may inspect source now but cannot close the gate until all required final receipts are available and judged; no downstream implementation or task Done is allowed early. Keep the original implementer solely responsible for verification and any later fixes.
ADR required: no new ADR
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
Reason: workflow scheduling only; existing product/storage/security decisions remain unchanged.


## K5 review fix1

Ruling: Run K5 fix1 only in the Settings card and its UI regression file while the already-started backend covering run finishes. The complete review found one Important invalid-draft/action coupling and no core indexing defect; both literal corpora are already retired. Cost is explicit historical-source labeling for the old covering UI subset, superseded by a complete post-fix card-file run. Core/scanner/store/provider hashes remain unchanged, so backend/scale receipts stay applicable; final idle waits all competing tests and final gate waits scoped rereview plus every pending receipt.
Parse and validate draft processing only for Enable/Apply; other actions carry the saved immutable policy and captured scope/job CAS. Add mounted invalid-draft Clear/Disable and validation-preservation regressions. Do not relax ProjectIndexPolicy or change core indexing. ADR required: no new ADR; direct correctness implementation of ADR240.


## K5 fix1 qualification addendum

Ruling: Resolve K5 I2 as a local lifecycle-test factory adapter obligation, preserving unconditional production _AdmittedStoreFactory.assert_retired. Source confines runtime composition to that richer factory; the failing test privately injects a standalone callable-only Rig service after its real drain. Cost is a truthful fixture proof seam rather than silently supporting arbitrary private injection; preserve held native writer/pipes/thread/app-shutdown assertions and prove the adapter assertion runs after retirement. No production getattr/no-op fallback, generic protocol redesign or runtime source change is authorized.
Additional allowed fix path: Tests/Workspaces/test_project_index_service.py, only the failing local injected-factory lifecycle fixture and proof assertions. Run exact failed node and directly affected retirement/cancellation cases; keep original mixed covering result and corrected targeted results separately. Core/scanner/store/runtime/provider production source remains unchanged. ADR240 applies; no new architecture decision.


## K5 final idle workload/fixture qualification

Ruling: Qualify final K5 path+text cached candidate performance with common/rare/miss workloads on BOTH retained completed20k caches,50samples after5warmups each, while keeping20 actual marker Capture/whole-tool samples separately. Current probe measures only the rare marker and has a pre-provider borrowed registry connection from fixture lookup; it cannot close the expensive common-term or complete-owner cost gate. Cost is one corrected bounded query-only probe after preserving/retiring the adverse run, not a folder rebuild or repeated suite. Keep <=500ms p95, current admission/filter/authority paths and exact match counts, no guard/SQL bypass. Close fixture lookup and candidate-only registry scopes before actual public provider invocations so successful outer-owner retirement/cost is real.
Detailed query-only proof contract is retained in the final report/evidence. This refines existing performance verification and creates no product/storage/permission change or new ADR.


## K5 query performance qualification and bounded correction

Ruling: Retain two explicit performance boundaries. The original accepted spec target remains cached canonical lexical_candidates p95 <=500ms, including that method's production activation, guard, SQL filtering/ranking and return. The later root-added admitted-candidate <=500ms target additionally includes the real registry/root/chain fence, management-status read and floor merge. Source audit confirms the expanded benchmark is faithful, not a fixture error; all six expanded cells failed and nested/common canonical577.305ms also failed. Accept the expanded target as an explicit unmet requirement deviation for this delivery rather than weaken authority or change generic storage/config admission; preserve and report both boundaries and all adverse receipts. Cost is roughly0.5-1.2s measured cached admission before live validation on this Windows machine and an unfulfilled stronger target. Original canonical500 remains a gate, and final whole-branch review must independently judge this deviation. Address only the measured repeated exclusion normalization with a per-query immutable predicate; no cross-query cache, shared admission change or new store/service API.
The canonical metric does not include initial store/open/read-pin acquisition, current registry fence or live Capture; those real costs are separately retained as admitted-candidate and whole-tool timings. No raw-SQL-only timing replaces canonical decorated lexical_candidates. The source audit and fix2 brief are retained with the delivery evidence.


## Final integration correction contract

Ruling: Address the complete final whole-delivery review in one consolidated index-specific fix wave: ImportantI1-I4 and MinorM1-M3. ReopenK5 and extend its AC before code; original task gates/receipts remain historical rather than retroactively green. Complete admitted operations must remain owned through outer native registry closure, detached derived caches need durable cleanup-only ownership/retry without source authority, processing prefixes use the already accepted scanner casefold rule at both boundaries, and committed compatible Building rows remain searchable with partial results. Also bound proven-completed run/error history, correct scanner public types/docs and retain dirty processing input on status reload. Cost is focused service/runtime/registry/provider-controller/UI integration and new native/behavior regressions; no generic storage/operation framework, permission broadening, Library mutation or semantic work. I2 concrete durable cleanup-record format/placement and retry entrypoint require a source-backed proposal and root record BEFORE that implementation, not a guessed global collector.

Disable deliberately revokes index use while retaining its owned cache until explicit Clear. Detach removes read/publication authority immediately and must leave sufficient bounded durable cleanup ownership to reclaim already-owned derived namespaces after restart, even when source binding/root authority is gone. Cleanup-only metadata must never grant source access, query eligibility or key reconstruction from a matching path; native capabilities, validated catalogs, exact accounting group, current revocation/replacement checks, positive retirement and quota headroom remain mandatory. Any new cleanup receipt is control metadata only and charged to the binding budget; no project bodies/Library copies or general collector. Its exact storage/permission/migration treatment must be recorded before code.

Index close must reject new finite operations and drain already admitted management, query, key/UI/preparation operations through their outer native registry proof as well as cache/worker retirement. A finite lifecycle record is not retained authority. Repeated waiter cancellation leaves the event loop responsive while physical owners drain; unknown closure remains represented/reported.

Processing include-prefix comparison preserves the current scanner casefold behavior in SQL, including Unicode, exact directory and slash-boundary descendants. Mandatory/run/current exclusions and actual filesystem containment are unchanged. Building is query-eligible only for committed compatible rows under the existing exact scope/root/policy/manifest/liveCapture gates; results are partial, never complete Ready coverage. No uncommitted read or silent scope adoption.

ADR required: yes, amend existing ADR240; no duplicateADR. Reason: clarify index-owned lifecycle/cleanup contract and direct correction of existing accepted producer-consumer/search behavior; no global storage/permission interface redesign.


## Detached cache cleanup ownership receipt (final I2 correction)

Ruling: Accept the narrow I2 cleanup ownership design: fixed binding-group cleanup-owner-v1.lock holds at most4096UTF8JSONbytes {version:1,workspace_id,binding_id,authority_revision}, enrolled before cached content under live exact ownership and actual data/native admission. It is immutable cleanup-only control metadata, never a source key/read grant. Reuse native BindingLifetime handle I/O and existing namespace cleanup; no new schema, generic collector or pathname inference. Post-commit detach/archive hints queue offthread attempts; Settings Workspaces Retry removed-folder cache cleanup exposes bounded cold-restart retry with truthful completed/pending/refused counts. Cost is a small durable private control file, index-only lifecycle/background retry integration and one Settings action; missing/corrupt/unproven legacy detached receipts remain refused until proven ownership can be enrolled while live, not silently erased.

The exact stable accounting group is the existing hash(workspace_id,binding_id) under admitted DataRoot/project_index. Receipt JSON has a closed version1 schema, strict nonempty bounded IDs/revision and maximum4096bytes; it contains no locator/root/source bodies or Library records. The owned cache budget includes this file and native creation headroom. Existing native receipt handle is opened no-follow/single-link under the verified private group; receipt-specific bounded native read/write/flush/truncate uses the held handle, never pathname replacement or a general agent file API. Missing detached receipts are opened existing-only/refused, never created by retry. Interrupted creation may be repaired only with exact live ownership and all existing group native lifetime proofs; malformed detached ownership refuses. New native I/O must preserve private-parent/alias assumptions and positive actual handle retirement.

Acquire receipt and stable native lifetimes outside registry fences. Capture exact live revision before enrollment, perform bounded write/flush outside the registry transaction, then recheck live exact ownership/revision before configure/publication; a changed binding cannot gain index authority from a receipt. Existing proven v2/v3 namespaces may enroll only while live; validated legacy cleanup/Clear must remain usable under capacity pressure without requiring metadata allocation or migration first. Unproven detached legacy receipt cases remain explicit refusals, not cleanup success.

Detached cleanup retains current data admission, receipt capability and all observed generation lifetimes plus the CURRENT data-generation lifetime, repeats bounded complete namespace/catalog inventory, and verifies receipt IDs/group plus a later durable tombstone with absent binding or the same archived workspace. Active/recreated binding refuses cleanup; a distinct replacement binding at the same path has a different group. Current revocation checks and native capability serialization must prevent a same-ID recreate/Enable race from deleting new state. Registry fences span only bounded ownership/revocation acceptance, never native acquisition, joins, worker I/O or bulk reclamation. Existing cleanup_eligible, clear_owned_namespace/reclaim and whole-binding journal headroom remain mandatory. Unknown layouts/native closure keep ownership/pending failure rather than query permission or successful removal.

Attempts/retries run offloop under the complete I1 service operation owner and drain during close; no retained authority, source scan, auto-Refresh, watcher, generic collector or new Library behavior. Durable receipt remains available across process exit so UI retry does not require the old folder row or source reattachment. Disable deliberately retains its cache until live Clear; detach/archival gets revoke-first cleanup/retry. Retained empty control/CAS/receipt metadata is distinct from cached source bodies.

Additional authorized exact product path: tldw_chatbook/Workspaces/project_index/lifetime.py, fixed-purpose bounded receipt capability operations only. New focused regression file: Tests/Workspaces/test_project_index_detached_cleanup.py. No new production file/schema/dependency authorized by this amendment; request exact path if a clean local module is concretely needed. ADR required yes, existing ADR240 amended before code.


### Building compatible coverage proof

Ruling: Permit a narrow bool-only committed-compatible-row existence check for Building scope/query eligibility, sharing the exact SQL processing/exclusion predicate with lexical ranking. Raw indexed counts include newly excluded rows and cannot satisfy I4 zero-compatible coverage. Preserve exact scope/current and applicable run floors/native admission, never fabricate production authority/guards; no names/counts are disclosed by the check. The earlier fix2 no-new-API restriction applied to its performance-only correction, not this final I4 requirement. Cost is additional admitted SQL only on Building eligibility and a small internal store/service seam; Ready lookup remains on its normal predicate path and is requalified on final source.

Final I1 wave source audit: failed supervisor launch must retire its still-running writer outside the service dispatch lock. Preserve retained owners and exact failure/close proof; add held-writer regression before the narrow correction. The current covering run remains earlier-source evidence and only affected checks are rerun. Existing ADR240 and task AC8 apply; no new architecture boundary or second wave.

Final implementation/review: source3b616994e6ff3d3237a08f2802f25c5b5633e693, one consolidated wave and one scoped rereview completed. Four Important and M2/M3 addressed; M1 native-retired history bounded but still-running supervisor-tail pruning remains a real accepted nonblocking Minor, recorded with exact interleaving/future correction in QA keyword/final-re-review.md and execution-notes.md. Final12affected checks pass, all6canonical p95 targets pass; expanded admitted common667/535ms still misses500ms, independently accepted as explicit deviation. Historical20k builds retain source applicability qualifications. ADR240 applies; no further architecture/dependency/Library/semantic change. Branch/worktree/corpora retained; no integration/full sweep.
