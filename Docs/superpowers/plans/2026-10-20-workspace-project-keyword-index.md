# Optional Workspace Keyword Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide opt-in, folder-bound keyword discovery over large project trees with bounded resource use and live file authority.

**Architecture:** A project-index service owns per-binding SQLite/FTS caches, a durable scan frontier, and a contained scanner job. A read-only local-tool spec returns bounded, live-validated results. Settings manages indexing and Console only projects its status; Library ingestion and retrieval remain independent.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, stdlib SQLite FTS5/os.scandir, SQLiteStoreKernel, existing pinned-root/process-containment primitives, existing ChunkingService, pytest/Ruff. No new dependency or embedding requirement.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), optional-index sections and acceptance items 2-3 and 5-12. Deliver after [workspace guidance](2026-10-20-workspace-project-guidance.md).

**Reference:** [Codex CLI source findings](../specs/2026-10-20-codex-cli-project-context-reference.md), especially path-search sessions, cancellation, and instruction-cache revocation. This is a durable text index; do not copy Codex's in-memory fuzzy path catalog.

**ADR required:** yes, existing accepted decision.
**ADR path:** `backlog/decisions/240-workspace-project-context-and-folder-bound-index.md`.
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
- Refresh and restart Resume are explicit. No filesystem watcher, automatic whole-tree prompt injection, or scan per send.
- Register storage/activation/maintenance ownership before production opens; no sync/export/ordinary-backup payload for the cache.
- Use only targeted tests unless the user separately requests a full sweep.
- At execution, create/read each atomic Backlog task, set In Progress, then add its task-file Implementation Plan and ADR check. K1-K5 are plan-local labels.

## Review Focus

- Root replacement, case aliases, and junctions can reuse a pathname: K1/K2 test filesystem identity and canonical path rejection.
- Unreadable subtrees and directory mutation can look like deletions: K2/K3 test explicit incomplete-directory receipts and non-destructive reconciliation.
- A blocked `fs_read` can coexist with Allowed index search: K4 tests the narrowing permission floor and Ask review.
- Exclusion/detach can occur while a search or build is finishing: K3/K4 hold the actual publication/disclosure barrier and revoke first.
- A restored profile or delayed status callback can resurrect old readiness: K1/K5 test restoration invalidation and view/job epoch fencing.

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
| `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py` | Existing-token management card with no lifecycle ownership. |
| `Tests/Workspaces/project_index_support.py` | Real temporary-store/binding fixtures shared by the new test modules. |

Existing seams modified by their owning tasks: `Workspaces/recovery.py`,
`DB/private_sqlite.py`, `Backup_Recovery/participants.py`,
`Agents/local_tool_provider.py`, `Chat/console_agent_bridge.py`,
`Tools/workspace_file_roots.py`, `UI/Screens/settings_screen.py`,
`Widgets/Console/console_workspace_files_modal.py`, and `app_lifecycle.py`.
Keep the existing one-shot agent tool-worker protocol unchanged.

## K1 Versioned cache and authority contracts

**Files:** Create `models.py`, `DB/Workspace_Project_Index_DB.py`, and project-index `recovery.py`; extend `Workspaces/recovery.py`, `DB/private_sqlite.py`, and `Backup_Recovery/participants.py`. Create `Tests/DB/test_workspace_project_index_db.py`, `Tests/Backup_Recovery/test_workspace_project_index_owner.py`, and `Tests/Workspaces/project_index_support.py`.

**Interfaces:** Define frozen records with no generic owner names accepted from user data:

```python
ProjectIndexKey(data_authority_id: str, workspace_id: str, binding_id: str,
                locator_fingerprint: str, root_identity_fingerprint: str)
ProjectIndexPolicy(include_prefixes: tuple[str, ...], text_suffixes: tuple[str, ...])
ProjectIndexScope(key: ProjectIndexKey, epoch: int, processing_revision: int)
ProjectIndexAuthority(key: ProjectIndexKey, root: LocalRoot, chain: DirectoryChain,
                      exclusions: tuple[SensitiveExclusion, ...])
ProjectIndexStatus(scope: ProjectIndexScope, state: str, discovered: int,
                   indexed: int, cache_bytes: int, last_scan_at: str | None,
                   job_id: str | None)
```

`state` is validated against Off/Building/Partial/Ready/Paused/Interrupted/Error.
Use existing LocalRoot, DirectoryChain and SensitiveExclusion types. Define
`IndexAuthorityReader = Callable[[ProjectIndexKey], ProjectIndexAuthority | None]`;
the reader checks that exact workspace/binding, not the global active selection.
K1 also defines `ScanEntry(relative_path: str, kind: str, identity: str,
mtime_ns: int, size_bytes: int)`, `FileCapture(relative_path: str, identity: str,
mtime_ns: int, size_bytes: int, content_sha256: str, text: str)`, and
`ScanPage(directory: str, directory_identity: str, entries: tuple[ScanEntry, ...],
captures: tuple[FileCapture, ...], directory_complete: bool)`.
`kind` accepts file/directory only. K1 defines the K4 `IndexCandidate`,
`ProjectIndexHit` and `ProjectSearchResult` records as part of this shared
contract; K4 supplies their query behavior. K2 consumes these page records rather
than defining a competing transport/store type.

Store these control/cache files under the active private data authority at
`project_index/<opaque binding namespace>/workspace_project_index.db`.
The namespace is a fingerprint of the complete key, never a raw project name.

`WorkspaceProjectIndexDB(path: Path)` owns schema v1. Methods:
`load_status(key) -> ProjectIndexStatus`,
`configure(key, policy, *, enabled: bool) -> ProjectIndexScope`,
`commit_scan_page(scope, job_id: str, page: ScanPage) -> bool`,
`lexical_candidates(scope, query: str, *, limit: int) -> tuple[IndexCandidate, ...]`,
`revoke(key, *, clear: bool) -> ProjectIndexScope`, and `close() -> None`.
Use typed `ProjectIndexKey`/`ProjectIndexScope` on every method. Before the page
transaction, transform captures to chunks using ChunkingService's default
400-word/100-word-overlap policy and retain character-to-line mappings; metadata,
text, chunks and directory completion still publish in one transaction.

- [ ] **Write store/owner tests first.** The support module supplies `make_index_key(tmp_path) -> ProjectIndexKey` and `open_index_store(tmp_path) -> WorkspaceProjectIndexDB`. Pin `assert store.load_status(key).state == "Off"`, a committed chunk discoverable by FTS, `assert not store.commit_scan_page(old_scope, job_id, page)` after revoke, and rollback of manifest/text/FTS/frontier together. Test duplicate page delivery, hostile SQL-looking queries, case/path normalization, key mismatch, corrupt schema, and restore generation invalidation. Owner tests assert cache declarations are intentionally excluded, admitted, and retired under maintenance rather than unknown files.
- [ ] **Run:** `python -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py -q`. Expect missing production contracts or the intended assertions to fail.
- [ ] **Implement contracts and schema v1.** Tables cover singleton identity/settings/epoch, scan jobs, directory frontier/completion, files, chunks/FTS, and cleanup receipts. Manifest/text/FTS/frontier updates share a parameterized transaction. Keep raw locators out of persisted keys/logs. A namespace contains only one binding, making per-binding physical-byte accounting and cleanup possible. Register the exact private-SQLite owner and Workspaces recovery declaration before opening production handles. Reuse installed admission/activation and finite owner retirement; do not declare an archive payload for rebuildable data or seed an unrelated database migration.
- [ ] **Verify the new files plus owner inventories:** `python -m pytest Tests/DB/test_workspace_project_index_db.py Tests/Backup_Recovery/test_workspace_project_index_owner.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q`. Expect PASS, including failed-open/maintenance cleanup. Run scoped Ruff and `git diff --check`.
- [ ] **Commit:** `git commit -m "feat(workspaces): add folder-bound project index storage"` with exact task paths and complete Backlog evidence/ADR notes.

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
stop: threading.Event) -> ScanRunResult` retains and retires one process tree.
K2 defines `ScanRunResult(status: str, directories_completed: int,
discovered: int, issues: tuple[str, ...])` in the shared models; status accepts
complete/partial/paused/interrupted/error.
`ScanPage` carries at most 256 metadata entries, bounded text captures, the
directory identity, and whether enumeration actually completed. False from
`emit` stops admission/publication. Authority is frozen for the job and checked
again by the parent before page application; it is never inferred from a path.

- [ ] **Write real-worker tests.** `test_wide_directory_is_not_listing_truncated` places a marker beyond 200 and 10,000 entries. Assert `all(len(page.entries) <= 256 for page in pages)`, `max_queued_text_bytes <= 4 * 1024 * 1024`, and `max_active_reads <= 2`. Cover changed root identity, junction/symlink escape, invalid frame size, excluded subtree, invalid UTF-8, 1 MiB+1 file, unreadable directory, and depth that would overflow recursive traversal. An eligible 1 MiB file made of JSON-escaped characters must round-trip in bounded parts. A Unicode filename containing newlines must remain a single data field. Use OS-supported fixtures; report platform skips explicitly.
- [ ] **Run:** `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q`. Expect missing scanner behavior to fail.
- [ ] **Implement the fixed scanner/transport.** Use os.scandir and persisted frontier work, bounded pipe frames/acks, the existing root pin and ExecutorProcessTree. Do not import ML/Textual/database code into the worker. Enforce sensitive/exclusion checks before descent; apply the spec's processing filters without adding a gitignore dependency. Emit incomplete-directory receipts on I/O failure/mutation. Restart re-enumerates an interrupted directory idempotently; no iterator offset certifies completion. Add fixtures that count actual worker exit and inherited handle retirement after cancel.
- [ ] **Verify:** Run the two new test files and existing `Tests/Tools/test_workspace_root_pin.py`. Run `python Tests/Performance/run_project_index_profile.py --files 20000 --directories 20000 --wide-files 20000` against generated temporary corpora; it reports counts, peak RSS, queued bytes, throughput, cancellation latency and corpus/machine metadata. This command is a new harness defined by this task, not an existing benchmark claim. Results must prove the bounds; no tree-sized path list is allowed even when model output is only ten hits.
- [ ] **Commit:** `git commit -m "feat(workspaces): scan project trees in bounded resumable pages"`; record actual Windows/POSIX qualification and any unsupported platform cases in task notes.

## K3 Job service incremental publication and revocation

**Files:** Create `service.py` and `runtime.py`; extend `models.py`, the store, `Workspaces/registry_service.py` (`save_runtime_binding`, `remove_runtime_binding`, `set_folder_binding_access`, `add_binding_exclusion`, `remove_binding_exclusion` notifications), and `tldw_chatbook/app_lifecycle.py::_shutdown_app_owned_lifecycles`. Create `Tests/Workspaces/test_project_index_service.py` and `Tests/Workspaces/test_project_index_recovery.py`.

**Interfaces:** `ProjectIndexService(store_factory, scanner, authority_reader: IndexAuthorityReader)`;
`enable(key, policy) -> ProjectIndexStatus`,
`refresh(key) -> str` (job ID), `pause(job_id: str) -> ProjectIndexStatus`,
`resume(job_id: str) -> str`, `disable(key) -> ProjectIndexStatus`,
`clear(key) -> ProjectIndexStatus`,
`status(key) -> ProjectIndexStatus`, `revoke(key) -> None`, and
`async aclose() -> None`. Application methods run off the UI thread.
`get_project_index_service(app) -> ProjectIndexService` is lazy after data-authority
admission; `async close_project_index_service(app) -> None` is called by app
shutdown, independently of whether a Console view exists.

- [ ] **Write lifecycle tests with real stores and worker barriers.** Start a page, hold immediately before CAS publication, detach/exclude/retarget/clear, then release. Assert `published is False` and no ready state returns. Test unchanged files, rename/delete, interrupted resume without duplicates, incomplete subtree preservation, narrowing processing policy versus widening+Refresh, repeated cancellation, cleanup failure, and two sessions sharing one index without retargeting. Physical worker exit and held-connection retirement must be observed separately from coroutine completion.
- [ ] **Run:** `python -m pytest Tests/Workspaces/test_project_index_service.py Tests/Workspaces/test_project_index_recovery.py -q`. Expect the intended missing service/publication assertions to fail.
- [ ] **Implement the single lifecycle owner.** One writer owns transactions and status publication; no UI component owns a job. Recheck live authority/exclusions and epoch at every batch. Registry notifications invalidate readiness but are not a substitute for these fresh checks. Content hashes establish capture identity; stat hints only avoid work. Reuse K1's chunk publication path. Only completed directories retire unseen paths. Enforce physical cache bytes including SQLite sidecars/temp headroom before publication; quota overflow pauses with Partial coverage. Disable/Clear revoke before terminating workers or deleting cache data. Clear removes enablement/consent. Startup leaves interrupted jobs inert until Resume; restore/authority changes invalidate readiness.
- [ ] **Verify:** Run the two new test files plus K1/K2's targeted files. Observe the app lifecycle with no Console instantiated, a held writer, and a second profile. Confirm there is no background reopen/rebuild after Clear. Run scoped Ruff and `git diff --check`.
- [ ] **Commit:** `git commit -m "feat(workspaces): own project index refresh recovery and revocation"`; update the Backlog AC/notes after actual cancellation and maintenance evidence exists.

## K4 Permission bounded index search through local tools

**Files:** Extend `service.py`, `models.py`, `Agents/local_tool_provider.py`, `Chat/console_agent_bridge.py`, and `Tools/workspace_file_roots.py`. Create `Tests/Agents/test_project_index_tool.py` and extend `Tests/Chat/test_console_workspace_folder_note_contract.py`.

**Interfaces:** Consume K1's `IndexCandidate(relative_path: str, file_identity: str,
content_sha256: str, line_start: int, line_end: int, rank: int)`,
`ProjectIndexHit(relative_path: str, line_start: int, line_end: int, excerpt: str,
origin: str)`, and `ProjectSearchResult(status: str, hits: tuple[ProjectIndexHit, ...],
coverage: ProjectIndexStatus, refresh_required: bool)`. Query status accepts
ready/partial/unavailable; origin is `workspace_file`.
`ProjectIndexService.search(scope: ProjectIndexScope, query: str, *, limit: int = 10) -> ProjectSearchResult`
filters candidates then validates live bytes via the existing pinned read executor.
The provider receives optional `project_index_service` and an immutable per-run
scope map. Add `fs_search_index` to the read/path spec sets, never the mutating
set. Arguments: existing alias contract, nonblank query <=2048 characters,
strict integer limit 1..10 (booleans rejected). Keyword mode is the only query
implementation in this delivery.

- [ ] **Write integrated provider tests.** Use real LocalToolProvider invocation and real store/worker reads. Assert `len(result.hits) <= 10`, serialized bytes <=16384, correct relative paths/lines, and no fabricated `[S#]` labels. `test_indexed_instruction_text_stays_tool_evidence` places instruction-like text in a file and asserts it creates no system/developer guidance or implicit tool grant. Test `fs_read` Blocked with index Allowed yields no delivery; fs_read Ask still creates review. Add late exclusion/detach/file-rewrite barriers, a missing file with stored matching text, traversal-shaped aliases, zero compatible entries, and more unauthorized candidates than top-k. Assert no revoked names, counts, or snippets are revealed and that a live stale read queues Refresh rather than returning cached text.
- [ ] **Run:** `python -m pytest Tests/Agents/test_project_index_tool.py -q`; expect the missing spec/guard assertions to fail before wiring.
- [ ] **Implement service search and provider composition.** Apply eligible binding/generation/path filters before ranking, then live identity/hash checks before disclosure. Search at most 100 candidates per call to fill at most ten hits; report Partial if stale candidates exhaust this budget. Construct excerpts from validated live text. Apply both search review and the effective fs_read narrowing floor without persisting a new grant. Preserve schema-hash invalidation and existing root path-target/project-instruction preparation. Register/advertise no index tool for Off, incompatible, SSH, or unreadable roots. Add the capability note only after final tool filtering; no global active-workspace lookup.
- [ ] **Verify:** `python -m pytest Tests/Agents/test_project_index_tool.py Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Chat/test_console_library_runtime_policy.py -q`. Profile cached lexical candidate lookup on the K2 corpus: p95 <=500 ms on the recorded reference machine; measure live validation separately. Prove ordinary file tools and all Library policy values still work unchanged.
- [ ] **Commit:** `git commit -m "feat(tools): search authorized project index candidates"` with task-local files and evidence.

## K5 Settings management Console status and release qualification

**Files:** Create `Widgets/Settings_Widgets/project_index_binding_card.py`; integrate it through `UI/Screens/settings_screen.py::_render_workspace_folder_bindings`. Extend `Widgets/Console/console_workspace_files_modal.py` with status and a Settings navigation action. Create `Tests/UI/test_workspace_project_index_settings.py`, `Tests/UI/test_console_project_index_status.py`; update the K2 performance harness, `Docs/Architecture/README.md`, `console-file-authority.md`, and `rag.md`; create `Docs/Architecture/project-indexing.md` and update the Console user guide.

**Interfaces:** `ProjectIndexBindingCard(binding_id: str)` projects immutable
`ProjectIndexStatus`; an action message carries the exact binding ID, epoch,
and requested action. Enable/Refresh/Pause/Resume/Disable/Clear and inclusion
controls call K3 via workers. Apply a completion only if the view's binding,
job, and epoch still match. Widget construction/compose and Console status
projection perform no filesystem/database/network reads.

- [ ] **Write mounted UI regressions.** Off/Building/Partial/Ready/Paused/Interrupted/Error display the correct controls, last scan and discovered counts. Assert a late callback from binding A cannot update B, blocked/unavailable status stays truthful, SSH explains unsupported indexing, and Settings is reachable from Console. Exercise keyboard focus and readable error copy; do not add conflicting global/terminal keybindings.
- [ ] **Run:** `python -m pytest Tests/UI/test_workspace_project_index_settings.py Tests/UI/test_console_project_index_status.py -q`. Expect absent controls/dispatch to fail. Resolve the current workspace row renderer by its existing `registry.list_folder_bindings` use before editing; put management behavior in the new card rather than enlarging the screen's business logic.
- [ ] **Implement the token-backed controls and background projections.** Reuse existing settings/status/action patterns and the existing Settings navigation contract. Show approximate storage, last completed scan, partial coverage and actionable limit/error state. Use discovered counts until totals are known. Semantic controls are absent until the separate semantic delivery. Rebuild source CSS only if a source stylesheet actually changed.
- [ ] **Qualify the release in an isolated app/profile.** Enable, build, query a deeply placed marker, pause/resume, restart+Resume, refresh modified/deleted files, detach during a held query, and Clear. Run the two 20,000-item corpus shapes and retain memory/disk/latency/cancellation receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/`. Prove missing embeddings and blocked sockets do not affect keyword operation. Run all new project-index test files plus `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_component_pattern_governance.py`, and `Tests/UI/test_css_bundle_sync_guard.py` only if relevant styles changed. Scope Ruff; `git diff --check` must pass.
- [ ] **Commit:** `git commit -m "feat(settings): manage optional workspace keyword indexes"`. Finish task notes, measured limits, documentation, self-review and ADR linkage before Done.

## Delivery gate and next deliverable

Keyword indexing ships when the management journey, live-result permission and
revocation races, physical worker/DB retirement, and both scale corpora are
qualified. Embeddings/Chroma must not be imported or required by this path.
Opt-in semantic work begins only under its separately reviewed plan. Recommend
Subagent-driven execution for the storage/worker/service/tool boundaries if the
user chooses delegation; otherwise execute natively with explicit task gates.
