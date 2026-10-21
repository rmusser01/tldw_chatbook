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

- [ ] **Write mounted UI regressions.** Off/Building/Partial/Ready/Paused/Interrupted/Error display the correct controls, last scan and discovered counts. Hold Enable before its worker begins, complete Clear, release Enable, and assert no new enablement/job/read; also test an old Clear against a later Enable. Assert a late callback from binding A cannot update B, blocked/unavailable status stays truthful, SSH explains unsupported indexing, and Settings is reachable from Console. Exercise keyboard focus and readable error copy; do not add conflicting global/terminal keybindings.
- [ ] **Run:** `python -m pytest Tests/UI/test_workspace_project_index_settings.py Tests/UI/test_console_project_index_status.py -q`. Expect absent controls/dispatch to fail. Resolve the current workspace row renderer by its existing `registry.list_folder_bindings` use before editing; put management behavior in the new card rather than enlarging the screen's business logic.
- [ ] **Implement the token-backed controls and background projections.** Reuse existing settings/status/action patterns and the existing Settings navigation contract. Show approximate storage, last completed scan, partial coverage and actionable limit/error state. Use discovered counts until totals are known. Semantic controls are absent until the separate semantic delivery. Rebuild source CSS only if a source stylesheet actually changed.
- [ ] **Qualify the release in an isolated app/profile.** Enable, build, query a deeply placed marker, pause/resume, restart+Resume, refresh modified/deleted files, detach during a held query, and Clear. Hold a real production query operation and prove its admitted storage pin delays maintenance/restore replacement until actual native read retirement; isolated encrypted restore/rollback must invalidate old scope/key. Run the final service/store/tool pipeline on at least20,000 text files across20,000 nested/sibling folders and a separate20,000-file single directory (literal spec AC6); retain baseline/peak-extra-RSS, whole-binding disk/sidecars, throughput, candidate p50/p95, live-validation and physical retirement/cancellation receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/`. Prove missing embeddings and blocked sockets do not affect keyword operation. Run all new project-index test files plus `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_component_pattern_governance.py`, and `Tests/UI/test_css_bundle_sync_guard.py` only if relevant styles changed. Scope Ruff; `git diff --check` must pass.
- [ ] **Commit:** `git commit -m "feat(settings): manage optional workspace keyword indexes"`. Finish task notes, measured limits, documentation, self-review and ADR linkage before Done.

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
