# Folder-bound project indexing

Optional keyword indexes belong to an explicit local workspace folder binding. They are off by default, have no filesystem watcher, and never create Library, Media, or Notes records. Semantic discovery is a further per-binding opt-in with separate model/destination disclosure and approval. Governance: [ADR-241](../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md), [ADR-103](../../backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md), ADR-126 (storage admission), ADR-174 (binding exclusions), and ADR-150/161 (UI).

## Source map

| File | Contract |
| --- | --- |
| `Workspaces/project_index/models.py` | Immutable key, scope, policy, status and evidence records |
| `Workspaces/project_index/runtime.py` | Lazy admitted application service and exact selected-binding key capture |
| `Workspaces/project_index/service.py` | Explicit management CAS, single writer supervision, live-validated search, native retirement |
| `DB/Workspace_Project_Index_DB.py` | Derived per-binding SQLite/FTS manifests, frontier, durable scope/job revisions and policy |
| `Workspaces/project_index/scanner.py` | Bounded wire operations and actual worker-tree retirement |
| `Tools/project_index_scan_worker.py` | Isolated directory enumeration and UTF-8 capture |
| `Workspaces/project_index/lifetime.py` | Native binding lifetime ownership, including inherited child ownership |
| `Workspaces/project_index/allocation.py` | Binding-wide native SQL-unit gate, inventory and held-connection page admission |
| `Agents/local_tool_provider.py` | `fs_search_index` schema/permissions, run exclusion floor, Capture reader and wire budget |
| `Widgets/Settings_Widgets/project_index_binding_card.py` | Exact-binding Settings management, status/policy workers and visit/request fencing |
| `Widgets/Console/console_workspace_files_modal.py` | Captured-binding status lane and safe Settings handoff |
| `UI/Console_Modules/workspace.py` | Console-owned status reader and existing Workspaces Settings navigation |

## Authority, persistence and execution

Keys include data-authority generation, workspace/binding identity, canonical locator fingerprint and full pinned root-identity chain. A matching path string cannot restore authority. Registry mutations and sensitive publication/disclosure share the registry fence across processes. Application callbacks accelerate invalidation and queue one off-thread detached cleanup attempt; durable registry witnesses remain authoritative.

The derived cache is registered with storage activation and maintenance before production opens. It is excluded from sync/export/ordinary backup payloads. A retained storage pin spans each writer or query's actual native work. Cancellation of an asynchronous observer does not establish retirement. Service and runtime close acquire their stop/composition snapshots off the UI loop inside the shielded drain, then await actual owners before surfacing repeated caller cancellation. Complete finite index operations are admitted against the same close fence through their final registry-native close, including Clear, key capture, provider invocation, Settings/Console and controller preparation. Nested calls on the physical owner thread borrow that admission; no authority result is cached. Close waits admitted operations before its final writer snapshot. Unknown native retirement keeps its operation, actual owners and pins and refuses subsequent operations, including reuse of the failed owner thread. Workers close native SQLite handles, pipes and process-tree ownership before admission is released; uncertain retirement remains blocked.

Enable configures a policy without scanning. Refresh explicitly starts a scan. Pause requests native settlement; Resume explicitly continues a retained job, including after process restart. Policy changes can leave coverage Paused until Refresh. Disable revokes use and retains cached content until explicit Clear. Clear revokes publication and reclaims proven owned derived namespaces across old roots/generations, leaving minimal durable control/lifetime records. Cleanup failure must remain visible rather than being reported as successful Off. Unknown legacy retirement is not inferred.

Every management effect carries an expected scope; job actions additionally carry the expected job ID/revision. `policy(scope)` reads saved choices under registry/store admission and refuses stale scopes. Management surfaces use `management_snapshot(key)` to read immutable status, exact-scope policy and a validated upgrade-required flag without allocating an older cache upgrade. A new binding with no cache yields a coherent in-memory Off/default-policy snapshot; a later status read cannot pair a newly created nondefault cache with invented defaults. A legacy cache never appears query-ready; Clear remains available even when migration headroom is insufficient. Processing prefixes compare casefolded path/prefix text at both scan and SQL boundaries, with exact-or-slash-descendant matching; physical containment and exclusion floors remain independent. Empty include-prefix or suffix tuples mean all eligible folders/text; Settings preserves those and partly empty choices exactly. Settings captures these values before dispatch, so an Enable queued before Clear cannot silently reacquire a newer scope. Each card also fences completion by mounted visit, request and exact workspace/binding. Status refresh preserves dirty local input, including edits made before initial load finishes. Successful Enable/Apply updates input only if it still matches the submitted text. Status polling while Building does not scan. Console status uses its captured selected binding and binding generation; Settings navigation occurs after modal-owned lanes retire and the modal closes.

`last_scan_at` is unit publication activity, displayed as **Last scan activity**. New scan-activity and cleanup-receipt timestamps use the shared canonical millisecond UTC `Z` format (ADR-173); existing offset timestamps remain readable. It is not a separate last-success timestamp while a newer build is in progress. Committed compatible Building rows remain discoverable through Console and public search, always reporting partial evidence. A build with no compatible committed rows under the current policy and requesting run exclusion floor is not advertised as useful coverage. The boolean metadata check and ranked query share the same eligibility predicate. Every candidate and final result retains exact scope, authority, manifest and live identity/hash checks. Ready certifies completed enumeration under the current policy; Partial, Paused, Interrupted and Error remain distinct.

## Bounded units and cache upgrades

The index-only wire protocol is version2; ordinary filesystem tools keep their own protocol. Before every bounded unit, the service verifies current job, binding identity, authority revision and exact exclusions. Preparation computes chunks and the complete replay digest outside the final registry/store fence. Publication is one transaction, applying entries before all per-directory identity/completion receipts. Only completed directories prune missing children. A bounded eligible-seed callback queries the next durable page after publication; unstarted seeds remain eligible.

Cache schema3 adds `directories.attempt_revision` and the `(complete,attempt_revision,path)` eligibility index. A terminal incomplete attempt is skipped for the rest of that job revision; explicit Resume advances the revision and retries it once. Continuations remain eligible. This replaces the tree-sized in-memory attempted set and avoids missing new children behind a lexical cursor. Existing singleton preparation/publication APIs remain compatible adapters.

Cache schema4 adds independent semantic consent/control, generation and member tables. Persisted settings accept versioned catalog-key references only: provider:v1:<provider_id> and model:v1:<model_id>, with bounded provider/model key syntax; destination, config and optional approval fingerprints are sha256:v1: followed by 64 lowercase hexadecimal characters. Task1 validates syntax before persistence; code-issued catalog resolution and fingerprint construction belong to the later embedding scope. No credentials or indexed source text belong in these fields. Semantic revision changes independently of the keyword scope; full binding revoke disables semantic use. The current implementation includes settings, code-issued embedding admission, immutable generation building, live-validated vector search and mounted per-folder consent controls. Fresh production status projects Off in memory without creating cache control. Existing v3 keyword reads remain available without a migration; semantic state projects Off until an explicit admitted upgrade/configuration. v2 remains cleanup-only and v1 requires rebuild/quarantine.

The same FTS indexes path and bounded text. A valid captured file that yields no ordinary chunks receives one zero-length path-search chunk, including empty and whitespace-only files. Candidate ranking still applies the requesting run's exclusions before LIMIT. Paths never substitute for live identity/hash validation.

The file-backed003 migration preserves control, claim and frontier state and rebuilds/backfills FTS in SQL. Only legacy migration takes the exact binding capability. Before its transaction, the runtime counts all namespaces/sidecars and reserves DELETE-journal headroom plus1 MiB; native page limits bound allocation. SQL failure rolls back to the validated v2 catalog, which remains available to nonmigrating cleanup. Current-v3 readers do not acquire the migration capability. An unproven failed-constructor native close retains the counted connection and one binding capability in the runtime factory, blocks further affected construction and reports unresolved retirement on close.

The file-backed004 migration preserves the validated v3 keyword catalog and adds only semantic metadata/derived tables. Each current binding SQL unit acquires a try-only native group gate before its registry fence, then closes its actual SQLite connection and inventories settlement before release. The source-pinned semantic allowance uses actual page count/size and whole-group file lengths to reserve every old rollback page, journal headers and up to4096 bytes of control growth. It refuses unknown files, recovery sidecars, aliased namespace directories and observed other busy native owners. Before each write-capable finite unit opens its target, the held group gate inspects existing namespace catalogs serially through the project SQLite owner's must-exist read-only mode, positively closes each handle and inventories settlement. Any enabled semantic consent, ready reference, or retained generation/member row qualifies all group writers for source-checked admission; disabled settings alone do not. An absent gate with any existing v4 namespace refuses before creating control, including repeated attempts; validated v2/v3 migration performs a separate gate-held recheck. Pure keyword transactions retain the earlier aggregate two-copy cap without requiring semantic SQLite qualification. Scanner/provider waits, file Capture, native joins and retry sleeps occur outside SQL units. Cleanup holds a control gate while sequential namespace units borrow its exact lease; no two independent stores overlap. Existing-only v3 reads inventory strictly without creating a gate. The analytical bound applies only to the qualified SQLite build, runtime settings and cooperating current-version writers; it is not a filesystem quota or a claim about unobserved older writers.

Registry connections created by writer, supervisor, complete index-tool invocation, controller preparation and UI operations have finite owning-thread scopes. Scope exit requires positive closure of the exact native handle and registered-core retirement; generic helper exit alone is insufficient because ordinary DB close can swallow failure. Borrowed connections retain their caller's ownership. Outer registry leases can outlive an inner query pin but block maintenance independently until the provider returns; failed close cannot return successful evidence.

## Bounds and evidence

Scan units span multiple directories but contain at most256 discovered entries,256 directory attempts,256 raw iterator inspections and256 pending paths. One native directory iterator survives only an explicit, freshly admitted continuation. At most2 simultaneous file reads and4 MiB of combined retained/source scratch text are allowed; wire frames stay within2 MiB. Eligible UTF-8 files are at most1 MiB. Binaries, archives, generated/dependency/VCS content, symlinks, junctions and reparse targets are skipped. Mandatory sensitive paths and exact binding exclusions apply before descent, reads, ranking and disclosure. Default processing exclusions include `node_modules`, `.venv`, `venv`, `__pycache__`, `dist`, `build`, `.git`.

Each binding is limited to250,000 eligible discovered entries and1 GiB of cache-owned storage, including staging, sidecars, temporary data and future vectors. Exhaustion means Partial coverage. Search returns at most10 hits and16 KiB serialized output, with smaller run budgets taking precedence. Lexical candidates carry only locations/identity; live Capture rereads content and checks current identity, digest, authority and exclusion floor before any evidence is disclosed. Stale or deleted files request Refresh and never disclose cached text.

Keyword search needs neither embeddings nor provider access. `fs_search_index` remains tool evidence with `workspace_file` origins and line locations; indexed text is never project instructions or an authority grant. Existing saved Library retrieval scopes and controls are independent.

Release receipts belong under `Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/`. `Tests/Performance/run_project_index_profile.py` distinguishes the historical scanner-only profile from the runtime/service/store/tool pipeline. Full qualification uses at least20,000 text files across20,000 nested/sibling folders and a separate20,000-file single directory, retaining sampled memory, whole-binding disk, throughput, candidate/live-read timing and physical retirement evidence. Windows native qualification does not imply POSIX or GUI approval coverage.


## Detached cleanup ownership

Before Enable/Refresh can publish content, the live exact binding enrolls a bounded
`cleanup-owner-v1.lock` in its stable accounting group. The version1 JSON contains
only workspace ID, binding ID and enrolled authority revision (at most4096 UTF8
bytes); it contains no root locator, source text or query grant. The existing
native capability opens the file no-follow/single-link under the verified private
group. Handle-based bounded I/O, truncation and flush precede a fresh exact live
witness check. Cleanup does not introduce a cache schema migration. The receipt and creation
headroom count against the binding quota; validated legacy Clear does not require
receipt allocation. Missing/corrupt detached receipts and unproven legacy catalogs
refuse rather than infer ownership from an opaque pathname.

Detach/archive revokes admission immediately. A coalesced off-thread attempt and
the global Settings Workspaces **Retry removed-folder cache cleanup** action use
the same cleanup-only path. The action is available without the former folder row
and can retry after a process restart. It reports completed, pending and unverified
groups. Cleanup retains the current admitted data pin, receipt capability, every
observed generation lifetime and the current-generation lifetime, repeats bounded
inventory/catalog preflight, and requires a later durable detach/archive witness.
It never scans or readmits the source. A live/recreated binding refuses cleanup;
a distinct replacement ID at the same path has a different group. Holding the
native lifetimes prevents same-ID Enable/publication during reclamation. Existing
headroom, `clear_owned_namespace` and incremental reclamation apply; empty control,
CAS and ownership metadata remain. Failed native close retains the actual owners
and fails service close; an ordinary reclamation failure with all owners retired
remains explicitly retryable.

Fully retired writer history keeps sixteen diagnostic records, without exception
tracebacks. Unresolved native owners and physically live supervisors retain their
shutdown join records beyond that tail; close joins them despite repeated observer
cancellation. Scanner typing documents
both singleton `ScanPage` compatibility and production callback/`ScanUnit` handoff.

## Issued semantic embedding scope (Task 2)

`Workspaces/project_index/semantic.py` resolves a selected current embedding catalog key and issues `ProjectionExecutionScope` only after existing-only semantic settings, exact approved destination, semantic revision, scope and live binding checks. The carrier keeps the actual admitted registry/data parent live through a finite embedding operation; all project SQL/gate/fence handles close before model loading or provider waits. Invalid, disabled, paused, legacy, stale, copied or dead authority refuses without configuring or migrating the cache. Project routing precedes Library source inference; a missing carrier still uses the Library path.

The actual wrapper/factory accept an explicit project carrier and `local_files_only`. A project Hugging Face model must be a complete existing snapshot and loads offline with dynamic fallback disabled. A later independent build/query issues a new accepted carrier and calls `EmbeddingsServiceWrapper.rebind_execution_scope(successor)` only after the old carrier and its native children have closed; compatible cached model state may remain but no idle project pin or source permission does. Credential rotation requires a fresh wrapper. Construction and accepted async/native transfer use `RAG_Search/activation.py`; callers finish native work before each carrier closes and close a retained wrapper under its final live carrier.

On Windows, `RAG_Search/model_recovery.py` uses the existing `Utils.platform_files.os` descriptor adapter to enumerate the pinned model directory. This preserves ADR-126 handle-relative closure, alias, hardlink and manifest checks; it does not introduce a path fallback. Offline factory and issued-project composition qualification are distinct from semantic generation/query quality and scale verification.


Selected configured embedding catalog entries precede built-ins. Keyless OpenAI-compatible destinations are valid candidates; the actual default OpenAI provider refuses a missing key before session creation. A selected Hugging Face cache root precedes the global cache and participates in safe compatibility identity. Each issued call rechecks the current safe catalog identity after its finite DB/fence closes. The existing model owner still reviews its retained artifact closure/receipt before encoding; project admission does not duplicate that full-byte review. Full artifact identity remains bound at issuance and rebind.

Failed project model close is classified and strongly retained by the existing `ProjectIndexService._Operation.projection_close_pending` map. It roots the actual wrapper, factory record, carrier and admitted parent across lost caller references. Only successful same-wrapper close clears its token; pending closure prevents new semantic use and retirement of that wrapper's own carrier while keyword search remains independently admitted. An already-admitted peer carrier can retire under its issued identity, original execution, live parent and zero-child checks even while another wrapper is closing. Service shutdown reports known pending closure before waiting indefinitely. Library close behavior and unrelated native-retirement uncertainty are unchanged.


## Semantic generations, search and physical retirement

ProjectIndexService owns semantic jobs and finite synchronous queries; semantic.py and semantic_vectors.py implement catalog admission, finite float32 codec, heap100 exact cosine and RRF60. One captured keyword manifest fences immutable members, durable claim/checkpoint and final Ready CAS. Only local Hugging Face source/provider pages may contain up to 128 contiguous chunks and 1 MiB encoded source text; hosted/other providers remain at most 32 rows / 1 MiB. Validate the complete provider result count and dimensions before append. Each native append transaction remains at most 32 rows / 1 MiB with fresh source/manifest/scope/consent checks and checkpoint advancement after its successful slice. Query vector pages remain at most 32 rows / 1 MiB; exact cosine ranking keeps its unchanged single outer `store.sql_unit()`/registry fence across all pages, rather than a transaction per page. Retained/queued source totals <=4MiB per binding; total owned regular files remain subject to the existing1GiB source-conditioned allocation contract. Failed staging stays Error; only independently compatible priorReady may remain available with a safe refresh-failed reason.

Query preflight uses the production-issued current artifact identity to check compatibleReady before wrapper/model/provider work, with later validation retained for races. Exact current and run-floor policy/exclusions apply before rank; existing liveCapture/hash/permission validation and10-hit/16384-byte disclosure limit remain final. Semantic unavailable differs from successful zero hits; contention or unavailable semantics keeps keyword candidates usable. One wrapper serves a whole build and one finite query; no idle model cache is retained, so release qualification must report construction cost.

Before model construction/encode/close or retry waits, the existing owning-thread registry seam positively retires newly owned/replacement WorkspaceDB handles. Borrowed live caller handles are preserved and semantic admission refuses. No SQL handle/fence/allocation gate or search read slot spans provider/native work. OpenAI factory records explicitly close their already-owned Session; idle/LRU eviction closes before removing records and retains a failed callback.

Known model-close failure roots the actual wrapper/factory/carrier/operation and binding native lease. At most one immediate same-owner retry precedes an explicit non-spinning retry wait. semantic_close_pending(key) is model-free; retry_semantic_close(key) only signals the original owner, including after failed service shutdown. Clear revokes the epoch before reporting pending cleanup. Repeated cancelled build/query observers cannot release real owners or publish staleReady. Generic uncertain registry/native retirement remains separately retained and may require process exit. Positive wrapper/carrier/native retirement precedes successful publication or shutdown.


Semantic source disclosure applies current processing policy and authoritative binding exclusions in bounded keyset SQL before loading text, including retained rows in Partial keyword coverage. Append checks the next eligible IDs and exact source data; final completeness and member eligibility use that same set. Local128/hosted32, native32/1MiB and existing allocation limits remain unchanged. Project OpenAI-compatible wrappers force their finite transport to refuse redirects before resend; default Library transport retains its prior behavior. This nonpersisted choice does not change saved catalog fingerprints.

Candidate and final public-search fences compare the captured semantic revision/settings and exact compatible Ready generation. Revocation or reconfiguration during live Capture returns unavailable without semantically selected hits, even when keyword scope/manifest did not change. Native-positive semantic tests explicitly request the exact SQLite3.49.1/source/options fixture; portable refusal, model-free status and keyword controls remain active on other builds. The required Windows lane first asserts actual qualification and must execute its positives without skips.
