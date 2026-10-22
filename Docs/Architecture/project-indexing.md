# Folder-bound keyword indexing

Optional keyword indexes belong to an explicit local workspace folder binding. They are off by default, have no filesystem watcher, and never create Library, Media, or Notes records. Semantic indexing is a separate delivery. Governance: [ADR-240](../../backlog/decisions/240-workspace-project-context-and-folder-bound-index.md), ADR-126 (storage admission), ADR-174 (binding exclusions), and ADR-150/161 (UI).

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
| `Agents/local_tool_provider.py` | `fs_search_index` schema/permissions, run exclusion floor, Capture reader and wire budget |
| `Widgets/Settings_Widgets/project_index_binding_card.py` | Exact-binding Settings management, status/policy workers and visit/request fencing |
| `Widgets/Console/console_workspace_files_modal.py` | Captured-binding status lane and safe Settings handoff |
| `UI/Console_Modules/workspace.py` | Console-owned status reader and existing Workspaces Settings navigation |

## Authority, persistence and execution

Keys include data-authority generation, workspace/binding identity, canonical locator fingerprint and full pinned root-identity chain. A matching path string cannot restore authority. Registry mutations and sensitive publication/disclosure share the registry fence across processes. Application callbacks only accelerate invalidation.

The derived cache is registered with storage activation and maintenance before production opens. It is excluded from sync/export/ordinary backup payloads. A retained storage pin spans each writer or query's actual native work. Cancellation of an asynchronous observer does not establish retirement. Service and runtime close acquire their stop/composition snapshots off the UI loop inside the shielded drain, then await actual owners before surfacing repeated caller cancellation. Workers close native SQLite handles, pipes and process-tree ownership before admission is released; uncertain retirement remains blocked.

Enable configures a policy without scanning. Refresh explicitly starts a scan. Pause requests native settlement; Resume explicitly continues a retained job, including after process restart. Policy changes can leave coverage Paused until Refresh. Disable and Clear revoke publication before cleanup; Clear reclaims proven owned derived namespaces across old roots/generations, leaving minimal durable control/lifetime records. Cleanup failure must remain visible rather than being reported as successful Off. Unknown legacy retirement is not inferred.

Every management effect carries an expected scope; job actions additionally carry the expected job ID/revision. `policy(scope)` reads saved choices under registry/store admission and refuses stale scopes. Management surfaces use `management_snapshot(key)` to read immutable status, exact-scope policy and a validated upgrade-required flag without allocating an older cache upgrade. A legacy cache never appears query-ready; Clear remains available even when migration headroom is insufficient. Empty include-prefix or suffix tuples mean all eligible folders/text; Settings preserves those and partly empty choices exactly. Settings captures these values before dispatch, so an Enable queued before Clear cannot silently reacquire a newer scope. Each card also fences completion by mounted visit, request and exact workspace/binding. Status polling while Building does not scan. Console status uses its captured selected binding and binding generation; Settings navigation occurs after modal-owned lanes retire and the modal closes.

`last_scan_at` is unit publication activity, displayed as **Last scan activity**. It is not a separate last-success timestamp while a newer build is in progress. Ready certifies completed enumeration under the current policy; Partial, Paused, Interrupted and Error remain distinct.

## Bounded units and cache upgrades

The index-only wire protocol is version2; ordinary filesystem tools keep their own protocol. Before every bounded unit, the service verifies current job, binding identity, authority revision and exact exclusions. Preparation computes chunks and the complete replay digest outside the final registry/store fence. Publication is one transaction, applying entries before all per-directory identity/completion receipts. Only completed directories prune missing children. A bounded eligible-seed callback queries the next durable page after publication; unstarted seeds remain eligible.

Cache schema3 adds `directories.attempt_revision` and the `(complete,attempt_revision,path)` eligibility index. A terminal incomplete attempt is skipped for the rest of that job revision; explicit Resume advances the revision and retries it once. Continuations remain eligible. This replaces the tree-sized in-memory attempted set and avoids missing new children behind a lexical cursor. Existing singleton preparation/publication APIs remain compatible adapters.

The same FTS indexes path and bounded text. A valid captured file that yields no ordinary chunks receives one zero-length path-search chunk, including empty and whitespace-only files. Candidate ranking still applies the requesting run's exclusions before LIMIT. Paths never substitute for live identity/hash validation.

The file-backed003 migration preserves control, claim and frontier state and rebuilds/backfills FTS in SQL. Only legacy migration takes the exact binding capability. Before its transaction, the runtime counts all namespaces/sidecars and reserves DELETE-journal headroom plus1 MiB; native page limits bound allocation. SQL failure rolls back to the validated v2 catalog, which remains available to nonmigrating cleanup. Current-v3 readers do not acquire the migration capability. An unproven failed-constructor native close retains the counted connection and one binding capability in the runtime factory, blocks further affected construction and reports unresolved retirement on close.

Registry connections created by writer, supervisor, complete index-tool invocation, controller preparation and UI operations have finite owning-thread scopes. Scope exit requires positive closure of the exact native handle and registered-core retirement; generic helper exit alone is insufficient because ordinary DB close can swallow failure. Borrowed connections retain their caller's ownership. Outer registry leases can outlive an inner query pin but block maintenance independently until the provider returns; failed close cannot return successful evidence.

## Bounds and evidence

Scan units span multiple directories but contain at most256 discovered entries,256 directory attempts,256 raw iterator inspections and256 pending paths. One native directory iterator survives only an explicit, freshly admitted continuation. At most2 simultaneous file reads and4 MiB of combined retained/source scratch text are allowed; wire frames stay within2 MiB. Eligible UTF-8 files are at most1 MiB. Binaries, archives, generated/dependency/VCS content, symlinks, junctions and reparse targets are skipped. Mandatory sensitive paths and exact binding exclusions apply before descent, reads, ranking and disclosure. Default processing exclusions include `node_modules`, `.venv`, `venv`, `__pycache__`, `dist`, `build`, `.git`.

Each binding is limited to250,000 eligible discovered entries and1 GiB of cache-owned storage, including staging, sidecars, temporary data and future vectors. Exhaustion means Partial coverage. Search returns at most10 hits and16 KiB serialized output, with smaller run budgets taking precedence. Lexical candidates carry only locations/identity; live Capture rereads content and checks current identity, digest, authority and exclusion floor before any evidence is disclosed. Stale or deleted files request Refresh and never disclose cached text.

Keyword search needs neither embeddings nor provider access. `fs_search_index` remains tool evidence with `workspace_file` origins and line locations; indexed text is never project instructions or an authority grant. Existing saved Library retrieval scopes and controls are independent.

Release receipts belong under `Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/`. `Tests/Performance/run_project_index_profile.py` distinguishes the historical scanner-only profile from the runtime/service/store/tool pipeline. Full qualification uses at least20,000 text files across20,000 nested/sibling folders and a separate20,000-file single directory, retaining sampled memory, whole-binding disk, throughput, candidate/live-read timing and physical retirement evidence. Windows native qualification does not imply POSIX or GUI approval coverage.
