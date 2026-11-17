# Backup and recovery

This document describes the ADR-126 complete local backup and restore subsystem: inventory and data groups, admission/quiescing, capture and encrypted archives, replace/isolated restore with journaled publication and encrypted rollback, recovered-media review, and the startup fence. "Recovery" elsewhere (notes-sync crash recovery, server-sync account recovery, fleet worktree recovery) belongs to those subsystems' own docs.

Qualification keeps native identity and operation verdicts fresh on every call. Only parsed installed evidence is cached, by its path and descriptor-verified file generation (device, file identity, size, nanosecond modification and change times). Windows uses native ChangeTime. Changed, missing, malformed or concurrently replaced evidence is refused; a prior refusal is never retained as an authorization cache.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Service | `Backup_Recovery/recovery_service.py` | `RecoveryService` — one worker composing all engines: backup/restore previews, inspection, extraction, rollback, recovery-copies, profiles |
| CLI + startup unlock | `Backup_Recovery/launcher.py` | `python -m tldw_chatbook.Backup_Recovery` (`inspect / extract / restore / recover / profiles / copies`); `startup_preflight` / `startup_unlock` (master-password prompt, rekey, reset); `minimal_recovery` host |
| Admission | `Backup_Recovery/storage_admission.py`, `admission.py` | `admit_startup()` (called before runtime imports — the fence), `MaintenanceSession`, capture scopes/leases; nested admission forbidden |
| Participants | `Backup_Recovery/participants.py` | `run_finite_local_worker`, installed persistence participants, exact-repository-type → owner-id mapping, participant-coverage requirement |
| Agent worktree directory | `Backup_Recovery/agent_worktree_storage.py` | `prepare_agent_worktrees_base` — ordinary fresh admission for the fixed profile directory while retaining and restoring an enclosing core DB operation |
| Inventory & groups | `Backup_Recovery/inventory.py`, `data_groups.py` | `discover` (no guessed targets on config failure), the 16 `BACKUP_GROUPS` and owner→group map |
| Capture & archives | `capture_service.py`, `capture.py`, `archive_writer.py`, `archive_reader.py` | coherent capture, strict-manifest archive publication, bounded inspection (`CompressionReviewRequired`) |
| Encryption | `crypto.py`, `age_worker.py` | package-owned **age v1**; isolated `python -I` single-scrypt pipe worker; no PATH lookup, no plaintext fallback |
| Restore | `restore_plan.py`, `publication.py`, `journal.py`, `replacement.py`, `later_rollback.py`, `isolated_restore.py`, `inert_extraction.py` | reviewed plans (restore/retire/preserve/issues), journaled publication ("never releases admission"), replace-mode with held original + encrypted rollback, separate-profile restore, inert bytes-only extraction |
| Credentials & copies | `credentials.py`, `rollback_credentials.py`, `recovery_copies.py` | credential-aware staged copies, `--acknowledge-credential-issue`, retained rollback copies |
| Post-restore gating | `activation.py`, `recovered_media*.py` | `execution_allowed` per-owner review gating (imported by app.py), recovered-media review (keep/delete) |
| Profiles | `profile_catalog.py`, `profile_open.py`, `profile_paths.py` | restored-profile locators, open/acknowledge, effective path resolution |
| Schema catalogs | `DB/recovery_core_schema.py`, `recovery_core.py`, `recovery_operations.py` | exact installed `sqlite_schema` SQL per core DB (validation compares actual schemas, not version labels); per-owner adapters; ~20 subsystem `recovery.py` modules feed inventory |
| UI | `UI/Screens/backup_restore_screen.py` (+state), `app_lifecycle.py` | the in-app Backup & Restore screen; `action_backup_restore` entry (Settings Storage + Overview); `request_recovery_restart` guarded quit → CLI restart |
| Legacy helpers | `Backup_Recovery/settings_backup_helpers.py` | the retired settings-backup UI's surviving contracts (named-DB manifest publication, character-card serialization) kept for tests/TTS-profile backup |

Control root is fixed at `<config dir>/recovery/control` (`service_storage.py`); there is no `[backup]` config section — destinations are chosen per operation.

The fixed agent worktree directory is prepared under its own ordinary native
storage admission. Its no-argument Backup_Recovery helper retains a counted outer
AgentRuns operation, suspends only its discovery during that concrete directory
preparation, then restores and revalidates it. The DB operation never grants
filesystem authority; new acquisition still refuses on pause, profile selection
changes or native private-path checks. A refused preparation returns a typed
worktree storage refusal so the retained parent can terminalize an unstarted run.
See ADR-126/228's TASK-35582 AC23-B amendment.


## Backup creation (dataflow)

1. Settings → Storage "Backup & Restore" (or the Overview button) pushes the screen.
2. Preview → `inventory.discover` over the selected configs; group resolution via the owner→group map (unknown owners get no group).
3. `start_backup` → admission: a `MaintenanceSession` with participant coverage checks → coherent capture (SQLite through lock-safe checked capture).
4. Optional age-v1 encryption via the isolated worker.
5. `write_archive` stages payloads under strict manifest names and publishes atomically; publishing inside a selected external source requires an explicit exclusion.

## Restore (dataflow)

1. `start_inspection` → bounded archive inspection (groups, profiles, scope).
2. `preview_restore(mode = isolated | replace)` → a reviewed plan (`restore/retire/preserve/issues/safety_scope`); the CLI prints it and requires typing `restore`.
3. Replace mode additionally requires verified capability, an explicit `--target-config`, typed confirmation, and a **fresh rollback password**.
4. `start_restore` → journal evidence → `finalize_candidate` (fsync, retire originals); the original capture is held and an encrypted rollback archive created. Unselected stored data is **preserved**, not deleted (ADR-126 rev 4).
5. Post-restore: `activation.execution_allowed` gates features per owner until restored generations are reviewed; recovered media lands under `<user_data>/recovered_media` with a keep/delete review flow.

Interrupted operations recover through `recover <operation-id> --finish|--rollback|--abort` with typed confirmation; rollback needs the rollback password; retained copies are listable/inspectable.

## Startup fence and unlock

`admit_startup()` runs before runtime/config imports (literally the first lines of `app.py` and `Web_Server/serve.py`, and several package `__init__`s — `Web_Server`, `RAG_Search`, `TTS`); enrollment failure is a hard `SystemExit("Recovery required: …")`. When config encryption is enabled with a verifier, `startup_unlock` prompts pre-TUI: a wrong-but-readable password offers rekey; an empty one resets encrypted config values (keys removed; chats/notes untouched). Served children refuse the unlock with a plain-language error rather than hanging. Non-unlock recovery reasons get the minimal recovery host. A recovery restart (app → CLI handoff) checks no running operation and no unsaved editors, quits guarded, and spawns a fresh process into the minimal recovery host.

Same-process initialization followers wait on the existing `_changed` condition (TASK-35595). Initializer retirement/error, acquisition close and local-pause cancellation notify under the same lock. The original acquisition/path/provenance checks run at entry and after waking; native admission still follows before any dependent effect. Removing the 10 ms wake timer avoids repeating native path checks under the shared lock while the leader is unchanged. This does not change native cross-process lock polling, deadlines, source validation or lease retirement.

Creation-intent ancestry validation uses the internal visitor of `native_files.pinned_directory` to inspect every existing parent in a single nofollow traversal (ADR-126, TASK-35595). Each descriptor reaches the visitor only after its original owner/mode check and is borrowed only for that call. Missing descendants do not hide existing ancestor intents; visitor errors and native close errors propagate. No settled verdict or directory handle is cached between acquisitions.

Ordinary Windows storage admission can reuse an allowed derivation on the same
live hold after two full derivations agree. Each use counts a lease, then
checks the current NTFS path metadata and authority content under the native
registry gate. A changed or unavailable proof goes through full derivation.
Path proofs belong to the selector evidence that established their scope. See
the [TASK-40608 amendment to ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md#task-40608--ordinary-windows-admission-reuse).

Within one Windows metadata-tree observation, each drive root receives fresh
local-NTFS qualification. A child opened relative to an exact live parent birth
from that same observation may use its structurally established volume proof.
The child still receives fresh identity, disk-type and reparse checks, volume
consistency, security descriptors and ChangeTime. Bottom-up named-path checks
and exact native retirement remain. The proof ends with the observation;
ordinary opens and borrowed descriptors retain full volume qualification.
See [ADR-126 Task32](../../backlog/decisions/126-complete-local-backup-and-recovery.md#finite-metadata-tree-volume-proof-task-40608-task32).

## Failure behaviors and boundaries

| Case | Behavior |
| --- | --- |
| Config parse failure at inventory | No guessed targets — discovery yields nothing |
| FIFO/non-regular config | Refused (non-blocking open, regular-file check) |
| Journal/publication crash | Admission never released by record labels; recovery resumes via `recover` |
| Restore without review | `execution_allowed` blocks feature execution per owner |
| Extract request | Inert bytes to a chosen directory — never profile/schema restoration |
| Fresh-install securing failure | `~/.tldw_cli-data` fallback (ADR-127) with a plain-language diagnostic |
| `run_finite_local_worker` cancel | Only that worker's file-backed caches retire; in-transaction/foreign handles survive |

## Native workflow history

AgentRunsDB schema 29 includes immutable plans, decisions, task results, native
verifier checkpoints, change-window references, reservations and active-time
accounting. Complete profile capture carries these required records through
the existing exact-SQL owner catalogs. Optional agent logs may be absent.
Restored history requires current profile activation and explicit Resume
through the current app owner. The original startup audit projects stopped,
drained and uncertain attempts conservatively. A restored approval supplies
history only. Unknown active intervals keep their charge.

The [workflow qualification receipts](../superpowers/qa/2026-11-01-native-phased-agent-workflows/README.md)
identify actual process boundaries, archives and source versions.

## Governing decisions

ADR-126 (`126-complete-local-backup-and-recovery.md`, rev 4 with amendments), ADR-127 (`127-fresh-install-private-data-root-recovery.md`), ADR-125 (`125-lock-safe-private-sqlite-validation.md`). Specs/plans: the `2026-09-07-complete-local-backup-restore*` series, `2026-09-15-selectable-backup-groups*`, `2026-09-12-python-backup-encryption*` (under `Docs/superpowers/{specs,plans}/`). Tests: `Tests/Backup_Recovery/` (331 files).

## Verified gotchas

1. Prompt archives at schema v4 and v5 are retained and migrate to v6 on a staged copy; v6 is validated against the installed catalog. The schema catalogs are **exact-SQL**: requalification compares actual installed schema text, not version labels — relabeling is forbidden by the module header.
2. `run_finite_local_worker` closes only caches that worker acquired; exact repository classes confer authority ("subclasses confer no authority").
3. `settings_backup_helpers.py` documents a **retired** UI kept alive by tests and the TTS-profile backup contract — the user-facing surface is the Backup & Restore screen.
4. Restoring an old DB does not auto-activate its contents — imported history stays inert until fresh live claims are allocated.
5. The Settings Storage card warns that path changes apply on next launch (active DB handles keep current paths) — backup/restore works on the admitted set, not mid-flight handles.

## Related docs

- [database-layer.md](./database-layer.md) — the stores being captured
- [server-sync.md](./server-sync.md) — the separate server-sync account/key recovery
- [notes-sync.md](./notes-sync.md) — the notes-sync recovery executor

## Archive chronology schema integration (ADR166)

Core recovery retains the frozen predecessor catalogs and appends exact schema85 successors after the preserved schema81/82/83/84 lineages and installed shared Subscriptions layouts. Canvas payload validation remains limited to those exact catalogs. Restricted backup migration admits only the fixed chat84→85 chronology step and its declared column/index/trigger effects. Workspace backup validation retains frozen schema8/9 catalogs and admits the fixed9→10 chronology step before exact successor qualification. The existing schema8 migration refusal remains unchanged. Existing source, quiescence, payload and native-handle guards remain binding.


Portable detached relocation (ADR-256) records outer publication intent only after
`restore_isolated` has durably written `candidate_staged` and retired its source
lease, before namespace registration or publication. A pre-receipt interruption
leaves an inert retained attempt; a completed outer marker resumes the existing
receipt-bound operation. Malformed or ambiguous older pending states still refuse.

The installed `tamagotchi.config` owner includes only the portable copied roaming
canonical JSON and exact dated backups. Publication retains its existing shared-home
owner policy inside the private attempt home. After committed profile validation,
both fresh selection and launch derive portable home/roaming selectors from that
attempt, ignoring inherited APPDATA. A subsequent move acquires those pet files
again. Other files under the pet directory remain unsupported durable content.

## Bounded MCP operation admission (TASK-40608)

An installed MCP raw operation admits its exact ordered member set through the existing related-path acquisition. A recovery-selected source retains one freshly admitted canonical lease during preparation, validates its original binding, process/thread/task and live custody on each observation, then transfers it into the operation's existing retirement handling. Custom/uninstalled sources retain separate ordinary member admissions. No permission payload is cached, and preparation grants no raw-effect authority. Source changes, maintenance pauses and uncertain closes retain their original refusal/custody behavior.

Warm stock Windows Agents guard entries share one finite ordinary activation observation across their ordered source rows. Existing native readers must prove absence of pending/restored state, and a full fresh final observation must match before the body begins. Each unique path keeps an independent lease; every row keeps its owner and namespace check. Cold, restored and custom routes retain their original gates. A new nested entry makes a new observation. A failed native proof is a refusal, never a reason to retry through fallback.

For a qualified top-level entry, `storage_admission._admit_agents_vector` also gathers the common authority inputs and union of all confirmed path metadata once. The first ordinary proof runs under a counted pending attempt before later leases are issued. Every path lease is counted independently before the native union observation, and the existing final ordinary proof precedes execution-state publication. The original once-captured source tuple stays authoritative; this route does not repeatedly invoke service getters. Exact source definitions, path selections, live hold/evidence and lease custody are checked before acceptance. Native metadata handles retire before the result transfers to the guard's existing reverse-order cleanup. Nested-active, cold, mixed and customized entries use the sequential route. The original journey trace's scalar admission counters do not include vector tokens; complete-entry native tests count actual inputs, metadata observations and resources separately.

Warm repository path proofs use a local `_OperationPath` receipt. The coordinator captures exact operation/resource ownership, releases its lock for the original parent `stat` and path resolution, then rechecks ownership, source and dynamic selection under the lock before allowing access. The adjacent same-owner nested entry shares one named proof; different-participant selection/restoration, connection setup, getter/access and admission-wait checks retain their separate proofs. Unregistered operations or participants refuse before repository callbacks. This removes native path waits from those coordinator sections without retaining evidence between commands or changing SQLite resource retirement. Cold scope and startup derivation retain their existing path.

The same ordinary input reader may serve an exact MCP canonical preparation's generation selection before its source lock. All original source/binding/custody checks still run; final fresh validation precedes lock acquisition and discards the shared observation. No observation crosses a lock wait or survives into a later operation. Within one ordinary generation checkpoint, one complete control scan and registry read supply the existing source-coverage, startup and pairing predicates. Pending, restored and custom readers retain their original route; a selected native failure never becomes a retry.

Within each stock Windows observation, `bootstrap._pinned_ordinary_inputs` owns a checked bootstrap directory and opens admission/control records relative to it. It shares the original parsers and retains each newly opened handle's native checks. The original shared registry lock still brackets the registry read and releases before selector fingerprinting. Before returning, the reader checks current named root/admission ancestry, control names and contents, enrollment marker, registry contents/lock and selector fingerprint. Standalone control writers are not serialized by the registry lock, so the final content checks remain necessary. This owner ends with the observation; it does not combine separate permission-command phases or grant atomic-snapshot semantics against arbitrary external writes.

Stock Windows permission load and kill-switch commands select one explicit read operation before generic nested helper setup. The existing raw scope owns admission, the exact source/mutation lock, member and canonical leases, parent pins and physical retirement. Three fresh post-lock proofs cover entry, the protected payload open, and publication after the payload descriptor has positively retired. Metadata-only helpers consume explicit phase inputs and check current source/live custody on each borrow. The original parser and normalization body are shared; corruption repair ends the read phase and uses the original checked write/backup/fsync route before a new final proof.

Each command reads fresh policy. Public load/getter commands retain their publication observer; the captured Console reader consumes the checked receipt without adding that observer effect and retains its inside-lock/current-source checks. Missing-file defaults still require a fresh final absence proof. Customized generation readers, load methods or observers retain the original route. Approval/provider waits and later dispatch commands start new evidence lifetimes. These contracts and work-count controls do not establish the absolute journey target; TASK-40608 remains open until the original paired journeys qualify.

Console display checks use detached presentation data under owner/revision fences between bounded reconciliations; they never carry storage or dispatch authority. See [Console](console.md) and the TASK-40608 follow-up in ADR-126.

The complete ordinary Windows permission persistence surface now selects the
contained permission process before parent raw admission. Its synchronous
checkpoints run the issuing parent's exact source/config/callback predicates,
then refresh the child's original native proof after the wait. Parent reservation
and publication identity remain shared with external `mutation_fence()` composites;
session grants and Tool Packs registry writes remain local. Every command settles
its native resources before returning; startup enrollment remains until the
explicit `Shutdown/Drained` handshake. The runtime maintenance census includes
direct-store helpers, closes intake after upstream producers settle, and resumes
with lazy fresh processes. Weak receiver associations permit healthy idle teardown;
unresolved commands retain their exact source/capture and refuse replacement.

Native predicates retain their original scope: permission parent ACL/ancestry
posture is checked at admission, while protected-open/final parent checks refresh
the named/pinned identity (the stricter mode predicate there applies to config
companions). The process boundary does not add a new continuous parent-ACL policy.
Control/registry, nofollow, leaf identity, missing-destination and retirement
checks retain their fresh original boundaries. See the
[accepted permission-owner contract](../superpowers/specs/2026-11-14-task-40608-isolated-permission-design.md).

The stock Windows manual primary run groups its initial AgentRunsDB row creation and lifecycle sequence reads in one finite core operation. The clock, step constructor and its observers run after that operation retires; the lifecycle insert retains its own transaction and admission. Both original history reads and independently visible commits remain. Exact database selection, wrapped reader definitions and allocator checks guard this route. Custom, memory, child, resumed and workflow paths retain the scalar route. Later events separated by provider, tool, disclosure or approval effects start their own operations.

## Prompt mix-in operational content (ADR-286)

The four conversation-owned `console_prompt_mixin_*` tables belong to the existing
whole-ChaChaNotes backup cohort. They need no sidecar owner, new source authority,
separate data group or reference-counting service. Maintenance capture retains
preferences and exact used bodies even when diagnostic trace capture is off.

The installed v86 schema catalog includes the operational tables, immutable-row
and marker-preservation triggers, dispatch marker, and corrected conversation FTS
purge trigger. Retained v81–v85 catalogs stay historical. Disposable candidate
migration executes only the installed `chachanotes_prompt_mixins.sql` through the
existing restricted authorizer; validation compares the complete resulting SQL
catalog and foreign keys. Runtime constructors and source files are not used as
restore candidates. Existing restored-profile admission and activation gates
remain authoritative.

## Plugins owner (TASK-40132)

`Plugins/backup.py` installs the `plugins` owner: one tree over `<user data>/plugins`
in the `tools` group, shaped like `recovered.media` (`inventory.sqlite_tree_member`
names each mixed tree's single SQLite member for sidecar declaration and capture
dispatch). `registry.sqlite3` is captured through the `recovery.plugins.registry`
SQLite owner (backup route, WAL folded) against the frozen v5 catalog
`REGISTRY_SCHEMAS`; everything else is raw bytes. Excluded: `runtime.lock` and,
inside each `staging/<name>/`, every member but `snapshot/` (an allow-list: only
`snapshot/` and the custody record serve recovery). The owner passes
`_staging_excluded` to the tree walk as `prune`, so each excluded member is one
row whose contents are never listed (a local-source `repository.git` can hold
tens of thousands of files), and replace restore keeps such a directory whole
when the archive excluded it too (`restore_plan`'s unknown-children rule skips
it). A fetch folder the archive lacks that holds excluded members cannot be
retired around them, so `plan_restore` calls `refuse_added_fetch_folders` and the
review refuses with `plugin_fetch_added_since_backup` (owner decision
2026-11-12: refuse early; automatic handling is a follow-up). Capture refuses rather than waits: an unsettled `processes` row or a
held runtime lock reduces the owner to one unavailable root plus a
`writers_active` row, and the inventory adds `plugin_writers_active` after group
selection; each item's copy creates if needed and holds the lock. Restore is in
place only: `staging._config_targets` refuses the owner for a different profile,
`relocate_restore` refuses a registry whose data-root or package paths name
another root (`owner_relocation_unverified:plugins`; the destination root is
resolved as `PluginRuntimeOwner` names it), and detached import refuses
a plugin root by name. The owner is activation-required:
`PluginAdmission.capture` refuses `plugin_restore_review_required` while a restored
generation that lists `plugins` is unapproved, and `PluginService` approves it once
its authority recovery completes. The keyring generation marker is never captured,
so a restored profile without it reports `recovery_required`.
