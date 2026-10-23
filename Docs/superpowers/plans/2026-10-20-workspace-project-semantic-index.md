# Optional Workspace Semantic Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. The user already approved the feature and subagent execution; proceed without another preference/approval round.

**Goal:** Add explicitly enabled, folder-bound semantic discovery while preserving saved Library choices and keyword operation without embeddings.

**Architecture:** Add immutable semantic membership/vector generations to the existing project SQLite cache, with bounded streaming exact cosine. Reuse the real embedding wrapper under code-issued project admission and offline/destination consent; keep default Library admission unchanged. Merge eligible candidates through the existing live validation and output boundary.

**Tech Stack:** Python>=3.12, existing SQLiteStoreKernel/BindingLifetime and optional embedding dependencies; float32 BLOBs and bounded numerical cosine, no new persistent backend or dependency.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), Index modes, refresh publication, acceptance2-3/5/8-12. [Pinned actual Codex CLI source reference](../specs/2026-10-20-codex-cli-project-context-reference.md).

**ADR required:** yes, amend existing accepted decision before code.
**ADR path:** `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`.
**Reason:** Implements the accepted isolated projection/consent/generation contract with current schema and bounded SQLite storage. This amendment replaces the stale v1->v2 and Chroma allocation assumptions; feature intent remains approved.

**Merged baseline:** PR3204 merged as6ac357de; branch starts from dev2070d9453bcf4f2cacb0b007cff5a41861aee188. Ten relevant store/kernel/lifetime/embedding source files match preflight. Cache is v3, so add004 and fresh v4; preserve prior migrations.

## Global Constraints

- Semantic indexing is a further per-binding opt-in; enabling keyword indexing never implies semantic consent.
- Saved Library choices and item scopes remain unchanged. No shared Library RAG service, Library source tables, or globally searchable project collection.
- File text goes only to the explicitly selected embedding destination under index-specific consent. A destination change pauses semantic work until approved; credentials never enter identity, logs, or stored payloads.
- Missing optional dependencies, a missing local model, or embedding/provider failure leaves keyword search usable; no implicit model download or chat-generation analysis.
- Reuse the keyword file/chunk revisions, authority keys, exact exclusions, inclusion policy, epoch, quota and cleanup owner.
- Fence source pagination and ready publication to one committed keyword manifest revision; a changing source cannot produce a mixed Ready generation.
- <=1 GiB total per binding includes ready/staging vectors and temporary files; preserve <=4 MiB queued source text and all existing scanner bounds.
- Search still returns <=10 results and <=16 KiB, with live file/hash/permission validation before any disclosure.
- Only compatible ready generations are queried. Revocation precedes cleanup, and old workers cannot publish after Clear/detach or an epoch change.
- No embeddings/vector import or model construction on widget compose, ordinary keyword search, or index status projection.
- Targeted test selections only; real-provider qualification is distinct from deterministic test embeddings.
- Sequential implementation ownership: S1a, S1b, S2, S3. No shared-file parallel writers, installer/model download, source-text egress, or automatic semantic PR merge.

## Current-source decisions and interfaces

The original plan is retained in merged Git history. Its Chroma adapter/strict-VectorStore changes are superseded: vectors stay in project SQLite, so unused Chroma/genericVectorStore/generation.query_scope paths remain unchanged. Strict project query outcomes distinguish unavailable from ready-zero. The embedding wrapper's actual constructor, activation, direct/native/model entrances still require exact issued project scope before effects.

`ProjectSemanticSettings` keeps provider_id, model_id, destination_fingerprint, config_fingerprint and approved_destination_fingerprint, adding restart-safe secret-free provider/config references. No raw credentials or credential-bearing endpoints are persisted. `ProjectSemanticState` holds independent semantic_revision, enabled/settings, safe state/reason and optional generation/coverage. Existing keyword state keeps its meaning; semantic changes do not call keyword configure/revoke or advance its epoch. Full Clear/revoke affects both.

`ProjectSemanticGeneration` records generation_id, complete scope, semantic_revision, settings/model/artifact/format/dimension identity, captured manifest_revision, durable claim/checkpoint, state and chunk_count. Immutable members own relative path, filesystem identity, content hash and line map plus float32 vector; no foreign key to mutable keyword chunk IDs. `page_chunks(scope, *, expected_manifest_revision, after_chunk_id: int|None, limit, max_utf8_bytes)` returns a checked immutable page; integer IDs are stable only within that checked manifest.

A source/vector page is <=32 records and <=1MiB of its encoded payload; check SQLite byte lengths before materializing a BLOB/text. Oversized records refuse explicitly. Source retained/queued across work remains <=4MiB. No cursor/transaction spans embedding await, no fetch-all/full matrix/OFFSET pagination. Query scores use a heap of at most100, exact policy/run/current exclusions before rank, deterministic ties, RRF k=60 and path/hash/line deduplication before the existing live choke point.

## Allocation and settlement contract

Use a project-only `allocation.py`, the existing native BindingLifetime primitive and one fixed binding-group `allocation-v1.lock` (zero-payload control file, known/counted cleanup inventory). This avoids repurposing the cleanup receipt and its existing lock order. All cooperating current-version SQLite units use a nonblocking allocation gate outside registry fences. A retained job authority lifetime may precede that try-only gate. Every gate holder uses try-only receipt/lifetime acquisition, unwinds on busy, and never waits/joins while holding it; cleanup returns Pending for a live job. Keyword and semantic writers remain sequential. Read/status/search paths may not silently allocate/reconcile durable control; create/reconcile it inside an admitted mutation.

A finite cleanup/receipt control unit may own the group gate while sequential namespace SQL units borrow that exact live lease. Borrowing is limited to the same group and process/thread/execution; stale or copied contexts and other groups refuse. The control owner holds no SQLite handle. At most one native SQL store/unit is active in the group at a time (same-store nested methods may reuse it); each namespace closes and inventories before the next opens. Overlapping independent reservations could budget against stale other-file lengths and are forbidden. Admit and count bounded receipt growth under the gate, then inventory again. Any child store, receipt, capability or gate settlement uncertainty retains/quarantines the exact outer gate and operation owner. No borrow extends over joins, retries, scanner/provider waits or file Capture; cleanup capability acquisition remains try-only and a live owner yields Pending.

Public read admission uses the unchanged 3s outside-unit retry bound. Measured consecutive finite writer units can starve status at its first constructor admission, so ProjectIndexService may record an ephemeral pending-read count per exact binding group only after status/management/policy receives transient LifetimeBusy. Registration always clears on success, timeout, close or any other exit. After a successful background SQL operation has completely released its fence, native store and allocation gate and finished inventory, a same-service writer with a pending reader yields once for at most25ms using a stop-aware wait, unless stopped or closing. No unconditional per-unit delay, gate-held wait, cached authority/status or new persistent owner is introduced. At most one bounded yield follows a successful unit, so a stream of readers cannot indefinitely suspend the writer. This is a cooperative acquisition window for the existing5ms read retry, not a cross-process fairness or hard scheduling guarantee; foreign contention still reports Busy at3s. Native synchronized contention and physical stop/close tests must verify the chosen policy.

Review clarification for binding-wide admission: before a write-capable outer SQL unit enters its target allocation_store or opens its native handle, the existing group gate owns a bounded, serial, existing-only/read-only peer preflight through the established project SQLite owner. Positively close and inventory each peer before another peer or target opens. Cache only a boolean for this exact finite gate-held unit and discard it on release. Qualify the group when any validated namespace has semantic consent enabled, a non-null ready-generation reference, or generation/member rows. Disabled settings alone, schema-v4 tables alone, and an Off row without generation data do not force semantic runtime qualification; their bytes/freelist still count under the keyword allowance. Unknown/corrupt/aliased/recovery-sidecar peers or uncertain close refuse. There is no permanent semantic marker or new persistence owner. Preflight write-capable constructor/management units, including regular constructors that may migrate; keep cleanup-only/read units read-only, and refuse unexpected read-to-write nesting rather than opening a peer beside a live target. Pure keyword groups retain keyword admission without semantic source qualification. No gate waits, registry-fence reversal or overlapping native SQL store is permitted.

Persisted semantic settings v1 use configuration catalog keys, not raw model paths or connection settings: provider_id matches [a-z][a-z0-9_-]{0,63}; model_id matches [A-Za-z0-9][A-Za-z0-9_.-]{0,127}; provider_ref is exactly provider:v1:<provider_id>; config_ref is exactly model:v1:<model_id>. The model key selects the existing EmbeddingConfigSchema.models entry (which already contains its provider and configuration); no separate configuration-profile system is introduced. Project semantic configuration must refuse unsupported key syntax without rewriting saved Library configuration. destination_fingerprint, config_fingerprint and a non-null approved_destination_fingerprint are exactly sha256:v1:<64 lowercase hexadecimal characters>. Different well-formed approval/destination digests still project Paused. Syntax validation does not prove a caller's arbitrary ID is not a secret; Task2 must resolve code-issued actual provider/model catalog membership and construct fingerprints from non-secret canonical fields before effects or persistence. Task1 rejects endpoint/userinfo/query/assignment/freeform reference material before persistence and exposes no raw invalid values in errors. These metadata formats do not change schema-v4 ownership or create a catalog/runtime in Task1.

For semantic-enabled projection allocations and their competing writes, require verified single-main DELETE journal, NORMAL locking, FULL/EXTRA synchronous, incremental auto-vacuum, secure_delete, foreign_keys, memory temp and cache_spill=OFF on the actual held connection. Forbid ATTACH, WAL transition, savepoints, disk temp, VACUUM and unbounded pruning. Before SQLite open, refuse nonempty journal/WAL/SHM/unknown sidecars rather than trigger unbudgeted recovery or delete evidence. Stop issuing SQL after any SQLite error.

With budgetB=1073741824, old page countN, page sizeP, sector upper boundS=65536, other owned regular-file lengthsO, separately reserved control growthE and observed journalJ0:

`J = 2*S + N*(P+8)`; choose integerC>=N with `O + E + C*P + max(J0,J) <= B`.

Verify actual file length=N*P and actual returned/read-back max_page_count<=C and>=N; never clamp a negative allowance to one page. Recompute each transaction; freelist/deleted rows still count. Preserve old Ready while staging. This reserves every old page, not an estimated touched set. The initial source proof is pinned to probed SQLite3.49.1/source873d4e... plus actual connection/compile/VFS conditions; qualify another build before using the certificate. Unsupported semantic allocation reports setup/unavailable, while pure keyword-only operation retains its established behavior and limits without embedding imports. Do not turn a semantic-only runtime requirement into a keyword dependency.

The envelope covers cooperating current-version processes after old native owners retire. It does not certify concurrent older binaries ignoring the new gate, filesystem cluster/directory allocation overhead, arbitrary extensions or external file writers. Refuse observed mixed/unretired writers; do not claim an OS quota. Existing keyword reserve was unproven for arbitrary new vector transactions, not a demonstrated keyword overrun.

The project transaction owner retains native connection/gate through body, COMMIT, rollback, native close and post-settlement inventory. Each finite SQL unit (including reads that could trigger recovery) closes positively before releasing the gate. No gate, SQLite handle/transaction or cursor spans scanner IPC, embedding/provider await, file Capture, native joins or retry sleeps; only existing job/data/native lifetime owners may span those waits. Acquire the gate before entering a registry fence. Use timeout=0/busy_timeout=0 on the actual project opener/connection; retry SQLITE_BUSY only after the unit has closed/unwound. Refactor constructor pre-open, scanner frontier/admit/emit callbacks, search capture boundaries, revoke joins and supervisor final status into these finite units. This preserves keyword search during real scanner/provider waits. Generic kernel commit failure is outside its body rollback arm; do not change the generic kernel or assume it settled. Uncertain rollback/close/inventory retains ownership/quarantine. If commit logically ambiguous but physical settlement proved, reread CAS in a new admitted operation before reporting/retrying. Failure-status/control writes need admission too; no allocation after full quota merely to record an error. Cleanup v2/v3/v4 stays nonmigrating; v1 quarantine remains.

Primary proof references: [SQLite3.49.1 release](https://www.sqlite.org/releaselog/3_49_1.html), [pinned pager](https://raw.githubusercontent.com/sqlite/sqlite/version-3.49.1/src/pager.c), [temp files](https://www.sqlite.org/tempfiles.html), [PRAGMAs](https://www.sqlite.org/pragma.html). Retain exact source proof and actual native peak evidence separately.

## Review Focus

- Hidden status/control writes, native open hot-journal recovery or commit error can bypass quota; S1a tests actual entrances and retains uncertain ownership.
- Embedding construction can resolve Library sources before activation; S1b instruments actual wrapper/factory/native guards and default compatibility.
- A destination/model change or same-dimension model can reuse stale Ready; S1b/S2/S3 compare full compatibility and consent before effects/publication.
- Current exclusions alone omit a requesting run floor; S2 tests dominant forbidden matches and old-run exclusion removal before semantic top-k.
- Cancellation may finish a coroutine before native work retires, and stale mounted actions can affect a new binding; S2/S3 hold actual workers/owners and captured revision fences.

## File map

| Area | Files |
| --- | --- |
| Allocation/settings foundation | New `Workspaces/project_index/allocation.py`, `DB/migrations/workspace_project_index_004.sql`; existing project models, DB, lifetime, recovery, runtime and service; small `tldw_chatbook/DB/project_semantic_store.py` companion if cohesive SQL needs separation, same owner/kernel |
| Explicit embedding scope | New `RAG_Search/projection_scope.py`, `Workspaces/project_index/semantic.py`; existing activation/native guard plumbing, embeddings_wrapper and Embeddings_Lib; no unused vector adapter changes |
| Generations/search | `semantic.py`, new `semantic_vectors.py` for codec/bounded ranking, same project store/service/runtime/recovery; no second scanner |
| Consent/qualification | Existing ProjectIndexBindingCard, model-free file-root wording, architecture/user docs and retained semantic QA |

## Task 1: S1a cache state and allocation — TASK36420

**Files:** Allocation/settings foundation row above; new `Tests/Workspaces/test_project_semantic_settings.py`, `Tests/DB/test_project_semantic_schema.py`, `Tests/Workspaces/test_project_index_allocation.py`; relevant existing DB/migration/owner/cleanup/native retirement nodes.

**Produces:** v4 exact catalog and immutable semantic records; `load_semantic_state(scope) -> ProjectSemanticState` (read-only); `configure_semantic(scope, settings, *, enabled, expected_semantic_revision) -> ProjectSemanticState`; project allocation/native settlement helper consumed by all new projection mutations, with stable gate and legacy keyword-only availability.

- [x] Write meaningful RED tests for fresh/1->2->3->4 preservation, DDL rollback, independent semantic revision and no eager dependencies.
- [x] Write actual held-SQLite/native RED cases: two namespaces competing for one binding allowance, keyword lookup while a real scanner waits outside a SQL unit, negative cap/no clamping, all control/migration/status entrances, unknown sidecars before open, commit/body/rollback/close uncertainty, old Ready retained and v2/v3/v4 cleanup with v1 refusal. Use scaled budgets for exhaustion plus the analytical formula; sampled peaks alone are insufficient.
- [x] Run only these new nodes first; confirm failures identify absent behavior rather than bootstrap/typo.
- [x] Implement the narrow allocation owner and v4/settings interfaces; preserve keyword rows/policy/epoch/frontier/claims and native ownership. No generic kernel change, semantic job, embedding construction or UI.
- [x] Verify new files and directly affected keyword DB/service/recovery/cleanup controls, scoped Ruff/format/source guards and diff. Preserve child receipts outside numbered pytest temp immediately. Record actual SQLite/VFS proof limits.
- [x] Commit exact task paths and report to parent; independent task review before S1b. ADR241 is amended before code.

## Task 2: S1b issued embedding scope — TASK36421

**Consumes:** S1a state/revision, scope and allocation contracts. **Produces:** `ProjectionExecutionScope`; keyword-only `execution_scope: ProjectionExecutionScope|None=None` and `local_files_only: bool|None=None` in actual wrapper/factory; `_issue_project_projection_scope(service, scope, settings, semantic_revision)` in semantic.py. Binding to exact target/model occurs before constructor effects. None keeps Library behavior.

**Files:** Explicit embedding scope row; `Tests/RAG/test_project_projection_scope.py`, focused factory/offline/logging tests, actual existing activation/native lifetime regressions.

- [ ] Write REDs through actual constructor, source admission, synchronous/async/native entrances, wrong target/forged scope/stale revision/copied or dead parent, repeated cancellation and default Library behavior.
- [ ] Write actual factory offline missing-snapshot/blocked-network and controlled-HTTP credential-bearing success/retry/failure tests. Replace old raw-endpoint logging expectations with safe behavior assertions.
- [ ] Implement exact issued routing before Library inference, retained accepted-parent transfer and actual native/model retirement, offline-only complete-snapshot loading, secret-free config/destination resolution and safe error classification. Do not unwrap decorators, impersonate Library service, weaken broad optional-dependency guards or construct unused Chroma.
- [ ] Verify new scope/offline/privacy tests plus affected existing wrapper/factory/activation defaults. Record true setup-required conditions; deterministic embeddings are not quality evidence. Commit/report/review.

## Task 3: S2 generations and search — TASK36422

**Consumes:** S1a records/allocation and S1b wrapper/issuer. **Produces:** `ProjectSemanticIndex` bounded builder/candidate outcome, service owned semantic jobs and `enable_semantic(key, settings, *, expected_epoch, expected_semantic_revision)` / `disable_semantic(key, *, expected_epoch, expected_semantic_revision)`; model-free separate state.

**Files:** Generations/search row; new `Tests/Workspaces/test_project_semantic_generations.py`, `test_project_semantic_search.py`, `Tests/Backup_Recovery/test_project_semantic_lifetime.py`; affected actual project tool/recovery controls.

- [ ] Write real-SQLite deterministic-vector REDs for synonym corpus, failed rebuild/old Ready, mutation between pages/final CAS, same-dimension model changes, identical-text binding separation, ready-zero versus unavailable, corrupt/oversized/nonfinite vectors and quota refusal.
- [ ] Hold actual work across repeated cancellation, Clear/detach/exclusion/settings changes, competing keyword build and app close; publication must fail and physical borrowers retire.
- [ ] Implement immutable bounded keyset pages, <=4MiB total source ownership, finite float32 codec and heap100 exact cosine. Apply shared exact eligibility predicate before ranking, RRF60/dedup before unchanged live validation, 10hit/16384byte output. No full matrix, source replay scanner or per-tool asyncio.run.
- [ ] Build uses actual async embedding pages on one owned job lifecycle with durable claim/checkpoint; synchronous query embeddings use the existing finite tool worker. Consent/config checks apply at service boundaries before UI exists. Register semantic jobs in the same close/revoke snapshots; sequential keyword/semantic writes.
- [ ] Verify new files and directly affected tool/keyword/native controls, codec/rank/filter behavior, source guards/lint/diff; commit/report/review. Scale/real quality follows in S3 after source settles.

## Task 4: S3 consent and release — TASK36423

**Consumes:** Stable service actions/state/resolver/jobs. **Files:** Consent/qualification row; new `Tests/UI/test_project_semantic_consent.py`, existing mounted project card tests; `Docs/Architecture/project-indexing.md`, `rag.md`, user guides, semantic QA.

- [ ] Read design-language/component patterns before UI. Write mounted REDs for independent controls/selectors, separate readiness, dirty drafts, stale semantic Disable after Clear/re-enable, no model work on compose/status, missing dependency and all saved Library combinations.
- [ ] Exercise cancellation of external disclosure (zero calls), approved destination hold/change/publication refusal and unchanged keyword/Library state. Implement existing-token controls with explicit provider/config refs and captured semantic revision; no generic query(Input) assumption about two processing fields.
- [ ] Qualify actual cached local model through real production factory/issued scope and app: paraphrase discovery, refresh, failed rebuild, restart, semantic disable and full Clear/detach. Candidate snapshot: already cached MiniLM c9745ed...; artifact presence is not loader/quality proof. Missing sentence-transformers requires an isolated dependency setup, never modifying the shared environment or silently downloading model files.
- [ ] Run one source-frozen semantic scale qualification on retained20k shapes, preserving corpora and analytical/native storage envelope; record model/config/dimension, bytes, throughput, memory, query latency and physical cancellation. Test keyword with missing embeddings independently. No favorable timing reruns or universal latency claim.
- [ ] Retain raw RED/GREEN/native/provider/visual receipts as they retire, exact hashes/limitations/rulings and documentation; self review, independent task review, then one whole-branch review/fix wave. Close tasks only on actual evidence. User has not authorized auto-merging a future semantic PR and PR3204 Qodo waiver does not carry.

## Delivery gate

Real-backend quality, supported storage/runtime envelope, exact project admission, compatible atomic generations and physical revoke/cancellation must be qualified. Keyword and saved Library behavior stay intact when semantics is Off/missing/degraded. Deterministic vectors, presence-only model inventory or observed disk peak alone cannot satisfy those gates.


R1 implementation refinement (recorded before owner-policy edit): the actual first GREEN failed because db.workspace_project_index permits only PRIVATE_FILE and refuses READ_ONLY_URI. Authorize only this existing owner's allowed kinds to become _PRIVATE_AND_READ_ONLY, using the existing checked mode=ro, must_exist=True path for serial peer inspection. Keep its production_module, private-path validation, backup/sync exclusion, foreign-source flag, recovery flags, memory exclusion and every other owner unchanged. This introduces no source grant or generic-kernel behavior. Record the new WorkspaceProjectIndexDB._probe_peer call as a separate read_only_uri C99 row in backlog/docs/sqlite-private-owner-inventory.md and deliberately extend only the exact connection census range to include C99 in Tests/DB/test_private_sqlite_inventory.py, retaining its guards. Add actual project-owner read-only/missing-target/write-refusal controls and run matching inventory/registry nodes; no inventory regeneration or unrelated baseline absorption.

Missing allocation gate bootstrap: before first gate creation, a regular write-capable constructor must inspect every existing namespace serially through existing-only read-only handles, with validated inventory and positive native close. An existing v4 target OR peer refuses without creating the missing gate, including repeated/restarted attempts. Only an empty or fully validated legacy v2/v3 group may proceed to a separate gate-held write-capable unit, which rechecks current peers before opening the target. Read/status never creates a gate. Ambiguous aliases, unknown catalogs, recovery sidecars or close uncertainty refuse; uncertain ownership is retained. The preliminary scan grants no write authority and cannot substitute for the gate-held recheck. Costs if wrong: safe refusal or additional finite peer-open work; no repair-by-retry of missing v4 control.


Demonstrated Windows offline-loader correction: actual qualified factory reached model_recovery._closure and failed because that owner imports stdlib os while pinned_directory returns the existing platform filesystem descriptor. Python Windows stdlib os.listdir(fd) is unsupported. Reuse tldw_chatbook.Utils.platform_files.os in model_recovery.py, as native_files already does, so listdir/open/fstat use the existing handle-relative no-follow Windows implementation and unchanged stdlib POSIX behavior. This is a narrow import/adapter correction under existing ADR126 and ADR241, not a new storage/source permission or Windows backend. Do not add path-based enumeration fallback, weaken alias/single-link/identity/manifest/closure limits, change the generic platform shim or bypass model/recovery gates. Retain the actual offline13/16/17 failures and run native closure success/refusal tests plus directly affected model-reference/recovery controls and actual blocked-network offline factory. Report any additional owner need before edits.

### Task2 review correction: retained project model close failure

Ruling: Use only the existing ProjectIndexService operation owner to strongly retain failed project model retirement. Add a private projection_close_pending token-to-actual-wrapper map to _Operation and narrow private mark/settle/check helpers under the service lock. Register the actual project wrapper before factory/model close; keep its actual factory/cache record and carrier/retained parent on any failure, raise/log only classified project_embedding_close_failed without raw cause, and remove only its token after positively successful retry. Carrier close and wrapper use/rebind refuse while its retirement is pending; the service-rooted operation -> wrapper -> carrier -> actual parent chain must survive caller reference loss and GC. A self-contained wrapper/carrier cycle alone is insufficient. Default Library close behavior is unchanged.

Keep projection-close pending separate from generic retirement_unproven. It must not blanket-deny registry_connection_scope or keyword reads merely because a model close failed; new semantic use on the affected binding refuses, while actual keyword lookup remains usable. During explicit service shutdown, detect/report a known unresolved projection close before waiting forever on its not-done operation; retain its real owner and permit positive same-wrapper cleanup retry. Never clear an unrelated native/store/capability uncertainty, claim cleanup success, or release the parent because an exception was swallowed. Serialize close/rebind and prevent embed reuse after partial closure. No SQLite handle/gate/fence spans model close, joins or waits.

Authorized additional exact product path is tldw_chatbook/Workspaces/project_index/service.py, limited to this existing operation-owner retention/reporting seam. Existing Task2 projection_scope.py, embeddings_wrapper.py and actual factory paths may consume it; no generic recovery framework, new persistence owner, source grant or Library behavior change. Test actual factory record retention after injected close failure, sanitized logs/errors, caller-reference loss/GC, carrier and affected-semantic admission refusal, actual keyword lookup availability, explicit shutdown unresolved result, and successful physical retry followed by parent retirement. Root inspected actual factory close (cache.clear only after all record closes), service operation roots/admission and shutdown waits before this amendment.

ADR required: yes, amend existing ADR241 before the cross-module owner change. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: narrow completion of accepted physical retirement and keyword independence, reusing existing project operation owner; no duplicate ADR.
