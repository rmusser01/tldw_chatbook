# Optional Workspace Semantic Index Implementation Plan

Current governance: [ADR-241](../../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md). Renumbered from project ADR-240 during 2026-10-22 dev integration; historical rulings and delivery notes below retain their original source-era labels.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicitly enabled semantic project discovery while retaining keyword search and the folder binding's live authority.

**Architecture:** Build a separately owned vector generation from the keyword index's committed chunks, then atomically publish a compatible ready pointer. Reuse embedding/vector primitives with an explicit code-issued project execution scope; their existing Library activation/readiness defaults cannot stand in for project authority. Merge eligible candidates before the keyword delivery's unchanged live-validation/output boundary.

**Tech Stack:** Python >=3.12, existing ChunkingService/EmbeddingsServiceWrapper/VectorStore and optional Chroma backend, existing model/egress/recovery controls, SQLiteStoreKernel, pytest/Ruff. Imports remain lazy behind `Utils/optional_deps.py`.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), Index modes, Refresh publication, and acceptance items 2-3, 5 and 8-12. Prerequisite: the shipped [keyword-index delivery](2026-10-20-workspace-project-keyword-index.md), including its permission/revocation/Settings contracts.

**Reference:** [Pinned Codex CLI source inspection](../specs/2026-10-20-codex-cli-project-context-reference.md). Codex's inspected path-search session is not a semantic content index; this feature is an optional Chatbook extension.

**ADR required:** yes, existing accepted decision.
**ADR path:** `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`.
**Reason:** Implements its isolated semantic projection, destination consent, and atomic-generation contract. The shared-adapter scope extension below is necessary to preserve those existing boundaries rather than impersonating the Library service.

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
- Before execution, allocate/read atomic Backlog tasks and add their plans only after In Progress. S1-S3 here are local plan labels.

## Review Focus

- Existing vector/embedding decorators can accidentally resolve Library owners: S1 instruments actual source admission and forbids that crossover.
- Direct activation calls, lazy native collection creation, and native-worker transfers must select the same issued project scope as decorated calls.
- Native query failure currently becomes an empty list: S1/S2 distinguish failure from successful zero hits without changing Library defaults.
- A same-dimension model change can incorrectly look compatible: S1/S2 fingerprint model, destination, chunker, configuration, authority, and epoch.
- A failed native query/build may continue after coroutine cancellation: S2 observes the physical borrower and retirement before publication/cleanup.
- Keyword and semantic candidates may refer to different file revisions: S2 validates revision before merge and uses the shared live-result choke point.
- The user can change embedding destination while an old build waits to commit: S3 holds that boundary and proves pause/revocation without changing Library or keyword availability.

## File structure

| Files | Responsibility |
| --- | --- |
| New `tldw_chatbook/RAG_Search/projection_scope.py` | Narrow code-issued execution/query-scope carrier for a second projection owner. |
| Existing `RAG_Search/activation.py`, `generation.py`, `simplified/embeddings_wrapper.py`, `simplified/vector_store.py` | Accept the explicit scope at every installed entrance; explicit strict query/offline options preserve current Library defaults. |
| Existing `Embeddings/Embeddings_Lib.py` and the wrapper's factory setup | Credential-free destination/error reporting and project-only offline loading. |
| New `tldw_chatbook/Workspaces/project_index/semantic.py` | Project embedding configuration, generation builder, candidate query and readiness. |
| Existing project-index models/store/service/recovery/runtime | Versioned semantic settings, generation pointers, job ownership and revocation. |
| Existing ProjectIndexBindingCard | Per-binding semantic choice/consent and separate keyword/semantic status. |
| New semantic/adapter/generation tests | Source admission, real SQLite generations, deterministic embeddings and dependency-failure controls. |

## S1 Explicit projection ownership and versioned semantic settings

**Files:** Create `RAG_Search/projection_scope.py` and `Workspaces/project_index/semantic.py`; extend the four RAG adapter/guard files in the table, `Embeddings/Embeddings_Lib.py`, `project_index/models.py`, `DB/Workspace_Project_Index_DB.py`, and project-index `recovery.py`. Create `Tests/RAG/test_project_projection_scope.py` and `Tests/Workspaces/test_project_semantic_settings.py`; extend focused wrapper/vector/factory tests.

**Interfaces:** Add keyword-only `execution_scope: ProjectionExecutionScope | None = None`
to `EmbeddingsServiceWrapper.__init__` and `ChromaVectorStore.__init__`; forward
it through `create_vector_store`. `ProjectionExecutionScope` is an immutable,
code-issued carrier with `operation() -> ContextManager[None]` and
`query(target: object) -> ContextManager[None]`. Only internal composition issues
one; no config/tool/archived payload supplies callbacks, owner names or grants.
The project's issuer lives in `semantic.py`, under the existing recovery admission
guard and K3's current key/epoch checks. Scope transfer to awaited native work
retains accepted lifetimes. Default None preserves current Library behavior.
Bind the carrier to its exact cache/generation/model targets, not only an owner
label. `activation.execution`, `_call_scope`, `native_worker`, nested lazy
client/collection creation, and `generation.query_scope` all select that same
carrier before effects. The scope owns context-checked transfer of an accepted
live parent into retained native work; copied/dead context or the wrong target
cannot fall back to Library source inference. Keep ancestor-liveness, task/thread
identity and native borrower checks; no parallel untracked worker path.
Its exact issuer is `_issue_project_projection_scope(service: ProjectIndexService,
scope: ProjectIndexScope, settings: ProjectSemanticSettings,
semantic_revision: int) -> ProjectionExecutionScope`.

Add `ProjectSemanticSettings(provider_id: str, model_id: str,
destination_fingerprint: str, config_fingerprint: str,
approved_destination_fingerprint: str | None)` and
`ProjectSemanticGeneration(generation_id: str, scope: ProjectIndexScope,
settings_fingerprint: str, manifest_revision: int, state: str, chunk_count: int)`.
Generation state accepts building/ready/failed/cancelled/incompatible. Add store
schema v2 transactionally: device-local semantic settings, a semantic revision, generation
metadata, durable build checkpoints, and a ready pointer. Existing v1 keyword
rows and enablement remain intact; no semantic objects are constructed on upgrade.
Semantic settings/consent changes advance that separate revision; they do not
advance the binding epoch or invalidate the keyword generation. A full Clear or
binding revocation still advances the shared binding epoch and revokes both.

Add explicit `local_files_only: bool | None = None` to the embedding wrapper's
factory configuration: None keeps existing callers' behavior; project local
models set True, disallow dynamic remote model acquisition, and resolve only an
already complete local snapshot. This applies on ordinary and restored profiles.
A missing snapshot returns a setup-required state before loader/network effects.
Add `raise_on_error: bool = False` to the vector `search` protocol and its actual
implementations. Project semantic queries set True and propagate a classified
failure; default False preserves existing callers. Do not infer success from
`[]` after an adapter has swallowed an exception.
Production project composition selects the persistent backend explicitly; an
unknown/unavailable backend must not silently become an in-memory Ready index.

- [ ] **Write adapter/upgrade tests.** `test_project_scope_never_admits_library_sources` instruments the actual EmbeddingsServiceWrapper and ChromaVectorStore through constructor, first lazy client/collection creation, async embedding native worker, nested query and retirement. Assert the project declaration/key and model-artifact owner are used and no Media/Notes/Prompts or `db.rag_indexing` source is inferred. `test_default_scope_keeps_library_admission` pins current behavior. Reject a serialized/fake scope, wrong target, copied/dead transfer context and untrusted owner name before any effect. Test repeated cancellation with accepted parent retirement. On an ordinary isolated profile with network blocked, a missing local model cannot download and the real loader receives offline-only settings. A native collection query exception raises in strict mode; a successful zero-hit query remains empty and default behavior stays compatible. Assert `old_model_fingerprint != new_model_fingerprint` even when dimensions match. Upgrade tests compare keyword rows and policy values before/after v1->v2, prove a semantic-revision change preserves keyword readiness, and roll back an injected DDL failure.
- [ ] **Run:** `python -m pytest Tests/RAG/test_project_projection_scope.py Tests/Workspaces/test_project_semantic_settings.py -q`. Expect the missing execution-scope/semantic contracts to fail.
- [ ] **Implement the explicit scope and opt-in adapter contracts.** Constructor decorators must see the supplied scope before first filesystem/model effects. Route every listed decorated/direct/native entrance to that exact typed carrier without falling through to Library default source discovery/reconciliation. Retain native borrower tracking and storage admission. Do not unwrap decorators, monkeypatch global guards, forge `_projection_service`, duplicate the vector adapter, or broaden default guards for all callers. The project scope checks its own storage generation, current authority and exact target; local models retain readiness checks plus the explicit offline-only loader contract. Add the strict search option at the actual exception handler, preserving its default.
- [ ] **Verify:** `python -m pytest Tests/RAG/test_project_projection_scope.py Tests/Workspaces/test_project_semantic_settings.py Tests/RAG/test_local_citation_capture.py Tests/RAG/simplified/test_vector_store.py Tests/RAG/simplified/test_embeddings_wrapper.py Tests/Backup_Recovery/test_projection_generation.py Tests/Backup_Recovery/test_rag_projection_lifetimes.py Tests/Backup_Recovery/test_activation_local_embedding.py -q`. These are focused existing regressions for the shared guards being changed. Include a real private store and an installed optional vector backend; deterministic/memory tests alone do not establish Chroma lifetime behavior. Run scoped Ruff and `git diff --check`.
- [ ] **Commit:** `git commit -m "feat(rag): support explicit project projection ownership"`; document exact default-Library compatibility and the new owner boundary in Backlog notes.

## S2 Bounded semantic generations and live validated search

**Files:** Extend `semantic.py`, project models/store/service/scanner accounting, and project recovery/runtime. Create `Tests/Workspaces/test_project_semantic_generations.py`, `Tests/Workspaces/test_project_semantic_search.py`, and `Tests/Backup_Recovery/test_project_semantic_lifetime.py`.

**Interfaces:** `ProjectSemanticIndex(scope, settings, store, embeddings, vector_store)`
consumes the exact S1 scope and the existing `VectorStore` protocol.
`async build(job_id: str, stop: threading.Event) -> ProjectSemanticGeneration`
reads committed keyword chunks in pages of <=32 chunks and <=1 MiB source text.
K1's store adds `page_chunks(scope, *, expected_manifest_revision: int,
after_chunk_key: str | None, limit: int, max_utf8_bytes: int) -> ChunkPage`;
each page checks the revision transactionally and uses stable unique keyset
ordering. A changed revision stops/discards staging or starts an explicit new
build; never keep an SQLite read snapshot open across embedding/network work.
The captured manifest revision belongs to generation metadata/checkpoints and
the final ready-pointer CAS, alongside scope, authority and semantic settings.
`candidates(authority: ProjectIndexSearchAuthority, query: str, *, limit: int) -> ProjectSemanticCandidates` returns
only revision-bearing candidate records, never cache excerpts. This frozen
result distinguishes ready (including zero hits) from unavailable with a safe
reason code. It consumes S1's strict vector query mode.
Apply K4's run-specific exclusion floor before semantic ranking too. A generation
is queryable only when its recorded eligible path set is no broader than that
call's authority, or a bounded exact backend filter can narrow it before top-k.
The existing metadata allowlist cannot be replaced by an unbounded list of every
authorized file. If exact bounded pushdown is unavailable, report semantic
unavailable and use keyword search; do not post-filter a global semantic top-k.
Test exclusion removal plus refresh from an old run, and an added exclusion
whose matches otherwise dominate top-k. This is the same K4 authority contract,
not a separate semantic permission model.
`async aclose() -> None` drains issued work and retires native borrowers.
Extend `ProjectIndexService.search` internally to merge keyword and semantic
candidates by reciprocal rank with fixed k=60, deduplicate by path/hash/line,
and then use K4's unchanged 100-candidate/live-validation/output boundary.
Keyword search proceeds when the semantic result is unavailable.
`candidates` uses `EmbeddingsServiceWrapper.create_embedding` on the existing
finite tool worker; `build` uses `create_embeddings_async` for its bounded pages.
Do not introduce a new event loop with `asyncio.run` for each tool call.

- [ ] **Write generation/race tests first.** With real temporary SQLite and deterministic test embeddings, build a known synonym corpus, stage a changed generation, fail before ready-pointer publication, and assert the prior compatible pointer remains. Mutate keyword rows between semantic pages and immediately before final CAS: no mixed membership may become Ready. Assert incompatible model/config/epoch generations are not queried, no content crosses two bindings with identical text, updated hashes are not merged as the same revision, and the serialized result still fits 16384 bytes/ten hits. Inject an exception into the actual vector collection query, assert semantic unavailable with keyword fallback, then recover the query and distinguish a real zero-hit result. Hold native work across cancellation, exclusion/detach/Clear, then release: publication must fail and actual borrowers must retire. Force storage budget exhaustion including staging vectors and assert Partial plus keyword usability.
- [ ] **Run:** `python -m pytest Tests/Workspaces/test_project_semantic_generations.py Tests/Workspaces/test_project_semantic_search.py Tests/Backup_Recovery/test_project_semantic_lifetime.py -q`. Expect the new semantic build/query assertions to fail.
- [ ] **Implement batching and atomic readiness.** Reuse keyword chunks and their content hash/line mappings, not a second scanner or an independent re-chunking policy. Collections are opaque per binding/epoch/config/generation under that binding's cache namespace. Build in staging from one checked manifest revision, persist revision-bearing checkpoints, then under K1's registry-then-store fence validate scope/config/authority/exclusions/manifest revision and atomically swap the ready pointer. Keep old compatible ready generations during rebuild, reporting their coverage revision; changed/deleted files are still rejected by live candidate validation. Source membership changes invalidate a new build's publication but do not falsely rewrite an already published generation's snapshot. Run query embeddings under the same destination consent and source admission as build embeddings. Degrade to keyword on semantic failure with a truthful status rather than an empty-success semantic claim.
- [ ] **Verify:** Run the three new files, K4's `Tests/Agents/test_project_index_tool.py`, K3's recovery tests, and affected node IDs from `Tests/Backup_Recovery/test_projection_generation.py`, `test_rag_projection_lifetimes.py`, and `test_activation_local_embedding.py`. With optional dependencies installed, exercise the real vector store's close/maintenance path using the shared lifetime observer. Repeat the scale harness in semantic mode, recording model/config, corpus bytes, cache+staging peak, indexing rate, query latency and cancellation. Deterministic vectors test lifecycle, not semantic quality; retain a separate real embedding-quality receipt.
- [ ] **Commit:** `git commit -m "feat(workspaces): build optional semantic project generations"` after exact-path staging, scoped lint, and `git diff --check`; update Backlog coverage and measured limits.

## S3 Semantic choice destination consent and release qualification

**Files:** Extend `ProjectIndexBindingCard`, project service/runtime, `Tools/workspace_file_roots.py` status wording, `RAG_Search/simplified/embeddings_wrapper.py`, `Embeddings/Embeddings_Lib.py` endpoint/error reporting, and `Tests/UI/test_workspace_project_index_settings.py`. Create `Tests/UI/test_project_semantic_consent.py`; extend the actual embedding factory's controlled-HTTP tests. Update `Docs/Architecture/project-indexing.md`, `rag.md`, and the Console/Settings user guides. Retain receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/`.

**Interfaces:** `ProjectIndexService.enable_semantic(key: ProjectIndexKey,
settings: ProjectSemanticSettings, *, expected_epoch: int,
expected_semantic_revision: int) -> ProjectIndexStatus`
accepts only a current explicit Settings action and valid destination approval.
`disable_semantic(key, *, expected_epoch: int, expected_semantic_revision: int) -> ProjectIndexStatus` revokes semantic query/build
eligibility while leaving keyword mode enabled. Expose separate `keyword_state`,
`semantic_state` and coverage in status, together with `semantic_revision: int`;
no worker or model construction occurs
when projecting them. Resolve secrets through the existing embedding factory;
persist only nonsecret references/fingerprints.

- [ ] **Write mounted consent/availability tests.** Assert missing embeddings offers keyword operation and an actionable optional-feature state; switching chat providers changes no index setting. Select an external embedding destination, cancel its disclosure, and assert zero provider calls. Approve, hold the build before publication, change destination, then assert paused semantic work, unchanged keyword state and byte-identical Library policies. Send credential-bearing custom endpoints through actual wrapper/factory construction and controlled HTTP success/retry/failure responses; inspect captured logs and propagated status/error strings, not only UI labels. Prove no raw credentials/query secrets escape. Test stale semantic Disable after Clear/re-enable and ordinary-profile missing local models with network blocked.
- [ ] **Run:** `python -m pytest Tests/UI/test_project_semantic_consent.py Tests/UI/test_workspace_project_index_settings.py -q`. Expect absent semantic controls/guards to fail.
- [ ] **Implement existing-pattern Settings controls and safe factory reporting.** Show provider/model, local/external destination, separate semantic readiness/coverage, and its storage share. Reuse existing optional-dependency and provider selection seams; do not mutate the global Library RAG profile or capture a Library Allowed setting as consent. Replace raw endpoint logging in both the wrapper and `_openai_embedder` success/retry/failure sites with a credential-free destination label and safe error classification; propagated request exceptions must not re-expose the URL. Changing provider destination invalidates semantic consent/build publication before requesting new approval. Keep keyword results and the file tools available through failure, disabling and cancellation.
- [ ] **Qualify with the real app and selected real embedding backend.** Exercise local semantic discovery for a paraphrased query, refresh, failed rebuild, restart, destination change, semantic disable, and full Clear/detach. Verify returned project facts through live file reads. For external embeddings, capture only credential-free destination/call receipts under explicitly selected consent. Run the S1/S2/S3 targeted tests plus the G3 policy-preservation test. Scope Ruff and token/CSS checks if relevant UI sources changed; require `git diff --check`.
- [ ] **Commit:** `git commit -m "feat(settings): opt into semantic project indexing by destination"`. Close the Backlog task only after actual provider, lifecycle, documentation, self-review and ADR evidence is recorded.

## Delivery gate

Ship when semantic opt-in has a recorded real-backend qualification, project-only
source admission, compatible-generation publication, and cancellation/revocation
evidence, while missing/degraded semantic capability leaves keyword search and
saved Library choices intact. The user selected Subagent-driven execution.
The [pre-implementation review](../qa/2026-10-20-workspace-project-context/plan-review.md)
records the source-snapshot, adapter-failure, direct/native-scope, offline-model,
factory-reporting and recovery-regression corrections included here.
