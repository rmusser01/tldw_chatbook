---
id: TASK-36422
title: Build bounded semantic generations and live-validated project search
status: Done
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
updated_date: '2026-10-23 20:40'
labels: []
dependencies:
  - TASK-36421
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide optional semantic project discovery with truthful readiness and keyword fallback under the existing folder authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Build pages are at most32chunks and1MiB, total retained/queued source is at most4MiB, and one checked keyword manifest produces immutable generation membership with atomic compatible Ready publication.
- [x] #2 Interrupted/failed/quota-refused rebuild preserves the prior compatible Ready; stale scope, semantic revision, consent, destination or source cannot publish.
- [x] #3 Semantic ranking applies exact run/current exclusions and policy before top-k; RRF60 merges bounded candidates through the existing100candidate/10hit/16KiB live identity/hash/permission disclosure boundary.
- [x] #4 Semantic unavailable failures remain distinct from successful zero hits and always leave keyword-only search usable.
- [x] #5 Durable job ownership, repeated cancellation, Clear/detach and shutdown retain real native borrowers through physical retirement; data cannot cross identical-text bindings.
- [x] #6 Keyword lookup remains usable while actual embedding/provider work is held outside finite SQL units; binding-wide keyword/semantic job admission stays sequential and total retained/queued source across jobs stays at most4MiB.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read this task, the focused Task3 brief, reviewed Task1/2 producer handoffs, project-index/RAG/native ownership architecture and existing ADR241. Consume schema4, finite SQL allocation, code-issued embedding carrier and failed-close retention without changing Library choices.
2. Write meaningful real-SQLite deterministic-vector REDs for bounded source/vector pages, immutable membership/manifest CAS, prior Ready preservation, same-dimension incompatibility, exclusions before top-k, binding isolation, unavailable versus zero and quota/corrupt-vector refusal.
3. Implement ProjectSemanticIndex and service-owned semantic generations/jobs under existing binding/native lifetimes. Verify cross-mode/cross-namespace whole-job exclusion and at most4MiB total queued source; SQL/allocation/fence units close before actual model/provider waits, Capture, joins or retry. No second scanner/backend or generic owner rewrite.
4. Implement finite float32 streaming exact cosine/heap100, current plus run-floor eligibility before ranking, RRF60/dedup and unchanged100candidate/10hit/16384byte live file/hash/permission boundary with keyword fallback. Keep separate truthful model-free semantic state and consent CAS.
5. Exercise actual provider/native wait with keyword hit, repeated cancellation, stale/exclusion/settings/Clear/detach/app close, durable claims/checkpoints and physical retirement/publication refusal. Qualify production cached local model through real async builder using reviewed accepted-transfer lifecycle, reusable offline bootstrap and blocked provider network. Deterministic vectors do not certify quality or scale.
6. Verify only new/directly affected DB/service/tool/keyword/native targets plus scoped lint/format/source guards/diff, retaining actual parent/child/raw/source hashes and adverse outcomes. Commit exact paths and report interfaces/limits for independent review before mounted consent and release qualification.
ADR required: yes, existing amended ADR.
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted generation, allocation, source ownership and disclosure contracts; no new persistence owner. Report concrete need before any new schema, durable control or cross-owner mechanism.
7. Retain a background semantic build owner through actual model close failure and expose only retry_semantic_close(job_id) as a same-owner retry signal under the existing ADR241 amendment. Verify wrong-job refusal, pending shutdown, keyword lookup and positive physical retirement; report the synchronous query close-failure disposition. No general executor or storage/control change.
8. Superseding the earlier unimplemented job-id signature, retry_semantic_close(key) signals only owned pending build/query closes for the exact binding. Keep one nonblocking semantic query/service on the existing sync tool worker; persistent close failure retains that worker without a SQL/read-slot/gate/fence, and contention falls back to keywords. Verify same-owner positive retry, cancellation, wrong-key refusal and keyword availability.
9. Use the existing runtime/service registry context yielded owner for finite newly owned/replacement WorkspaceDB retirement with actual native/core proof before semantic model/provider/close/retry waits. Preserve/refuse live borrowed caller handles before effects. Exact added path RAG_Search/projection_scope.py integrates current-state retirement; no generic DB or keyword behavior rewrite.

### Task3 explicit OpenAI transport retirement

Ruling: The held-provider lifecycle test exposed that the OpenAI factory record has close=None while its embedding closure owns a requests.Session. Reference loss or factory-cache clearing is not deterministic transport retirement. Extend only the private _openai_embedder seam in tldw_chatbook/Embeddings/Embeddings_Lib.py to return the embedding callable and the existing session.close callback; the single production _build caller stores that callback in its existing CacheRecord. Adapt the direct consumer in Tests/RAG/test_embeddings_lib_logging.py and focused factory/held-provider controls. The existing factory refcount/condition, cache record, same-owner wrapper close and pending-retirement path remain the owners; do not introduce a transport manager, executor, new persistence or permission boundary. Requests/response behavior, destination approval, credentials, saved Library policy/item scopes, and keyword operation remain unchanged. Shared factory retirement now explicitly closes its already-owned OpenAI session for both project and default callers; do not silently substitute project-only no-op closure.

Before source changes, retain the actual None-callback diagnosis and the aborted test as adverse evidence, not proof that the original callback failed. Test explicit real Session close on the original worker after in-flight embed completion, injected failure retention and successful retry, exact cache/wrapper/carrier retirement, and default-factory closure plus sanitized provider logs. No SQL handle/gate/fence spans close. No new model quality run is required for this HTTP-only seam. Cost if wrong: close failure remains truthfully pending under the existing owner; independent Task3 review includes this producer delta.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: complete deterministic transport retirement at the existing embedding factory boundary without a new owner or storage contract.

### Task3 eviction retains failed transport retirement

Ruling: The new explicit OpenAI session callback makes both existing idle and LRU eviction paths part of deterministic project transport retirement. They currently pop a zero-refcount CacheRecord before close, losing the owned callback if close raises. Correct only those branches in Embeddings_Lib.py to invoke the existing callback before removing the exact record under the existing factory condition. A failed close propagates while the actual record/callback remains retained for the existing wrapper/factory close path; no replacement model is built and no new ownership framework is introduced. Successful eviction retains existing capacity, refcount and concurrency behavior. Project wrappers already prevent embedding reuse after close starts. Preserve failure evidence, test both idle and LRU actual callback failure/retention and successful physical close retry, and cover the directly affected factory concurrency contract. Do not repeat cached-model quality or held HTTP qualification when this isolated ordering delta does not reach those previously exercised paths.

ADR required: yes, amend existing ADR241. ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md. Reason: retain the already-owned session across failure at both existing cache retirement entrances. Cost if wrong: safe failed embedding and retained resources until explicit existing close succeeds; never a lost native owner or false retirement.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immutable bounded semantic generations in the existing project SQLite owner and ProjectIndexService, finite float32 exact cosine/RRF60 before live validation, separate consent/readiness and keyword fallback. Reused actual issued carriers and native lifetimes; same-owner pending-close retry, finite registry retirement, explicit OpenAI Session.close and close-before-eviction preserve real owners. The plan-named ProjectSemanticIndex class was intentionally replaced by methods on the existing service plus codec/ranking helpers; no additional owner, backend or Library setting mutation.
Targeted realSQLite, allocator, manifest/CAS, artifact preflight, live tool/query cancellation, Clear/detach/exclusion/settings and actual cached MiniLM async/held HTTP controls passed with retained raw/hash/child evidence. Scoped lint, two new test-file format checks, compilation and diff checks passed. Initial independent review0C3I1M; all three Important findings fixed in538faa103e and fresh re-review reports0new issues. Source commits805701f9,5ad64a28,286be157,538faa103e. Intermittent scanner fixture startup, pre-fix aborts and inherited warning owners remain documented; no full suite or app relevance/20k claim. Existing ADR241 amended before owner-boundary corrections; architecture updated. Mounted consent and release quality/scale coverage remain a separate deliverable.
<!-- SECTION:NOTES:END -->
