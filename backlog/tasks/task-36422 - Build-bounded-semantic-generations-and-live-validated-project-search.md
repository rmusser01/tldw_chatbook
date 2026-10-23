---
id: TASK-36422
title: Build bounded semantic generations and live-validated project search
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
updated_date: '2026-10-23 18:21'
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
- [ ] #1 Build pages are at most32chunks and1MiB, total retained/queued source is at most4MiB, and one checked keyword manifest produces immutable generation membership with atomic compatible Ready publication.
- [ ] #2 Interrupted/failed/quota-refused rebuild preserves the prior compatible Ready; stale scope, semantic revision, consent, destination or source cannot publish.
- [ ] #3 Semantic ranking applies exact run/current exclusions and policy before top-k; RRF60 merges bounded candidates through the existing100candidate/10hit/16KiB live identity/hash/permission disclosure boundary.
- [ ] #4 Semantic unavailable failures remain distinct from successful zero hits and always leave keyword-only search usable.
- [ ] #5 Durable job ownership, repeated cancellation, Clear/detach and shutdown retain real native borrowers through physical retirement; data cannot cross identical-text bindings.
- [ ] #6 Keyword lookup remains usable while actual embedding/provider work is held outside finite SQL units; binding-wide keyword/semantic job admission stays sequential and total retained/queued source across jobs stays at most4MiB.
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
<!-- SECTION:PLAN:END -->
