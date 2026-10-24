---
id: TASK-36423
title: Offer per-folder semantic consent and qualify optional search
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:28'
updated_date: '2026-10-24 05:19'
labels: []
dependencies:
  - TASK-36422
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
  - backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make optional semantic project discovery understandable and explicitly approved per folder while keeping keyword operation independent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounted Settings provides per-binding model/destination choice, disclosure approval, separate keyword/semantic readiness and coverage, and preserved dirty/stale-action fences without model work during compose or status.
- [x] #2 Cancelled external disclosure causes zero provider calls; destination changes pause old semantic work before approval and stale semantic actions after Clear cannot affect a newer binding state.
- [x] #3 Saved Library policies/scopes and keyword readiness stay unchanged through semantic enable/disable, missing dependencies, provider failure, restart and chat-provider changes.
- [x] #4 A selected real local embedding backend is qualified in the real app for paraphrase discovery, refresh, failed rebuild, restart, disable and Clear/detach; no implicit model download or unapproved external project-text transmission occurs.
- [x] #5 Actual bounded scale, storage peak, memory, query latency and cancellation evidence plus all limitations, docs, ADR linkage and independent reviews are retained; deterministic vectors alone never certify semantic quality.
- [ ] #6 Critical new semantic schema/settings/native ownership and project-admission regressions plus current keyword-without-embeddings controls execute in a bounded required PR lane on the actually qualified runtime; the new mounted consent tests are classified in the governed UI census, without spoofed SQLite qualification or optional ML in the PR gate.
- [x] #7 The selected local backend honors explicit device selection: CPU model loading uses FP32 even on a CUDA-capable host, while automatic and explicit CUDA behavior remain correct; focused producer tests and actual offline synthetic qualification cover the contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Task36423, focused Task4 brief, reviewed Task3 API/retirement handoff and needed Task2 producer seam; read complete design-language/component patterns, relevant live/testing lessons and ADR241/103. Preserve binding authority, saved Library choices and embeddings-free keyword operation.
2. Write mounted meaningful RED controls for perbinding model/destination consent, cancelled disclosure with zero provider work, destination pause before approval, independent keyword/semantic choices, dirty drafts, stale CAS afterClear, no model/vector import on compose/status and capturedowner pending Retrycleanup.
3. Extend existing project binding card and canonicalSettings using existing tokens; explicit setup resolution in workers, captured epoch/semanticrevision and actualservice/key cleanup. Separate readiness/coverage and previousindexavailable-refreshfailed; signal alone never means retired. No new settings destination or generic owner.
4. Add bounded required PR coverage for critical schema/settings/native/admission/missing-embedding keyword controls on pinned WindowsPython3.12.10. Print actual SQLite source/options and run positive real allocation contracts without spoof/skip. Preserve existing lane caps and13UIshards; update exact aggregation/target contract and newUIfile census intentionally, measuring added targets/headroom before final lane placement. Amend existing ADR103/241 before workflow/UX boundary changes; no optionalML in PRlane.
5. Qualify actual cached MiniLM through production realapp/factory/issuedscope for paraphrase, refresh, failedrebuild, restart, disable, Clear/detach and native closure with blocked network/no download. Independently qualify current keyword with optionalembeddings absent and savedLibrary matrix/chatproviderchanges.
6. Freeze production source once and run one semantic scale qualification on retained20k tree/single-directory corpora in place, no favorable repeats/reconstruction. Record model/artifact/config/dimension, wholebindingregularfile peak, RSS, throughput, query modelconstruction/querylatency, cancellation and explicit limits. Report adverse results honestly; propose measured minimum corrections before any additional run.
7. Retain sourcehashes/raw parent-child/visual/native/storage proof under tracked semanticQA without model/tempDB payloads; update architecture/userdocs/ADRs. Targeted tests/lint/format/guards only, selfreview and commit/report for independentTask4review, then one final wholebranchreview. Parent owns externalPR/CI actions and taskclosure; no fullsuite/semanticautomerge.
ADR required: yes, amend existingADR241 andADR103.
ADR paths: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md; backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
Reason: complete existing consent/retirement UI and requiredPRruntime/aggregation boundary; no newstorage, modelowner or dependencybackend.

8. Complete the selected-backend device contract exposed by real qualification: in existing Embeddings/Embeddings_Lib.py resolve the effective device before loading the HuggingFace model, select FP32 for CPU and FP16 only for actual CUDA, and preserve automatic/device fallback behavior. Write focused producer-contract RED/GREEN controls and one owned cached offline synthetic probe. Keep this correctness fix separate from throughput attribution; do not start another corpus qualification or alter its bound until retained timing decomposition supports a scoped diagnostic decision. No new provider, model owner, schema, persistent cache, global thread setting, or source authority. Amend existing ADR241, accepted Superpowers plan and Task4 brief before producer source edits. ADR required: yes, amend backlog/decisions/241-workspace-project-context-and-folder-bound-index.md; reason: complete the existing explicitly selected embedding-provider runtime contract, without a new boundary.

9. Address Task4 independent review round1 from task-4-review-report.md: reload or explicitly invalidate revoked semantic display after same-owner cleanup, surface/retry semantic read errors without losing valid drafts/state, replace category-save tautologies with before/after transition assertions, cover stale Disable after Clear/re-enable, and diagnose actual Settings executor ownership before any minimal lifecycle repair. Preserve old adverse child warnings; no warning suppression, global grace increase or fabricated retirement. After final source changes, qualify the existing four-case Console policy selection once because its pre-final source bytes were not preserved, and rerun actual offline app quality/lifecycle once after CPU FP32 correction. Keep scale and hosted gates open. ADR required: no new ADR for these existing-contract bug/test fixes; existing ADR241/103 continue to govern. A separate measured source-batch amendment, if adopted, requires prior ADR241/plan/task documentation and focused native/checkpoint/cancel evidence.
10. Apply the measured local Hugging Face source/provider page amendment recorded in ADR241 and the accepted plan before source edits: at most 128 contiguous chunks and 1 MiB encoded source text for local HF only; hosted/non-HF source/provider pages and every native append/query transaction remain at most 32 rows and 1 MiB. Validate all provider vectors before any append, split contiguous commits with fresh live/consent/epoch/revision/destination checks and per-slice checkpoint advancement, and preserve 4 MiB queued-source, 1 GiB binding/native formula, existing authority and physical retirement. Focused RED/GREEN covers 129-plus tail, source-byte trim, max-dimension vector split, wrong count before append, revoke/source-change/stop between slices and final CAS, actual native hold/retire, and RSS. The prior operator-stopped 20k run remains adverse; require separate root ruling before another full retained run under its unchanged 3,600-second bound. ADR required: yes, amend existing ADR241; no new owner, schema, provider or dependency.

11. Precision ruling for step 10: the 32-row and 1 MiB limits apply to native append transactions and query vector pages. The unchanged outer store.sql_unit and registry fence contain the complete paged cosine rank; do not claim the whole query transaction reads only 32 rows. Preserve the existing query source and measure actual 20k latency. Existing ADR241 applies; no new boundary or quota relaxation.

12. Address fix-round1 re-review P2 by preserving original receipt bytes through Git: narrow raw-receipt directory attributes and deliberate restage, then compare all committed blob lengths and SHA256 hashes with the original manifest. Keep existing originals and hashes intact; correct the direct-versus-child command-table attribution. No runtime or test source change, model replay, or new ADR; existing ADR241 evidence-retention contract applies. AC5 and final integration/hosted qualification remain open.

13. Correct the hosted Windows shared Rig cache setup to use verified secure_private_directory creation as production does, preserving ownership refusal and core dependencies. Retain five hosted setup failures, rerun exact core12 locally and on PR CI, and review the narrow test-only fix. Diagnose and preserve finalscale01 invalid full-status comparison; exclude only shared cache_bytes from future keyword invariants, reuse the persisted nested Ready for unexecuted query/cancel, and build wide once after reviewed harness preparation. No new ADR; existing ADR241/103 apply.

14. Complete AC1 Ready coverage exposed by actual 20k data: load the existing referenced Ready generation count in the same model-free read connection, requiring current scope/revision/settings/format identity; retain unknown None versus known zero, prior compatible Ready after failed refresh, and safe None for Off/Paused/stale/mismatched rows. Write focused generation/projection RED and GREEN tests. No schema, write, provider/model import, artifact resolution, or new ADR; existing ADR241 governs this missing read projection.

15. Address the complete final branch review in one coordinated fix wave: filter semantic source disclosure and completion by current processing policy and binding exclusions including retained Partial rows; refuse project HTTP redirects outside approved destination while preserving Library behavior; separate admitted carrier retirement from peer close-pending admission guards; revalidate semantic consent before final public search disclosure; classify native-positive tests on the exact qualified SQLite runtime while preserving portable refusal and keyword controls plus fail-closed Windows positives. Add focused RED/GREEN regressions, amend existing ADR241 contracts before interface edits, retain actual reports and run one scoped final re-review. No new backend, dependency or schema. Existing ADR241 applies; root owns final source qualification, PR CI and task closure.

16. Within the explicitly approved existing Windows project-index job, add the four measured final-review selectors for public Disable/reconfigure disclosure, peer carrier retirement and project versus Library redirects. Keep the 1200-second job bound, dependencies, runtime assertions, shards and aggregation unchanged. Existing normal128 and whole settings selectors cover source privacy and qualified versus portable lease controls; expected 44 cases, conservative284 seconds. Update the exact CI contract and existing ADR103 coverage detail before the final targeted16-selector qualification. No new job, skip-based positive qualification, or new architectural boundary.

17. Complete the approved CI integration against concrete hosted d012 Fast Lane failures: align the exact workflow job/verdict contract with the already-added project-index job, update current ADR103 runner counts to actual20 (21 with perf), and safely handle that count in its statement test. Preserve historical dated counts, workflow/job/dependency/cap behavior, source/model code and all qualification originals. Run only the five failing nodes plus directly related CI contracts, retaining RED and GREEN. This is routine correction of existing AC6 accounting under ADR103, not another semantic production fix wave or a new approval boundary.

18. Complete the retirement fix before the pending single scoped review verdict: nested issued carriers may share one existing operation, so admitted retirement must consider all pending wrappers in that actual parent operation while ignoring genuinely independent peers. Reproduce with actual issued/native/factory owners, retain the pending model on outer-close refusal, prove same-owner successful retry and both carrier/operation retirement, and retain existing independent-peer/own-failure/native-child controls. Include this critical regression within the existing bounded Windows job and exact target contract without changing caps/dependencies/shards. Existing ADR241/103 apply. No new owner/framework or model/20k replay solely for hashes; prior real app qualification remains attributed to441bb9.

19. Complete direct transport-seam compatibility identified by the pending scoped review: the real Library endpoint-cache regression test patches requests.Session.post with a strict old signature. Reproduce its unexpected allow_redirects keyword failure, adapt only that boundary double to the actual Requests contract and assert the Library default remains allowed. Run the exact endpoint-isolation node RED/GREEN with current core runtime, preserve physical SQLite/factory closure and source receipts, scoped lint/format, and a test-only commit. No product/workflow/model change, new ADR or model replay. This covers existing saved-Library compatibility AC3 and closes scoped test breakage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Optional semantic project search is implemented with per-binding consent and model/destination identity, folder-bound storage/access, independent keyword operation and saved Library policy/item-scope preservation. Settings handles setup, readiness/coverage, destination reapproval, stale actions and same-owner cleanup retry. Existing ADR241 and ADR103 govern the implementation; no new architecture record is needed.

Final review fixes filter retained Partial sources before provider disclosure, refuse project HTTP redirects, revalidate semantic consent before disclosure, classify native-positive SQLite tests and isolate physical retirement by the actual parent operation across binding keys. Exact CI consumers/current20/21 runner counts and one Library HTTP-double signature were corrected from actual hosted/review evidence. The final scoped review at660a438 reports all findings addressed and no new Critical/Important breakage.

Targeted evidence: at441bb9, core44passed/0skip314.19s without SentenceTransformers, compatibility26passed156.19s, and actual cached CPU MiniLM app lifecycle1passed241.88s with durable Library choices, zero attempted HTTP, native/service closure and source34 equality. After the retirement refinement, aa42cc0 targeted ownership/CI19passed185.96s. Test-only660a438 Library endpoint/cache control1passed10.11s. Scoped lint/format/AST/diff checks passed. Adverse/RED originals and exact-byte source/runtime/retirement receipts are retained separately; no full suite, installs/downloads/shared-environment changes or model replay solely for hashes.

Historical20k nested/wide qualification remains attributed to d012, including62-74second query latency and sampled rather than universal peak/OS-quota claims. Broader platform/relevance, unfinished Settings startup and a mounted positive Ready-count observation remain unqualified. See Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/final-reviews/README.md and sibling final-fix/integration-fix/cache-double evidence.

AC6 remains open: the bounded approved Windows job now has17selectors/45cases, preserving1200s/job dependencies/shards/aggregation. Local critical regressions passed; the earlier hosted d012 job33passed187.06s is historical and final-head hosted CI is pending. Old-head Fast Lane five stale contract/count failures were reproduced and corrected. The separate Windows Eval45s isolated-restore timeout is in unchanged test code, with underlying delay unproved. Keep task In Progress and PR3237 draft/auto-merge off pending current-head CI; no semantic merge is authorized.
<!-- SECTION:NOTES:END -->
