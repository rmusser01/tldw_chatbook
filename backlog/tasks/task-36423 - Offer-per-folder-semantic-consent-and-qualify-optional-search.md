---
id: TASK-36423
title: Offer per-folder semantic consent and qualify optional search
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:28'
updated_date: '2026-10-24 00:39'
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
- [ ] #1 Mounted Settings provides per-binding model/destination choice, disclosure approval, separate keyword/semantic readiness and coverage, and preserved dirty/stale-action fences without model work during compose or status.
- [ ] #2 Cancelled external disclosure causes zero provider calls; destination changes pause old semantic work before approval and stale semantic actions after Clear cannot affect a newer binding state.
- [ ] #3 Saved Library policies/scopes and keyword readiness stay unchanged through semantic enable/disable, missing dependencies, provider failure, restart and chat-provider changes.
- [ ] #4 A selected real local embedding backend is qualified in the real app for paraphrase discovery, refresh, failed rebuild, restart, disable and Clear/detach; no implicit model download or unapproved external project-text transmission occurs.
- [ ] #5 Actual bounded scale, storage peak, memory, query latency and cancellation evidence plus all limitations, docs, ADR linkage and independent reviews are retained; deterministic vectors alone never certify semantic quality.
- [ ] #6 Critical new semantic schema/settings/native ownership and project-admission regressions plus current keyword-without-embeddings controls execute in a bounded required PR lane on the actually qualified runtime; the new mounted consent tests are classified in the governed UI census, without spoofed SQLite qualification or optional ML in the PR gate.
- [ ] #7 The selected local backend honors explicit device selection: CPU model loading uses FP32 even on a CUDA-capable host, while automatic and explicit CUDA behavior remain correct; focused producer tests and actual offline synthetic qualification cover the contract.
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
<!-- SECTION:PLAN:END -->
