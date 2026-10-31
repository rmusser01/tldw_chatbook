---
id: TASK-34650
title: Fix review-B non-Console efficiency findings
status: Done
assignee:
  - '@Robert'
created_date: '2026-10-07 02:57'
updated_date: '2026-10-09 06:11'
labels:
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the 24 efficiency defects unique to the 2026-10-06 review-B performance sweep (complement of the sibling remediation): unbounded RAG content assembly, per-sample model reloads in Evals, O(BxF) notes-sync hashing, serial ingestion/research LLM pipelines, missing conversations browse index, legacy Prompts_DB N+1 searches, per-keystroke UI rebuilds, video-store walk multiplication, fork/import batching. Spec: Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RAG conversation content assembly bounded with order preserved,B4 semantic model constructed once per process,Notes sync identity fallback O(B+F) digests per pass,Ingestion chunk analysis bounded concurrency order-preserving,Research gate overlaps scrape+summarize under semaphore,conversations(last_modified DESC,id DESC) index serves browse pages,Prompts_DB searches use FTS subquery+batch keywords+LIMIT,In-memory vector store uses dict index+matrix cosine+OrderedDict LRU,UI legacy widgets debounced and render-capped,Video store single snapshot per save and RecoveredMedia reuse per call,Fork commit and history import statement counts bounded,All targeted tests green with counted evidence
- [x] #2 The rebased PR preserves dev changes, all confirmed code-review findings are resolved, and targeted regression checks plus required CI pass before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md waves A-F (tasks B1-B27) via per-wave implementation agents; commits per wave; PR against dev.

PR #3043 integration and review (2026-10-08):
1. Rebase onto the latest dev while retaining the ADR-224 v79 migration followed by ADR-216 v80 and dev's newer Notes serialization conventions.
2. Review the complete rebased PR with independent code reviewers; reproduce and fix every confirmed finding in the affected callers and tests.
Confirmed review fixes, with failing regressions before implementation:
- RAG: reuse the query embedding across union allowlist entries and replace the fake embedding-count claim with a real-engine/store check; enforce the total conversation-content fetch cap in SQL while retaining ranked conversation and chronological message order; prevent vector cache publication from racing mutations.
- Data: release observation snapshots on every cancelled Notes planning/token await; serialize eval memo publication, preserve DB admission on cache hits, and invalidate after committed result/run-group writes or connection retirement; update current-head timestamp migration checks to v80 while keeping historical v79 checks; avoid SQL variable-limit failures in shared prompt keyword enrichment; report the actual conflicting message ID in bulk imports.
- Ingestion/research: bound retained parsed payloads to the worker window and persist in input order; stop queued scrape/summary work at cooperative cancellation boundaries.
- UI: suppress selection events during checkbox synchronization, settle the final timeline drag repaint, and clear the Chatbooks pager for zero matches; verify actual mounted event flows; reapply settled filters after threaded tree population and reject oversized reasoning before allocating its encoded bytes.
3. Reconcile the existing index-census pins with reviewed installed DDL and diagnostic-boundary hashes with the approved inventory delta; run targeted regression suites and the derived-artifact preflight, update this task's implementation evidence, and publish with an explicit force-with-lease.
4. Wait for required GitHub checks, address any newly posted review findings, and merge the verified head into dev.
ADR required: no new ADR
ADR path: backlog/decisions/216-conversations-browse-order-index.md; backlog/decisions/224-conversation-timestamp-normalization.md
Reason: rebase reconciliation and review fixes preserve the existing storage and runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Approach:** Executed plan `Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md` (tasks B1–B27, waves A–F) via six per-wave implementation agents in an isolated worktree branched from `origin/dev @ 1086b9f8aa`; strictly complementary to the sibling PR `perf/nonconsole-efficiency-remediation` (verified: zero sibling-owned files/regions touched — `git diff --name-only` checked against the exclusion list).

**Highlights (evidence in plan tasks + agent records):**
- B1: hybrid/keyword conversation content leg bounded (5,000 → ≤80/conv, ≤400 total rows fetched); citations scan the 1,000-char preview only; FTS recall and oldest-first order unchanged.
- B3: in-memory vector store — dict index + single matmul cosine + OrderedDict LRU: 100 queries over 5,000×16 vectors 1.54 s → 0.098 s (~15.7×), oracle-identical results.
- B4/B5/B6 (Evals): SentenceTransformer constructions 3 → 1 per run; per-sample `store_result` off the event loop; rail failure-count scan cached until next `store_result`.
- B7 (ADR-216): v79→v80 migration adds `idx_conversations_last_modified` + `idx_conversations_archived_browse_order` after dev's ADR-224 timestamp normalization at v79 (EXPLAIN: TEMP B-TREE eliminated on every scope; 3.1 ms → 0.28 ms/query on the original 5k fixture). Recovery catalogs, restricted migration actions, and current-head tests follow the same v79/v80 lineage.
- B8/B9: Prompts_DB legacy searches ported to FTS subquery + batch keywords + LIMIT (22 statements → 1 for a page of 20); `search_media_db` DISTINCT dropped (VM steps 21,121 → 17,950, golden-identical).
- B10/B11/B12 (notes sync): digest computations per pass 200 → 50 (O(B×F) → O(B+F)); plan executions 2 → 1, off the event loop, token byte-identical; idle watcher ceiling 10 s → 30 s (walks/10 min idle 62 → 23).
- B13–B16: Docling converter once per process (3 → 1 per batch); chunk analysis bounded-concurrent (2.65× wall on 30×50 ms fixture); CLI batch ingest parse stage concurrent 4× with serial persistence (thread-safety gate: MediaDatabase is thread-local/`:memory:`-per-thread — persist stays on caller thread); research gate scrapes+summarizes overlapping under semaphore(3) with sequential relevance gating.
- B17–B22 (UI): activity log rebuilds 2,000 mounts/burst → 0 (debounced, 200-row render cap); SmartContentTree/config search/repo tree/trajectory/chatbooks per-keystroke and per-frame work bounded; trajectory render byte-identical (golden). B20 side-fix: descendant checkbox sync was silently broken (raw path vs sha1 DOM id) — now works via direct handles.
- B23–B27: video store walks per save 3 → 2 and RecoveredMedia per slug allocation n → 1 (on-disk golden identical); fork-commit SELECTs 303 → 4 per 300 messages (decisions golden-identical, hop cap preserved); 1,000-message history import 2.34 s → 0.47 s with row-for-row DB equivalence (sanctioned fallback strategy: sidecar coordinator contracts not bulk-expressible — documented); BUILTIN_PIPELINES mutation bug fixed (deepcopy); per-chunk OpenRouter log removed; think-filter byte counting via encode (30-case oracle identical); loop-invariant hoists in grep tool + EPUB spine map.

**Tests:** 30 new test files, 160 passed / 0 failed in one consolidated run (`-p no:cacheprovider`); all regression comparisons A/B-verified against pristine base by each wave (pre-existing `RecoveryRequired: raw_source_selection_changed` env failures and known flaky tests unchanged). Targeted suites per AGENTS.md; no full-suite sweep run.

**Deviations from plan:** documented per-task in the plan file and agent records — notably B7 second (archived-prefix) index added after EXPLAIN showed the single index insufficient for scoped browses; B8 legacy methods now default to LIMIT 20 (plan-mandated; no external callers); B14 failure semantics corrected (loops already continued per-chunk); B15 partial fix per thread-safety gate; B25 fallback strategy per feasibility read; B26 hybrid site already deepcopy in HEAD.

**Modified/added files:** 39 modified + 33 new (production: 31 files; tests: 30 new files + 8 pin updates; ADR-216; 1 migration SQL; shared helper `Local_Ingestion/analysis_concurrency.py`).


## PR #3043 review remediation

Rebased onto `dev` at `0254a6bd34352633ac7746baf561fc288b536bc6`, retaining dev's Notes serialization handling and the ADR-224 v79 normalization before ADR-216 v80 browse indexes. Restored merge-only fleet migration checks that the linear rebase would otherwise omit. No new ADR is required: these changes implement the existing migration decisions and repair existing runtime contracts.

Qodo was unavailable because its workspace lacked credits. The user explicitly requested the `requesting-code-review` skill instead. Three reviewers covered the full PR by domain; each confirmed bug was reproduced before its fix, then another reviewer checked the fix group. All confirmed findings are addressed:
- Real scoped semantic searches reuse one embedding across allowlist entries, with both citation paths covered. Ordered SQL limits fetch at most 80 matching messages per conversation and 400 overall; the real 600-row fixture now fetches 400 without changing conversation ranking or message chronology.
- One in-memory store lock protects cache construction, mutations, stats and collection operations. Deterministic concurrent tests prevent stale similarities, stats indexing failures and collection mutation failures.
- Evals memo hits retain DB admission; generation checks and post-commit invalidation cover overlapping readers, run-group reassignment and reopened connections. Result inserts and completed-sample updates retain their original atomic transaction.
- Notes retire only the observed pass bundle after cancelled or failed off-loop token/planning work; normal passes still plan once off-loop. Shared prompt keyword enrichment uses one JSON parameter at the live SQLite variable limit. Atomic bulk-import failures identify the actual colliding message.
- Batch ingestion retains only its four-worker payload window and writes serially in input order. Research stops queued work and checks cancellation before additional scraping or paid summaries.
- The complete-PR fatal lint pass also caught and corrected the missing Textual `Widget` type import in the cached settings-search helper.
- Mounted UI regressions cover sibling selection and row highlighting, settled timeline painting, empty-result paging and filters that finish before threaded tree loading. Oversized thinking is rejected before an encoded copy is allocated, with UTF-8/surrogate behavior retained.

Diagnostic inventory review compared exact statements against `09357e62150bd52e2bf7c7c7a043653065bec470`: vector-store diagnostics only moved or were re-indented; two batch error branches became one with the same existing path/error content and no new sink. Regenerated `Docs/security/production-diagnostic-inventory.json` only after that review. All derived-artifact preflight checks pass. Scoped fatal Ruff checks pass; comparisons with the reviewed head found no new default lint or formatter debt.

The wider integration run exposed sixteen existing index shapes absent from the already-touched census. Reviewed their v69-v78 migration DDL and the canonical index-plan census before adding explicit name/table/unique/column pins; production schema is unchanged. The original PR already carried a stale non-owner manifest hash (`99f482…` versus its actual `1c72b7…` normalized inventory); verified that its checked inventory equals the rebased head before applying the two reviewed owner deltas. Updated only the two normalized manifest-boundary hashes in the summarization review fixture to incorporate the approved ingestion/vector inventory delta; the site ledger, owner declarations and mutation guards are unchanged.

Final local verification: **255 passed** across all 30 PR-added regression files plus the current-head timestamp and index-census checks; **6 passed** in the privacy manifest/mutation subset; **3 passed** in the affected settings-search suite after its import correction; all derived-artifact preflight checks passed. The independent review of the final census and diagnostic hashes confirmed the installed DDL and unchanged guard assertions/ownership ledger.

A broader targeted integration run reported **1,045 passed / 26 failed** before pin reconciliation. Two index-census and three manifest failures were repaired and verified above. Eighteen admission-sensitive fixture cases passed in a separate process with the existing bootstrap-profile marker. Two unchanged RAG fixture failures were reproduced against pinned `dev`: profile B is never constructed after raw-source admission fails, and the public-search fixture omits the required search configuration. The remaining query-plan failure is inherited from dev's v75 hook-receipt schema: its assistant/parent message locators lack indexes. These baseline issues were not hidden by weakening production admission or readiness guards.

GitHub validation: reviewed code head `6352503461b9a44ff0d3b713680b98d82012763d` passed the [Derived Artifacts workflow](https://github.com/rmusser01/tldw_chatbook/actions/runs/37889728610): PR Fast Lane, all four UI Fast Lane shards, and the required source-artifact gate. The [UI latency guardrails](https://github.com/rmusser01/tldw_chatbook/actions/runs/37889728651) also passed. No new actionable PR review comments were posted.

Recorded the concrete false embedding-count incident in `backlog/docs/lessons-testing-evidence.md`: caller forwarding is not evidence about the real engine's expensive work. The final completion note changes documentation only; its published head must pass the same required gate before merge. No full-suite sweep, native-terminal session or live provider/model download was requested.
