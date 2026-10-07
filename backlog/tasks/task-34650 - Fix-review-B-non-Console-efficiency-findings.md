---
id: TASK-34650
title: Fix review-B non-Console efficiency findings
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-10-07 02:57'
updated_date: '2026-10-07 02:59'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md waves A-F (tasks B1-B27) via per-wave implementation agents; commits per wave; PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Approach:** Executed plan `Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation-review-b.md` (tasks B1–B27, waves A–F) via six per-wave implementation agents in an isolated worktree branched from `origin/dev @ 1086b9f8aa`; strictly complementary to the sibling PR `perf/nonconsole-efficiency-remediation` (verified: zero sibling-owned files/regions touched — `git diff --name-only` checked against the exclusion list).

**Highlights (evidence in plan tasks + agent records):**
- B1: hybrid/keyword conversation content leg bounded (5,000 → ≤80/conv, ≤400 total rows fetched); citations scan the 1,000-char preview only; FTS recall and oldest-first order unchanged.
- B3: in-memory vector store — dict index + single matmul cosine + OrderedDict LRU: 100 queries over 5,000×16 vectors 1.54 s → 0.098 s (~15.7×), oracle-identical results.
- B4/B5/B6 (Evals): SentenceTransformer constructions 3 → 1 per run; per-sample `store_result` off the event loop; rail failure-count scan cached until next `store_result`.
- B7 (ADR-216): v78→v79 migration adds `idx_conversations_last_modified` + `idx_conversations_archived_browse_order` (EXPLAIN: TEMP B-TREE eliminated on every scope; 3.1 ms → 0.28 ms/query on 5k fixture). NOTE for merge order: this PR owns schema version 79; sibling PR is still at 78.
- B8/B9: Prompts_DB legacy searches ported to FTS subquery + batch keywords + LIMIT (22 statements → 1 for a page of 20); `search_media_db` DISTINCT dropped (VM steps 21,121 → 17,950, golden-identical).
- B10/B11/B12 (notes sync): digest computations per pass 200 → 50 (O(B×F) → O(B+F)); plan executions 2 → 1, off the event loop, token byte-identical; idle watcher ceiling 10 s → 30 s (walks/10 min idle 62 → 23).
- B13–B16: Docling converter once per process (3 → 1 per batch); chunk analysis bounded-concurrent (2.65× wall on 30×50 ms fixture); CLI batch ingest parse stage concurrent 4× with serial persistence (thread-safety gate: MediaDatabase is thread-local/`:memory:`-per-thread — persist stays on caller thread); research gate scrapes+summarizes overlapping under semaphore(3) with sequential relevance gating.
- B17–B22 (UI): activity log rebuilds 2,000 mounts/burst → 0 (debounced, 200-row render cap); SmartContentTree/config search/repo tree/trajectory/chatbooks per-keystroke and per-frame work bounded; trajectory render byte-identical (golden). B20 side-fix: descendant checkbox sync was silently broken (raw path vs sha1 DOM id) — now works via direct handles.
- B23–B27: video store walks per save 3 → 2 and RecoveredMedia per slug allocation n → 1 (on-disk golden identical); fork-commit SELECTs 303 → 4 per 300 messages (decisions golden-identical, hop cap preserved); 1,000-message history import 2.34 s → 0.47 s with row-for-row DB equivalence (sanctioned fallback strategy: sidecar coordinator contracts not bulk-expressible — documented); BUILTIN_PIPELINES mutation bug fixed (deepcopy); per-chunk OpenRouter log removed; think-filter byte counting via encode (30-case oracle identical); loop-invariant hoists in grep tool + EPUB spine map.

**Tests:** 30 new test files, 160 passed / 0 failed in one consolidated run (`-p no:cacheprovider`); all regression comparisons A/B-verified against pristine base by each wave (pre-existing `RecoveryRequired: raw_source_selection_changed` env failures and known flaky tests unchanged). Targeted suites per AGENTS.md; no full-suite sweep run.

**Deviations from plan:** documented per-task in the plan file and agent records — notably B7 second (archived-prefix) index added after EXPLAIN showed the single index insufficient for scoped browses; B8 legacy methods now default to LIMIT 20 (plan-mandated; no external callers); B14 failure semantics corrected (loops already continued per-chunk); B15 partial fix per thread-safety gate; B25 fallback strategy per feasibility read; B26 hybrid site already deepcopy in HEAD.

**Modified/added files:** 39 modified + 33 new (production: 31 files; tests: 30 new files + 8 pin updates; ADR-216; 1 migration SQL; shared helper `Local_Ingestion/analysis_concurrency.py`).
