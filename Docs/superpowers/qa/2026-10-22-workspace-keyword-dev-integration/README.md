# Workspace guidance and keyword index: dev integration

PR [#3204](https://github.com/rmusser01/tldw_chatbook/pull/3204) targets `dev`. The original query-qualified product source is `19d38fd32ed9009d9fd7c9ffb04e38fd989d370d`, rebased onto `2a5afa03b31f60e00bdcd10a161e3e351fad023b`. The [current CI correction](ci-correction.md) is at product source `e8c841d48ba35f99433842e5e7552f989867c01f`; measurements below retain their original source identity. Workspace guidance and optional folder-bound keyword indexing preserve saved Library choices/scopes. Semantic indexing is a separate delivery after actual merge.

The fresh independent source review found one Important failed-start ownership race. Its scoped correction review is **APPROVE: 0 Critical, 0 Important, 0 Minor open**. The earlier supervisor-history tail was also fixed in this integration. [Review](review.md), [implementation report](implementation-report.md), [chronological ledger](execution-ledger.md), [all 18 integration rulings and their costs](rulings.md), and [byte/hash manifest](manifest.json). QA-local Git attributes preserve immutable receipt/report bytes; their artifact diffs do not exempt product code or authored summaries from text checks.

## Verification and provenance

| Selection | Source | Actual outcome |
| --- | --- | --- |
| Focused supervisor/history/store controls | `e16100e2b3` | 10 passed, 1 warning, 363.34s |
| Guidance/Library/Settings/Console/shutdown/dev seams | `e16100e2b3` | 2 failed, 20 passed, 2 warnings, 617.23s |
| Failed-start race, held-writer/dispatch controls and both corrected retry cases | `19d38fd32e` | 5 passed, 1 warning, 137.63s |
| Single same-cache final query qualification | `19d38fd32e` | 1 passed, 1 warning, 244.90s |

The two covering failures occurred during controller construction: per-test config redirection conflicted with the already bound raw config participant. Only the two retry wrappers now retain pytest's private bootstrap profile using the established marker. Both actual retry assertion paths subsequently passed; config/controller admission and assertions were not bypassed. The original mixed outcome remains recorded rather than relabeled an all-green aggregate.

The new causal race test holds the real ownership iterator while a test thread removes an actual physically retired run under the production mutation lock. RED produced a real dictionary-iterator RuntimeError, replaced the original injected start failure and left the production storage pin open. GREEN preserves the original error, closes that pin exactly once outside the lock, retires the operation and allows real storage-pause drain. The actual history collector is separately covered by the existing 20-completion test; the new mutation is explicitly test-triggered.

Corrected current-source index-plan, Textual-worker and timestamp guards passed. Scoped source/test Ruff, format, AST/compile and whitespace checks passed; the shared controller-test lint comparison retains three inherited diagnostics and introduces none. Historical child-only success/outer timeout, initial capture/guard failures and meaningful RED outputs remain adverse in raw evidence. No full local test suite, clean-install or universal platform claim is made.

## Final large-cache query measurements

The retained nested and single-directory 20k caches stayed in their original locations. This run rebuilt neither corpus nor index. Five warmups precede 50 measured samples for each candidate cell, plus 20 actual public marker calls per shape. All 24 recorded source hashes remained unchanged.

| Shape | Query | Canonical decorated lexical p95, ms | Expanded admitted p95, ms |
| --- | --- | ---: | ---: |
| Nested | Common | 273.264 | 644.722 |
| Nested | Rare | 205.671 | 549.382 |
| Nested | Miss | 169.099 | 454.642 |
| Wide | Common | 244.297 | 617.648 |
| Wide | Rare | 168.701 | 474.645 |
| Wide | Miss | 164.260 | 435.502 |

All six canonical cells pass the original 500ms gate. The broader admitted measurement includes real registry/root/status/floor work and has **three accepted misses**: both common cells and nested rare. It remains an unmet broader target. The independent review accepts this explicit deviation without authority shortcuts or favorable-timing reruns. Instrumented public whole-tool p95 was 1948.863ms nested and 2443.248ms wide, including retirement-proof bookkeeping; these values are separate from the canonical gate and are not a universal latency promise.

All 42 actual Capture retirement proofs were positive. All 40 public calls returned with the actual registry retired. Service close left zero searches and zero operations, and the task-owned query process exited. [Exact timing/retirement data](raw/integration-idle-latency.json), [summary](raw/integration-idle-summary.json), [command and source hashes](raw/integration-idle-source.json), and [unchanged probe](raw/test_integration_idle_latency.py).

The [historical delivery evidence](../2026-10-20-workspace-project-context/keyword/execution-notes.md) retains literal 20k build duration, sampled memory, quota experiments, simulated dependency absence/blocked sockets, Windows/headless scope and staged encrypted recovery qualifications. Those build timings are not current-head build measurements. This integration does not establish simultaneous replacement of an active index owner, real-provider/model reliability, POSIX execution or human-terminal approval behavior.

## Governance and external integration

[ADR-241](../../../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md) governs project context and folder-bound indexing. Landed MCP ADR-240 retains its identity; only the unmerged project decision and active links were renumbered. Historical raw hashes/source-era labels remain intact.

The human explicitly waived the current-head Qodo requirement **for PR #3204 only** after Qodo reported an out-of-credits billing block. That comment is not a review. The waiver does not alter independent review, required hosted CI, unresolved-thread or merge-queue requirements. At this metadata checkpoint the task remains In Progress and the PR is not claimed merged; actual hosted/thread/queue closure is recorded externally before semantic implementation starts. The original checkout and installed dependencies were not modified; private test profiles/corpora remain in place.
