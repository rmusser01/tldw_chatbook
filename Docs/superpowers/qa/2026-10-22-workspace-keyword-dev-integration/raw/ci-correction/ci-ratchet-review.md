# Scoped CI correction review — PR #3204

Reviewed exact range `007521567a9e961d2cd854432b357b545d8605d4..00a1e1495236203517570ac8cae2ee39b84da110` from the 71,120-byte package with SHA-256 `a5b8f3b739474d12670d72a0ed1b3ffdef15e6a2c281902559be3f02891f9dc8`. Source HEAD and index were clean at review. This is the TASK-36006 AC15/16 mechanical CI correction only.

## Verdict

- **Spec compliance:** Pass for this correction. The three PR-introduced size overruns are removed, their pins lowered to actual counts, and lifecycle is below fetched dev while its inherited failure remains explicit. I found no changed execution boundary for Settings, project scope preparation, evidence prefix, local review hooks, or shutdown.
- **Task quality:** One Minor issue to address before scoped approval. No Critical or Important findings. After the Minor is fixed and its source checked, this correction is scoped ready to merge, subject to required current-head CI, resolved review threads, and the configured queue.

## Findings

- **Minor — import order, `tldw_chatbook/Chat/answer_citations.py:6`.** The new `ConsoleMessageRole` project import sits between the standard-library `dataclasses` and `typing` imports. Repository AGENTS.md requires stdlib imports before local imports. Move it below `typing` with the other project imports. Ruff's reported zero new diagnostics does not cover this repository style rule.

Counts: **0 Critical, 0 Important, 1 Minor.**

## Basis for the scoped decision

The hosted failure on the pre-extraction head was Settings 35,764 / 35,732 with 2,828 other passes. The exact package changes only the named Settings, Console/controller/interrupt, citation, and lifecycle files plus budget/task/diagnostic metadata; no project-index service, store, scanner, DB, tool query, or authority file changed. The moved Settings generator retains the same widget order, IDs, per-row attributes, project card and result placement, and the screen remains its caller. The controller retains the registry-scope invocation and catches unavailable preparation; the helper keeps the guarded loop and exclusions. The evidence-prefix function still uses the controller's text-preservation method and is invoked by the existing static wrapper. The review-hook controller wrapper supplies all global getters as lambdas at invocation; the new helper imports controller types only under `TYPE_CHECKING`, and the interrupt module re-exports it. The lifecycle method still awaits the durability drain on the original shutdown task, through a lazy helper import. I found no runtime import cycle or early global capture in those paths.

The exact size/AST receipt reports Settings 35,671 / 35,671, controller 29,043 / 29,043, interrupt 6,307 / 6,307, and lifecycle 2,338 versus fetched-dev 2,347 (inherited pin 2,134). Package inspection supports the body's mechanical move; AST equality alone was not used to infer call ownership. The 10-path dev/PR matrix attributes eight current broad-ratchet failures to fetched-dev baseline; the post-correction selected Console ratchet has 5 passes and the single inherited lifecycle failure (+204, improved from dev +213). The initial full local ratchet 10-failed/61-passed and hosted failure remain adverse receipts, not passing runs.

Retained focused outputs show mounted Settings 5 cases and its exact ratchet 2 cases passing; grounding/retry and four saved Library policies 6; interrupt/review/citation controls 10; and native app shutdown/durability 3. The source AST/compile receipt, scoped static parity, and diff check support the mechanical claim. Whole-file formatting failures in controller, Settings, and lifecycle were present at the pre-extraction head. Diagnostic inventory changed only for the two literal project-preparation debug calls: controller 71 to 69 and new helper 0 to 2; statement digests and text match, final inventory diff passes with no sink change.

The report's later provenance correction supersedes its earlier mistaken claim that the successful six-case grounding command used `-p Tests.conftest`. The retained successful command JSON has no `-p`; it records the child profile paths and exit 0, and raw output has 6 passes. The earlier setup attempt's argv and traceback were not retained, so its cause cannot be independently reconstructed. This limits the preliminary-attempt claim, not the successful run's result.

The earlier same-cache query result remains applicable to unchanged backend/query/store/scanner code and unchanged controller scope-preparation body. Its 19d-source timing is **not** a new latency measurement for this head. Canonical six-cell lexical p95 met the 500 ms gate on that source; admitted misses remain separately recorded.

## Declined judgments

- **Current-head hosted CI and mergeability:** No current-head hosted run, review-thread state, or queue outcome is in this read-only package; external gates remain for the integration owner.
- **Whole feature security and functionality:** This review is limited to the exact extraction delta after a prior whole-feature review. The absence of backend/query/authority changes and focused checks supports continuity, but does not retest every original K5 constraint.
- **Current-head performance and scale:** No benchmark was repeated for this extraction; the retained same-cache result is source-qualified to 19d and cannot prove current end-to-end latency or a new 20k build.
- **All broad local ratchet failures:** Eight inherited over-budget modules are outside AC15/16 and were neither changed nor re-pinned here; their failing receipts remain adverse.
- **Preliminary grounding setup failure:** Its raw runner, original argv, and traceback were not retained, so I cannot judge the exact setup failure. The saved successful run can be judged independently.
- **Unselected platforms/providers and full suite:** Only the specified targeted evidence was supplied; repository guidance precludes a full sweep without user opt-in. No test was rerun during this review.

## 2026-10-22 import-order follow-up

Reviewed exact `00a1e1495236203517570ac8cae2ee39b84da110..e8c841d48ba35f99433842e5e7552f989867c01f` package (`ci-ratchet-import-review-package.txt`, 970 bytes, SHA-256 `00f3a0c5cc2ab45a08232bdd47840a4e4a65f9f33a97886e51138976fa55a087`). The only source change moves `ConsoleMessageRole` below the standard-library `typing` import in `tldw_chatbook/Chat/answer_citations.py`. That **addresses the sole Minor**. I found no new breakage: the import multiset and non-import module AST are unchanged, and the retained compile, scoped Ruff check/format, and diff check passed. No runtime test was repeated for this import-only change.

**Final scoped verdict:** Spec compliance **Pass**; task quality **Pass**; **0 Critical, 0 Important, 0 Minor open**. The consolidated correction is scoped ready to merge, qualified by required current-head hosted CI, resolved review threads, and the configured merge queue. All declined judgments and source-qualified adverse receipts above still apply.
