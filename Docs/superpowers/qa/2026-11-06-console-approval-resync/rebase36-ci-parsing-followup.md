# PR3036 rebase and CI parsing follow-up

The branch was rebased onto dev `36a21dc4d270940e3576b49487a71b37c78896f8`. Independent review verified the incoming census prefix and all seven approval additions: 1,813 + 7 = 1,820. The incoming deletion of the retired MediaViewer test remains documented. The RAG facade retains lazy loading and exposes the incoming `VectorStoreSearchError` through its original owner. The remote worker was regenerated from the combined canonical sources with UTF-8/LF output.

The published `99de8b0e` [PR Fast Lane](https://github.com/rmusser01/tldw_chatbook/actions/runs/42443701734/job/128640834847) exceeded its existing 35-minute limit at 92%, with no assertion failure visible before cancellation. The derived-artifact job failed its upstream-success gate; no artifact drift was observed. A nearby identical 4,177-item contract had passed in 33m55s.

Profiling one existing census regression recorded 5,507 AST parses and 93.73 cumulative seconds in parsing out of 118.83 profiled seconds. The checker now shares successful ASTs within one `main()` invocation. Each lookup still reads the complete current UTF-8 body and requires body equality; standalone resolver calls retain uncached defaults. Fixture precedence, diagnostics, membership, shards and time limits remain unchanged.

Targeted verification:

- Final raw-source checker file: 87 passed, including all original 84 cases.
- Existing P0 resolver consumers: 70 passed; PR-step resolver consumer: 1 passed.
- RAG metadata and public-owner compatibility: 41 passed.
- Canonical worker UTF-8/LF, rebuild, prefix stamp and actual loopback ping stamp: four passed.
- Original-source reuse-count control failed as expected. A deliberately stale helper failed both equal-size, restored-mtime source-change cases; exact candidate restoration preceded final tests.

| Observation | Pytest seconds | Older-member case seconds |
| --- | ---: | ---: |
| Original checker, 84 cases | 164.07 | 105.02 |
| Same candidate AST before layout cleanup, 87 cases | 24.66 | 11.82 |
| Final raw-source checker, 87 cases | 40.65 | 19.50 |

These are individual local observations with ordinary run-to-run variation. The separate final isolated older-member case passed in 20.98 pytest seconds, with an 18.88-second call; it has no matched isolated unprofiled baseline. Fresh hosted CI is required to establish the effect on the required lane.

Exact CI-pinned Ruff 0.16.6 reports the same existing B905 finding in the checker and no findings in its tests. Checker formatting passes. The test file retains its pre-existing formatting differences; all introduced layouts match the formatter. Static reversal preserves the original checker outside the declared cache plumbing and all original test ASTs. Independent source review of the CI change is clear.

Task 34569 remains In Progress. Qualified native/browser samples remain zero. The physical geometry/theme/Inspect matrix, comparable timing distributions, Windows dispatch qualification and final-head hosted checks remain required before auto-merge. The 28 selected incoming ownership/fingerprint cases are still pending local execution. The headless diagnostic remains separate, with no physical timing evidence.

The adjacent source manifest binds the relevant checked contents using UTF-8/LF normalization. Detailed commands, expected negative controls, raw logs and process audits remain in private QA receipts.
