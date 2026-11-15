# Native Windows Console validation and performance evidence

The change strengthens Console Close retirement, removes an unnecessary endpoint startup import, repairs current validation fixtures, and reduces repeated CSS candidate-union work. The native journey comparison is **not qualified**: its final baseline cohort failed. No overall application latency improvement is established.

## Product and targeted validation

- Close drains the runtime and captured controller preparation-read owners through the existing physical-retirement primitive. Stock context reads reject a committed close fence. Repeated cancellation still retires native work and deletes the session before delivering cancellation. Existing storage admission, source-currentness, permissions and custom-callback behavior remain intact.
- The canonical endpoint resolver owns its existing suffix table; setup facts re-export that exact object. UI-ready startup remains within the unchanged **1040-module** limit, including mandatory Windows file checks.
- CSS candidate collection lazily prehashes subject buckets under the existing parsed-stylesheet cache owner and uses native set unions. A 500-rule, three-index control across ten distinct shapes reduces Python rule-hash callbacks from **15,000 to at most 1,500**, preserving rule identity and source order.
- Full-screen upstream style parity passes in **85.65 s**. The final native/CSS contract selection passes **10 tests in 15.37 s**; the readiness/P0 selection passes **179 tests**. Three native tab controls, connected context controls and the unchanged startup census also pass. Counts overlap and must not be summed as unique coverage. No full suite was run.
- Scoped static checks pass with pinned **Ruff 0.16.6**; the recorded ratchet is **1090 <= 1105**. Findings produced by the ambient Ruff 0.4.5 were tooling-version mismatches, not product test failures.

Historical assertions, current corrections and raw validation references are indexed in [CI_INVENTORY.md](CI_INVENTORY.md) and [ci-inventory.json](ci-inventory.json).

## Bounded CPU experiment

[bench_candidates.py](bench_candidates.py) records 900 candidate-LRU misses across nine rounds per process. Parsed indexes and subject buckets are populated before timing. The median round mean changes from **0.816 ms to 0.318 ms**, about **61% lower** in this synthetic workload. First-use bucket construction, application startup, complete Console return latency and terminal painting are excluded.

The identity-dictionary experiment measured **1.157 ms** and was rejected. Its raw packets and the earlier newline-mismatched comparison remain retained. Callback-count reduction alone is not a general speed claim.

## Serial native journeys

The attempted order is A2 / C1 / C2 / A3. Baseline source is `80710424ceebe3c545160fa368dfe344f9c83f78`; candidate source is `f53f8c7bebf5fbee0148f546a1ac693054089899`.

| Cohort | Completed returns | Exit | Navigation to observed draft, ms | Navigation to attachment, ms |
| --- | ---: | ---: | ---: | ---: |
| A2 baseline | 6/6 | 0 | 82–367 | 2786–7570 |
| C1 candidate | 6/6 | 0 | 39–677 | 2504–8893 |
| C2 candidate | 6/6 | 0 | 45–734 | 2608–6360 |
| A3 baseline | 1/6 | 1 | 5700 | 73.8 |

A3's first posted key took 5647 ms to appear in the draft, after attachment had completed. Its second return observed the key at 88.4 ms, then failed the existing attachment wait with `TimeoutError`. That pending return is retained separately from completed samples. The baseline failure does not establish a candidate regression or improvement.

[comparison.json](comparison.json) records `qualified: false`, every completed row, A3's pending return and failure, and each cohort's exit and guard state. [compare.py](compare.py) validates exact heads, manifest hashes, runtime settings, flags, matching source inventories and the sole changed product module, then exits nonzero and withholds arm aggregates when a cohort is incomplete. All four packets retain zero network/profile guard counts, stopped observer threads and unchanged product/CSS sources.

## Provenance and limits

- Exit receipts retain exact commands, heads, harness hashes and before/after CSS hashes. Product-source manifests retain before/after hashes for all 3012 Python files. The only product-source difference between arms is `Utils/textual_css_fastpath.py`.
- [executed-harness.json](executed-harness.json) identifies exact executed copies in `executed-harness/*.py.txt`; their hashes match the cohort receipts. The current probe received cosmetic pinned-formatter changes after the frozen runs.
- These are identically stack-sampled, headless journeys with a periodic traceback thread. They do not measure terminal paint latency, and source hashing warms filesystem caches.
- The harness explicitly focuses the composer before posting its key. Automatic arrival focus is outside this measurement. `draft_retained` means that posted key survives attachment; it does not establish prior-draft continuity across navigation.
- Two processes per arm are insufficient for statistical significance. The failed baseline prevents a complete matched comparison. Diagnostic spans, the perturbing cProfile run, failed setup/attachment runs and rejected experiments remain retained separately.

At publication, **PR #3572 is open, auto-merge is off, and required CI is queued**. Remote CI status will be updated separately; targeted local passes do not establish a remote green result.
