# Historical CI failures checked on current sources

Historical head: `9f07ca956d` in merged PR3166. Fresh source base: `cc6006fcf59328353004206bba8724c399216749`. Exact historical jobs/nodes remain in [ci-inventory.json](ci-inventory.json). Historical failures are not assumed to remain failures on current dev.

| Failure group | Current evidence |
| --- | --- |
| App lookup counts, architectural imports/layers, CSS count/sheet-size ratchets, focus contrast themes | `validation/current-ci-static`: 123 passed; its sole new modal omission was corrected and passed in the connected boot batch. |
| Modal reverse inventory and wide-tier classifications | `PluginRestoreAdoptionModal` uses the existing 90% viewport plugin shell; add its verified classification only. Connected registry checks pass. |
| Boot worker census, first-Send preimports, runtime custody fields | `validation/current-ci-boot-runtime`: nine passed. Its only failure was the separate module census below. |
| UI-ready module count | Initially 1041 > 1040. Move the existing endpoint suffix table into the already-resident endpoint owner and preserve exact setup-facts re-export identity. New import-closure RED then connected GREEN; `current-ci-final-contracts` passes unchanged 1040/1040. Mandatory Windows file checks remain resident. |
| Navigation, settings snapshot preferences, changed-source preparation, durable commit quiet gaps, runtime size | Eleven checks pass in `validation/current-ci-final-contracts`. Runtime size remains exactly 7078 lines. |
| Workbench/composer Send agreement | Final batch failed in the fixture's four-second polling helper before readiness assertions. Use its sibling tests' checked-publication helper; `validation/readiness-and-p0-fixed` passes 179 checks, including all P0 gate inventory tests. Refusal/replay tests retain bare polling. |
| Markup interpolation census | `python scripts/check_markup_interpolation.py` exits 0; no baseline rewrite. |
| CSS semantics and candidate work | The accepted native-bucket control records 15,000 original Python hash callbacks versus at most 1,500 across ten distinct shapes, preserving every distinct rule in source order. Seven connected cache/selector checks pass; repaired full-screen upstream parity passes in 85.65 s. Final native/CSS contracts pass 10 tests in 15.37 s. Earlier zero-hash identity-dictionary evidence belongs to the rejected slower experiment. Unchanged live-class and parsed-rule ratchets pass. |
| Native context-read Close lifetime | Causal controls expose a controller-owned read surviving the runtime-only drain and a new stock read admitted after close fencing. Drain both captured owners with the existing primitive and reject fenced stock reads. Native lifetime and repeated-cancellation controls pass; physical retirement and session deletion precede cancellation delivery. |

The historical Perf Guard failure was module/worker/CSS census growth; its log did not report an actual latency-budget assertion failure. Latency is measured separately in the guarded native packets.

These are targeted validations. No full test suite was run. Raw failures, source patches, observer errors and later passing results are retained; do not add their counts as though they were unique tests. Scoped static checks pass with pinned Ruff 0.16.6; the recorded ratchet remains 1090 <= 1105. Ambient Ruff 0.4.5 findings are tooling-version mismatches, not product test failures.

The serial native journey comparison is unqualified: A2/C1/C2 complete six returns each; final baseline A3 exits 1 during its second attachment wait. The failure, partial rows and guard/source metadata remain in [comparison.json](comparison.json), with no arm aggregate or overall latency-win claim. The separate synthetic candidate-LRU-miss benchmark uses warm parsed indexes/buckets: 0.816 ms original versus 0.318 ms accepted candidate; first-use construction is excluded. The rejected identity dictionary measured 1.157 ms. See [README.md](README.md) for exact scope and limitations.

PR #3572 is draft with auto-merge off; its CI is queued at this evidence snapshot. Remote CI status will be updated separately.
