# K5 fix round2 re-review

Exact FIX_BASE `3820fe11572696f62d27291971f5c8d1dba0faec` -> HEAD `5d927a98ffc695c281465b09a8020d1db196490a`; full package `review-fix2-3820fe1157..5d927a98ff.diff` (7395 bytes), exactly two authorized paths. Read the re-review contract first, fix brief/report, original implementer status and metric alignment review. This is the same review seat, scoped to the predicate remedy and new breakage in this diff.

**Final performance remedy disposition: ADDRESSED. Spec verdict: PASS WITH EXPLICIT ACCEPTED DEVIATION AND QUALIFICATIONS. Quality verdict: PASS WITH QUALIFICATIONS.** New fix-diff findings: **0 Critical, 0 Important, 0 Minor**. Original I1/I2 are addressed; original Minor M1 remains deferred to final whole-branch review. The final evidence closure below supersedes the historical pending checkpoint; the expanded admitted500 ms target remains unmet.

## Source and behavior

The product change is confined to `WorkspaceProjectIndexDB.lexical_candidates` (`tldw_chatbook/DB/Workspace_Project_Index_DB.py:967`). It normalizes this invocation's current admitted exclusion tuple once, then uses immutable exact-path/name/direct-parent sets and subtree prefix tuple inside the existing SQL eligibility callback. There is no retained state across calls.

The predicate remains equivalent to unchanged `_path_allowed`:

- Exclusion values use exactly replace-backslash-with-slash, strip outer slashes, then casefold. Candidate paths are casefolded, without introducing new separator normalization.
- File and subtree exact matches share a set; subtree descendants require the slash boundary. Empty subtree denies every candidate. Similar peer prefixes remain allowed.
- Name matches use the final slash component; direct_children uses the parent from rpartition, including empty parent/root. Unknown kinds still have no matching rule. Duplicate exclusions remain equivalent.
- Unicode casefold expansion, case-insensitive comparison and existing processing prefix/suffix behavior are preserved. Empty subtree's empty prefix is safe because deny_all already short-circuits.

The existing per-query authority guard, decorated activation/admission, transaction and current control/scope checks are unchanged. The service's fresh current/run exclusion-floor union is unchanged, as are registry/root/chain/native gates. Filtering still occurs in the same SQL WHERE before ranking/LIMIT; SQL and deterministic ordering are unchanged. No schema/API, generic authority cache, live Capture or ownership change is present.

The new behavioral test (`Tests/DB/test_workspace_project_index_db.py:879`) publishes18 logical paths into a real store, obtains canonical candidate order, and compares15 successive exclusion floors at limits1 and100 with the unchanged live oracle. Cases cover all four kinds, Unicode, slashes/backslashes, empty subtree/root, subtree peers, direct children versus descendants, mixtures and a final empty floor after restrictive queries. This exercises filtering before top-k and fresh next-call state, rather than counting the implementation's normalization calls. Existing narrowed policy/suffix and actual provider removal/Refresh/native Capture cases supply complementary coverage.

## Examined evidence and scope

Read raw receipts, not just report counts:

- `k5-fix2-semantic-baseline.txt`: new behavior test **1 passed, 1 warning, 7.70 s** on3820 before product edit. A preserving optimization appropriately passes the same semantic oracle before and after; the original nested/common577.3054 ms measurement is the performance failure motivating it.
- `k5-fix2-scoped-green.txt`: exact authorized six DB/provider nodes **6 passed, 1 warning, 75.58 s**. Includes the oracle, exclusion-before-top-k, suffix oracle, narrowed processing policy, old-run floor after removal/Refresh, and filename/empty/whitespace live Capture.
- `k5-fix2-ruff.txt`: all checks passed; `k5-fix2-format.txt`: two files already formatted. Independent read-only `git diff --check` for the exact commit range was clean.
- `k5-fix2-source-scope.json`: remaining product AST equal outside lexical_candidates, with old/new file hashes. Exact git stat/package confirms only authorized DB/test paths changed (143 insertions,1 deletion).

No tests, profiling, rendering or source mutations were performed by this reviewer. The earlier source-timed comprehensive, native, UI and scale qualifications remain as qualified in the original review and fix1 report; this narrow query-only change is not represented as rerunning them. Original I1 and I2 remain **ADDRESSED** per `task-5-re-review-1.md`. Original scanner typing/docs M1 remains deferred to whole-branch review, not a new fix2 finding.

## Metric boundary and pending evidence

Root's explicit pre-code ruling in `task-5-fix2-brief.md` preserves canonical decorated lexical p95 <=500 ms as the gate and accepts the stronger admitted <=500 ms target as an **unmet deviation** for this delivery, subject to independent whole-branch review. Both original failed metric boundaries and all adverse receipts remain retained. This is a documented requirement amendment, not an invalid-fixture or corrected-pass claim.

The final task-private probe source matches that contract:50 measured samples after5 warmups for common/rare/miss on each same retained20k cache; lexical timer encloses the complete original decorated method; admitted timer retains initial guard, actual fence/root/chain/status/scope/floor and fence exit. Guard now precedes fence as production does; corpus-only state/count assertions are outside. Store/read-pin setup and final owner closure remain outside the explicitly cached candidate clock. The20 real public marker samples per cache retain separate instrumented whole-tool/native Capture and empty-entry/positive-retirement assertions without a test outer ownership scope.

Final source/load/sample receipts and actual native retirement are **pending at this checkpoint**. No six-cell canonical pass is inferred from code quality or scoped correctness tests. The accepted admitted limitation must continue to be reported numerically, regardless of final lexical outcome. See `k5-idle-metric-alignment-review.md` for precise historical contract comparison, instrumentation limits and the reviewer disposition on this explicit deviation.

## Declined to judge

No new broad source review, proposed generic storage/config admission optimization, POSIX qualification or fresh full suite is in this scoped loop. Earlier declined/qualified matters remain recorded in `task-5-review.md`; no new deferred behavior was introduced. This report does not certify a planned timing result, change a budget, authorize Done, or replace final whole-branch assessment of the admitted-latency deviation.
## Final evidence closure — source5d927

The actual query-only run has completed: `k5-fix2-idle-latency.txt` records **1 passed, 1 warning in256.56 s**, with255.50 s test-call time. Root reports no active probe. The test pass is the canonical lexical gate plus correctness/retirement assertions; it is **not** an admitted500 ms pass. Read raw `.superpowers/k5-scale/fix2-idle-latency.json` and `fix2-idle-source.json`, not a planned result. Their source commit is exactly5d927a98ffc695c281465b09a8020d1db196490a, started2026-10-22T01:16:24.810599Z. Independently compared all29 recorded source SHA256 values plus the probe hash to current bytes:30 matches,0 mismatches.

Both same retained physical completed20k caches/profile were queried without rebuilding. For each shape the raw data contains common/rare/miss50 measured samples after5 warmups (330 total candidate calls,300 retained measurements), with all expected counts100/1/0. Recomputed canonical p95 as sorted sample[47] agrees with every recorded value. These are the recorded-machine empirical samples, not a confidence bound or a universal latency guarantee. No competing K5 test/build/render ran; light read-only review/coordination could overlap and unrelated OS background was uncontrolled.

| Corpus/workload | Canonical p50 ms | Canonical p95 ms | Admitted p50 ms | Admitted p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Nested/common | 208.1393 | 222.3257 | 495.2744 | 534.4660 |
| Nested/rare | 157.6595 | 172.1625 | 439.7766 | 469.0829 |
| Nested/miss | 156.7933 | 181.9234 | 441.0551 | 493.7689 |
| Wide/common | 212.3573 | 232.7950 | 502.7325 | 561.9087 |
| Wide/rare | 157.2588 | 177.2331 | 444.5825 | 479.2464 |
| Wide/miss | 157.7064 | 168.1995 | 443.0204 | 467.3531 |

**All six canonical decorated lexical p95 cells satisfy the original <=500 ms target.** The largest is232.7950 ms. The original nested/common577.3054 ms miss is addressed on the final qualified source. Both common admitted p95 cells still exceed500 ms; the stronger target remains an explicitly accepted unmet deviation under the pre-code root ruling. Its actual authority/status/floor costs were executed and retained, not removed or reclassified as fixture overhead. Four passing admitted cells do not make that broader target pass.

Public measurements remain separate,20 actual marker calls per shape:

| Corpus | Canonical lexical p50/p95 ms | Native Capture p50/p95 ms | Instrumented whole-tool p50/p95 ms |
| --- | ---: | ---: | ---: |
| Nested | 162.5927 /214.5572 | 240.5639 /281.9572 | 1994.4823 /2703.4715 |
| Wide | 156.0074 /168.9987 | 229.7040 /257.1561 | 1909.2825 /1957.3418 |

Each public call produced the fresh expected marker. All40 entry/return registry checks are true, all40 public native-retirement records pass, and both authority warmups bring the total to42 positively gone/retired/pipes-closed native Capture records with non-null returncode. Both candidate/setup owners retired before public measurement. Both shapes end with zero searches; final service_closed is true and searches_remaining_after_close is0. Capture timing includes actual production worker/pipes retirement; whole-tool timing additionally includes the probe's psutil retirement assertion/bookkeeping, as disclosed. Neither live Capture nor whole-tool latency has the500 ms canonical budget. The seeded provider root/guard limitation remains; this is not every possible real Console guard workload.

### Consolidated task gate and retained qualifications

The source/security remedy and actual canonical metric gate are now satisfied. **K5 task spec gate passes under the explicitly amended contract and accepted admitted-latency deviation; task quality gate passes with the retained evidence qualifications.** No blocking Critical/Important issue remains in this task review. Original I1 invalid-draft management and I2 injected shutdown-fixture proof are ADDRESSED by fix1; fix2 adds no new findings. Original Minor M1 scanner public typing/docs remains deferred for root/final whole-branch triage. This closes the original task review's pending metric evidence, not the final whole-branch review or integration/Done procedure.

The following qualifications still apply; no claim of an all-green full suite or exact-final-byte rerun of every receipt is made:

- Original eight-file covering is1failed181passed3skipped1deselected; the sole I2 failure is superseded by the corrected exact shutdown/related5-pass receipt, and pre-fix card evidence by the full post-fix23-pass card receipt. The deselected actual junction case separately passed. Windows filename/permission/rename skips and absent POSIX/Windows ACL-denial qualification remain.
- Both literal scale builds completed and retired. Later formatter-only source was AST-qualified, fix1 changed only three authorized card/fixture paths, and fix2 changes only the query predicate/test. Original scale/build receipts retain their historical source timing; final query metrics are on exact5d927 bytes. Sampled RSS/disk/throughput limits remain samples, not exact physical maxima.
- Staged recovery proves held Capture/pin, full native and query-process retirement, actual encrypted rollback and fresh-child old-key refusal. It does not prove simultaneous restore against the same live service.
- The latest visual pairs retain per-image source hashes and theme qualification. Clear focus/completion frames do not independently establish settled Off controls; mounted state assertions do. The startup-overridden light-theme image remains adverse historical evidence.
- Existing lint/governance failures retain unchanged-source attribution rather than an exact baseline runtime reproduction or an all-green claim. Deprecation warnings, omitted full suite, no human approval UI qualification, and other earlier declined scope remain recorded in the original reviews.
- All earlier failed scale/setup/performance receipts stay retained. In particular, no original admitted failure is erased by the contract amendment, and the final whole-branch reviewer must independently assess the accepted limitation and observed approximately1.9–2.7 s instrumented public-call p50/p95 values.

No duplicate source review or test execution was performed for this closure. Subsequently read the completed final append in task-5-fix2-report.md and task-5-report.md: it agrees with the raw result and preserves both failed admitted common cells. Also read fix2-idle-process-retirement.json (2026-10-22T01:23:02.766926Z): exact final source, no matching idle/scanner processes, service_closed true and searches0. k5-fix2-private-child-results.json retains both provider child XML records/hashes, each one test with zero errors/failures/skips. No final task-review evidence remains pending. Root still owns QA/metadata retention, deferred Minor triage and the independent whole-delivery review.