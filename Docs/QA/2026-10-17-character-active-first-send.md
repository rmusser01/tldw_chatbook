# Character active display: fresh first-send comparison

Baseline: actual PR3141 merge `6d25091df9e276aa1b6a8d60eaab144c0ec2c748`.
Candidate: `12c803b093a03b2dde11cccc97dd5e01cef3d780`.

Stock active Character display now passes its original current-owner closure into the existing finite batch. Direct actions retain four fresh callbacks; idle, active and queued-to-active displays use two, with the same three metadata pairs and recent-chat read. Original native callbacks/retirement, source/owner fences, one-second run check/invalidation and two-second display memo remain.

## Results

Eight unprofiled fresh profiles, four per version, balanced B/C/C/B/B/C/C/B; one actual Enter each,120x35, DeepSeek Flash, identical synthetic CEDAR input. Post-save means durable-save-success to trace-reservation-entry, distinct from actual Enter-to-HTTP-send.

| Run | Version | Post-save (ms) | Enter to HTTP (s) |
| --- | --- | ---: | ---: |
| 1 | baseline | 1440 | 2.679309 |
| 2 | candidate | 933 | 1.940536 |
| 3 | candidate | 1065 | 2.082305 |
| 4 | baseline | 782 | 1.785013 |
| 5 | baseline | 1415 | 2.975148 |
| 6 | candidate | 1032 | 2.492981 |
| 7 | candidate | 1165 | 2.157907 |
| 8 | baseline | 1439 | 2.738223 |

Post-save median **1427 ->1048.5ms**,378.5ms lower (26.5%); baseline range782-1440ms, candidate933-1165ms. Enter-to-HTTP median **2.708766 ->2.120106s**,588.66ms lower (21.7%); baseline range1.785013-2.975148s, candidate1.940536-2.492981s. Three of four pairs improved; the adverse pair is retained.

| Baseline/candidate runs | Post-save saved (ms) | Enter-to-HTTP saved (ms) |
| --- | ---: | ---: |
| 1/2 | 507 | 738.773 |
| 4/3 | -283 | -297.292 |
| 5/6 | 383 | 482.167 |
| 8/7 | 274 | 580.315 |

All8 HTTP200/exact replies, complete verified-equal traces, zero pending checkpoints. Exactly27 ordered tool schema hashes, captured headers and full supplied settings (normalizing only private profile paths) match the actual-merge diagnostic. Supplied temperature0.7 and captured temp0.6 were compared separately to their respective unchanged reference, not substituted. All8 have actual normal quit/Chat+App unmount logs, server/driver exit0, observed owned PIDs absent and loopback ports closed. Whole tracked Python source maps contain8603files per version; unchanged within each run, differing only in the five reviewed Python paths between versions. No detailed profiler/sampler in this cohort.

## Validation and limits

Exact-candidate targeted7-file cohort: **210passed177.86s**, no errors/skips/profile/network guard refusals, selected before/after source maps equal. Causal active/queued baseline failures4!=2 follow all original native reads/retirement; candidate controls pass, including owner/source drift, dynamic no-argument custom dispatch, repeated cancellation and force_fresh. Five touched Python files lint clean; four test files format clean; pre-existing production formatting drift retained. Independent whole-branch and frozen8-helper prelaunch review:0Critical/Important/Minor findings.

This is four local pairs, not population p95, OS cold-cache, native paint, Windows, later-send, large-history or broad-provider qualification. The live chain uses disposable environment plus native admission, without a separate real-home audit hook. Separate app-child exit status and arbitrary native-worker retirement are not collected; observed PID closure is the stated evidence. An inherited RichLog missing-widget error remains in all8 logs. The earlier instrumented actual-merge diagnostic (915ms logged post-save,1.876188s Enter-toHTTP) is a separate cohort and is excluded from this comparison. The parent one-second goal remains open: candidate median1048.5ms still exceeds it.

Raw closed audits, source maps, original logs, schema/settings comparison and independent review are retained in the local pr-review-wave2 evidence directory. No credentials, real user content or raw logs are included in this report. Required hosted CI and merge remain release gates.
