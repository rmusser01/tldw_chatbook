# Windows Console attachment and boot admission

TASK-35679.2, PR #3572. Investigation starts from `7dbeff56aba6032f43aba2f4bbab803d3f133471`. Existing native storage admission, source ownership, path/permission checks, fresh effect-time gates and physical retirement remain authoritative.

The selected UI starter falls from **637.492 ms to 7.308 ms** in original-code diagnostics. The matched timing comparison completes **24/24 returns**, but attachment is not uniformly faster: its first return is slower in both candidate processes. No general latency win is established.

## Historical timeout

The prior packet's A3 (`80710424ceebe3c545160fa368dfe344f9c83f78`) completes one return, then fails the existing ten-second attachment wait on its second. Its first observed key takes 5647 ms after posting; attachment had already completed. During the timed-out second return, 520/575 main-thread samples (90.43%) are IOCP polling. The timeout dump catches a workspace read opening a checked SQLite connection through Windows filesystem verification and a context read qualifying native admission evidence. This supports awaited native preparation; it does **not** identify a unique historical timeout owner or establish executor starvation/deadlock. The original adverse packet remains in `../windows-console-2026-11-15/`.

## Measured correction

The fresh original-code diagnostic completes all six returns and observes the subscription starter's original `execution_allowed` on the main thread: **637.161 ms wall, 125 ms thread CPU**. The enclosing starter lasts 637.492 ms and that return's maximum sampled loop overshoot is 608.328 ms. Wall minus CPU includes native waits and descheduling; it is not a measurement of mutex wait alone.

The correction runs the complete informational gate and original callback in one finite App-owned invocation (`app_boot_workers.py`). Both existing App shutdown entries drain its physical owner; the backfill's fresh effect-time `execution_scope` remains. Retry/error policy, actual Worker context, exact handle ownership and callback results are preserved. A stale terminal event cannot release a replacement's boot slot.

[diagnostic-admission-proof.json](diagnostic-admission-proof.json) retains the exact original rows. The after starter measures 7.308 ms wall and zero CPU at the coarse thread clock. A positively paired background invocation runs on thread 12388 for 4100.419 ms wall / 875 ms CPU across resume 1 and 2; resume 2 posts and observes its key in 78.900 ms while that invocation remains active. This proves background execution and responsive draft delivery during its lifetime. The worker includes admission **and backfill**, so its duration is not comparable to the original gate alone. Five main-thread admission spans remain during startup; this correction concerns the subscriptions starter.

The empty-chat context probe is cheap (median 0.027 ms in this diagnostic), while its surrounding native scopes cost hundreds of milliseconds. Skipping those scopes was rejected: they also enforce pause, path/source and authority refusal. Workspace and readiness native reads remain measured costs for future investigation. Overlapping spans must not be summed as journey time.

## Comparison protocol

`run_probe.py` launches the original headless six-return journey with identical main-thread stack sampling, explicit composer focus, Windows-clock-safe polling, ten-second attachment bound, 300-second external deadline, network/profile guards and native private fixture ownership. Each row checks both the newly posted key through attachment and prior-draft continuity across navigation. Product-source maps and CSS hashes are captured before/after; receipts hash all four executed harness files. Diagnostics use original-code monitoring and remain separate from timing cohorts.

The diagnostic precedes an added prior-draft assertion and EOL normalization of two baseline files; it is attribution evidence, not a matched timing arm. The plain cohorts use the same frozen harness. Candidate source differences are the two App wiring modules and the added finite boot helper; all other product bytes must match their baseline maps.

The preliminary `after-1`/`after-2` runs also changed an unchanged startup file from CRLF to LF. `preliminary-source-qualification.json` proves that sole extra difference is EOL and preregisters replacement `after-3`/`after-4` runs. Preliminary results remain retained outside the matched aggregate. The replacement restores the exact baseline startup bytes and verifies the complete source delta before launch. Evidence-local Git attributes preserve the executed harness bytes, including CRLF, while retaining whitespace checks; receipt hashes must match the committed harness blobs.

These are sampled headless measurements, not terminal paint or automatic arrival-focus measurements. Our runs are serial, but unrelated native test workloads on this Windows host mean host-wide contention is uncontrolled. Small cohorts do not establish statistical significance.

## Matched timing results

Product baseline: `7dbeff56aba6032f43aba2f4bbab803d3f133471`. Product candidate: `ea5a104807a53c2ba69c2434f6a4a1b22118b139`. The final checkout's 3013 product hashes equal the measured candidate.

| Cohort, in comparison order | Returns / exit | Median navigation to observed draft, ms | Median navigation to attachment, ms |
| --- | --- | ---: | ---: |
| before-1 | 6/6 / 0 | 120.993 | 4272.336 |
| after-3 | 6/6 / 0 | 95.363 | 3460.905 |
| after-4 | 6/6 / 0 | 121.677 | 3608.778 |
| before-2 | 6/6 / 0 | 126.515 | 3609.550 |

[comparison.json](comparison.json), produced by [compare.py](compare.py), records `qualified: true`: exact heads, identical executed harness and runtime settings, all product/CSS maps unchanged within runs, matching maps within each arm, only the intended three product paths differing across arms, six complete returns, draft checks, stopped observers, zero network/profile guard counts and unchanged 10-second/300-second bounds. Qualification establishes valid packet comparability; it does not establish statistical significance.

Across twelve returns per arm, the median navigation-to-draft is **120.993 → 102.127 ms** and attachment **3683.187 → 3460.905 ms**. The worst observed sampled loop overshoot is **360.003 → 367.724 ms**. First-return attachment is **3299/2565 ms baseline versus 4924/4022 ms candidate**; first-return draft is **280/486 versus 476/440 ms**. Keep these adverse first-return results visible rather than inferring a uniform gain from aggregate medians. Source hashing warms filesystem caches, runs are separated by diagnostics/validation, and this is not an exclusive or OS-cold host.

[first-return-analysis.json](first-return-analysis.json) preserves the adverse samples and their sources. Plain first-return main stacks poll in IOCP for 79–88% of samples and contain no sampled subscription-starter/admission frame; main-only sampling cannot exclude background overlap. Existing diagnostics show longer CSS work and awaited native preparation, with the selected candidate subscription worker starting after the diagnostic first return completes. This does not establish a unique cause for the plain first-return worsening.

## Validation and retained failures

The causal RED observes the original native gate on `_MainThread`, without replacing it. GREEN coverage observes that original gate off-loop, actual Worker/Future ownership, fresh Context, pause refusal, body result/error policy, exact local-pause retry, queued close, stale terminal ownership, fresh body admission, and physical drain through both shutdown entries under repeated cancellation.

The connected Actor Pack fixture now waits until the original drain yields with its exact retained read before issuing cancellation. Its old single-loop-turn assumption could instead interrupt the earlier project-index drain. The shipping outer shutdown retries that cancellation; no shutdown code changed for this fixture correction. Native connection/lease/retirement/reopen assertions are retained.

All unsuccessful runs are retained, including a wrong test-node collection error, whole-process child timeouts and the original Actor fixture race. A child's success marker before a process deadline does not establish prompt process exit. Test receipt diffs predate staging the new test file; the final committed test is the reproducible source, and those early receipts alone do not hash its then-untracked contents.

The final full native selection passed ten cases and hit two 60-second process deadlines: the subscription thread control had completed its assertions, and the Actor child was still seeding content before its exercise. These two test files now allow 120 seconds for the whole child process, including imports/seed/exit. Internal held-operation and cancellation/checkpoint deadlines remain unchanged. This is a test startup-budget adjustment; neither the comparison's ten-second attachment wait nor its 300-second process bound was increased. Both failed child logs are retained alongside the reruns.

The final policy/ratchet/P0 selection passes **391 tests**. The native selection passes ten cases, and both whole-process timeout cases then pass with the documented 120-second process budget; the corrected Actor checkpoint reaches and retires its exact original owner. Earlier connected coverage passes twenty cases and exposes the subsequently corrected Actor sequencing assumption. The targeted startup census passes at the unchanged **1040/1040** limit. These selections overlap and are not a unique-test total. Scoped F checks and new-file format checks pass with pinned **Ruff 0.16.6**.

Independent source review found no remaining Critical/Important issue after correcting fixture assumptions and preserving every existing module ceiling. Evidence review additionally required hashing the actual published harness and a complete normalized-byte/manifest witness for the excluded preliminary cohorts; both are now checked. No full suite was run. Required remote CI remains a separate result; PR auto-merge stays off.
