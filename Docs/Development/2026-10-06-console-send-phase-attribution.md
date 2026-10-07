# Console Send phase attribution

Status: original-body attribution; no new product optimization or performance acceptance claim.

Candidate product source: `f70cacae3b225378573e9e0903422fe0e607553f` on `codex/console-send-preparation-plan`. The prior [shared preparation qualification](2026-10-06-console-shared-preparation-verification.md) retains matched observer-free samples and the failing original native regression.

## Method

`Tests/Performance/console_send_phase_spans.py` selects original Python code objects through local `sys.monitoring` events. It retains bounded scalar names, timestamps, thread/task identifiers, and source hashes. It does not replace product functions or retain arguments, results, receivers or frames. Global monitoring events are zero. Original application bodies, enabled tools, timers, private-profile guards and completion deadlines remain intact; only final provider network I/O is immediate.

Each sample executes three Sends sequentially, then verifies three replies and complete linked traces, zero pending dispatch checkpoints, source equality and diagnostic process/monitor retirement. Runs are sequential. The separate UAT chat's last confirmed state was native-idle with its agents paused; renewed requests received no fresh acknowledgement. Therefore these runs are attribution only, including that coordination limitation.

All timings below are inclusive elapsed wall time. Nested spans include children and waits and must not be added as predicted savings. These diagnostics do not qualify terminal paint or prove ordinary production-native cleanup.

## First two passes

| Run | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| spans-1, action to adapter | 11.137 s | 10.754 s | 10.537 s |
| spans-2, action to adapter | 12.353 s | 11.419 s | 9.276 s |

The following original stage intervals do not overlap. Each cell lists Send 1 / 2 / 3, in seconds.

| Interval | spans-1 | spans-2 |
| --- | --- | --- |
| UI action to UI submit | 1.565 / 1.911 / 1.850 | 1.900 / 2.567 / 1.515 |
| UI submit to controller | .028 / .101 / .103 | .040 / .176 / .106 |
| Controller to provider resolution | 1.314 / 1.351 / 1.676 | 1.544 / 1.459 / 1.431 |
| Resolution to commit start | .661 / .738 / .776 | .508 / .693 / .733 |
| Durable commit stage | .319 / .150 / .147 | .411 / .119 / .118 |
| Commit success to trace reservation | 6.956 / 6.266 / 5.583 | 7.350 / 6.173 / 4.978 |
| Trace reservation to actual adapter | .295 / .238 / .402 | .600 / .233 / .394 |

The partitions equal action-to-adapter time within 0.00015 seconds of the probe's separately sampled origin. Saving is a small part of these totals; the evidence does not support blaming six seconds on the acceptance transaction or removing the save-before-dispatch barrier.

Postcommit function spans across these passes locate awaited prompt history at .726–1.206 seconds, a later hook-admission check at .434–.968 seconds, and shared provider composition at .828–.913 seconds. None is a savings estimate. Checkpoint completion to trace reservation remains 2.065–3.518 seconds.

Pass 2 divides that last interval:

| Interval | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Checkpoint end to bridge body | .796 | .552 | .508 |
| Bridge body to run_turn body | .505 | .377 | .417 |
| run_turn to _run_one | .355 | .260 | .307 |
| _run_one to model adapter body | 1.269 | 1.440 | .380 |
| Adapter body to trace factory | .560 | .573 | .447 |
| Trace factory to trace reservation | .033 | .004 | .005 |

These boundaries alone cannot separate filesystem admission, database work, thread scheduling or contention. Original source maps the middle intervals to nested agent admission, run creation/lifecycle/run-log writes, and independent model-loop admission. A narrower pass observes admission entry separately from its full lifetime.

Warm personal-context and budget lookups are poor optimization targets in these samples: all budget calls are at most .213 ms, warm personal service lookup .008 ms, warm profile-tool composition .033 ms, and profile snapshots .044 ms. Cold personal service construction is .140 seconds. Repeated static call sites alone do not establish a bottleneck.

## Narrowed third pass

The final attribution pass completed three Sends in 11.033 / 8.064 / 10.112 seconds. It records 526 scalar events: 254 complete start/return pairs and 18 original admission-generator yields. All 18 first-entry pairs are complete, the event cap did not overflow, and the local monitor retired. Source manifests match; three replies and linked complete traces remain, with zero checkpoints. Diagnostic Job retirement is positive with no forced cleanup, identity overflow or PID lookup race. General production-native cleanup remains explicitly unqualified.

The observer measures the original `RecoveryAdmissionGuard.execution` generator from entry to its first yield, separating admission from the subsequent work held inside that context. Six such entries precede each adapter call; their measured entry costs range from .184 to .568 seconds. This is now direct evidence of admission cost, rather than attributing a whole context-manager lifetime to its checks.

| Original operation | Send 1 | Send 2 | Send 3 |
| --- | ---: | ---: | ---: |
| Admission immediately before `_run_one` | .230 s | .195 s | .213 s |
| Admission immediately before model-loop `_consume` | .414 s | .353 s | .565 s |
| Run-log binding | 1.091 s | .907 s | .211 s |
| Run creation | .014 s | .015 s | .014 s |
| Both context lifecycle rows, inclusive callback | .034 s | .029 s | .043 s |

The first two admission rows account for almost all elapsed time in their corresponding previously unexplained gaps. The observed DB row writes are a much smaller consolidation candidate than admission/run-log preparation. Neither the six admission durations nor overlapping parent spans are advertised as achievable savings: required fresh checks remain required.

Source inspection identifies one already-supported finite-operation reuse opportunity inside run-log binding: its base-directory and run-directory containment checks independently resolve the same sensitive-path context. `is_within(..., context=...)` already supports sharing that context within one synchronous invocation while continuing to resolve/check each candidate. This is a bounded candidate for targeted controls; it does not justify a global path cache or skipping admission gates.

## Evidence custody and limits

Pass 1 has 262 scalar events/131 paired spans; pass 2 has 368/184. There are no malformed rows, unmatched starts/returns, timestamp reversals or observer overflows. Both 7,741-file before/after source manifests match. Selected production sources still match their retained hashes.

Both passes prove containment Job emptiness at parent exit, release of the Job identity, pipe/identity-monitor task retirement, unchanged containment source, no forced retirement, private-profile removal and local monitoring retirement. The receipt explicitly leaves `ordinary_app_native_cleanup_proven=false`; diagnostic containment is not a general proof of every production resource lifetime.

Pass 2 has one PID diagnostic lookup race and therefore fails a strict zero-race timing qualification criterion despite positive Job retirement. Pass 1 retains the inherited cosmetic source-receipt kind `original_composition_counts`; the actual plugin/output clearly records function spans. Do not alter either raw receipt to hide these limits.

Evidence is retained under `.superpowers/sdd/2026-10-06-console-shared-tool-preparation/native-pairs/candidate-spans-{1,2,3}.*` in the candidate worktree. The original subsecond Send target, actual 100 ms paint target, native regression and other-host qualification remain open.

## Next architecture work

The [retained ordinary commit plan](../superpowers/plans/2026-10-06-console-native-commit-ownership.md) specifies a prerequisite for safely showing earlier Preparing feedback. It does not move durability or history later and does not claim save-speed gains. Atomic admission/promotion, screen-free capture and the runtime initial-hook-review bridge still precede enabling receipt. Measured preparation/admission consolidation remains the main throughput work; warm personal/budget lookups are excluded from speculative optimization.

ADR: [ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md).
