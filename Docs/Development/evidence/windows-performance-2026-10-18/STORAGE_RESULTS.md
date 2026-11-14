# Windows native storage initialization wait

TASK-35595; PR #3166 follow-up to [FULL_RESULTS.md](FULL_RESULTS.md).

## Cause and bounded fix

`_Acquisition.initializing` serializes same-process first-use registration.
While another initializer owned the root, its followers woke every 10 ms and
repeated `self.check(path)` under the global `_changed` condition lock. For a
genuinely admitted nested database operation, that check performs native parent
identity and resolved-path validation. The preceding FULL diagnostics captured
followers in this wait and another thread performing those Windows checks.

The new regression holds an original initialization owner while an actual
file-backed database operation requests its nested storage lease. It observes
the original check and condition wait without replacing either callback. On
`579de70851ca4c655947f9962671be774296c80f`, checks increased **1 → 16 in 250 ms**
despite no owner-state notification; every wait had the original 10 ms timeout.
The test fails specifically on this repeated validation.

The product change replaces only `_changed.wait(0.01)` in this initializer with
`_changed.wait()`. Leader retirement/error, acquisition close, local pause and
resume already notify under the same lock. Predicate checking and waiting remain
atomic. Checks still run on entry and after every notification; a spurious wake
cannot admit a follower. Path/source changes while parked are rejected before
dependent effects when the owner releases. There is no cached safety verdict.

Native cross-process waits, `hold.ready` polling, maintenance deadlines,
recursive-entry refusal, path identity, execution selection and physical lease
retirement remain unchanged. Directly setting the private cancellation Event is
not the production cancellation protocol: `_begin_local_pause` sets it and
notifies. Its native registry-lock test remains selected separately.

The timer-only replay still stalled. Worker sampling then identified repeated
ancestry traversal inside `admission_authority`: `_require_settled_ancestry`
reopened the entire root-to-parent chain for each ancestor. A native regression
measured **91 directory opens for 13 path components**. The final implementation
uses the existing `native_files.pinned_directory` nofollow traversal with an
internal visitor after each original owner/mode check. It checks the same
creation-intent marker at every existing ancestor, including when descendants
are missing. This requires **13 opens**, an **85.7% reduction** for that fixture.
No settled verdict, permission or directory handle is cached.

ADR required: yes for the bounded visitor contract; the existing
[ADR-126](../../../../backlog/decisions/126-complete-local-backup-and-recovery.md)
is amended under TASK-35595 rather than introducing a duplicate decision.

## Native regression verification

Seven new subprocess cases pass using genuine Windows user-owned private
fixtures. The quiet follower retains **one check across the 250 ms parked
interval**, then completes original native admission after release. Other cases
verify pause cancellation before leader release, spurious wakeups, leader error,
path/source drift, recursion refusal and complete acquisition/operation/lease
retirement. The assertions use the original guards and native scopes.

The final combined focused run is **16 passed**: the seven initializer controls,
eight ancestry controls and one repaired pending-authority fixture. Ancestry
tests check every visited parent and fixture-owned marker location, missing
descendants, a real Windows junction, a real unsafe Everyone-write ACL, visitor
exceptions and native-close error propagation. All observed descriptors retire.

The broader lifetime/lock-order/companion run initially had **51 passed / 12
failed**. The same batch against the pinned original code had **52 passed / 11
failed**, with eleven identical failures. They include stale connection-lifetime
expectations, two Windows non-socket `select` errors and missing symlink privilege;
see `storage-lifetimes-baseline-comparison.json` for exact identities and limits.
Baseline code was loaded in both pytest and every product Python `-c` child;
`storage_baseline_tests.py` preserves that reproducible mechanism.

The unmatched pending-authority case passed twice on both versions without
entering the changed initializer wait. Its original 10 ms remap timeout did not
witness durable pending publication. The fixture now observes the original write,
cancels through the native remap API after publication, and reads back pending
state before the unchanged refusal assertion. It passes on both versions.
The original mismatch is consistent with timeout before publication; those
isolated reruns did not directly reproduce that timing race. No storage check
was bypassed to repair the fixture.

Existing native-files, pending-directory-durability and companion-eligibility
verification finished with **42 passed / 6 failed / 7 skipped**, exactly matching
the pinned original code. The six failures are platform fixtures (stdlib mode
expectations, four unavailable symlink-privilege cases and unavailable `mkfifo`);
the seven skips are explicitly POSIX-only reuse cases. Every selected pending
directory durability case passes. Together with the final focused run, **58
checks pass across the two final targeted batches**. No full suite was run.

New production helper changes, new tests and evidence scripts pass targeted
Ruff checks. `storage_admission.py` retains eight pre-existing E721 findings and
`control_records.py` retains one, unchanged by code/message; exact type checks
were preserved. The existing unrelated formatting debt in those files is not
rewritten. Independent review found no missing notification or removed native
validation; see `storage-lint-comparison.json` and the baseline comparisons.

Run targeted tests with the repository's Windows fixture runner:

```powershell
$testRoot = Join-Path $env:USERPROFILE '.tmp-windows-settings-tests'
$env:TEMP = $testRoot
$env:TMP = $testRoot
& .venv/Scripts/python.exe -m Tests.windows_private_fixture_runner Tests/Backup_Recovery/test_storage_initialization_wait.py -vv --timeout=90
```

## Profiling protocol

The existing guarded probe mounts the actual application at 170×48 in a fresh
private Windows profile, then performs six Home → Console returns. It injects
composer keys, measures draft update and attach completion, and samples the main
thread plus a 10 ms heartbeat. No pytest process runs beside timing probes.
Provider/network and real-profile guards remain active.

`storage-before` was recorded before the edit. `--storage-baseline` loads
`storage_admission.py`, `control_records.py` and `native_files.py` from the pinned revision in memory, retaining every earlier
Console performance fix. `--attach-poll` reinstates the extra 5 ms observation
task from the prior failed runs as a separate diagnostic workload; it is not the
normal-profile latency benchmark. Draft update and attach timing do not measure
paint or completed Send/Stop actions. Overlapping samples are not additive costs.

The timer-only normal run worsened median input arrival **795 → 930 ms**; with
both fixes it was **835 ms**. The extra-observer replay timed out in original,
timer-only and final code, so it does not establish that these product changes
resolve the observed ten-second stall. `storage-initialization-discovery.json`
retains the measured worker hotspots and diagnostic limits.

This machine's Python 3.12 event loop uses `GetTickCount64` with a **15.625 ms**
clock resolution. Multiple diagnostic pollers below that resolution can make
timers ready early and keep the loop repeatedly polling without blocking.
`--clock-safe` sets only the probe's polling and heartbeat intervals to at least
twice that resolution; it does not change product timers or storage behavior.
Final paired profiling uses this mode to separate measurement interference.

`asyncio_polling_probe.py` reproduces the mechanism without importing application
code. Over two seconds, 5/5/10 ms pollers caused **261,448 loop iterations**, every
selector call with zero timeout, and **1968.75 ms main-thread CPU**. Clock-aware
pollers caused **92 iterations**, with CPU below the 15.625 ms accounting tick.
An independent ctypes worker made 25,628 calls under short polling versus
7,258,397 under clock-aware polling. This is scheduling evidence, not a storage
throughput claim; the worker only calls `GetCurrentProcessId`. Raw results are
in `asyncio-polling-results.json`.

```powershell
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --tag storage-after
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --storage-baseline --tag storage-contention-control
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --tag storage-contention-verified
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_full.py storage
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_full.py storage-contention
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --storage-baseline --tag storage-clock-control
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --tag storage-clock-verified
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_full.py storage-clock
```

## Final paired results

Both clock-aware runs completed all six returns with drafts retained and attach
reconciled. Both recorded zero network attempts and zero real-profile refusals.
The original-code control also completes once observation is clock-aware: the
earlier extra-observer timeout cannot be credited as a product fix.

| Measurement | Original | Both storage fixes |
| --- | ---: | ---: |
| Median return-to-input | 940.33 ms | 744.42 ms |
| Worst return-to-input | 1791.58 ms | 890.50 ms |
| Median return-to-attach | 944.00 ms | 749.41 ms |
| Worst return-to-attach | 1797.40 ms | 895.18 ms |
| Median per-return worst heartbeat overshoot | 289.32 ms | 275.62 ms |
| Worst heartbeat overshoot | 330.55 ms | 311.46 ms |

Median input readiness improved **20.8%** and the worst observed return **50.3%**
in this matched six-return pair. This is a headless empty-profile measurement,
not a general tail-latency guarantee; the earlier short-poll normal pair regressed
about 5%. The deterministic reductions in parked checks and native directory
opens are stronger causal evidence than one timing pair. Clock-aware observations
have 31.25 ms polling/heartbeat granularity and must not be mixed with earlier
10 ms heartbeat results. Main-thread sampling still uses 10 ms intervals.

Remaining main-thread samples are led by Textual CSS application (61), followed
by inspector and workspace/browser presentation (about 10 each; overlapping).
Those are the next profiling candidates. Completed Send/Stop actions, rendered
paint and large histories are outside this workload.
