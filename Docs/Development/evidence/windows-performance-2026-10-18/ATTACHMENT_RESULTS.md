# Console attachment completion — TASK-35615

Baseline: `3742efae9428dc10f879606b519215913263ac4e` (the prior PR head).
Native Windows, Python 3.12.10, Textual 8.2.8, headless 170×48 viewport.

## Diagnosis and correction

The return trace identifies `console-manual-read-return` entering FULL while
`console-attach-reconciliation` is already waiting on configuration. The duplicate
call sets the coalesced request flag, causing another FULL after the first one.
In one diagnostic return that second pass also crossed the character-context
recheck deadline and spent approximately 714 ms before configuration presentation.

`acknowledge_explicit_console_return` requested that FULL even when its freshly
captured visit was absent, the read refused, or there was no unread token. The
four-line correction returns in those already-ineligible acknowledgement cases.
Ordinary attachment still performs its original FULL. Saved chats still read the
current reminder; marked chats retain their original FULL and after-paint
acknowledgement with generation/profile/service/session/painted-owner checks and
the exact compare-and-clear token. No storage check, lifetime or permission cache
changes. Existing ADR-171 and ADR-126/230 apply; no new ADR is required.

## Rejected scheduling experiments

These separate instrumented discovery runs are not the uninstrumented A/B gate.
All six returns per run completed, retained drafts and reported zero network
attempts/profile refusals; their subprocesses exited zero.
`attachment-diagnostics.json` retains journey values, original native-span samples
and the decisive FULL worker trace. Detailed local stack dumps remain diagnostic
artifacts; published evidence does not require those large dumps.

| Diagnostic | Median input | Median attachment |
| --- | ---: | ---: |
| Original queue trace | 159.99 ms | 1190.57 ms |
| Cooperative timer experiment | 209.84 ms | 2278.82 ms |
| 100 µs handoff experiment | 982.11 ms | 2247.10 ms |

The Windows default pool has 16 workers on this 12-core host. One captured dump
showed 15 occupied by Textual timer waits. Config/workspace native queue medians
were 75/119 ms. Using cooperative long timer waits cut those to about 6/1 ms but
increased callback execution times and total attachment latency. Lowering the
existing finite-read interpreter handoff limit also regressed this workload.
Neither experiment is in production code. Diagnostic flags remain only to
reproduce the rejected experiments. Queue delay is one remaining hotspot, not
evidence that increasing concurrency improves the complete interaction.

## Serial plain ABBA results

Six Home/Console returns per fresh process, no diagnostic method wrappers or
timer/handoff experiments enabled. Tests finished before these profiles started.

| Run | Median input | Median attachment | Worst input | Worst attachment |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 232.90 ms | 1315.90 ms | 307.25 ms | 1743.12 ms |
| Updated 1 | 175.31 ms | 1036.77 ms | 212.34 ms | 2106.49 ms |
| Updated 2 | 153.73 ms | 970.42 ms | 171.06 ms | 1179.40 ms |
| Baseline 2 | 177.38 ms | 1069.05 ms | 248.39 ms | 1332.45 ms |

Across 12 returns per arm, input median **191.33 → 169.09 ms (11.6% lower)** and
attachment median **1149.08 → 979.41 ms (14.8% lower)**. Both updated run medians
are lower than either baseline median. All **24** returns retained drafts and
completed attachment, all four subprocesses exited zero, and all network-attempt
and real-profile-refusal counts were zero. Raw packets and exit receipts are
retained alongside `attachment-comparison.json`.

The updated worst attachment was **2106.49 ms**, worse than the baseline worst
1743.12 ms. This correction reduces unnecessary work and the observed medians;
it does not establish an improvement in tail latency. The earlier 727/976 ms
comparison used a different baseline and time window; these new results must not
be presented as recovery of all that earlier regression.
The outlier was updated run 1's first return: typing was available at 212 ms,
maximum loop overshoot was 146 ms, and 113/129 main-thread samples were in the
event-loop poll. That supports waiting rather than a two-second continuous UI
block; the sampler does not identify the exact awaited callback or prove its cause.

Remaining samples include upstream CSS application (24/27 inclusive main-thread
samples in updated runs) and fresh config posture (4/9). Counts overlap and are
not measured CPU durations. Native preparation and executor contention remain;
neither bypassing their checks nor broad concurrency changes are justified by
these results. See `attachment-hotspots.json`.

PR head advanced concurrently to `48339e4d6d` with evidence and an opt-in diagnostic
flag only. It was committed at 14:15 PDT, before this 14:24–14:28 PDT ABBA run;
the process census during profiling found only this task's profiler. Integration
preserves that update. Its product code is identical to this comparison baseline.

## Verification

The new reminder regression initially hit the repository's source-selection
guard because its fixture lacked `bootstrap_profile`. After correcting fixture
selection, original code failed the three unnecessary-refresh cases; no-session
and actual-unread controls passed. The fixed code passed all 57 targeted checks
covering reminder branching, attachment retries/stale visits, typing, original
FULL/config/workspace reads, cancellation and physical worker retirement.
Ten additional mounted real-SQLite reminder checks passed, including marked and
unmarked explicit returns and all existing newer-token, wrong-profile/service,
navigation, unpainted-session and wrong-chat acknowledgement fences. Six unrelated
compact-row geometry cases were deselected. Total targeted GREEN: 67 passed.

`attachment-safety-sources.json` compares nine complete storage/config/reminder
sources and seven original reminder methods with the baseline. All are unchanged.
Independent read-only review found no product safety/correctness issues.
New regression/diagnostic files pass Ruff lint and formatting. Existing session
and unread-test lint counts remain 10 and 5 respectively, with no added findings;
see `attachment-lint.json`. The existing product module's formatting debt is
unchanged; only the four-line product fix is included.
No full-suite sweep was run.
All 14 exact task-owned disposable fixture roots were removed after completion;
their recorded identities and cleanup count are in `attachment-verification.json`.

## Reproduction

```powershell
python Docs/Development/evidence/windows-performance-2026-10-18/run_attachment_profiles.py
python Docs/Development/evidence/windows-performance-2026-10-18/summarize_attachment.py
python Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --attachment-observe --tag attachment-request-trace
```

The serial ABBA runner substitutes only the exact baseline session module in
memory for controls; it never swaps working files. Each subprocess records exit
status and the checkout's session-source hash. Discovery spans add instrumentation
and cannot be directly compared with plain runs. The 31.25 ms observer floor is
retained to avoid sub-resolution Windows polling. These small empty-chat return
workloads measure posted-key observation and reconciliation completion, not
terminal paint latency, long conversations, provider traffic or whole-app p95.
