# Diagnostic exit and Console presentation stalls — TASK-35612

Original production baseline: `5629f0bdc52be71a4152fdc4c6e9c6336006a9f1`.
Python 3.12.10, Textual 8.2.8, native Windows, same 170×48 headless viewport.

## Diagnostic exit investigation

The earlier `matching-after1` run stopped halfway through its first 60-second
`faulthandler.dump_traceback_later` dump, with no Python exception or final report.
The shell wrapper lost the actual native exit code. Its original partial JSON and
stack dump remain committed; a frame name in that dump does not identify a fault
in the storage routine it happened to interrupt.

CPython documents a GIL-free traceback-timer race involving invalid/freed frames
in [issue 140815](https://github.com/python/cpython/issues/140815) and
[fix 140921](https://github.com/python/cpython/pull/140921). This is a plausible
diagnostic failure mechanism, **not proof of this run's cause**. A bounded,
storage-free stress comparison (three timer and three synchronous runs, four
seconds each) exited zero in all six cases: `traceback_stress.py` and
`traceback-stress.json`. The original exit remains unreproduced and unclassified.

The probe now invokes synchronous dumps from a Python thread, stops/joins it
before closing the log, and keeps synchronous failure/timeout dumps. This avoids
the suspect timer mechanism but cannot diagnose an indefinite GIL-held hang;
that requires an external debugger. The ABBA runner records each subprocess's
exact exit code and source hashes, including failures. Both comparison arms use
the corrected diagnostic machinery.

## Changes and focused evidence

- Hidden Home artifact/content/cache completion previously rebuilt its dashboard
  behind Console, including notification admission. All three now retain pending
  presentation until a successful visible cache warm; unchanged snapshots replay
  and failed warming keeps the pending work. Native reads and the TTL are intact.
- One real inspector build invoked settings loading nine times. The existing
  synchronous derivation scope reduces that to one. The next build reads current
  config again, retaining the session's model override. Nothing spans an await;
  FULL's fresh post-await rail and hidden inspector content remain.
- Checked display no longer waits for the storage coordinator. Failed nonblocking
  acquisition refuses/defer-retries presentation. Nested pricing/capacity helpers
  also refuse rather than taking a blocking native fallback. Expired displays
  that already entered native presentation retire fully before retry. Cleanup
  errors remain failures. Admission, source/owner/pause checks, display expiry,
  native/action routes and physical lifetimes are unchanged.

Initial regressions failed on the original code: four hidden Home cases and four
contention cases; inspector measured nine settings loads. Independent review
found a native-fallback sentinel escape; two genuine-expiry regressions failed
before the retirement-safe retry fix and passed afterward. One inspector test
assertion was corrected to compare the model row, not unrelated runtime changes
such as the bridge becoming available between builds.

46 distinct focused cases passed across serial runs: seven Home completion
regressions, six contention/retirement regressions, five provider/inspector
derivation cases, three existing Home contracts, hidden-inspector freshness,
inspector sharing, twenty pricing contracts, and three native config lifetime
cases (cleanup failure, source change, pause). No full suite was run.
The final combined new-regression/inspector/rail batch passed all 15 cases on
the completed code (101.67 seconds); earlier corrected failures are recorded in
`stalls-verification.json` rather than omitted from the history.

The existing inspector-sharing test initially failed before the UI ran with
`raw_source_selection_changed`, both in the group and alone. The exact original
production modules reproduced the same failure. Giving this one test the existing
`bootstrap_profile` marker keeps its selected config stable; it then passed with
the real storage guard. No assertion or admission check was removed.

Ruff checks/formatting pass for new tests and diagnostic helpers. The changed
legacy files retain their baseline lint/format debt with no added lint diagnostics
(`stalls-lint-comparison.json`). Independent read-only reviews found no remaining
blockers after the native-fallback fix.

## Measurements

Serial ABBA results (six returns per fresh process):

| Pair / code | Median input ms | Worst input ms | Median attach ms | Median largest heartbeat overshoot ms |
| --- | ---: | ---: | ---: | ---: |
| Original, first | 900.9 | 1298.6 | 905.7 | 220.6 |
| Updated, second | 748.1 | 1111.2 | 748.1 | 176.1 |
| Updated, third | 821.3 | 1163.1 | 823.7 | 183.3 |
| Original, fourth | 828.1 | 1288.6 | 830.9 | 215.6 |

The first pair's median improves 17.0%; the reverse-order pair improves only
0.8%, effectively flat at this sample size. Worst observed input improves 14.4%
and 9.7%; heartbeat overshoot medians improve 20.2% and 15.0%. All 24 drafts
survive and attaches complete; all four subprocesses exit zero, with zero network
attempts and zero real-profile guard events. These small headless samples do not
establish a universal median speedup or a crash fix.

Checked-display stack samples fall from 6/5 in the originals to 0/0 updated.
Fresh config-posture samples remain (7/8 original, 11/7 updated), as does CSS
matching (17/16 original, 15/10 updated). Hidden Home notification reads happen
in neither fresh control this time; their removal is covered by the saved earlier
traces and deterministic completion regressions, not an isolated latency claim.
The inspector's 9→1 read count is a direct original/current regression result;
sampling counts are too sparse to estimate its standalone time saving.

All timing processes finish in 50–52 seconds, before their 60-second dump. A
separate one-second dump stress run exercises the replacement; it is excluded
from the timing table. No app profile runs concurrently with tests.
That accelerated run completed 49 dumps and six returns, retained every draft,
completed every attach, and exited zero with no network/profile-guard events
(`plain-ready-stalls-dump-stress.json`, `stalls-dump-stress-exit.json`). Its full
stack log is retained locally. The earlier unexplained exit remains unresolved.
`stalls_baseline.py` restores the exact four original modules in memory without
changing the checkout. The clock-safe polling floor and native storage checks
are identical in both arms. Resume-to-input means observed draft input, not paint
completion or a Send/Stop latency measurement. Stack samples overlap and are not
additive elapsed timings.
Exit receipts hash the unchanged checkout files; original runs load the pinned
baseline revision recorded in their plain report rather than those checkout files.

Reproduce with `.venv/Scripts/python.exe`:

```text
Docs/Development/evidence/windows-performance-2026-10-18/run_stalls_profiles.py
Docs/Development/evidence/windows-performance-2026-10-18/summarize_stalls.py
Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --traceback-seconds 1 --tag stalls-dump-stress
```

The runner retains failed/incomplete receipts and stops on nonzero exit. Use the
private-profile test runner `matching_tests.py` for the targeted files named above;
ordinary shared Windows TEMP is intentionally refused by native admission.
