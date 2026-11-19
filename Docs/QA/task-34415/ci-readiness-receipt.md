# PR3034 coalesced-readiness correction receipt

Base reviewed head: `97e64dd93b4aaa2c264ee8d73465376daef825e4`.
Dev: `6a08c6a18add254751023387d6e97203552efa2b`.
Only source change: 23 added lines in
`Tests/UI/test_console_mcp_approval.py::_wait_for_production_console_ready`.
No production/profile/CI or qualification change.

The final-head [UI1 job](https://github.com/rmusser01/tldw_chatbook/actions/runs/37577168073/job/112648535231)
failed the original geometry node's requested=False assertion before layout:
650 passed, one failed, two warnings in 554.27s. PR/UI2/3/4 and Perf succeeded;
the required aggregate FAILED. Hosted CI did not attribute the lock.

## Controlled evidence

Every controlled row runs the original private-profile geometry node and
unchanged assertions. The real REBUILD holder is joined; no valid control
expired its holder watchdog. The probe delegates original sync/render methods,
not fake config values, flags or projection results.

| Raw label | Actual outcome | Credit |
| --- | --- | --- |
| native-red | Original requested=True assertion; one failed in 5.73s | Valid exact-assertion RED on base helper |
| native-green | Installed replay rendered; one passed in 6.36s | Native GREEN after bounded flag wait |
| worker-red | Four flags clear, real Worker unfinished; helper returned before installed replay; one failed in 5.82s | Valid enqueue-gap RED on intermediate flag-only wait |
| worker-green | Same real pending Worker and native holder; installed replay rendered; one passed in 6.36s | GREEN on final helper, no pytest warnings |
| stuck-bounded | Frozen trailing replay; original requested assertion; one failed in 15.12s including startup | Expected rejection after original ten-second phase, not a passing test |
| ordinary-geometry | Uninstrumented original child; one passed in 6.86s | Original geometry control, no pytest warnings |
| observed-red | Call-through observation only; one passed in 5.98s | Misleading label; NOT RED or hosted lock attribution |

Earlier `red`, `held-red`, `replay-red`, `actual-red` attempts failed
probe setup (busy priming, callback-pump polling, original assertion during
priming, or a lazy optional flag). They are retained but are NOT controlled RED.
The first `stuck` attempt had a diagnostic-watchdog bookkeeping teardown error;
`stuck-bounded` corrects only that bookkeeping and supplies the valid negative
control. No failure or teardown error is reclassified as a pass.

The five other original helper consumers passed separately in the existing
bootstrap-profile fixture mode: finishing-card count/focus 12.76s; Alt+A pending
Select focus 12.00s; no-pending notification 11.97s; 80-column Inspector-closed
focus 11.93s; single-row button geometry 12.16s. Each is one test; no pytest
warnings. Tests, getters and admission remain unchanged.

All eleven artifact guards pass. Helper format passes. Base and final Ruff
reports have the same four inherited findings (I001, RUF012, RUF015, I001);
this is no-additions evidence, not whole-file lint cleanliness.

## Review-bound correction

Fresh review accepted the flag/worker predicate but found the initial poll
could exceed the phase budget: installed `Pilot.pause(0.05)` first calls
`_wait_for_screen(timeout=30.0)`. The monotonic condition only ran before
that await. The final helper instead uses a non-draining `asyncio.sleep`
capped by remaining time; it does not cancel real workers or extend the bound.

The [phase guard](ci-readiness-budget-guard.txt), used with the same native
probe, rejects entering Pilot.pause from this final projection phase.
`budget-red` failed that guard in 4.97s with clean holder retirement;
`budget-green` passed in 6.05s and observed the actual pending Worker with
all four flags clear, then the installed replay. This is a direct call-path
guard backed by installed source, not an observed hosted 30-second timeout.
`yield-stuck` still failed the original requested assertion after the
ten-second phase (14.28s total including startup), no fixture error.

All six ordinary consumers were rerun after this one-line correction:
private-profile geometry child one passed in 5.85s; the five original separate
bootstrap-profile cases passed in 11.49s, 11.35s, 11.84s, 11.35s and 11.24s
respectively. Positives have no pytest warnings. Current format and all eleven
guards pass; the same four inherited Ruff findings remain. New `budget-*`,
`yield-*` and guard-source hashes augment the original manifest without
overwriting earlier evidence.

## Retention and rerun

Latest-dev replay: all eight authored commits replayed without conflicts onto
`5a607a14bffe9f9d52ec348985819e2c927a1fdd`, producing
`4382dafadc15a3a945fd67cc1c0b367cb69d9990`. Every range-diff entry is identical;
authored Console/test and script/package/workflow inputs are byte-identical
to reviewed `84d53ecfbb7087a69904f08113a9a4055748de43`.
Incoming PR3016 changes File Notes history/retention/delete safety; no authored
path overlaps. Library's factory is lazy and its shutdown checks for a real
workspace. No File Notes redesign or whole-tree identity claim.

Combined-tree focused selection: nine passed in 17.72s (six real-lock cases,
two Console size checks and ordinary geometry; its private child passed in
5.94s). Five original separate bootstrap-profile controls passed in
12.05s, 13.40s, 13.32s, 13.71s and 13.69s. No pytest warnings; all eleven
combined-tree guards pass. Complete originals remain in
`/private/tmp/pr3034-rebase-5a607-qtopba`; sixteen appended hashes identify
them without publishing environment-rich dumps. Total manifest entries: 111.
New exact-head CI is still required.

Complete raw parent/child logs, XML, static outputs, observation and runnable
probe source remain in `/private/tmp/pr3034-ready-sync-probe-9Zk8Xa`.
The complete hosted log remains at
`/private/tmp/pr3034-final-ui1-failed-37577168073.log` (1064 lines) and in the
linked hosted job. [Hashes](ci-readiness-sha256.txt) identify these originals.
Environment-rich diagnostic dumps are not published; this audited summary
does not pretend to be a normalized full copy. Existing warning/failure
receipts and qualification gaps remain unchanged.

[Exact diagnostic source](ci-readiness-probe.txt), SHA-256
`cfda68215e5e169b512b0f6cd5c6055405b3b801420ca0d3cb765a7772a609c2`,
can be saved as `pr3034_ready_sync_probe.py` in a disposable directory.
Set `PYTHONPATH` to that directory plus the worktree and
`PYTEST_PLUGINS=pr3034_ready_sync_probe`, then run the original geometry node
with the repository interpreter and a fresh `--basetemp`/`--junitxml`.
Default mode exercises native deferral; `PR3034_PROBE_MODE=worker` holds the
actual replay Worker; `PR3034_PROBE_MODE=stuck` must reject the frozen replay.
To exercise the phase guard, also save its source as
`pr3034_ready_sync_budget_guard.py` in that directory and add
`,pr3034_ready_sync_budget_guard` to `PYTEST_PLUGINS`.
Do not substitute this diagnostic for ordinary controls or exact-head CI.
