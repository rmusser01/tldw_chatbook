# Native preparation waits and Console attachment — TASK-35645

Baseline: `d24919ab402d503ca8210dc5ee6b0fd186adbd64`. Candidate product:
`bf1463078c`. Native Windows 10, Python 3.12.10, Textual 8.2.8,
170×48 headless Textual viewport. Auto-merge remains **off**.

## Diagnosis

The previous combined experiment's 22.3% worse attachment median remains valid
for its recorded window; these new observations do not establish which earlier
change caused it. Fixed warmup/away delays do not ensure that background native
work has settled. A new original-code diagnostic reproduced a 2.12-second return:
config preparation queued for 249 ms and then ran for 599 ms; FULL deferred after
that preparation. Workspace preparation later queued for 312 ms before running.
Inclusive spans overlap and cannot be added as CPU budgets.

A separate identity-preserving wall/thread-CPU trace identifies 17 display
refusals at the coordinator's nonblocking acquisition branch (line 682 of
`console_spend_projection.py`), plus expiry refusals during pending refresh.
The fresh configuration reads themselves took 377–992 ms wall time and about
94–203 ms thread CPU. Windows thread CPU measurements are coarse, in 15.625 ms
increments here; they distinguish substantial waiting from continuous CPU work,
not exact kernel/GIL/lock contributions. Samples show other Home reads waiting
on serialized native initialization. Executor queueing, live native checks,
coordinator refusal and the existing 200 ms FULL replay all remain real costs.

Tracing original Home lifecycle and content seam entries confirms **15 of 21**
observed content calls began after Home's suspend event. These later count/list
queries continue opening/reading stores while the user has returned to Console.
That is demonstrably avoidable background demand, rather than a reason to weaken
native admission or accept a busy/expired display proof. Prior cooperative-timer
and shorter-handoff experiments were adverse and are not applied here.

## Correction and boundaries

Only `UI/Screens/home_screen.py` changes in the product. Mounted hidden Home
stops issuing further content queries. A read already issued is still awaited
through its original result/cleanup, and a hidden partial build does not replace
the preceding complete snapshot. The next resume starts its ordinary refresh.
Direct unmounted builder callers retain their original behavior. No native
operation, authority check, Console refusal/retry, cache TTL or permission path
is removed or cached differently.

This change covers suspension without cancelling the current read. It does not
redesign preexisting `@work(exclusive=True)` replacement on a rapid subsequent
resume, which can cancel an old coroutine while its thread continues.

Separate corrected-observer runs verify **22 actual content-query worker
entries with old Home versus 8 with updated Home** across six returns each.
Both exited zero, retained all drafts/attachments and had zero guard events.
Updated Home still entered the cheap seam helper 24 times; 16 entries were
already hidden and returned before issuing a worker. A worker queued while
Home was visible may begin after suspension; this change does not cancel it.
This is observed query-demand reduction, not a paired latency qualification.
The updated trace still records 11 coordinator-contention display refusals and
config callbacks ranging from 221 to 1136 ms. Other background work and native
preparation remain material.

## Verification

The seven focused checks first produced **6 failures / 1 pass** on original
product code, then **7 passes** after the correction. They hold each of readiness,
count, notes, conversation and media work; verify retirement before return,
skip later queries, retain the exact previous snapshot object, and refresh all
counts on a visible revisit. Hidden-before-start and direct unmounted callers
are also covered. Two earlier attempts were refused during fixture setup: the
new module initially lacked the existing `bootstrap_profile` marker, and then
mistakenly named it as a fixture. The final run uses the marker and preserves
source-selection guards; those setup errors are not product RED evidence.

The final targeted native batch passed **50 checks in 158.23 seconds**, including
the seven regressions, Home cache/threading/presentation and Library-rail seam
compatibility, held attachment keyboard input, original FULL/config preparation,
workspace source/session/view/maintenance drift and physical retirement after
cancellation. No full suite ran. Lint/format checks pass for changed Python.

Independent read-only review found no product correctness blocker and requested
the now-updated Home architecture description. Its diagnostic finding concerned
missing exception-unwind observation. Attempting to add `PY_UNWIND` to local event
masks was refused by Python 3.12 before app construction (exit 1, retained as
`native-waits-verified-exit.json`). A minimal stdlib RED/GREEN probe confirmed
that `PY_UNWIND` requires a global mask; the observer now enables it globally,
filters unknown code objects and clears it on finish. Earlier discovery packets
used ordinary-return observation only; successful callback spans and Home
entries remain evidence, without claiming complete exception accounting.

## Serial comparison

Controls restore **only Home** from the exact baseline
in memory. All other PR optimizations remain active in both arms. Six identical
Home/Console returns run per fresh private profile, in control/after/after/control
order, with 31.25 ms clock-safe polling/heartbeat and the original guards.
No method wrappers/native observers, tests or discovery runs overlap these
plain timing processes. Exact exits and product hashes accompany each arm.

| Run | Median input | Median attachment | Worst input | Worst attachment |
| --- | ---: | ---: | ---: | ---: |
| Control 1 | 288.27 ms | 3194.34 ms | 566.69 ms | 4339.17 ms |
| Updated 1 | 261.86 ms | 1541.16 ms | 731.16 ms | 2126.77 ms |
| Updated 2 | 223.84 ms | 1672.15 ms | 342.64 ms | 2739.78 ms |
| Control 2 | 166.24 ms | 1573.85 ms | 444.36 ms | 2404.83 ms |

Across 12 returns per arm, pooled input median was **262.72 → 238.43 ms (9.2%
lower)** and attachment median **2152.17 → 1639.77 ms (23.8% lower)**. The forward
attachment pair improved 51.8%, but the reverse pair was **6.2% worse**. The first
control was substantially slower than the last control. These results show a
lower pooled median in this window, not a consistent latency gain or recovery
of the earlier experiment's regression. All raw journeys remain in
`native-waits-comparison.json` and four compact plain packets.

Worst attachment improved **4339.17 → 2739.78 ms**, while worst input regressed
**566.69 → 731.16 ms**. Every one of the 24 returns retained its draft and finished
attachment; all four processes exited zero with zero network-attempt or
real-profile-refusal counts. Auto-merge stays off. Significant native
preparation/replay waits remain, and no overall latency threshold is qualified.

The updated worst-input journey was iteration 4 of updated run 1. Its sampled
main-thread stacks include the direct coalesced control-bar path:
`_run_coalesced_control_bar_sync` → `_sync_console_control_bar` →
`run_console_presentation_sync` → `run_console_config_sync` → native config
operation → initialization wait. Unlike FULL's awaited preparation, this direct
fallback can still enter native work on the event loop. The samples establish
that remaining path, not its exact contribution to the 731 ms input result.
The control's 4.34-second attachment outlier had 235/273 samples in event-loop
poll and 10 inclusive CSS samples, supporting substantial waiting rather than
continuous styling. Counts overlap and cover input plus attachment, not CPU
durations. Compact stacks and journeys are in `native-waits-outliers.json`.

`native-waits-safety.json` verifies twelve unchanged storage/config/ownership
modules against the preceding PR head, including the Windows native filesystem
layer. Fourteen exact completed disposable profile/test roots were removed only
after validating every absolute target, parent, prefix and absence of reparse
entries, and confirming no owned process remained. The cleanup receipt records
the exact targets. An earlier setup-refused fixture with no retained exact root
was not inferred or deleted.

## Reproduction

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_native_wait_profiles.py
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_native_waits.py
```

Discovery separately adds `--native-wait-observe` to the existing `probe.py`
ready/resume-only/attach-poll/clock-safe journey. `native-waits-diagnostics.json`
retains scalar events/results; full sampled stacks and instrumented packets stay
local. Do not substitute diagnostic timings for the plain comparison.

ADR required: no. ADR path: N/A. This is a bounded display-demand correction
within existing ADR-126/230 boundaries, without storage/service lifetime changes.
