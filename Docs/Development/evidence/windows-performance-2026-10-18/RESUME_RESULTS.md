# Windows Console resume: attention and workspace reads

TASK-35592 / TASK-35593; continuation of PR #3166.

## Workload and baseline

Run `probe.py --ready --resume-only --tag resume-before` using the repository
Python on the native Windows user account. The probe creates a fresh private
profile, blocks provider/network requests, guards the real profile, and mounts
the actual application at 170×48. After the initial Console settles, it performs
six Home → Console transitions, injecting one composer key on each return and
sampling a 10 ms event-loop heartbeat. No pytest process runs beside the probe.

The baseline is commit `efb71994b80efc0ba578979c3f4d7b91f456eeaa` plus the diagnostic
`--resume-only` option, before either product change. These measurements include
other resume/rendering work; they are not isolated database timings. The
separate sampler inspects only the main thread. Overlapping stack counts are
evidence of where the loop spent time, not additive elapsed-time budgets.

Input latency ends when the focused composer's draft changes; it is not a
pixel-presentation or completed Send/Stop-action measurement. Textual still
awaits the existing attach callback on the Console screen's message pump.
Preparing the registry read frees the event loop and other pumps, while screen
actions and repaint work may still wait for attach completion. This preexisting
FULL-sync/callback boundary remains a separate profiling candidate.

Baseline median return-to-input latency was 589.18 ms (worst 984.02 ms).
Median per-transition worst heartbeat overshoot was 346.06 ms (worst 520.46 ms).
During the six transitions, the sampler captured 51 main-thread stacks containing
`list_console_unseen_marks` and 29 containing `get_active_workspace`.

## Boundaries

The optimization prepares original stock reads through existing finite worker
ownership. It does not remove native storage admission or change permissions,
receipt acknowledgement, workspace mutation or authoritative FULL sync.
Custom callbacks retain their original synchronous route. The public synchronous
attention recompute API remains available; UI notifications and projections stay
on the loop. Receipt-outcome classification for hidden receipts remains on its
existing route and is outside this measured empty-profile optimization.

Workspace preparation supplies one result to the immediately resumed stack.
The original reconciler still applies session changes at its original point,
before subsequent resume actions; the result is discarded afterward. No reusable
workspace cache or display proof authorizes reconciliation.

An already-answerable partial resume retry keeps its synchronous read. Introducing
a new await there would open a gap between approval restoration and workspace
repair; ordinary visits prepare before runtime answerability instead.

ADR required: no. Existing ADR-126/230/231 finite-lifetime and authority boundaries
apply; no storage format, service boundary or permission contract changes.

## Verification and results

Native regression controls hold the original query with `sys.monitoring`, without
replacing its storage callback. They check finite read ownership/retirement,
loop progress, cancellation, source/session/visit drift and original custom
callback affinity. The workspace controls also exercise full runtime disposal,
native operation/lease retirement, a new visit during a held read, and the
already-answerable retry. Attention controls cover coalescing, exact acknowledgement,
source/authority replacement, reentrant force projection and disposal before
worker issuance.

New helper/test files pass Ruff lint and formatting. The touched existing files
retain the same 33 Ruff findings as pre-change HEAD (same file/code/message counts;
no additions). `git diff --check` passes. No full-suite sweep was requested or run.

The initial combined run encountered two stale transcript harness assertions,
then timed out waiting for a render-entry event after its producer had already
failed. Both assertions reproduced on pre-change `efb71994`; the fixture now
provides the existing store/rate-chip seams, and the event wait also observes
producer failure. All three focused checks then passed. A subsequent combined
run completed with 139 passes and five failures: four tests selected a new
private config without rebinding the real config source, and one badge assertion
ignored existing reserved padding. All five reproduced against the same
pre-change sources (`attention_baseline_tests.py`). Storage rejection was retained.

The two widget source fixtures and badge expectation then passed (3 passed).
Combined evidence is **142 selected checks passing across runs**, including the
17 new native attention/workspace controls; it is not a single all-green sweep.
The two unchanged full-app navigation tests remain baseline-proven
`raw_source_selection_changed` setup failures. An attempted private-profile
fixture migration reached a later three-second first-provider-chunk timeout,
before navigation. That partial migration is excluded from this change; it does
not justify weakening storage rejection or claiming those journeys passed.
A direct private-profile baseline replay of the first migrated case hit its
180-second test deadline; that inconclusive replay is not regression evidence.

### Native timing results

| Six-transition run | Median return-to-input | Worst return-to-input | Median worst loop overshoot | Worst loop overshoot |
| --- | ---: | ---: | ---: | ---: |
| Initial baseline | 589.18 ms | 984.02 ms | 346.06 ms | 520.46 ms |
| Initial updated | 769.99 ms | 1091.84 ms | 406.77 ms | 598.65 ms |
| Paired original control | 793.21 ms | 989.91 ms | 459.86 ms | 564.81 ms |
| Paired updated repeat | 743.48 ms | 1060.74 ms | 441.14 ms | 600.20 ms |

The first after-run was slower overall despite removing both target reads from
the main thread. To check host variation, repeat the original and updated code
back-to-back with no concurrent tests:

```powershell
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --resume-baseline --tag resume-control
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --tag resume-verified
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_resume.py
```

The baseline option loads the eight changed existing modules from the pinned
pre-change commit using `attention_baseline_tests.py`; it does not swap working
files or replace native reads. The updated helper is unused by that control.
Import time is excluded from transition measurements. Native fixture ownership,
real-profile guards and network guards are identical in both runs.

Paired median input latency improved **6.3%** and median worst-loop overshoot
improved **4.1%**, but both worst-case values worsened. Six transitions per run
and substantial between-run variation do not establish a general or tail-latency
speedup. The narrower result is reproducible: main-thread attention-read samples
fell **51 → 0** initially and **79 → 0** in the pair; workspace-read samples fell
**29 → 0** initially and **36 → 0** in the pair. All six drafts were retained and
attach completed in each instrumented after/control run. Every run recorded zero
network attempts and zero real-profile refusals. Full scalar results and sampled
hotspots are in `resume-comparison.json`; each run also has its compact JSON report.
Raw stack files remain local diagnostic outputs.
The four explicitly recorded disposable profile directories were removed after
their probe processes exited; scalar reports and stack samples remain available.

### Remaining measured work

The updated repeat still sampled `_sync_native_console_chat_ui` 81 times,
presentation/config synchronization 73 times, `_sync_console_rail_and_controls`
70 times and config raw-participant scope 65 times (overlapping stacks).
Authoritative FULL synchronization and checked config presentation are now the
dominant resume stalls. The screen-message-pump boundary described above and
large histories need separate profiling before further scheduling/cache changes.
This change does not claim that Console resume is fast overall.

The subsequent FULL/config preparation change and its native controls are
documented in [FULL_RESULTS.md](FULL_RESULTS.md).
