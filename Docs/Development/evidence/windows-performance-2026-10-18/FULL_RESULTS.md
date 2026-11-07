# Windows Console FULL/config presentation

TASK-35594; continuation of PR #3166.

## Cause and change

An expired display projection starts its checked worker refresh, but the first
synchronous presentation call deliberately retains the original native config
fallback. During FULL synchronization this put config admission and filesystem
checks on the UI thread. Original return-site observations confirmed that expiry
route; pricing validity was not the cause. The direct-call contract is retained.

FULL now awaits the existing stock `ConsoleReadinessConfigProjection.warm()`
before synchronous rail/control presentation. The await is outside native config
operations and the workspace build scope. Existing source/session checks and a
post-await visit/maintenance check gate rendering. Refusal retains whole-FULL
replay and cannot declare attach complete. Cancellation retains the actual read
and leases until physical retirement.

No storage admission, permissions, live action checks or authoritative FULL
effects are removed. Custom/direct presentation keeps its original fallback;
the display lifetime is unchanged. ADR required: no; existing ADR-126/230 apply.
General narrow polling remains Proposed (ADR-231).

## Reproduction

Run sequentially from the repository root with the native Windows user account:

```powershell
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --full-baseline --tag full-control2
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --tag full-verified
& .venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_full.py
```

The baseline loader restores only `config_sync.py` and `chat_screen.py` from
`2a9cf92464982cb0d53e6452ade220e4e4884976` in memory. Other earlier performance
fixes remain present in both versions. The original `full-before` run was taken
before the product edit. Each run uses a fresh private profile, actual application
at 170×48, six Home → Console returns, driver-delivered composer keys, a 10 ms
heartbeat and a separate main-thread sampler. Provider traffic and real-profile
access are guarded; no pytest process runs alongside timing probes.

Return-to-input ends at the draft update, not paint or completed Send/Stop.
Return-to-attach records `_console_attach_reconciled`; this metric was added
after `full-before`, so only the later runs have it. The final paired runs sample
it in the existing 10 ms heartbeat. The existing Screen message
pump still awaits its attach callback. Sample counts overlap and are not additive
time budgets. Diagnostic `--config-observe` results are excluded from timings.

## Verification

The new native expired-FULL regression failed on original product code with one
main-thread config scope and 854 native opens. With preparation it passes with
zero scopes and zero opens. Seven new cases cover that regression, loop progress
while the original native read is held, source/session/visit/maintenance drift,
and cancellation through physical lease retirement. Native read callbacks are
observed rather than replaced. Existing direct-expiry and custom fallback
controls remain selected in the checked-display test run.

Targeted evidence totals **107 passing checks across focused runs**: 7 new FULL
controls, 61 existing checked-display/readiness controls, 33 attach/config-lifetime
controls, and 6 repaired fixture cases. The initial existing runs had six
pre-assertion fixture failures: one missing mode-bar rate field, and five missing
attachment/runtime context fields. Each reproduced at the pinned baseline
(the five worker cases loaded baseline code inside their child processes).
Fixtures now supply those existing fields; original native admission, pause,
replay and retirement assertions are unchanged. All six repaired cases pass.
This is not a single full-suite run; no full sweep was requested or run.

New helper/tests and evidence scripts pass targeted Ruff lint/format checks.
`chat_screen.py` retains the same 22 pre-existing Ruff findings by code/message;
the changed helper and probe have none. Independent review found no change to
direct fallback or storage authority.

## Incomplete timing runs

The first updated run (`full-after`) and two diagnostic repeats timed out on the
first return-to-input check (10 seconds), before any completed cycles. The pinned
original-code control (`full-control`) reproduced the timeout. Async stacks in
both versions place FULL in character metadata preparation, before the new
helper; the updated version's config projection was already settled. Original
worker stacks show reads waiting in native `storage_admission.initializing`,
with another thread inside its Windows file-identity checks. No storage checks
were changed to make the probe pass.

Those runs used a separate 5 ms attach-observation task. The final paired probes
observe attach in the existing heartbeat instead. The original-code repeat
(`full-control2`) then completed six cycles. This does not prove the extra polling
was the sole cause, or that the broader native-admission stall is fixed. Failed
runs are excluded from latency medians and explicitly listed in
`full-comparison.json`; they are not zero-latency results.
`full-timeout-summary.json` retains the relevant await chains, native stack
excerpts and SHA-256 identities of the original local diagnostic reports.

## Paired native results

Both final runs exited successfully with six completed returns, all drafts
retained and attach reconciled, zero network attempts and zero real-profile
refusals. These are headless diagnostics on Python 3.12.10/Textual 8.2.8,
Windows 10 build 19045, not universal interaction guarantees.

| Measurement | Original (`full-control2`) | Updated (`full-verified`) |
| --- | ---: | ---: |
| Median return-to-input | 837.66 ms | 904.69 ms |
| Worst return-to-input | 1036.74 ms | 1696.51 ms |
| Median return-to-attach | 846.15 ms | 908.37 ms |
| Worst return-to-attach | 1829.79 ms | 1699.59 ms |
| Median per-return worst heartbeat overshoot | 435.53 ms | 243.36 ms |
| Worst heartbeat overshoot | 584.53 ms | 332.09 ms |
| Main-thread native presentation/config samples | 69 | 0 |
| Overlapping FULL synchronization samples | 80 | 11 |

The original pre-edit run independently recorded 69 native config samples,
733.04 ms median input arrival and 454.76 ms median worst heartbeat overshoot.
Its lack of the new attach observer makes it secondary to the final paired run.

The targeted UI-thread read is eliminated and median worst heartbeat overshoot
improves **44.1%**. Overall median input arrival worsens **8.0%**, median attach
worsens **7.4%**, and worst input arrival also worsens. This change therefore
demonstrates less main-loop blocking, not a general Console-return speedup.
Awaiting the original read still takes time, and the existing screen-message-pump
attach boundary remains. The baseline-reproduced native admission stall above
is also unresolved.

Remaining main-thread samples include Textual CSS fast-path application (57),
inspector-state construction (12), and provider/config presentation posture reads
(11). These overlap; they are candidates for separate profiling rather than
proof that any safety check should be cached or removed. Large histories and
actual paint/Send/Stop latency are outside this empty-profile workload.

Follow-up: [STORAGE_RESULTS.md](STORAGE_RESULTS.md) removes redundant native
initializer polling and repeated ancestry walks while retaining storage checks.
It also identifies the extra sub-resolution polling as a measurement problem on
Python 3.12 Windows: the pinned original code completes all six returns when the
probe polls above the event-loop clock resolution. The failed instrumented runs
above therefore do not establish an unresolved product deadlock. Their original
observations remain here; the follow-up contains the corrected paired profile.
