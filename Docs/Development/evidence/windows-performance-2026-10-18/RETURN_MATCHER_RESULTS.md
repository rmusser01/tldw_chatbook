# Console return diagnosis and subject-only CSS matching — TASK-35644

Baseline: `48339e4d6d2fa82b561e6fa1651f8e82928177b2`.
Python 3.12.10, Textual 8.2.8, native Windows 10, headless 170×48.

## What the slower comparison establishes

The preceding SUBJECT_CONFIRMATION comparison did not establish a general return
improvement: median input worsened 1.5% in one pair and 19.3% in the reverse pair,
while its focused CSS workload improved. Its sampled stacks combine time before
input and after input until attachment; they cannot place individual samples on
the critical input path or prove the reason for the historical difference.

New, separate diagnostic runs retain the real callbacks and their identities.
`return_observer.py` uses local Python monitoring events, retaining function names,
scalar times and task/thread integers, never frames, locals, callback bodies or
native handles. These instrumented runs are discovery evidence, not comparison
timings. They are excluded from the plain ABBA experiment.

- The first run records six original `Screen._on_screen_resume` spans of
  99.5–136.9 ms; their full `Stylesheet.update_nodes` calls take 97.8–134.8 ms.
  Synchronous presentation/config spans are substantially smaller in this run.
- The asynchronous phase run records configuration refresh waits of
  388–1,332 ms, followed by a separate original workspace read. One unsuccessful
  FULL attempt returns at 2.3 ms and the deferred attempt begins at 375.6 ms.
  Required refusal/replay and native waits account for delays that a faster
  selector alone cannot remove.
- A further worker/timer discovery observes original config read bodies taking
  238–743 ms on settled visits and workspace bodies taking 205–325 ms. A config
  read started while leaving Console and finished during return, taking 5.71 s.
  Thus a fixed three-second initial pause and half-second away interval do not
  prove that background work has retired. Up to 15 original Windows timer workers
  overlap in this instrumented run; this does not establish executor starvation.

These observations identify competing native work, delayed replay and styling
as contributors to fresh slow returns. They do **not** retrospectively prove
that the previous two optimizations caused the earlier regression, nor identify
all time inside admission as disk I/O. No TTL, warmup, native guard or timer is
changed to improve the comparison.

## Bounded change

The saved focused matcher breakdown makes 4,233 rule checks. Common rules such as
`Widget` (405 calls), `Static` (157), `Button` (182) and Button style compounds
still use Textual's ancestor-search stack despite describing only the subject.
The regression observes an ancestor check before the actual Button for all three
subject-only cases.

Single selectors beginning with DESCENDENT now call the original `Selector.check`
on the actual subject once. A compound is direct only when its first combinator
is DESCENDENT, every subsequent combinator is SAME, and every preceding advance
is zero. These conditions prove every successful check must be on that subject.
Ancestry and uncertain shapes retain the original matcher. Live pseudo-state is
evaluated anew; candidate rules, invalidation flags, style resolution, component
processing, source order and DOM query matching retain their existing owners.

Independent review found the first-combinator edge before publication. Four
CHILD/SAME single/compound controls reproduced false matches before the added
DESCENDENT guard and pass after it. The reviewer confirmed the blocker resolved.
No dynamic match cache or refresh suppression is added.

All eleven storage/config/finite-read ownership modules in
`return-matcher-safety.json` are unchanged from the pinned baseline after newline
normalization. No admission, freshness, source/view gate, cancellation, physical
retirement or Send authorization is weakened.

## Fresh measurements

The benchmark uses 513 mounted widgets in this run. Each arm performs a cold
update and five warm updates in each of four alternating rounds.

| Metric | Original | Updated |
| --- | ---: | ---: |
| Median of warm round medians | 94.32 ms | 91.84 ms |
| Warm round medians | 114.95 / 89.95 / 89.67 / 98.68 ms | 110.14 / 93.56 / 87.36 / 90.13 ms |
| Base/component style differences | — | 0 |

The observed median reduction is **2.6%**, with overlapping timings and one
round slower. This is a small component reduction, not evidence of a uniform
latency improvement. The deterministic regression establishes the removed
ancestor checks; the overall benchmark still includes all original styling.

The serial plain comparison retains six Home/Console returns per fresh process:

| Arm | Median input | Worst input | Median attach | Median largest heartbeat overshoot |
| --- | ---: | ---: | ---: | ---: |
| Original 1 | 189.77 ms | 270.55 ms | 1118.95 ms | 123.88 ms |
| Updated 1 | 170.35 ms | 223.98 ms | 1308.26 ms | 153.16 ms |
| Updated 2 | 166.20 ms | 201.78 ms | 1128.20 ms | 96.35 ms |
| Original 2 | 167.72 ms | 179.31 ms | 1237.48 ms | 99.81 ms |

Input median improves 10.2% in the first pair and only 0.9% in reverse order,
effectively flat in that pair. Worst input improves in the first pair and
worsens in reverse order. Attachment is 16.9% slower in the first pair and 8.8%
faster in reverse order. **No consistent end-to-end or tail improvement is
established.** All 24 drafts survive and all attachments complete. All five
processes exit zero; every run has zero network attempts and profile refusals.
No tests or instrumented discovery ran alongside the timing processes.

`return-matcher-comparison.json` and exact `return-matcher-*-exit.json` receipts
retain every result and the tested source hashes. The warm CSS benchmark and
input observer measure headless application work, not real-terminal paint time,
long-conversation performance or complete Send/Stop journeys. The still-variable
native preparation and replay waits remain the larger attachment target; this
change does not claim to fix them.

## Final integrated-code verification

Before publication, the remote PR advanced with TASK-35615's four-line
acknowledgement correction and its existing evidence. It was preserved by
rebasing onto `efdd8b2b21`; the reviewed matcher commit became `5c47d09273`.
The correction avoids an acknowledgement-only FULL when the original fresh
visit/read finds no unread token. Marked visits still perform FULL, paint,
original navigation/profile/service/session checks and token compare-and-clear.
Ordinary attachment still performs FULL. Twenty-six combined native rechecks
pass in 38.17 seconds; these overlap the earlier 48 and are not an additional
26 distinct cases.

A fresh, separate serial ABBA compares **both** fixes against their exact
`48339e4d6d` implementations. Only CSS and `UI/Console_Modules/session.py` are
restored in controls, explicitly recorded in `baseline_scope`; the rest of the
combined application is retained. This experiment cannot isolate either fix's
standalone time saving. It does not replace or blend the matcher-only results.

| Arm | Median input | Worst input | Median attach | Median largest heartbeat overshoot |
| --- | ---: | ---: | ---: | ---: |
| Integrated original 1 | 181.42 ms | 345.52 ms | 1250.77 ms | 120.28 ms |
| Integrated updated 1 | 226.99 ms | 517.00 ms | 1510.01 ms | 155.83 ms |
| Integrated updated 2 | 150.35 ms | 171.01 ms | 1140.14 ms | 105.30 ms |
| Integrated original 2 | 170.57 ms | 209.54 ms | 1127.89 ms | 94.23 ms |

Across 12 returns per arm, input medians are **170.57/170.14 ms**, effectively
flat. Attachment medians are **1127.89/1379.36 ms**, **22.3% worse** updated.
Worst attachment is **2243.36/3579.77 ms**, also worse. The prior attachment
experiment's observed median improvement does not reproduce consistently here.
The two updated runs differ substantially; retaining both is necessary.

The 3.58-second updated return accepts its key at 491 ms, including only 30 ms
after the driver posts it; 171/208 stage samples are in the event-loop poll.
The other slow-input return accepts its key at 517 ms and attaches at 2017 ms,
with 93/110 samples in the poll. CSS appears in 16/10 inclusive samples and
config posture in 1/1. These counts overlap, cover input plus attachment, and
cannot identify the exact awaited callback or turn into elapsed CPU budgets.
`return-matcher-integrated-outliers.json` retains the original leaf stacks and
journey scalars. No historical-regression or tail fix is claimed.

All 24 integrated drafts are retained and attachments complete. All four exits
are zero, with zero network attempts/profile refusals and zero periodic traceback
dumps. No tests or instrumented discovery overlap this experiment. Receipts hash
the final checkout's CSS, session and ChatScreen sources; controls additionally
identify the pinned substituted modules. Full run results remain in
`return-matcher-integrated-comparison.json` and four exact exit receipts.

## Verification

The final native targeted run passes **48 checks in 135.07 seconds**, covering
CSS state/style/component/invalidation, original FULL config preparation,
source/session/view/maintenance changes, held reconciliation input and exact
native retirement after cancellation. The real-app cross-screen style tour also
passed in the earlier broader run. No full suite ran.

That broader run initially had 45 passes and two teardown failures. Both CSS
assertions passed; the private-profile child refused config retirement because
native app workers remained uncertain. The exact baseline, audited in parent
and private-profile children, reproduces both failures. Those two tests now use
real `TldwCli` construction, matching their passing style-tour sibling, instead
of the app factory's mocked services. Every style/census assertion and retirement
gate remains. Both corrected fixtures pass in the final batch. This does not
claim to diagnose or repair the factory's broader lifecycle behavior.

The first baseline-runner attempt used the legacy default Temp and was refused
during collection. Reusing the established user-owned Windows fixture wrapper
provides the valid baseline result above; no admission check was relaxed.
`return-matcher-verification.json` retains this history and the red/green counts.

Independent review confirmed the first-combinator repair, unchanged storage
boundaries, corrected fixtures and baseline controls, with no remaining code
findings. Its measurement verdict was pending the final serial comparison.

## Reproduction

Run tests separately from timing processes. Both plain comparison arms retain
the same application and the existing warmup, posted-key/attachment observers,
31.25 ms clock-safe floor, real-profile guard and network guard. The original arm
loads only the pinned CSS module in memory, including its preceding pseudo-state
optimization; it preserves all later Console return and storage code.

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_subject_profiles.py --return-matcher
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_subject.py --return-matcher
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_subject_profiles.py --return-matcher --integrated
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_subject.py --return-matcher --integrated
```

The separate CSS benchmark alternates four pairs on the same mounted screen,
compares base/component rules and clears each arm's candidate cache. Exact exit
and source-hash receipts accompany every process. Raw sampled stacks and full
instrumented payloads remain local; compact scalar results are retained here.

ADR required: no. ADR path: N/A. This is a private matching implementation
correction within existing boundaries, with no storage or cache-lifetime change.
