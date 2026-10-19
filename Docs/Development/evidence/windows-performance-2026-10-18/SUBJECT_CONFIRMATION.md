# Config derivation and dynamic CSS: current-code confirmation

TASK-35613 follow-up on `3742efae9428dc10f879606b519215913263ac4e`.
Python 3.12.10, Textual 8.2.8, native Windows, headless 170×48.

The completed optimization and its earlier measurements were already on the
remote PR when this checkout was synchronized. The local draft versions were
preserved separately. This follow-up verifies the integrated code and adds a
comparison that retains the later Console return-responsiveness change in both
arms; it introduces no further product or safety changes.

## What is preserved

Independent provider/model and readiness derivations use the existing synchronous
scope: one current settings read per build instead of three, nested sharing,
fresh reads after Settings saves, and cleanup on exceptions. Dynamic CSS rejects
a false final pseudo-state before searching ancestors, while positive cases and
invalidation flags remain upstream. Dynamic results are not cached.

Nineteen focused native checks passed in 91.70 seconds: both actual config
derivations, dynamic CSS state/style/invalidation, static selector filtering,
candidate reuse, held-attachment input/answerability and recorded-cost compatibility.
No full-suite sweep ran. Seven storage/config safety modules are byte-identical
after line-ending normalization to the pinned pre-optimization code; hashes are
in `subject-confirmation-safety.json`. No native checks or permissions were relaxed.

## Fresh focused benchmark

Four alternating original/current rounds use the same 526 mounted widgets, with
one cold update and five warm updates per arm. The original CSS module comes
from `f152dfc4a6c4fbf2fcd277543757a46a0d2874e2`.

| Metric | Original | Updated |
| --- | ---: | ---: |
| Median of warm round medians | 144.37 ms | 117.70 ms |
| Warm round medians | 145.26 / 166.32 / 143.48 / 142.03 ms | 111.38 / 104.32 / 124.03 / 126.18 ms |
| Cold updates | 177.07 / 175.17 / 264.06 / 276.53 ms | 148.41 / 163.78 / 138.73 / 155.62 ms |
| Base/component style differences | — | 0 |

Warm CSS application improves **18.5%** in this repeated focused workload. Both
arms still perform the full Textual update; no rendering work or storage check
is replaced with a constant. Benchmark and plain-return runs are separate.

## Fresh serial return comparison

`--confirm` runs benchmark, original, updated, updated, original in separate fresh
private profiles. `subject_scoped_baseline.py` restores only the two original
derivation definitions into current ChatScreen, plus the original CSS module.
It asserts that the remaining ChatScreen AST is identical to current code.
The later return worker, composer gates and recorded-spend correction remain in
both arms. Loading the entire old ChatScreen would confound this comparison.

| Arm | Median input | Worst input | Median attach | Median largest heartbeat overshoot |
| --- | ---: | ---: | ---: | ---: |
| Original 1 | 165.69 ms | 205.74 ms | 1082.58 ms | 115.38 ms |
| Updated 1 | 168.12 ms | 424.86 ms | 1072.54 ms | 102.79 ms |
| Updated 2 | 233.70 ms | 505.58 ms | 1831.99 ms | 167.22 ms |
| Original 2 | 195.87 ms | 260.87 ms | 1363.26 ms | 135.39 ms |

This repeat **does not verify an overall latency improvement**. Median input is
1.5% slower in the first pair (effectively flat) and 19.3% slower in the reverse
pair. Worst input is slower in both pairs. These small samples do not establish
the cause of the variation, and the component saving must not be presented as
a general input or attachment saving. All 24 drafts were retained and screen
reattachments completed. All five subprocesses exited zero; network attempts and real-profile
refusals were zero. No tests ran alongside these timing processes. The five exact
profiling homes and one test-fixture directory were removed after their processes
exited; their scalar reports, receipts and local raw samples remain.

Exact scalar reports, source-hash/exit receipts and `subject-confirmation.json`
retain all results. Raw sampled stacks and the full instrumented benchmark payload
remain local diagnostic outputs. The summary refuses missing/failed benchmark
receipts and unfinished reports; a negative check confirmed a failed exit cannot
overwrite the completed summary. Sampling counts overlap and are not CPU budgets.

Independent read-only review found no product blocker. Its minor benchmark-exit
receipt finding was corrected and checked. Focused helper/test lint and formatting
pass; ChatScreen's prior lint/format debt is unchanged. The historical abnormal
diagnostic exit remains unclassified; this follow-up makes no crash-fix claim.

Reproduce serially with the native user account:

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/run_subject_profiles.py --confirm
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_subject.py --confirm
```

ADR required: no. Diagnostic isolation and verification of the existing ADR-126/
230/231 display boundaries; no storage, authority, source or cache-lifetime redesign.
