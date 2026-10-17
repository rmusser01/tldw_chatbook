# TASK-34406.11 — finite Console preparation handoffs

User contract: profile the permission wait, implement/benchmark one targeted fix, preserve every permission/freshness check, and merge only with an actual first-send latency improvement.

ADR required: yes, amend existing ADR230 before code. ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md. Reason: a bounded interpreter scheduling choice during existing finite callbacks; no native/permission/storage boundary changes. ADR126's native checks and retirement stay unchanged.

Evidence: qualified original live permission reads73.578/24.741ms and272.145/29.447ms wall/thread CPU. The latter records repeated individual stat/lstat calls waiting up to13ms while other threads run. An original-callback diagnostic with1ms interpreter handoffs during that read preserves313 posture stats,400 content stats,324 realpath lstats and27 mount checks, yielding26.175/16.253ms. Its whole post-save span1523ms versus1527ms does NOT establish release benefit. Local isolated native syscall probes show microsecond operations rather than this off-CPU gap. C monitoring does not expose BEFORE_WITH entry; no proof of a particular native lock wait is claimed.

## Task1 — scoped scheduling and qualification

1. Preserve diagnostic source/output. Write causal tests for a1ms maximum handoff while physical finite callbacks run, original/lower interval restoration, overlap, exceptions, repeated cancellation, submission failure and external interval changes. Expected RED: original callback still sees5ms.
2. Add one private context manager around the existing synchronous callback in Chat/console_preparation_reads.py. Reference-count overlap under its existing short bookkeeping lock; restore the captured interval on last physical exit only if it remains ours. Keep lower existing choices and later different external changes. No result cache, timer, pool, API setting, new work service, permission/native edits or moved checks.
3. Expected GREEN: scheduling tests and original real-native permission/source/retirement controls, plus relevant finite-read tests. Preserve exact original12/14 witnesses,15/19 execution-context checks,12 acquisitions and3 raw file-path admissions. Targeted checks only; no full sweep. Touched lint/format/diff and one fresh whole-branch review.
4. Commit exact sources; unprofiled8 fresh first sends in B/C/C/B/B/C/C/B order against current merged111f8e48, full matched config/27 schemas/traces/HTTP/normal Quit/PID/port/source guards. Register gate before reading data: both post-save and Enter-to-HTTP medians improve and at least3/4 matched pairs improve for each; no failures or changed checks. This small cohort is no population p95 claim. If gate fails, preserve data and keep unmerged.
5. With passing gate, publish reviewed PR, resolve review threads, run required CI and use current merge-queue policy. Rebase if needed, verify exact combined changes and merge only while the measured implementation is preserved.

Review focus: interpreter-wide effect despite finite lexical lifetime; unrelated global interval setters; overlapping callbacks and error/cancellation/submission ordering; source drift and original permission/native retirement; measurement source/guard fidelity and honest scope. No new task implementation outside these criteria.
