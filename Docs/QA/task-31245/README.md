# TASK-31245 qualification evidence

This directory records measured evidence, not blanket release acceptance.

## Standalone Keyword retrieval

[Raw manifest and 300 timings](keyword-scale-4fcbf7721.json) were captured on
source `4fcbf77219fc15c6d3928be63d50dc4596742253`, with an offline synthetic
profile: 10,000 conversations, 250,000 selected user/assistant messages and four
excluded-role/branch canaries. The host, exact query IDs, expectations, raw
timings and corpus/manifest digests are in the JSON. Five discarded warmups and
ten measured calls per query were used; index construction was not timed as
warm retrieval.

Warm nearest-rank P95 was **169.327 ms** against 300 ms; the maximum 5 ms sentinel
interval was **9.936 ms** against 50 ms. All identities matched. Regular file
descriptors were 7 before/after and registered database handles were 0 after
terminal corpus cleanup. Compatible qualification dependencies were isolated;
the shared development environment was not modified.

This is service retrieval evidence, not native input, compositor busy paint,
actual conversation activation, Windows, or moderated first-use evidence.

## Real UI and native status

The initial full-owner UI run at 52×20 passed 30 warm search cases (maximum busy
paint 26.402 ms; scheduling interval 19.189 ms), but failed first activation and
recorded over-budget preparation/activation intervals. A bounded diagnostic
confirmed that the owning modal blocked the canonical exposed-Console predicate
despite exact hydrated identity/transcript/focus. It also sampled synchronous
first-use tokenizer initialization. Those failures prompted the activation and
responsiveness repairs; they are not relabeled as passes.

Native macOS evidence on 4fcb verified Ctrl+K, pointer mode selection, typed
Keyword query, exact selected result and compact detail layout. Enter/F3 delivery
was not established. The dedicated synthetic-data process was subsequently
controller-interrupted and verified stopped; this is not native quit evidence.
Subsequent native macOS pointer activation on source
`f5b943c8f361c05cf53e61a67c3fb440b685cf4e` showed the exact Alpha transcript.
Manual Enter verification was not completed and its request was retracted.
The controller verified and SIGINT-stopped only the owned synthetic app
PID65085; its exit receipt reported no exception. This is partial native
evidence, not keyboard/F3 or native quit acceptance. Windows and
three-participant first-use verification remain external, unwaived gaps.

The full-owner UI attempt5 on that same source passed 30 narrow warm searches
and genuinely returned `OPENED` for `perf-00000`, but its instrument then waited
on Textual's lifetime `Widget.is_mounted` flag, which remains true after removal.
Its commit-wait observer also failed to forward the completion-owner keyword.
Bounded attempts6/7 confirmed exact Console exposure, modal absence from both
stack and app registry, and actual commit acknowledgement. The artifact now
checks those live boundaries and forwards supplied arguments unchanged, retaining
all exact target/transcript/composer/persisted-ID assertions and the 50 ms loop /
100 ms busy-paint limits. These are instrument corrections, not a passing full
matrix or permission to discard the failed attempts.

Performance remains **unqualified**: attempt5 preparation/activation intervals
were 79.873/127.801 ms, attempt6 activation was 131.468 ms, and attempt7 activation
was 87.733 ms. Attempt7's 79.257 ms automatic main-thread generation-2 collection
fell wholly inside that last interval (13,497 objects collected, zero
uncollectable). Its stylesheet rebuild and individual registration timings were
small; the collected objects' owners are not identified. Observer allocations
can shift GC timing, and capped detail records were dropped, so this is neither
a baseline Textual performance comparison nor proof of a retained-object leak.
No production GC/cache policy was changed. On 2026-09-07 the user moved the broader
lifecycle/GC investigation and its latency requalification to the separate
[TASK-31966 follow-up](../../../backlog/tasks/task-31966%20-%20Investigate-and-reduce-Console-activation-GC-pauses.md).
The measurements above remain failures, not retroactive passes. This scope decision
does not waive the remaining native workflow, Windows or participant checks, and
does not by itself establish release acceptance.
