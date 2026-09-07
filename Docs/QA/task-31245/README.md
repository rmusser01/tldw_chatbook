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
Fresh post-fix native and full-owner UI results are still pending. Windows and
three-participant first-use verification remain external, unwaived gaps.
