# Real-provider three-turn Console latency results

## Result

The pre-registered overall verdict is **inconclusive**. All 90 measured
conversations completed correctly, but the paired one-sided confidence bounds
do not establish every candidate non-regression gate at the required 10%
ceiling. This report therefore makes no overall performance pass or regression
claim.

The evidence compares pinned control
`5f720a40417eaa78f33619d5cbc82effc470104b` with candidate
`eb8225a32f88ea43c337aff99804d360384e7668` through the same local llama.cpp
model and mounted Console path. It includes three untimed warmups and 30
balanced measured conversations per arm.

## Validity

- 93/93 conversations completed three turns; the measured set is 30 complete
  rotation blocks and 90 unique samples.
- Every conversation used the exact `1/3/1` provider-round sequence, one
  `load_tools`, one confined `fs_write`, and one terminal tool-result follow-up.
- The third prompt was requested after the common turn-two terminal-provider
  completion and before turn-two release in every sample. Prompt loss was zero.
- All 93 confined mutations succeeded. Disabled samples recorded no Change
  Review work; control and enabled samples recorded their exact required review
  boundaries.
- All 465 provider calls have coherent prompt/completion/total token accounting;
  the 450 measured calls used 564,030 control, 569,080 disabled, and 569,080
  enabled total tokens.
- Every terminal ownership record reports zero live benchmark threads, zero
  pending shadow work, closed provider/SQLite resources, and zero target-source
  writes.
- The retained raw/manifest/summary evidence contains no absolute workstation
  path, credential/header/environment field, prompt/response/tool-result body,
  or file content. Successful child profiles and the detached control worktree
  were removed.

## Measurements

Values are `median / nearest-rank p95` across 30 measured samples per arm.

| Metric | Control | Disabled | Enabled |
| --- | ---: | ---: | ---: |
| Third send → worker | 565.98 / 831.18 ms | 169.71 / 309.07 ms | 270.26 / 456.96 ms |
| Per-sample event-loop-lag p95 | 14.21 / 33.49 ms | 7.40 / 22.93 ms | 13.64 / 37.09 ms |
| Assistant durable → turn release | 0.119 / 0.190 ms | 0.047 / 0.083 ms | 0.237 / 0.330 ms |
| Terminal provider → third provider | 2.307 / 2.697 s | 1.734 / 2.187 s | 1.945 / 2.360 s |
| Sum of provider-call durations | 5.840 / 10.177 s | 5.909 / 9.496 s | 5.918 / 8.490 s |
| Three-turn conversation wall time | 27.986 / 31.357 s | 27.137 / 29.199 s | 27.476 / 30.003 s |

Provider-call duration and whole-conversation wall time are descriptive only;
they are not evidence of application-owned improvement.

## Pre-registered gates

- **Disabled third-send-to-worker: pass.** The paired one-sided 95% p95-ratio
  upper bound is `0.4484`, below the `1.10` ceiling.
- **Disabled event-loop lag: inconclusive.** The upper bound is `4.3596`.
- **Enabled third-send-to-worker: inconclusive.** The upper bound is `2.3674`.
- **Enabled event-loop lag: inconclusive.** The upper bound is `3.2944`.

The wide bounds reflect high-tail scheduler variation. Descriptive p95s are
lower for disabled third-send and lag and for enabled third-send, but those
point estimates do not replace the paired confidence gates.

The separately pre-registered application intervals support narrower claims:

- Disabled assistant-durable-to-release improved; its upper p95-ratio bound is
  `0.8192`.
- Terminal-provider-to-third-provider improved for disabled (`0.8361` upper
  bound) and enabled (`0.9980` upper bound).
- No enabled assistant-durable-to-release improvement claim passed.

## Independent review

A separate standard-library-only recomputation read the raw JSONL without
importing the benchmark module. It reproduced every median, p95, 10,000-resample
paired confidence bound, improvement claim, and the final `inconclusive`
verdict exactly. It also independently verified sample cardinality, rotation,
queue and terminal ordering, tool calls, token arithmetic, mutations, review
events, ownership, candidate revision, profile cleanup, and privacy.
