# Confirmatory steady-state three-turn Console latency results

## Result

The separately pre-registered confirmatory verdict is **inconclusive**. All 90
measured conversations completed correctly. The review-disabled candidate
passed both primary non-regression gates, and review-enabled passed the
third-send-to-worker gate, but review-enabled event-loop lag had a one-sided
95% p95-ratio upper bound of `1.3179`, above the required `1.10` ceiling. Its
lower bound was `0.6037`, so the rule establishes neither a regression nor a
pass. This report does not reinterpret that result.

The comparison pins control
`5f720a40417eaa78f33619d5cbc82effc470104b` and candidate
`eb8225a32f88ea43c337aff99804d360384e7668`. It uses the same local llama.cpp
alias, server/runtime contract, fixtures, request settings, mounted Console
path, original statistics implementation, and gates as canonical TASK-20009.
The original benchmark remains separately retained and `inconclusive`.

The original evidence retained an endpoint-reported model alias but no digest
of the GGUF model weights. Consequently this confirmation cannot prove that the
historical and confirmatory model files are byte-identical. It instead fails
closed on the complete server/runtime contract preserved by the original run;
that equivalence preflight passed.

## Pre-registered exclusion

The complete schedule was three warmups, five balanced burn-in blocks, and 30
fresh balanced measured blocks. All 15 burn-in conversations passed the full
product, tool, privacy, cleanup, isolation, and ownership contract. Burn-in was
then excluded before statistical validation and summary generation. No burn-in
latency, ratio, interval, verdict input, or performance claim appears here or
in the machine summary.

## Validity

- The raw file has 222 rows: 111 child starts, three protocol-preflight
  results, and 108 terminal samples. The 108 sample start/result rows form 216
  rows and match the exact predeclared global order and positions.
- Phase counts are exactly 3 warmups, 15 excluded burn-in conversations, and 90
  measured conversations. Each measured arm has 30 samples in complete rotated
  blocks, and all 108 sample IDs and schedule positions are unique.
- Every conversation used the exact `1/3/1` provider-round sequence, one
  `load_tools(local:fs_write)`, one allowed confined `fs_write` to
  `measured/turn-two.txt`, and the terminal tool-result follow-up. All 108
  confined mutations succeeded and prompt loss was zero.
- All 540 provider calls have coherent prompt/completion/total token accounting.
  The 450 measured calls used 563,580 total tokens in each arm. Across warmup,
  burn-in, and measured phases, 2,028,888 total tokens were reported.
- Every terminal ownership record reports zero live benchmark threads, zero
  pending shadow work, closed provider and SQLite resources, and zero writes to
  either pinned source revision.
- The listener identity remained one process with fingerprint
  `3252eb7a4878ddf6f6523dd81327a4d57b8ccd49fc3a42fa9eea48ba4f90c45c`
  throughout acquisition and still matched immediately after verification.
- Successful sample/profile roots and detached target worktrees were cleaned;
  authenticated empty tombstones remain for all 111 child roots and both target
  roots. No campaign worktree remains registered and both campaign locks are
  absent after terminal state.
- The raw, manifest, summary, and attempt ledger produced zero privacy-scan
  findings: no absolute workstation path, credential/header/environment field,
  prompt/response/tool-result body, generated-file content, or personal
  workspace material is retained.
- `attempt-0001` is the first and only attempt. Its state is
  `complete_pending_review`, its raw SHA-256 is
  `2cdda7f369979fb1ac65f4f668bfd8ad4a28b4d2502f081eb767dc766d944a8c`,
  and no retry or sample replacement occurred.

## Measurements

Values are `median / nearest-rank p95` across the 30 measured samples in each
arm. Provider-call duration and whole-conversation wall time are descriptive
only; they are not application-owned improvement evidence.

| Metric | Control | Disabled | Enabled |
| --- | ---: | ---: | ---: |
| Third send → worker | 587.70 / 779.44 ms | 171.42 / 288.08 ms | 189.53 / 327.94 ms |
| Per-sample event-loop-lag p95 | 14.93 / 24.09 ms | 9.52 / 14.10 ms | 16.14 / 22.78 ms |
| Assistant durable → turn release | 0.142 / 0.234 ms | 0.049 / 0.069 ms | 0.280 / 0.386 ms |
| Terminal provider → third provider | 2.068 / 2.431 s | 1.490 / 1.837 s | 1.720 / 2.086 s |
| Sum of provider-call durations | 6.050 / 7.171 s | 6.041 / 7.170 s | 5.996 / 6.619 s |
| Three-turn conversation wall time | 27.776 / 28.541 s | 26.495 / 26.814 s | 27.117 / 27.425 s |

## Pre-registered gates

- **Disabled third-send-to-worker: pass.** The paired one-sided 95% p95-ratio
  upper bound is `0.4254`, below `1.10`.
- **Disabled event-loop lag: pass.** The upper bound is `0.8318`.
- **Enabled third-send-to-worker: pass.** The upper bound is `0.5171`.
- **Enabled event-loop lag: inconclusive.** The lower and upper bounds are
  `0.6037` and `1.3179`; the interval crosses the `1.10` ceiling.

The separately pre-registered application-owned critical-path intervals support
only these narrower improvement claims:

- Disabled assistant-durable-to-release improved; its upper p95-ratio bound is
  `0.3652`.
- Terminal-provider-to-third-provider improved for disabled (`0.7905` upper
  bound) and enabled (`0.9710` upper bound).
- No enabled assistant-durable-to-release improvement claim passed.

## Preparation recomputation and pending independent review

A preparation-side recomputation digest-checked and isolated the original
TASK-20009 runner, then invoked its `validate_sample`, `validate_run`, and
`build_summary` functions on exactly three warmups plus 90 measured rows. The
generated machine summary was byte-identical to the retained summary after its
declared excluded-count and burn-in-contract fields were added. Replacing every
burn-in latency and heartbeat value with an extreme sentinel left the complete
measured summary unchanged, demonstrating the exclusion boundary.

That recomputation also verified the complete raw schedule and order, 3/15/90
split, tool and provider-round contracts, token arithmetic, review events,
revision imports, ownership, listener identity, cleanup, original-evidence
hashes, protocol equivalence, lineage/raw-hash binding, and privacy. A separate
artifact-bound reviewer must still inspect these exact five artifacts and bind
its decision to their canonical digest before registration or publication.
