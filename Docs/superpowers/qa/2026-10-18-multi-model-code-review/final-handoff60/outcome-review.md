# Independent outcome review — one final qualification

**Outcome: failed; AC24 remains open.** The one approved run returned code 2 with
`live_report_incomplete_V1`. R1 and R2 sealed and were presented; V1 accepted four
dispositions but became stuck before completion. The 600-second wall deadline is
the runner's terminal stop, not the earliest demonstrated failure boundary.

Evidence root: `/private/tmp/task35582-final-handoff60/attempt-r72ljq_g`.
Source head: `ffe3cfdf5a2c81c103f973f80fb957085ecf3b6c`. Saved CI evidence records
successful completed normal `pull_request` run `40909862501`, verified at
19:31:08.836765Z before execution. This review read task-owned JSON and SQLite in
`mode=ro` with `PRAGMA query_only=ON`; it made no provider requests or source edits.

## First demonstrated failure boundary

The relevant boundary is V1's repeated invalid source ranges after its accepted
checkpoint. `source-snapshot.json` has seven worktree lines and five baseline
lines. `bounded-tool-results.json` preserves five V1 `invalid_source_range`
results: two reads of lines 1..8 before successful recovery, then three reads of
lines 1..10 after accepted dispositions.

The read-only `runs.db` events establish this chronology (2026-10-29 UTC):

| Time | Evidence |
| --- | --- |
| 19:40:32–33 | Two invalid V1 1..8 reads; later successful reads recover. |
| 19:40:50.561761 | `code_review_submit` succeeds: four dispositions accepted at revision 1. |
| 19:40:55.396349 / 55.651248 / 57.956611 | Three consecutive failed V1 source reads, matched to the saved 1..10 refusals. |
| 19:40:57.958108 | V1 error event, sequence 31. |
| 19:40:57.967003 | V1 `agent_run_failed`, terminal status `stuck`, sequence 10000011. |
| 19:40:58.268705 | Primary `wait_agents` already returns V1 `stuck`. |
| 19:41:44.624843 | Final primary model request starts, after inspecting the stuck verifier. |
| 19:41:56.357662 / 56.371312 | Primary model cancellation, then durable run cancellation. |

Three consecutive failures followed immediately by `stuck` fit the saved
`denial_circuit_breaker_limit=3`. **That named circuit-breaker cause is an
inference:** the durable error event deliberately omits its payload and text.
The concrete, retained boundary is repeated invalid source reads followed by V1
failure, about 58 seconds before the final primary cancellation. The earlier R2
out-of-range read and V1's first two refused reads were recovered and are not
themselves evidence of terminal failure.

`provider-error-01.json` identifies a **primary** `HostedChatProtocolError`, reason
`stream_terminal_missing`, original status null. It does not identify a V1 error
or carry an exception timestamp. Its file modification time is 19:42:03.447Z,
after the cancellation events and result refusal. This is consistent with a late
interrupted-stream observation; it does not establish an upstream cause or
precedence over the earlier V1 failure. All HTTP statuses being 200 does not
establish stream completion or usage settlement.

## Proven review progress and missing completion

- R1 (`gpt-6-luna`) and R2 (`deepseek-flash`) have complete capture, terminal
  `done`, revision 3, and nonempty seal receipts. Presented packets preserve their
  originals; every one of the seven V1 prepared requests records
  `originals_presented_before_verifier_request=true`.
- Three fresh children were launched on one work chain, distributed `[2,1]`
  across two primary turns; the native per-primary maximum stayed two.
- Four originals exist: `R1-F1`, `R2-F1`, `R2-F2`, `R2-F3`. V1's accepted
  `v1-batch-1-dispositions` checkpoint has four records, revision 1,
  `pending_unit_ids=[]`, and no pending checks. Two records confirm the empty-input
  regression; another confirms boolean acceptance while calling it pre-existing
  and interpretation-dependent; the fourth refutes silent string acceptance.
  These are actual accepted live submissions, not scripted replacement output.
- No accepted `verifier_complete` is retained. V1 ends `incomplete` / `stuck`,
  without coverage or a seal receipt; `rejected_submissions=[]`, so this is not
  the earlier submission-conflict failure. Final original expansion and unified
  pages refuse `stage_grant_unavailable`. Accepted unit decisions do not prove
  a complete host-expanded original mapping or a readable final unified report.

There is also a semantic evidence limit for any future diagnosis: saved R2-F3
states that the added comment's string-acceptance claim is **false**. V1's saved
string disposition instead refutes that quoted false premise, which does not
refute the reviewer's warning about the misleading comment. Without the final
admitted mapping/report, this cannot count as proof that the original finding or
the false-positive control was correctly adjudicated. No repair was attempted.

## Exact spending and accounting limits

- **44 physical generations:** Qwen 18, OpenAI 9, DeepSeek 10, Haiku 7. All 44
  transport records report HTTP 200. **3 product children** were created.
  Six separate metadata GETs are recorded; these are not generation requests.
- Historical 413/28 plus this 44/3 gives **457 requests / 31 children**. The
  unused 16 requests confer no retry authority; retiring them makes 98 retired
  unused requests in total, including the prior 82. There were no new preflights.
- The automatic continuation ledger records one committed generation, one
  committed child launch and eleven committed model calls. Ten token reservations
  settled at **96,990 actual tokens**; one **17,140-token reservation is uncertain**.
  The chain is `review_required` with `usage_unknown`; no reservation is in the
  ordinary `reserved` state. This ledger covers the automatic continuation,
  not all 44 requests.
- `chat.db` persists one complete assistant usage aggregate totaling 333,894 token
  components and a second, explicitly partial aggregate totaling 96,990. Together
  the recorded components are 110,638 uncached input, 289,145 cache read, 13,240
  cache write and 17,861 output (**430,884 recorded components**). This is **not an
  exact complete run token total or billing claim**: the last call's usage is
  unknown. The 17,140 uncertain reservation is an allowance, not actual usage.

## Cleanup and evidence limits

The manifest reports `native_shutdown_completed=true`, all observed owned fleet
threads stopped, and zero remaining workspace connections, workspace leases and
registered chat connections. It also honestly reports `retirement_uncertain=true`
because its snapshot still held the continuation primary as running. The later
read-only database state shows that primary cancelled at 19:41:56.371312Z and all
five runs terminal. Preserve both observations rather than rewriting the manifest
or claiming a running child leaked. The assistant message remains
`continuation_active` and accounting remains `usage_unknown`, so clean logical
completion and full usage settlement are not established.

Not checked: raw provider response bodies, unavailable terminal-error text, live
services, new tests or repairs. No retry, probe, extra budget or merge is justified
by this failed outcome; no such action was performed by this review.
