# Final handoff qualification: failed, 2026-10-29

**AC24 remains open, TASK-35582 remains In Progress, and PR3286 remains draft.**
One authorized Console qualification ran on
`ffe3cfdf5a2c81c103f973f80fb957085ecf3b6c`. It used **44/60 generation requests
and 3/3 product children**, with native launches `[2,1]`. The result gate exited2
with `live_report_incomplete_V1`. No qualification retry, extra budget request,
production repair or merge followed.

## CI and retained checks

[Normal PR CI](https://github.com/rmusser01/tldw_chatbook/actions/runs/40909862501)
passed on that exact head:54 successful Actions checks and3 skipped. One authorized
rerun of UI Fast Lane25 and its dependents passed. The original failure was a
five-second snapshot-settings save wait; its exact test passed unchanged locally
(1passed/15.01s). No timeout or application code was changed for this rerun.

The runner's30 focused offline controls and independent18 admission/result-gate
checks remain valid; its seven reviewed executable/helper hashes were rechecked
before invocation. R1's161 unique offline regressions, the eight instruction-
deferral regressions and their scoped reviews were reused without broad repeats.
Historical Haiku native tool-result/usage evidence was reused only for its17
unchanged boundaries. No additional live preflight was performed.

## First evidenced failing boundary

Both independent reviewer reports sealed and were presented before the fresh
Haiku verifier launched. R1 contributed one original and R2 three originals.
At19:40:50.561761Z, V1 successfully submitted four dispositions at revision1;
the accepted checkpoint had no pending units. It then made three consecutive
source reads with `start_line:1,end_line:10`. Captured worktree/base files have
seven/five lines. The native source owner correctly refused all three with
`invalid_source_range` at19:40:55.396349Z,55.651248Z and57.956611Z.

V1 emitted an error event and became `stuck` at19:40:57.967003Z. The primary's
next `wait_agents` result already reported that state at19:40:58.268705Z.
The configured denial circuit breaker is3; termination immediately after three
consecutive refusals is consistent with it. The durable error body is omitted,
so the exact named terminal reason is not independently retained. Two earlier
out-of-range reads were followed by the successful disposition submission and
are separate from the final consecutive sequence.

The600-second qualification deadline fired later. It is not the first V1
failure. V1 never submitted completion or obtained a seal; final original
expansion and unified-report reads refused `stage_grant_unavailable`. Four
accepted decisions and an empty pending-unit list do not establish a completed,
readable unified report or an independently verified every-original mapping.
The saved wording also leaves a semantic concern: R2-F3 warned that a QA comment
was false, while V1 refuted the quoted false premise. That does not establish
correct adjudication of the reviewer's warning; no mapping repair is inferred.

## Primary error and accounting limits

The observer retained the later primary exception before projection:
`HostedChatProtocolError`, reason `stream_terminal_missing`, no original HTTP
error status. It came from the hosted stream's `__next__` path and was recorded
after cancellation, consistent with the final primary request's interruption.
The observer does not retain an exception timestamp. All44 physical sends received
HTTP200; that does not establish clean terminal streams or successful execution.
The final stream interruption and deadline/cancellation coincide; this evidence
does not establish a separate server defect or identify who ended that stream.

Automatic-continuation accounting settled10 token reservations for96,990 actual
tokens. Its final17,140-token estimate remained uncertain and the chain paused
`review_required/usage_unknown`. Full token spend is therefore unknown; token
components from other turns are not a complete total. Native permissions,
accounting refusal and sealing checks were preserved.

Native shutdown returned successfully, with zero observed workspace connections,
leases, registered chat connections and live owned fleet threads. The original
manifest still captured the primary as `running` and `retirement_uncertain:true`.
A separate post-return read-only DB observation shows it cancelled at
19:41:56.371312Z and all runs terminal. Both observations are retained; the later
readback does not rewrite the manifest or resolve the uncertain usage reservation.

## Spending, preservation and scope

- New generations:Qwen18, Luna9, DeepSeek10, Haiku7 =44. Metadata GETs:6, separate.
- Cumulative actual use: **457 generation requests /31 product children**.
- This single invocation is consumed. Its unused16 requests retire; total retired
  unused historical requests are98. No new invocation is authorized by that balance.
- [Spending receipt](spending.json), [independent outcome review](outcome-review.md)
  and [external retention receipt](retention.json) accompany this summary.
- Private source packet:
  `/private/tmp/task35582-final-handoff60/attempt-r72ljq_g`.

Live model judgments remain distinct from the unchanged scripted duplicate,
false-positive, failed-reviewer, drift, truncation, small-context and overflow
controls. No new broad tests or speculative repairs were performed. The failed
run does not complete AC24. The follow-up automation was deleted to prevent
another run. Existing ADR-281 applies; no architectural decision changed.
