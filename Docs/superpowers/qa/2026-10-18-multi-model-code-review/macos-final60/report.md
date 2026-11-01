# Qualification after macOS CI repair: failed

**AC24 remains open, TASK-35582 In Progress, and PR #3286 draft.** The single
authorized qualification ran on `ac4816ac0350fe98f9b45e775854893cbb313750` from
2026-10-30 02:44:36 to 02:54:44 UTC. It consumed **27 of 60 generation requests
and 2 of 3 fresh children**. The result gate exited 2 with
`live_report_incomplete_V1`. No retry or new live probe followed.

## CI and valid checks

All normal same-source workflows completed before invocation: **54 successful
Actions checks and 3 skipped checks**. This includes
[Tool Recovery](https://github.com/rmusser01/tldw_chatbook/actions/runs/40954875538),
[Eval recovery](https://github.com/rmusser01/tldw_chatbook/actions/runs/40954875660)
and [Derived Artifacts](https://github.com/rmusser01/tldw_chatbook/actions/runs/40954875584).
The earlier green rollup alone was insufficient while Derived Artifacts was
pending; the retained all-workflow receipt confirms it finished successfully.
No CI rerun was requested for this head.

The macOS diagnostic change's ten focused tests and scoped review, V1 recovery's
205 review regressions plus 15 accounting controls, and unchanged runner/provider
checks retain their recorded scope. They were not repeated. Historical Haiku
tool-round-trip evidence was reused; no additional preflight was run.

## Failure boundary and diagnostic limits

Luna and DeepSeek each completed a sealed report, and both originals were
presented. The primary then made report reads, including refusals, without
launching the fresh verifier. The saved continuation contains an earlier pair of
failed reads (`call_e61b4bebe06a28d99513329a80995a69_0` and `_1`). The first
individually timestamped failed-read event is at 02:52:54.231886 UTC, followed by
a successful read at 02:53:47.202795. The durable failure is only
`review_refused`: the provider deliberately omits arguments and projects the
internal refusal reason to that generic label.

This establishes the first observed unsuccessful boundary, **not its exact
internal cause or a proven fatal error**. A wrong model-selected reference/view,
capacity or authority refusal cannot be distinguished from these artifacts.
The later successful read prevents attributing terminal failure to the first
refusal alone.

The configured 600-second wall deadline was reached. At 02:54:43.942346 the
primary recorded `Provider continuation could not be persisted`, followed by
terminal error and a session fence. The evidence does not identify the failing
internal persistence operation or prove whether deadline intervention caused,
exposed or merely coincided with that error. There is no provider-error observer
artifact. All 27 physical responses were HTTP 200 with no recorded transport
exception; those facts do not establish successful continuation.

V1 remained `not_started`, with no attempt, launch claim or Haiku request. The
five originals have no verifier dispositions or complete mapping. Final report
and expansion reads refused `stage_grant_unavailable` after the session fence.
That final fence does not explain the earlier refusals. The result gate correctly
failed; it is not the first live failure.

## What completed and what did not

Both reviewers found the seeded empty-input division-by-zero defect. R2 also
explicitly refuted the embedded false string-input claim. R2's bool and generator
observations describe pre-existing behavior with stated contract uncertainty;
they are not demonstrated introduced regressions. R1 contributed one original,
R2 four. No live verifier adjudicated these claims, grouped the duplicate or
produced unified output. Useful original reports do not complete AC24.

The automatic-work ledger contains zero reservations and no accepted wake. Its
chain remained unstarted with no pause reason. This run did not produce a
`usage_unknown` pause; it also did not exercise accounting through the fresh
verifier continuation. Historical uncertain usage remains historical, unresolved
evidence. Native shutdown completed with no running runs, owned fleet threads,
Workspace leases/connections or registered Chat connections, and no retirement
uncertainty. Source, temporary team and source configuration preservation
receipts passed.

## Spending and preservation

- New generations: Qwen 13, Luna 7, DeepSeek 7, Haiku 0; five metadata GETs separate.
- Actual cumulative use: **484 requests / 33 product children**.
- Native two-child limit remained intact; only `[2]` launched, not completed `[2,1]`.
- The single invocation is consumed. Its 33 unused requests and one unused child
  slot retire; total retired unused requests are 131. They authorize no retry.
- [Spending receipt](spending.json), [independent outcome review](outcome-review.md)
  and [external archive receipt](retention.json) preserve the result.
- Raw packet: `/private/tmp/task35582-macos-final60/attempt-f_yrr5n4`.

The archive excludes temporary credentials/configuration and verifies selected
bytes against the approved provider keys without retaining them. SQLite backups
and archive hashes were checked; standalone SQLite integrity checking was
unavailable without the application's custom validation function, as disclosed
in the retention receipt. Original evidence was not deleted.

The follow-up automation was deleted. No production repair, broad test run,
qualification retry, budget request, ready-for-review action or merge followed.
Existing ADR-281 applies; no architecture changed. Live observations remain
distinct from the previously recorded scripted fault controls.
