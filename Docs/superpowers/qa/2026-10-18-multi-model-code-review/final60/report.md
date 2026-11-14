# Single final qualification on the repaired delivery candidate

**Failed; AC24 remains open and PR #3286 remains draft.** The user authorized one
run capped at 60 new generation requests and three fresh children after CI.
The frozen runner ran exactly once on `0ab1e549f0c02b9621ce9be295baa969e0cb9e75`.
It exited 2 with `live_report_incomplete_V1`; no automatic retry occurred.

## First failing boundary

Both independent reviewers finished, sealed, and presented their originals.
R1 (Luna) reported no findings and missed the seeded empty-input regression.
R2 (DeepSeek) submitted four findings, including that regression, and explicitly
rejected the planted false string-input claim. Its coverage limitations remain
part of the original report. R2's previous incomplete-report failure did not recur.

The original primary exhausted its two-child allowance. The host correctly
accepted the already authorized verification continuation and created a fresh
primary on the same work chain. That primary received a successful response
requesting `code_review_read`, then failed on its next request before another
physical send. The UI classified this as a bad request/400. All 24 physical
transport receipts, including the final one, actually record HTTP 200; this is
not evidence of a server HTTP 400. The first established failing boundary is
local preparation/validation of the next continuation request. The retained
metadata omits message bodies, so it does not establish the exact malformed
field. Both prepared requests were below their reported input ceiling.

V1 never launched. No live Haiku generation, verifier dispositions, complete
original mapping, or unified report exists for this run (`verification_unavailable`).
The result gate correctly refused completion. The successful historical Haiku
native tool-result/accounting preflight was reused only for its unchanged
boundary; it cannot substitute for this missing end-to-end verification.

The [scoped outcome review](outcome-review.md) reproduced one narrow offline
incompatibility: project-context deferral rows retain a `name` key through native
serialization, while the strict hosted request validator rejects that extra key.
Removing only that key passes the synthetic control. The actual failed request
body was not retained, so this is a supported candidate cause, not a proven
attribution for this run. No production repair or live retry was started.

The automatic-work ledger settled the first continuation call at 14,897 tokens.
The locally failed next call left a 13,241-token reservation `uncertain` and the
chain `review_required/usage_unknown`. That accounting pause follows the
request failure; it does not establish a new streaming usage-extraction defect.
Native shutdown completed with zero running runs, owned fleet threads, workspace
connections/leases and registered chat connections. Resource cleanup and
successful accounting settlement are separate claims.

## CI and reused checks

[Normal PR CI](https://github.com/rmusser01/tldw_chatbook/actions/runs/40792192461)
on the exact source passed, including the repository's sole required check,
`Derived artifacts reproduce from their sources`, all 28 UI shards, the size
guards and Console P0. UI shard 9 initially failed an unchanged Notes row-mount
race; the exact test passed locally, and one targeted manual job rerun passed.
The failed first attempt remains recorded. No baseline test was edited.

The separate, non-required [Windows GGUF source evidence job](https://github.com/rmusser01/tldw_chatbook/actions/runs/40792192405)
timed out at 180 seconds in
`test_inventory_during_lazy_selector_mount_is_replayed_after_leaving_pane[True]`.
Its macOS/Linux jobs passed. This failure is disclosed, not relabeled green or
repaired under TASK-35582. Perf Guard, Tool Recovery and Eval Recovery passed.

Reuse the completed scoped source reviews, offline R2/V1 recovery checks,
17 size/helper controls, 29 UI lifetime checks, current-dev integration checks,
and separately attributed scripted fault/duplicate controls. No broad test sweep,
source repair wave, repeat live preflight or new qualification was performed
following this failure. Existing ADR-281 applies; no new architectural decision.

## Spending and retention

- New physical generation requests: **24/60** (Qwen 10, Luna 6, DeepSeek 8, Haiku 0).
- Fresh product children: **2/3**. Six readiness metadata requests are separate.
- Cumulative actual: **338 generation requests / 23 product children**.
- Unspent: 36 requests and one child; the single invocation is consumed and these
  do not authorize another run. The previous unused seven remain retired.

The private evidence directory is `/private/tmp/task35582-final60/`, with actual
run `attempt-1vlnx_0q`, the immutable approval/claim, required-CI receipts,
`final-spending-reconciliation.json` and the scoped outcome review. The raw
manifest SHA-256 is
`a6d78ff7d3a0e1e08917e8a3a062cccdc5ca1a74e9715daf7997c720ab037269`.
Raw evidence is preserved separately; no raw payload archive is added to the PR.
The original source, saved user profile and prior evidence remain unchanged.
