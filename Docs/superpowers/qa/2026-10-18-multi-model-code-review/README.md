# TASK-35582 delivery evidence

**Latest qualification failed; AC24 remains open and PR3286 draft.**
The [final handoff run after green CI](final-handoff60/report.md) used44/60
generation requests and3/3children onffe3cfdf5a, with native[2,1]. Both reviewer
reports sealed and were presented. Haiku accepted four dispositions, then became
stuck after three consecutive out-of-range source reads; V1 never sealed. The
later deadline interrupted the primary stream, leaving one usage reservation
uncertain. Final mapping/unified reads were unavailable. Current cumulative usage
is457requests/31children; the unused16requests retire with this consumed invocation.
No retry, new repair cycle or merge followed, and the follow-up was deleted.
[Independent outcome review](final-handoff60/outcome-review.md),
[spending](final-handoff60/spending.json), [retention](final-handoff60/retention.json).

The [previous run after green CI](v1-final60/report.md) used26/60generation requests
and2/3children onb35bb489ea. R1 received an accepted but contradictory completion
that could not seal; R2 sealed; the primary later reported an unresolved provider
error. V1 did not launch and unified output is unavailable.
[Independent outcome review](v1-final60/outcome-review.md),
[spending](v1-final60/spending.json), [raw retention](v1-final60/retention.json).

Before the latest run, cumulative usage was413generation requests/28children, including one
subsequent local diagnostic (zero cloud requests/children). The single
invocation is consumed; unused34requests/one child do not permit a retry.
The [instruction-deferral checkpoint repair](instruction-deferral-checkpoint.md)
reproduces a continuation defect against the exact saved checkpoint: deferred
calls remained pending and invalidated the next otherwise-valid response. Eight
new offline regressions pass after settling each deferral through existing
persistence/acknowledgment barriers. The missing historical response still limits
attribution. The latest run reached V1; AC24 remains incomplete.
The [R1 offline repair and provider diagnosis](r1-completion-offline/report.md)
now reject contradictory completion before acknowledgment, with same-ID repair
feedback and unchanged sealing checks. The failed live result is unchanged;
offline provider controls do not recover the missing historical response body.
The [provider diagnostics observer](provider-diagnostics.md) is now integrated
with seven maintained offline controls and a separately gated runner preparation.
The [October29 dev sync](dev-sync-2026-10-29.md) resolves the PR's integration
conflicts, preserves244 non-overlapping feature files, and records targeted
checks plus pre-existing size-test failures. Normal PR CI must be checked on
the resulting head; the earlier queued third-party review was not Actions CI.

The [shutdown reconciliation guard](ci-followup/shutdown.md) and earlier
[UI wait repair/Windows diagnostics](ci-followup/report.md) passed normal CI
before qualification:55success/3skips. All7native Windows full-app GGUF cases
passed; observed close time fell from142s to17s and CPU from136s to0.31s.
The [V1 conflict repair](v1-conflict-offline/report.md) remains verified offline;
the earlier26-request run never reached V1 and added no live evidence for that repair.
The [previous qualification](continuation-final60/report.md) and older failures
remain historical evidence with their original spending and limitations.

## Existing evidence reused

- [Whole-branch review](final-fix/final-review.md) and [scoped final fixes](final-fix/final-fix-review.md) cover the original implementation. Integration changes receive only their relevant follow-up checks.
- [Resource repair review](ac23/ac23-resource-review.md) and [AC23 closure review](ac23/ac23-regression-review.md) record finite native ownership and deterministic controls.
- [Haiku capability review](ac24/fresh42-after-usage-outcome/haiku-fix/review.md) covers the exact model-family request correction.
- The native scripted journey has4originals/3blinded units (exact duplicate, distinct trigger and false positive); fault controls cover failed reviewers, truncation, drift, small contexts and storage overflow. These remain scripted evidence, not live judgments.

Prior tests/reviews retain their source revisions and limits. Inherited profile
admission failures, descriptor/static warnings, a Minor precedence-fixture gap,
and bounded recovery limits are not relabeled green. Current delivery integration passed97 focused tests and2 current-dev Stop/stall
controls. The [integration report](integration-report.md) and
[scoped integration review](integration-review.md) record exact boundaries; the
review found no integration findings. The [bounded delivery recovery](delivery-recovery.md)
records the subsequent narrow CI corrections and preparation. The final failed
acceptance and newer CI blockers are recorded above.

## Preservation and review footprint

The PR retains required test fixtures (`seeded_average.py`,
`ac24/capacity-evidence.json`), concise reviews, and design documents imported by
the canonical ADRs. Redundant raw traces, payload snapshots, command logs and
private SDD reports are omitted from the delivery diff. They remain intact on
`codex/task-35582-multi-model-review` at3bce87f15c and in a verified6.1MB Git bundle.
The [retention manifest](retained-evidence.json) records paths, hashes, bundle
location and prerequisite. No original evidence or active worktree was deleted.

Historical artifacts call the feature ADR240. Current dev assigned that number
to MCP destination retirement, so delivery renames the feature decision to
[ADR-281](../../../../backlog/decisions/281-primary-agent-multi-model-code-review.md)
without changing its contracts. Verbatim historical reviews retain their numbering.


First dev sync (historical): the delivery branch was rebased onto
7884e11a496a677c8180cc5d67234a57b7d659b9. Only the lessons append and size-table
rows conflicted; both valid documentation additions and every stricter limit are
retained. All202 non-overlapping feature files and182 upstream files are byte
identical to their respective tips. Four affected ratchet cases pass. The existing
source reviews and Haiku preflight remain valid within their original scope; no
provider call was repeated. See the [small preservation receipt](offline-repairs/dev-sync.json).
GitHub Actions status must be read from the normal PR runs on the resulting head;
CodeRabbit's skipped review is not passing Actions CI.

The second integration candidate was rebased onto then-current dev
`3efe8476cebf70eceafa605d1abed76a4aad4e0e`. This second sync had a real runtime
overlap: it preserves upstream first-request schema builders, compact input
budgets and recoverable disclosure refusals alongside review role/spawn gates.
The extracted request factories match the combined upstream/feature bodies;
completion-repair source and tests remain unchanged. All 74 shared size ceilings
preserve the stricter parent limit (service 9,594; bridge 11,605).

Targeted integration checks passed: 54 upstream runtime controls, 34 review and
continuation controls, and five affected size/governance cases. The diagnostic
inventory guard passes after inspection of nine affected owner rows; 696 original
owner rows and all privacy/sink sections are unchanged. These checks do not
replace live AC24 evidence or the separately recorded baseline failures. Exact
commands, logs and preservation proofs remain in
`/private/tmp/task35582-dev-sync/runtime-conflict-report.md` and its adjacent
receipts; no new raw archive was added to the PR.

Normal PR CI on `ef56ce83f6` passed all 34 test jobs, including the eight
module-size guards, both admission lanes, all 28 UI shards and Console P0.
Perf Guard and Tool Recovery also passed. The final derived-artifact job failed
only its worker/DOM contract: five new review UI methods had post-await lookup
warnings. The [actual failed run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40787033395)
is retained as failure evidence, not relabeled green.

The follow-up addresses that boundary in the review modal and review-team
settings. Five mounted removal-race regressions failed with `NoMatches` before
the fix; the final two-file run passes 29 tests, including deferred read errors
and preserving a completed configuration save after panel removal. UI publication
now checks the lifetime of its original controls. The mutually exclusive Close
branch moves last to remove its lexical false positive. The exact worker guard
passes without any census edit, compilation/whitespace pass, and there are no
new lint findings. No runtime, permission, accounting, workflow, visual or budget
change was made. The scoped review found no actionable findings. Detailed red/green receipts remain in
`/private/tmp/task35582-dev-sync/worker-dom-repair/`; the next PR head must receive
its own normal CI result. AC24 remains open and PR3286 remains draft.
