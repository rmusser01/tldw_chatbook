# TASK-35582 delivery evidence

AC24 remains open. The [latest single qualification](continuation-final60/report.md)
ran after all CI passed on `a1b4b8dac95eed4d38186423ee06b71b3c555346`.
Both reviewers sealed/presented, and the repaired continuation launched Haiku
under native [2,1]. All three original findings have accepted dispositions and
readable unified output, but V1 resubmitted accepted units with changed content,
then ended without a seal after `disposition_conflict`. The result gate exited 2.
PR #3286 remains draft; no new source defect was established.

Actual latest usage: **48/60 requests, 3/3 children**. Cumulative usage:
**386 generation requests / 26 children**. Accounting settled and cleanup
completed. The invocation is consumed; no qualification retry or budget request.
See the [outcome review](continuation-final60/outcome-review.md) and
[spending receipt](continuation-final60/spending.json).

The [offline continuation repair](continuation-offline/report.md) now has live
evidence through the previously rejected local request and verifier launch.
The older [final60 failure](final60/report.md), [bounded run](bounded-final/report.md),
and their reviews remain historical failures. Their raw evidence is preserved.
The R2/V1 [offline repairs](offline-repairs/report.md) and scripted fault controls
retain their stated scope; they do not establish live V1 completion.

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

The current candidate is rebased again onto live dev
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
