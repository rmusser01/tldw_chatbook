# Final qualification after the continuation repair

**Failed: AC24 remains open and PR #3286 remains draft.** The one approved
qualification ran on `a1b4b8dac95eed4d38186423ee06b71b3c555346`, after all current
CI checks passed. It consumed **48/60 generation requests and 3/3 fresh children**,
then exited 2 with `live_report_incomplete_V1`. There was no qualification retry.

## What completed

- Both independent reviewers sealed their original reports and the native report
  reader presented them before verification. Luna returned no findings and missed
  the seeded empty-input regression. DeepSeek reported three findings: that
  regression, a refutation of the planted string-input claim, and explicitly
  pre-existing bool acceptance with uncertain contract intent.
- The repaired local continuation reached Haiku under native **[2,1]**, on one
  causal chain. No provider request rejection or usage_unknown pause recurred.
- Haiku's first submission was accepted at revision 1 for all three units. Every
  original maps exactly once to an accepted disposition. The unified report is
  readable across two pages ending with `next_offset=null`.

## First completion-blocking event

After the accepted batch, Haiku submitted the same three units under a new ID,
with changed reasons and a changed verdict for the finding that refutes the
planted string-input claim. The host refused this as `disposition_conflict`.
Its subsequent `verifier_complete` was refused as
`unresolved_submission_rejections`; V1 ended without a seal.

The actual next-request observation matches the 433-byte production checkpoint
note, including the instruction not to resubmit accepted decisions and to submit
completion when every unit is disposed. This demonstrates a model/protocol
violation after supplied acknowledgement and guidance. It does not establish a
new source defect. Earlier invalid-source-range reads recovered and were not the
completion blocker. V1's later claim that none of its submissions were accepted
contradicts the retained acceptance receipt. No fourth child launched.

The [scoped outcome review](outcome-review.md) independently checked this boundary,
semantic meaning, ordering, mapping, accounting and cleanup. A single existing
offline control, `test_batches_checkpoint_before_expansion_and_replay`, passed
and reproduced the changed-verdict/new-ID conflict while preserving the accepted
checkpoint. No source repair, new workflow or broad test sweep followed.

Readable unified output and complete mapping do **not** make an unsealed verifier
complete. The mechanical result gate correctly failed; AC24 and task status were
not promoted. Existing scripted duplicate, failed-reviewer, truncation, source
drift, small-context and storage-overflow controls remain separately attributed
historical evidence, not new live observations.

## CI, usage and retention

[Required normal PR CI](https://github.com/rmusser01/tldw_chatbook/actions/runs/40808844123)
passed on the exact qualified source, including all 28 UI shards, Console P0,
module-size guards and the required derived-artifact/static check. The performance
job first exceeded its 15-minute limit after dependency installation took 597
seconds. Its [one explicitly authorized rerun](https://github.com/rmusser01/tldw_chatbook/actions/runs/40808844015)
passed on attempt 2. Before qualification, the rollup was 55 successful checks,
three skipped, none pending or failed. Later closeout edits are documentation only.

Actual generation requests: Qwen 21, Luna 6, DeepSeek 9, Haiku 12. All 48 physical
receipts returned HTTP 200; this establishes transport acceptance, not AC24.
Six metadata requests are separate. Cumulative actual usage is **386 generation
requests / 26 product children**. The 12 unused requests do not authorize another
invocation; all three child slots and the single invocation are consumed.

The automatic-work chain remained active with no pause reason. All 21 token
reservations on that continuation chain settled, totaling 263,654 tokens; none
remain reserved or uncertain. This is not a replacement count for all physical
requests. All five run rows ended done. Native shutdown completed with zero
running runs, owned fleet threads, workspace connections/leases or registered
chat connections, and no uncertain retirement. Source, team and original config
bytes remained unchanged.

Raw evidence is preserved privately under
`/private/tmp/task35582-continuation-final60/`; `retention.json` identifies the
separate verified archive. [Spending receipt](spending.json) records actual counts.
No further provider call, budget request, qualification retry, merge or auto-merge.
Existing ADR-281 applies; no new architecture or ADR is introduced.
