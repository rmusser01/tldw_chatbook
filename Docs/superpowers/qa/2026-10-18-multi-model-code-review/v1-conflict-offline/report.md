# Offline V1 resubmission and completion repair

The recovery dead-end is fixed offline. TASK-35582 stays In Progress, AC24 stays
unchecked, and PR #3286 stays draft. This repair used **zero live requests and zero
product child launches**; cumulative live spending remains **386 requests / 26
children**. No qualification, allowance request, merge or auto-merge occurred.

## Cause and correction

The [last live run](../continuation-final60/report.md) accepted all three V1
dispositions. V1 then repeated them under a new submission ID with changed content.
Rejecting that write was correct: accepted decisions are immutable. The offline
reproduction establishes a separate host recovery gap: there was no supported
way to withdraw that rejected duplicate after all units had decisions. A new
completion ID remained blocked by the unresolved rejection. The live outcome
report's statement that no new source defect had yet been demonstrated remains
accurate for that earlier investigation; this follow-up supplies the reproduction.

The participant now receives bounded conflict feedback with exact retry arguments.
It can explicitly submit empty dispositions under only that outstanding conflict
ID at the current revision. The ordinary atomic publication path accepts an empty
acknowledgement and retires just that rejection. It never changes accepted
verdicts, evidence or original-finding mappings. Fresh IDs, other rejection reasons,
accepted-ID changes, other roles and closed attempts cannot use this path.
Stale-revision, acknowledgement-capacity or publication failures preserve the
original conflict's recovery eligibility and permit a later valid retry.

A rejected completion must still be retried under its own ID. Empty withdrawal
acknowledgements do not authorize history reduction: otherwise the new recovery
could erase the pending completion ID. Normal authority, capacity, quota, complete
unit accounting, terminal and capture checks remain in force. No runtime budgets,
permission policies, native child limits or provider integrations changed.
Existing ADR-281, the design spec, packaged verifier guide and launch/schema
instructions document this small protocol correction.

## Evidence

The tests use the real participant provider/service/store and real native/fenced
agent loops with scripted model responses, including actual next-request error
feedback and production history projection. These are offline protocol controls,
not evidence that a live model will follow the recovery instructions.

- Before the fix, both native/fenced reproductions fail at absent conflict feedback.
- A further native/fenced control fails when an empty withdrawal authorizes history
  eviction; retaining those rounds makes both complete normally.
- Final targeted suite: **285 passed**, including all 12 new controls and existing
  model, store, provider, verification, completion and report regressions.
- Module-size governance: **1 passed**; no size ceiling changed.
- New test lint and formatting pass. Models, store and tool provider formatting
  pass; the service's two existing formatter hunks are unchanged. Lint comparison
  reports **zero new findings**, with eight existing findings left untouched.
- `git diff --check` passes. The [scoped review](review.md) found no actionable issue.

The successful offline path keeps all three original dispositions byte-equivalent,
expands every original exactly once, preserves idempotent acknowledgements without
extra charging, and remains unsealed until capture plus terminal receipts arrive.
Failure controls cover stale revisions, disk/ack failures, other rejected IDs,
partial mixed batches, accepted-ID changes, closed attempts and other roles.

The exact focused pytest invocation is:

```sh
python -m pytest -q \
  Tests/Agents/test_code_review_conflict_recovery.py \
  Tests/Agents/test_code_review_completion_recovery.py \
  Tests/Agents/test_code_review_models.py \
  Tests/Agents/test_code_review_store.py \
  Tests/Agents/test_code_review_tool_provider.py \
  Tests/Agents/test_code_review_verification.py \
  Tests/Agents/test_code_review_reports.py --tb=short
```

Baseline was `4526f771b78e833182d62fe17c7660334ab7d7fd`. Raw receipts are
preserved separately as described in [retention.json](retention.json). The prior
live failure is unchanged, and this source candidate needs its own CI result.
No full test sweep or live qualification was run.
