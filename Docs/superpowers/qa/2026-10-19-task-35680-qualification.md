# TASK-35680 qualification

Continues [PR #3176](https://github.com/rmusser01/tldw_chatbook/pull/3176) from
head `da8f7496a5ffb8dcce030bc22f42f6c46b9834dc`, based on dev `8136af6f6f`.
Python 3.12 / Textual 8.2.8. Only targeted tests ran under private profiles.

## Cause and bounded correction

Programmatic `.value` assignment posted Input.Changed while the temporary
parallel-run flag was set, but the handler ran after it reset. With an unrelated
paste-threshold draft, that event added untouched max_parallel_runs:2. On
category return, the newly constructed Input also posted its initial value,
adding untouched max_parallel_runs:12.

The sync assignment now uses the existing `prevent(Input.Changed)` idiom.
The handler ignores values differing from the sending Input's current value
and values already represented by the effective draft/config. This covers
stale and initial/no-op events, while a genuine next edit to the previously
displayed value still stages normally. The temporary flag is removed.
Draft, validation, writer, keyboard and confirmation ownership remain under
ADR-033/031. No other control family or runtime contract changed.

## Automated evidence

- Initial permanent run on original product source: three failures in 125.79 s.
  The refresh case failed with exactly one extra max_parallel_runs:2 key.
- The first re-entry assertion compared the entire draft and also caught
  initialization values from unrelated controls. Narrowed it to the task's
  parallel-run key and unrelated threshold preservation; original source
  failed again specifically on `max_parallel_runs` presence (52.95s).
- First fixed-source selection: four passed, one compact setup failure.
  Keyboard typing had not reached the offscreen control, leaving the category
  clean instead of testing invalid 0. Used the existing field-search path,
  scrolled the control into view, and added actual focus/value assertions.
  No product code or expected validation behavior changed for this correction.
- Final permanent file: three passed in 159.41 s. Covers queued origin,
  unrelated draft preservation, first edit equal to the pre-refresh value,
  Input identity/focus/selection, invalid 0, actual private config Save of 12,
  category return and confirmed Revert at 90x30 and 180x50.
- Existing direct-call max-parallel save and both Settings size checks:
  three passed, 62 deselected, 58.14 s. The existing test body is unchanged;
  its private-profile marker/request argument were added.
- Both neighboring paste-origin cases passed on the same fixed product
  source in the earlier selection. They were not changed.

New test Ruff lint/format passes. Settings' inherited Ruff inventory remains
13 findings with none introduced, compared against current dev. Generated
stylesheet reproduction, worker contract, Backlog IDs/readability and PR UI
census guards pass. The screen's measured size and pin are 35,804 lines.
All 333 previous UI census entries stay in order; three new entries make 336.
No performance ceiling, safety guard or existing assertion was weakened.

## Live keyboard journey

A full TldwCli ran through the existing loopback verification server with a
disposable private profile and the real-home guard enabled. Field search found
Max parallel agent runs, initially 3. Typed 12, used F6 to reach the action pane,
then s saved it. Typed a later edit, used F6/r, observed the confirmation, and
Tab/Enter chose Discard changes. The field returned to 12, the UI reported no
changes, and the private TOML still contained 12. Zero unhandled exceptions
were logged. The app and server were retired after verification. Real user
configuration was not modified.

Independent read-only review found no actionable issue in the product or
tests; a final recheck approved the explicit compact focus/typing setup.

## Remote validation boundary

Before this update, the prior head's Perf Guard completed successfully and
Derived Artifacts remained pending behind an older run. There were no inline
review threads. Qodo reported that reviews were paused for exhausted workspace
credits; CodeRabbit reported that automatic review was skipped for this base.
These messages are service/configuration status, not review approval.
Current-head CI and any subsequent review feedback are checked after pushing;
no remote green result is inferred from local tests or the prior head.

## Independent review and CI follow-up (2026-10-20 UTC)

At the user's request, superpowers:requesting-code-review replaced Qodo for
this PR. A fresh read-only reviewer assessed the full 0fef489ece..df4e73c908
range against all four task records, the plan, applicable ADRs and lessons.
It found no Critical, Important or Minor issue requiring changes and
independently passed five panel/action/paste-origin cases. Its exclusions
(unchanged control families and broader decomposition/polling work with
existing owners) were explicitly accepted by the coordinator.

On that head, Perf Guard, PR Fast Lane, all ten UI shards, Dreams checks
and Console P0 passed. The admission-sensitive job failed one test with
545 passing and one expected failure:
[hosted assertion](https://github.com/rmusser01/tldw_chatbook/actions/runs/39375455061/job/118472787475).
`test_opening_console_during_a_headless_delivery_arms_the_poll` inspected the
poll with attachment still unreconciled, the wake still delivering and the
same controller. The Console paths and test were unchanged against dev;
dev's push workflow skips that job, so its green result is not a baseline
pass for this test. The unchanged exact node passed locally in 66.24 s.
Earlier failures recorded by TASK-34789 had different signatures; no exact
local reproduction or particular native deferral cause is claimed.

[TASK-35710](../../../backlog/tasks/task-35710%20-%20Wait-for-Console-attachment-before-checking-headless-delivery-poll.md)
owns the bounded test-only correction. Composer presence and a message-pump
pause do not witness completion of full attachment. The test now waits for
the existing completion flag/current screen at all three visits, using the
neighboring startup test's 50 polls at 0.1-second intervals. Both original
idle/active poll assertions and the wake/navigation/profile/test-timeout
contracts remain. Production source and polling policy are unchanged.

Eight targeted ownership checks passed in 93.02 s. Disposable negative
controls confirmed that never completing attachment fails the new setup
assertion (56.52 s), and suppressing poll arming fails the original active
assertion after successful reconciliation (53.19 s, reconciled=True).
No negative-control patches are committed. Ruff lint and modified-function
formatting pass; whole-file formatter debt is identical to the pre-change
head. The first comparison attempt used Windows' legacy subprocess encoding;
explicit UTF-8 completed the comparison. Backlog ID/readability guards and
diff whitespace checks pass. No full suite ran.

Independent read-only delta review of df4e73c908..8313dab824 found no
Critical, Important or Minor issue requiring changes. It inspected the
attachment lifecycle, neighboring startup pattern and both negative-control
receipts, and confirmed the wait establishes a legitimate precondition
without manufacturing readiness or weakening the polling oracle. New-head
CI remains a separate publication check; no hosted green result is inferred.
