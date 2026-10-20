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
