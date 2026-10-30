# CI cost reporting

The owner requested the remaining CI audit improvements after PR #3316. This
slice makes runner cost visible without adding a job or changing test selection.

## Design

Append a bounded, checkout-free `actions/github-script` reporting step to the
existing required aggregate, after its verdict steps. It runs with `always()`
and `continue-on-error: true`; only this diagnostic step may tolerate failure.
The aggregate receives actions-read permission. Application permissions stay as
they are. No checkout, installation, artifact upload, new service or credentials.

Read all job pages for the current workflow run and its current attempt through
the Actions API. Count each numeric job ID once. Sum actual started-to-completed
durations for completed, non-skipped jobs. Show current-attempt minutes and
cumulative minutes across recorded attempts separately. Exclude the reporting
job itself and explain that the running aggregate and later queue tick are not
included. These are observed job execution minutes, not GitHub billed minutes.

Show median and maximum created-to-start wait for completed jobs with valid
timestamps; label it job wait, since it can include dependency/concurrency waits.
Missing or inverted timestamps must produce an incomplete-data notice, never
silently become zero. Empty samples are unavailable, not a zero-cost pass.
Skipped jobs do not consume runner minutes. API failures produce a warning and
cannot alter the required qualification result.

Use an initial warning budget of 500 observed runner-minutes, compared with the
audit's 397-minute median successful first attempt (about 26 percent headroom).
Warn on cumulative observed cost above the budget; do not fail, cancel or omit
tests. Retain attempt and incomplete-data context so failed-job-only retries are
not mistaken for cheaper full qualifications. Reassess the budget using hosted
observations before expanding coverage; no automatic threshold ratchet.

## Evidence

Execute the actual inline script against recording API/summary fakes. Cover
pagination, repeated job IDs, retries, failed/cancelled/skipped jobs, missing and
inverted times, empty samples, the warning boundary and API failure. Contract
tests retain fail-closed behavior for every verdict and prerequisite while
allowing only the named diagnostic step to tolerate failure.

TASK-38100; amend ADR-103. Hosted cost output is qualification evidence still to
collect, not something local fake API tests establish.
