# TASK-32775 independent review

A read-only reviewer found that reloading the draft when final review closed could
fail, leaving an active cached session. Finishing again then rebuilt the review
from original answers and replaced applied edits. The real-coordinator regression
reproduced the hidden Review action with an unavailable resume operation.

The final correction projects the successful review transition into the cached
session and restores that known state on Close. Reopening calls coordinator.review
and preserves applied edits. The final ten-case run passes, including exactly one
Finish invocation. A second independent review found no remaining introduced
blocker. No service or persistence authority changed.

Preexisting worker callbacks that can outlive cancellation/dismissal remain a
separate lifetime audit; this review makes no whole-interview-lifetime claim.
