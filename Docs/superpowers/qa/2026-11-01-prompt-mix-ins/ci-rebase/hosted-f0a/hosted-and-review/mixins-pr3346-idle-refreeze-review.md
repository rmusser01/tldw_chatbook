# Independent UI18 idle-refreeze review

Verdict: no actionable Critical, Important, or Minor findings in the settled one-file fixture change against HEAD `f0a856d1b914c668a6d8ad28baafd91adb9b43a6`.

Read-only review of the exact mounted custody/GC test, IdleHeapRefreezer request/fire/maintenance lifecycle, app_idle_warmup completion callback, Console busy predicate/run-settled request, and retained controlled/ordinary evidence. No source edits or runtime tests were performed by this reviewer.

## Why this correction is sound

The old wait returned as soon as the weak reference died, then immediately required a first_send_preimport action. Those are separate asynchronous outcomes: a previously armed native run_settled timer can collect the cycle before the new warmup callback's quiet-period timer fires. IdleHeapRefreezer intentionally records an action only after maintenance and coalesces requests; weakref death alone does not establish completion of the requested reason.

The patch waits for the conjunction of both original outcomes within the original ten-second bound: the weak reference is dead and the exact `(refreeze, first_send_preimport)` action exists. The final exact-reason assertion remains. It does not accept any maintenance reason, replace physical GC, fire a timer manually, request repeated maintenance, or lengthen a deadline.

The actual held received-intent custody assertion and the 0.15-second no-maintenance check remain unchanged. The test still waits for real custody retirement and the real Console busy predicate, creates a cyclic object/weakref, performs no explicit collection after retirement of that object, submits the original warmup callback through the host UI-loop path, and relies on mounted timers and the real collector. No product busy/maintenance/weak-owner/generation/quiet-time logic or performance threshold changed. The whole test file is already in the UI census, so no new enrollment is needed.

## Evidence read

Directory: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-f0a-ui18-42od1l_t`.

The diagnostic probe delays only the second warmup submission until an already-armed real run_settled timer completes. Its request/fire/maintenance wrappers call the original bodies; it neither fakes collection nor manually fires timers. The timeline shows run_settled collecting the cycle, the original second callback requesting first_send_preimport while not busy, and the old assertion running with a dead cycle but that reason still pending. The controlled old test reproduces the hosted assertion in `ordered-red.log` (**1 failed in 8.87 seconds**).

With the corrected wait, the timeline continues through the actual first_send_preimport timer and maintenance before the test exits: `ordered-green.log` records **1 passed in 8.27 seconds**. The final exact test without the observer/ordering plugin also passes: `ordinary-green.log`, **1 passed in 5.59 seconds**. These are implementer-produced results read by this reviewer, not reruns, a full-file sweep, or hosted all-green proof.

`static-proof.json` records all original assertion ASTs and timeout arguments unchanged, lint zero before/after, and the same pre-existing formatter check status. The inspected source diff agrees: only the bounded readiness predicate and explanatory comment change.

Reviewed working-tree SHA-256 for `Tests/UI/test_console_idle_heap_refreeze.py`: `6b579c6ace596b17a192731d32914a0f6ccd0c6f2e0c1bc91dc39dbf7e73a085`.
