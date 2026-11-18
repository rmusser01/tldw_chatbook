# Independent UI26 readiness-close lifecycle review

Verdict: no actionable Critical, Important, or Minor findings in the settled change against HEAD `7ac214599d4969a4d1a57db8bef41ff02eb433c9`, including final regression enrollment.

Read-only review; no production/test edits or runtime tests were performed by this reviewer.

## Product correctness

`ConsoleReadinessConfigProjection.run` marks a read pending before scheduling `_refresh`. A real session can close before that coroutine enters. Previously `store.session_settings(key[5])` executed outside `_refresh`'s cleanup boundary, so its KeyError escaped as WorkerFailed and left pending true / the settled event unset.

The patch moves all existing setup into the existing try without changing its ordering or operations. An independent AST comparison confirmed: the former setup prefix plus original try body exactly equal the new try body; exception handlers, finally block, and post-try publication are each unchanged. Setup failure now follows the existing cold-return policy and clears `_read_request`, pending, and the settled waiter in finally.

There is no physical read to abandon when this initial lookup fails: the preparation-read submission occurs afterward. For already submitted reads, `run_preparation_read` still retains its shielded producer, consumes its outcome, and retires the physical read before propagating cancellation. `_refresh` still catches Exception only, so CancelledError/BaseException are not swallowed, while its finally still settles lifecycle bookkeeping. No native reader, source witness, proof seal, current-key comparison, runtime-identity fence, pricing/tokenizer fence, convergence rule, or publication path was changed.

An older cached value does not gain authority from this cleanup: run/warm still require exact current owner/key/runtime and freshness. The fix does not publish fallback data or bypass a failed source lookup.

## Deterministic regression

The new test uses the real ConsoleChatStore and real session controller via the existing fixture. It schedules warm, observes pending/unsettled state before worker entry, closes that actual session, and requires a False warm result, no pending/read request/physical reads, no value/native callback/publication, and a normally completed worker. It then creates a current session with established canonical settings and requires a successful new warm, exactly one read/publication for that owner, and its current session key.

The canonical settings fixture avoids an unrelated pristine-default convergence revision change; it preserves warm's strict currentness assertion rather than weakening it. Existing source/cancellation tests are unchanged.

## Retained execution evidence inspected

Directory: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-7ac-ui26-fjy9yzcp`.

- `original.log`: exact original ancestry-size0 native journey reproduced WorkerFailed / Unknown Console chat session, **1 failed in 45.38 seconds**.
- `race-red.log`: deterministic close-before-entry regression timed out waiting for the unset settled event before the fix, **1 failed in 2.60 seconds**.
- `race-green.log`: six existing repeated-cancel/physical-read, source-retarget, rebound-runtime, and closed-owner controls passed. That phase also had one failure in the new test's initial retry fixture, subsequently corrected using established canonical settings as described above; it is not described as an all-green run.
- `final-green.log` / `.json`: final race and exact original ancestry-size0 journey both passed, **2 passed in 14.31 seconds**.

These are phase-qualified implementer results read here, not a combined all-eight or broader CI success claim. Final performance qualification is separately coordinated by the parent.

## Gate enrollment

The exact new regression node appears once at the census tail, alongside the earlier single projection selector without overlap. The prior 1,332-entry census bytes match the immediately preceding reviewed SHA after removing only the new comment/node lines; the UI24 node stays in place. Current census/floor are exactly **1,333**. No whole-file enlargement, workflow, shard algorithm, or timeout change is included. Parent-owned census/unit verification is separate.

## Reviewed working-tree SHA-256

- `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`: `24b767b3a15ff6ab80b7daf704e7ebaaed5aeb5343dc758c5864d61c86f442e8`
- `Tests/UI/test_console_readiness_config_projection.py`: `59346b62a8494893623b2e6176fe7f7cd234cc771a07c4e0d8c913c91a5bdf43`
- `scripts/ui_pr_gate_census.txt`: `a92ce0a7602498007592d64218f6dd8d801f67c44019317da289212e2519f5b3`
- `scripts/check_ui_pr_gate_census.py`: `c187b0019d8e8a6cc172eefb9f708b252b7c5349bab1b01234b004c2833c1496`
