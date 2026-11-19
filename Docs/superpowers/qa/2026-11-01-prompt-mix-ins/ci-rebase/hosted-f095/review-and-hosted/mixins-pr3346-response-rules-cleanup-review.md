# Independent UI28 response-rules cleanup review

Verdict: no actionable Critical, Important, or Minor findings in the settled test-only patch against HEAD `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

Reviewed the fixture's scripted HTTP waits, the gated phase-event case's precondition/cleanup ordering, the added two native failure-retirement controls, and retained evidence. No product/test edits or test executions were performed by this reviewer.

## Cleanup correctness

`mounted_rules_console` owns the primary/helper release events used by its mock HTTP transport. Previously its outer finally only awaited `owner.dispose()`. The gated test can start a held helper request and then fail in preconditions before reaching its later local release block. The fixture is therefore the correct owner for unconditional release on exit.

The patch sets each non-None fixture-owned release event immediately before the original `await owner.dispose()`. Event.set is idempotent, so normal tests that already released their holds retain the same teardown. It neither substitutes a production disposal result nor suppresses exceptions, skips resource retirement, changes cancellation/timeout guards, or grants source/transport authority. No production file is changed.

The regression parameters exercise both primary and helper requests through the mounted Console and real scripted HTTP transport. Their observed Event.wait records the actual waiting task without replacing its wait. Each control confirms the task is running and the event is unset before raising a specific AssertionError inside the fixture scope. After exit, it asserts that exact error object survived, the event was released, the waiting task is done, and the helper manager has no unsettled work. Its bounded outer timeout prevents an unbounded regression; it does not turn timeout into success.

## Retained evidence inspected

Directory: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-f095-ui28-hud_keud`.

- The pre-fix diagnostic probe raises an injected assertion only after the helper has entered and remains unreleased, and schedules an observational eight-second release. Filtered task snapshots from `phases.jsonl` show the actual `http_transport` await at fixture line 94 still held at +1.30, +4.31, and +7.32 seconds after that failure. `injected.log` retains the injected failure. This is causal evidence for the fixture cleanup defect, not reproduction of the hosted failure's unknown initial precondition.
- `cleanup-controls.log`: both final primary/helper failure-retirement controls passed, **2 passed in 9.55 seconds**.
- `gated-controls.log`: all four original phase-event parameters (`none`, `rebuild`, `maintenance`, `no-controller`) passed, **4 passed in 14.64 seconds**. Their original assertions are unchanged.

These results were produced by the implementer and read here; no reviewer rerun or broad-suite success is claimed. The exact hosted early precondition remains unreproduced, so this report does not assert that its complete root cause is resolved. It confirms a demonstrated fixture-owned teardown gap is repaired while preserving the initial assertion failure.

## Reviewed working-tree SHA-256

- `Tests/UI/response_rules_fixtures.py`: `98272ace16c4b897d713911a826b2745f669dc2351ef88aebf31d2adfdf95e40`
- `Tests/UI/test_console_response_rules.py`: `1bfd03dd30ed0e57085fd1183f3f32f3c7bae24ffc9591facb735b900486a4bf`
