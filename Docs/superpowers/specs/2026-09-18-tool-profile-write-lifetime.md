# Tool Profile write lifetime — TASK-32787

Restore truthful pending/completed state across normal Settings recreation.
The mounted baseline removes the actual profile after its observer is cancelled,
while the replacement Settings screen retains the profile and an empty receipt.

## Ownership and interfaces

`Tool_Packs/operations.py` will own admitted asynchronous tasks, with one active
task per existing operation kind and one bounded terminal outcome. Admission is
synchronous before yielding. The task captures the existing service and exact
review/revision/destination arguments. The coordinator validates returned result
types and stores only compact service results or stable failure categories.
It never changes policy, binding or review authority itself.

`TldwCli` owns this coordinator lazily and drains it in the existing app lifecycle
pass before database closure. There is no new startup service scan or thread.
The existing exit watchdog is armed before the first app-owned drain, and Tool
Profile work drains before another owner can fail and advance resource closure.
No screen-level cancellation
loop or private executor manipulation is added.

Settings prepares and confirms as before. Only admitted writes move to the owner.
Its observer awaits through `asyncio.shield`, so screen destruction returns
promptly. Existing export cancellation remains a best-effort pre-publication
probe and shutdown also signals it. Import/removal report their actual outcome.
Recreated screens read coordinator revisions through the existing 250 ms
readiness timer, initial listing and originating completion. Separate completion
revisions trigger current-fact refresh even when newer progress supersedes a
receipt between ticks. No subscription retains a screen; newer focus and dialog
ownership remain intact.

The latest receipt is application-session state. A replacement screen projects
the latest activity or completion in event order. Repeated same-operation
actions cannot replace the active write even after navigation. Different operation
kinds retain their existing independent review/write semantics. Completion order
owns the latest receipt, matching the existing single-receipt surface; this is not
a history view. A stale preparation error must not override an admitted result.
Export `destination_changed` may reopen a picker only in its still-current
originating review; a replacement view receives recovery wording and requires a
fresh user action. No automatic retry or fresh confirmation is synthesized.

## Verification and limits

- Mounted dark/light production-CSS tests: destroy/recreate before and after
  completion; pending and terminal facts; repeated action; cancellation;
  other modal/focus retained; invalidation cannot retarget review authority.
- Coordinator tests: all three operation results and failures, per-kind admission,
  bounded outcomes, independent completion revisions, prompt cancelled observer
  and shutdown drain.
- Application shutdown ordering: admission closed and writes settled before
  dependent database closure, including cancelled shutdown and lazy unused owner.
- Existing Tool Profile lifecycle/review/export/removal tests, service boundary
  tests and governance. No full suite without explicit opt-in.
- Real private native journeys use the existing lifecycle lock to hold removal,
  navigate away and back, then release and verify the persisted tombstone, exact
  revision, visible receipt and focus. A controlled native shutdown journey releases
  the same real lock during normal shutdown and checks persistence/lifecycle.

No new schema, dependency, provider request or persisted job status. No guarantee
of a final receipt after watchdog/hard termination. Existing service reconciliation
remains authoritative on the next start. ADR required: yes;
`backlog/decisions/167-tool-profile-write-lifetime.md` amends the application
lifetime boundary while preserving ADR-107 and ADR-150.
