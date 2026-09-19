# ADR-167: Application lifetime for admitted Tool Profile writes

Status: Accepted
Date: 2026-09-18
Task: TASK-32787

## Context

Settings is recreated on destination navigation. Its exclusive workers observe
blocking Tool Pack writes through `asyncio.to_thread`; cancelling that observer
does not recall an admitted filesystem mutation. TASK-32787 reproduced a real
UI ownership gap: removal completed after Settings was destroyed, but a new
Settings instance retained the old row and had no outcome. TASK-32786's local
same-operation guard intentionally did not change this application boundary.

## Decision

The application lazily owns a narrow Tool Profile write coordinator. It captures
the exact admitted operation after the existing review/confirmation gates. It
retains at most one pending write per operation kind (import, export, remove)
and one latest terminal outcome. Outcomes contain existing compact result types
or stable error categories, never archive paths, arbitrary exceptions or review
payloads. No persistent job ledger, automatic retry or new notification surface
is introduced.

Preparation and modals remain screen-owned. Cancelling an observer returns
promptly without cancelling the owned task. Export preserves its pre-publication
cancellation probe, including cancellation from its original observer; a late
cancel cannot retroactively turn a committed publication into a cancelled result.
Import and removal observe their actual admitted outcomes. Exact snapshots,
revision validation, profile/binding authority and service locks remain unchanged.

Mounted Settings projects coordinator revisions through its existing 250 ms
readiness timer, initial listing and originating completion. A new instance
projects the latest progress/outcome. A separate completion revision ensures
current facts refresh even when another pending operation supersedes the receipt
before the next timer tick. Updates use existing focus-preserving local receipt
and listing paths; no observer subscription retains a screen. Duplicate admission is fenced
in the coordinator as well as before dispatching an exclusive UI worker.

Application shutdown closes admission, signals export's cancellation probe and
drains admitted work before another owner can fail and advance teardown to
dependent resource closure. The existing process-owned watchdog is armed before
the first app-owned drain, retaining the later unmount fallback. The existing application
shutdown cancellation handling and exit watchdog remain the only shutdown
policy; the coordinator does not swallow observer cancellation or invent a
second timeout. Hard termination does not guarantee a final UI receipt; existing
atomic service writes and startup reconciliation remain authoritative.

## Alternatives

- Keep writes screen-owned: reproduces lost outcomes and stale replacement views.
- Wait for threads in the departing screen: delays navigation and can keep
  teardown alive indefinitely; cancellation of a wait is not cancellation of work.
- Make Settings reusable: changes every category's lifecycle to fix one workflow.
- Add persistent jobs or reuse Console settings policy machinery: broader
  ownership/storage coupling than these three existing operations require.

ADR-107 remains the Tool Pack authority contract; ADR-150 governs presentation.
The [design](../../Docs/superpowers/specs/2026-09-18-tool-profile-write-lifetime.md)
and [plan](../../Docs/superpowers/plans/2026-09-18-tool-profile-write-lifetime.md)
record the implementation and verification scope.
