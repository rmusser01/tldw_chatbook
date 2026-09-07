# TASK-2061: Idle worker recycle for managed model deletion

**Status:** Approved design

**Date:** 2026-08-12

**Task:** TASK-2061

**Governing decision:** ADR-025, Shared STT Artifacts and Runtime Routing

## Context

The managed-model browser already deletes through `ModelArtifactService.delete()`.
That service takes an exclusive artifact lease and reports `ArtifactInUseError`
instead of bypassing a shared lease. A local STT worker intentionally keeps its
verified model closure leased while the native runtime remains resident, including
while the worker is idle.

This is correct for active work, but an idle resident can indefinitely block a
confirmed browser deletion. The artifact service cannot solve that problem because
the app-owned `LocalSTTExecutor`, not the store, owns worker lifetime.

## Goals

- Let one confirmed deletion ask an exact idle STT resident to unload, then retry
  normal lease-enforced deletion once.
- Never retire or cancel an active STT request.
- Match every exact artifact leased by the resident, including dependencies.
- Show idle-unload progress separately from a final hard blocker.
- Keep paths, exception details, and process details out of user-visible text.

## Non-goals

- No lease bypass, forced deletion, or active-job cancellation.
- No idle timeout, global lease-owner registry, or generic worker framework.
- No change to artifact-service deletion semantics.
- No automatic executor creation from the model browser.
- No second confirmation after the existing permanent-delete confirmation.

## Decision

The browser keeps ownership of the delete flow. It first calls the existing
`ModelArtifactService.delete()`. Only an `ArtifactInUseError` permits one recovery
attempt through an app-injected callback. The callback consults an already-created
`LocalSTTExecutor`; it never constructs one.

The executor exposes one narrow operation accepting a canonical managed artifact
key `(artifact_id, revision, variant)` and returning whether a matching idle
resident was safely retired. The operation runs under the executor's existing lock
and succeeds only when:

- the executor has a live resident generation;
- the worker has confirmed that the exact key is in its held managed lease set;
- no request is active;
- no retirement or shutdown is underway; and
- bounded worker-tree retirement proves the generation dead.

The operation reuses the executor's existing idle-retirement path. It does not call
`cancel()` or `force_stop()`. The app snapshots an existing executor reference
under its ownership lock, releases that lock, and only then calls the executor;
deletion therefore cannot introduce app-lock/executor-lock inversion.

## Worker-confirmed lease set

`ExecutorResident` will include canonical `managed_lease_refs`. The child derives
these references from the verified handle that owns the live leases:

- managed roots report `ArtifactHandle.closure`;
- external roots report `ArtifactDependencyHandle.references`; and
- unmanaged runtimes report an empty tuple.

The parent records the references only from a matching `ExecutorResident` envelope.
They remain valid for the resident generation, including across ordinary terminal
request failures that leave the runtime alive. They are cleared whenever the
generation starts, detaches, exits, recycles, or closes.

Reporting the verified closure is necessary because a resident managed root leases
both itself and its declared dependencies. Request fields alone cannot prove that a
dependency such as Silero VAD is held by that resident.

## Delete flow

1. The user accepts the existing delete confirmation.
2. `InstalledView` starts its existing off-event-loop deletion worker.
3. The worker calls `ModelArtifactService.delete(reference)`.
4. Any result other than `ArtifactInUseError` completes through the existing success
   or sanitized failure handling.
5. On `ArtifactInUseError`, the row renders `Checking for an idle model to unload…`.
6. The injected app callback asks the existing executor to recycle the exact key.
7. If the callback refuses or cannot prove retirement, the UI reports the existing
   hard blocker and does not retry.
8. After proven retirement, the row renders
   `Idle model unloaded; retrying deletion…`.
9. The source-policy deletion guard is checked again on the UI thread. A newly
   configured external source can therefore still preserve its required VAD.
10. If policy still permits deletion, the worker calls the same service deletion
    method exactly once more.

The first recoverable lease failure is not logged as an error. A final failure is
logged with the existing bounded artifact identity fields and rendered through
sanitized copy.

## Concurrency and failure behavior

The executor lock makes idle detection and worker detachment atomic with executor
submission. If an active request holds the slot, recycle returns false without
changing cancellation state, callbacks, generation, or process ownership.

Normal artifact leases remain the final concurrency authority. A new request or
another process can acquire the artifact after retirement and before the one retry.
If that happens, deletion remains blocked; the new work is never cancelled.

If process-tree death cannot be proven, the existing retirement path marks the
executor unavailable and retains scratch safety. Deletion remains blocked.

Shutdown is safe in either order: app shutdown may detach and close the executor
first, causing recycle to refuse, or executor retirement may finish before shutdown
continues. Both paths serialize on existing locks.

## UI states

The installed row keeps all lifecycle controls disabled throughout the confirmed
operation. Its operation text distinguishes:

- checking/requesting idle unload;
- proven idle unload and delete retry; and
- final hard-blocked notification.

The status uses only catalog identity and static copy. No local path, PID, lease
path, exception message, or provider detail is rendered.

## Verification

Focused tests will prove:

- worker envelopes report the full verified root/dependency lease set;
- idle root and dependency matches retire the worker and release their leases;
- active and nonmatching residents are unchanged;
- unproven death cannot authorize retry;
- deletion performs at most one recycle and one retry;
- policy is rechecked after retirement;
- mounted UI paints distinct checking/retrying and hard-blocked states;
- host wiring uses an existing app executor without lazy construction; and
- all new user-visible and logged recovery paths remain path-private.

Mutation checks will independently remove the active-work guard, exact-reference
match, worker-confirmed closure report, and one-retry fence. Each mutation must make
its focused regression test fail before restoration.

## ADR check

**ADR required:** no

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already assigns resident model leases and generation recycling
to the app-owned local STT executor. TASK-2061 completes the deletion integration
explicitly deferred by the TASK-596 browser design without changing that boundary.
