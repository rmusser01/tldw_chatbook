# TASK-762 Console RAG no-service offline repair

## Goal

Make the existing Console no-service regression exercise the actual missing-service
boundary and complete deterministically without constructing an embedding runtime or
accessing the network.

## Implementation

1. Explicitly remove the app-owned Library RAG search service in the no-service test.
2. Install a recording sentinel at the shared RAG factory boundary and assert it is
   not reached.
3. Wait for the final blocked UI state instead of an intermediate live-work card.
4. Keep the configured-service regression unchanged as the compatibility guard.
5. Run only the exact regression and focused Console RAG checks.

ADR required: no

ADR path: N/A

Reason: This is a stale-fixture correction against an existing service boundary. It
does not change application architecture, ownership, storage, or service contracts.
