# ADR-190: Console incremental display projections

Status: Accepted
Date: 2026-09-27
Related Task: [TASK-24300](../tasks/task-24300%20-%20Console-emptiness-checks-deep-copy-the-whole-transcript.md)
Preserves: ADR-088

## Decision

The Console store owns a process-local, constant-time display-projection revision per session. It advances when active transcript content, status, path, or usage may change, including usage attached after a terminal response and streamed text that has not been materialized. It is not a persistence revision or send authority.

The mounted Console screen reuses detached settled history, context token totals, and cost aggregates while the store revision and relevant session settings are unchanged. Draft text contributes separately to context and next-send estimates; typing never rebuilds settled transcript projections. A revision change may rebuild the affected session's projection once. Existing ADR-088 lifecycle filtering and provider serialization boundaries stay in force.

TASK-33081's one-second context estimate bound during active streaming is retained. Its cache key is checked using constant-time store counters and immutable run owners before materializing transcript rows. Draft, payload, settings, and run-owner changes recompute immediately; streamed text alone can remain at the last display estimate until that one-second bound expires. Settled sessions key directly on the display revision, so edits and branch or usage changes invalidate immediately.

## Context

TASK-24300 removed `messages_for_session` calls used as emptiness predicates, but the mounted settings and cost paths still build full transcript snapshots and walk them on every composer edit. Row token memoization avoids repeated tokenization but still performs O(N) snapshot, filtering, and aggregation work. `payload_revision` alone cannot invalidate spend because terminal usage may attach without changing provider payload.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Cache against `payload_revision` only | Late terminal usage leaves the cost chip stale. |
| Compare every row on each display tick | Exact but retains O(N) work per key. |
| Cache serialized provider history | Holds large media payloads and violates ADR-088's presentation boundary. |

## Consequences

The revision must advance at every store mutation relevant to display projections. Focused mutation tests and a mounted work census guard both invalidation and constant-time unchanged-key behavior. A cold rebuild and actual transcript mutation can cost O(N); unchanged typing cannot.
