# ADR-080: Trace v2 Exhaustive Event Projection and Collaboration Contract

- Status: Accepted
- Date: 2026-08-22
- Spec: `Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md`
- Related: ADR-031, ADR-066, ADR-067; TASK-19907, TASK-19908, TASK-19910–TASK-19913

## Context

ADR-066 deliberately chose a v1 active-path trajectory assembled from normalized
message history plus a local trajectory sidecar, and rejected a duplicate append-only
all-events table. ADR-067 defined a redacted-by-default v1 JSON export. The implemented
screen is useful but does not expose every agent/runtime event, durable child lineage,
approval/context causality, accessible narrow-terminal details, or a first-class export
workflow.

The approved product requirement is now: every observable event, Trace as the
canonical label, both usability and lineage in scope, and privacy-safe collaboration.
This changes the v1 completeness and sharing contract while retaining its privacy,
read-only, local-ownership, and legacy compatibility guarantees.

## Decision

1. **Canonical label.** User-facing surfaces say Trace. Existing trajectory module,
   class, table, and v1 format names remain where renaming provides no behavior.
2. **Exhaustive means observable.** The contract includes conversation, model, tool,
   approval, retrieval/context, compaction, feedback, agent lifecycle, steering,
   cancellation, retry, and failure transitions observed at Chatbook-owned seams. It
   excludes hidden chain-of-thought and provider-internal activity Chatbook cannot see.
3. **Keep normalized owners; add one projection.** We do not add a second all-events
   database. The pure trajectory projection gains adapters for messages/trajectory
   sidecar, AgentRunsDB runs and append-only steps, compaction/context repositories,
   retrieval provenance, and feedback. It outputs a deterministic causal Trace stream.
4. **Persist agent steps incrementally and idempotently.** `agent_run_steps` is already
   append-only. A UTC wall-clock distinct from the monotonic budget clock stamps each
   step; its runtime index is the durable per-run source sequence. The existing
   `on_step` service seam inserts by that explicit index. Terminal recovery inserts any
   missing indices with conflict-safe semantics, so failed live writes are recovered
   without duplicating successful writes. Existing legacy step blobs remain readable.
5. **Stable causal envelope.** Each projected event has a stable source-derived ID,
   conversation ID, kind/status, immutable owner `source_seq` distinct from display
   position, observed timestamps, actor/run/turn identity,
   parent/source/replacement links, safe summary, structured payload, sensitivity,
   and explicit per-field missing/redacted/truncated/omitted/capture-failed states.
6. **Deterministic order without false serialization.** Source order is authoritative;
   causal parents precede descendants. Concurrent unrelated events use observed time
   and stable ID as deterministic tie-breakers and remain visibly concurrent.
7. **Best-effort capture.** Trace instrumentation never fails a user run. Failures are
   logged with context and projected as incomplete diagnostics where a safe owner is
   still available.
8. **Local-only ownership.** No Trace or AgentRuns data enters sync triggers or sync
   payloads. Collaboration occurs only through explicit export.
9. **Trace format v2.** A self-contained JSON event bundle carries a manifest,
   normalized events, lineage, missing-data/redaction provenance, and SHA-256 digest.
   Export profiles are safe summary, redacted diagnostic (default), and explicit full.
   Credentials are forbidden in every profile. The digest is integrity, not identity.
10. **Read-only import.** V2 imports are ephemeral `READ-ONLY SHARED TRACE` screens,
    write no local domain/trace rows, retain v1 reader compatibility, and fail closed on
    unsupported versions or digest mismatch.
11. **Durable spawn correlation.** AgentRuns schema v14 adds nullable
    `spawn_event_id`. Parent spawn steps allocate their stable event ID before dispatch;
    child `create_run` stores it in fleet and inline paths. `parent_run_id` remains run
    lineage, while `spawn_event_id` identifies the precise causal event.
12. **Trace operations travel with collaboration state.** V2 export appends a synthetic
    `trace_export` operation event containing profile and privacy counts; per-field
    states record redaction/truncation/omission. Import returns snapshot plus manifest,
    integrity/privacy metadata, and an ephemeral `trace_import` operation event for the
    read-only screen; none are persisted locally.

## Compatibility with ADR-066 and ADR-067

- ADR-066's local-only sidecar, TOOL-marker invariant, nullable timing, pure projection,
  active-path legacy behavior, and read-only screen remain accepted.
- ADR-066's v1 event coverage is superseded by the exhaustive adapter contract above.
- ADR-067's redaction default, explicit full opt-in, atomic single-file write, and
  read-only import remain accepted.
- ADR-067's version-1 document shape remains supported for reading; v2 writes use the
  new Trace bundle contract.

## Alternatives considered

### New `trace_events` database/table containing every event

Rejected. It would duplicate messages, trajectory rows, agent steps, compaction, and
retrieval evidence; require dual writes at every seam; and introduce drift/recovery
questions. The existing append-only agent step table plus a pure multi-owner projection
provides exhaustive coverage with less code and preserves source ownership.

### Continue adding only `event_kind` rows to `message_trajectory_metadata`

Rejected as the sole mechanism. Its non-null message foreign key is a poor owner for
run lifecycle, approvals before persistence, and child-agent events. Those facts
already have better durable owners in AgentRunsDB and runtime repositories.

### Export raw source tables in v2

Rejected. It would leak internal schemas, make collaboration depend on database
migrations, and complicate privacy review. Export the normalized event contract while
retaining source IDs and provenance.

### Cryptographically sign trace files

Deferred. There is no product key-management or identity contract. A stdlib SHA-256
digest provides corruption/tamper detection without falsely claiming authorship.

## Consequences

- The projection has more adapters but stays pure and testable.
- Agent steps survive mid-run crashes and become live Trace inputs.
- Exhaustiveness is auditable through a documented event-family matrix and integration
  tests, rather than a claim based on row counts.
- Legacy and shared v1 traces remain viewable.
- Collaboration files can be meaningfully reviewed without importing data into the
  user's database.
- No ChaChaNotes schema bump is required by the foundation, avoiding conflict with
  concurrent schema work and a needless duplicate store.
