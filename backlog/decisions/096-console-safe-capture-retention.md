# ADR-096: Bound Safe exchange-capture history

Status: Accepted

Date: 2026-08-27

Related Task: [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md)

Related Spec: [Console Safe Capture Retention](../../Docs/superpowers/specs/2026-08-27-console-safe-capture-retention-design.md)

Amends: [ADR-092](092-console-full-semantic-capture-policy.md), which deferred
automatic pruning and retention policy. ADR-092 remains authoritative for capture
scope, Full consent and semantics, invariant credential/binary protections, export,
and conversation-scoped Full purge.

## Context

Safe exchange capture is enabled by default and currently persists the complete
accumulated provider message list on every call. A production-shaped 200-turn probe
retained 15.40 MB for one conversation because each later exchange repeated every
earlier message. Soft-deleted conversations retain these local blobs, and the only
existing purge intentionally targets Full rows.

This is both a storage and privacy-retention defect. Fixing it changes durable data,
migration behavior, and the meaning of Safe capture, so the decision must amend
ADR-092 rather than remain a local implementation detail.

## Decision

1. **Bound Safe history at capture time.** Safe retains the first system row, the
   most recent user row, and the final eight physical message rows, deduplicated and
   kept in original order.
2. **Use one content-free aggregate marker.** All other rows are represented by one
   versioned capture marker containing only original/omitted row counts, normalized
   role counts, and retained original positions. It contains no content, length,
   digest, identifier, timestamp, or reconstructable reference.
3. **Make elision visible and idempotent.** Generic capture inventories
   `messages_payload.history`; llama.cpp inventories
   `wire_payload.messages.history`. Re-projecting a stored Safe request reaches a
   fixed point rather than nesting markers or changing omission entries.
4. **Preserve protection order.** Credential exclusion, endpoint canonicalization,
   Safe project-instruction redaction, nested-credential removal, and binary stubbing
   occur before compaction; the existing shared size budget remains afterward.
5. **Apply one policy to both request shapes.** Generic `messages_payload` and the
   llama.cpp literal `wire_payload.messages` use the same pure compactor for primary
   and fallback calls.
6. **Leave Full semantics unchanged.** Full bypasses history compaction and remains
   the explicit mode for exact semantic diagnostic context within ADR-092's invariant
   protections. This decision adds no automatic Full retention or purge behavior.
7. **Rewrite existing Safe rows automatically.** A versioned ChaChaNotes data
   migration keyset-pages Safe `message_exchanges` in bounded batches and rewrites
   only captures changed by the same compactor. Full, small, and already-compacted
   blobs remain byte-identical.
8. **Fail narrowly and transactionally.** Recognized corrupt/unavailable Safe blobs
   are left untouched and counted without content-bearing logs. Unexpected code or
   SQLite errors roll back the entire migration and version stamp. Capture remains
   best-effort and cannot block a live provider call.
9. **Keep ownership local.** Compaction changes only `message_exchanges` capture
   bodies and their export projection. It adds no transcript, sync, FTS, Trace, or
   provider-request mutation.

## Consequences

- Safe per-call history becomes bounded and cumulative exchange storage becomes
  linear in call count rather than quadratic in conversation length.
- Safe answers what initial system context, latest user request, and immediate
  assistant/tool loop surrounded a call, but it is no longer an exact historical
  record. The marker and `omitted_keys` state this honestly.
- Readable historical Safe blobs are compacted on database open without a manual
  purge. The one-time migration adds startup work proportional to Safe exchange row
  count but holds only a bounded batch in memory.
- Corrupt or unavailable legacy blobs cannot be safely rewritten and may remain
  oversized. The migration reports only aggregate skip evidence and does not claim
  forensic erasure from SQLite free pages, WAL, backups, or exports.
- Full remains potentially sensitive and storage-heavy by deliberate user choice.
  Its existing consent, queryable provenance, and scoped logical purge remain the
  governing controls.

## Alternatives considered

### Retain one fingerprint per omitted row

Rejected. Metadata would still grow with full history on every call, leaving
quadratic cumulative growth, while hashes could confirm guesses about omitted text.

### Persist deltas or cross-exchange references

Rejected. Reconstruction, deletion, corruption, and export semantics would add a
second exact-history system when Full already serves that explicit diagnostic need.

### Retain only a newest-row tail

Rejected. Tool loops can push the current user request outside the tail, while the
initial system row remains important to explain provider behavior.

### Add timed deletion or a background pruning worker

Deferred. Bounded write-time Safe capture plus a one-time transactional rewrite fixes
the demonstrated defect without a scheduler or new retention-policy surface.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-27-console-safe-capture-retention-design.md)
- [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md)
- [ADR-092](092-console-full-semantic-capture-policy.md)
