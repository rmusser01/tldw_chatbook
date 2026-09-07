# Console Safe Capture Retention Design

**Date:** 2026-08-27

**Status:** Design approved; written-spec review pending

**Task:** [TASK-23026](../../../backlog/tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md)

**Decision:** [ADR-096](../../../backlog/decisions/096-console-safe-capture-retention.md)

**Amends:** [ADR-092](../../../backlog/decisions/092-console-full-semantic-capture-policy.md)

## Summary

Safe exchange capture currently stores the full provider message history again on every
provider call. A 200-turn production-shaped probe grew from 2.8 KB at turn 1 to 145.4 KB
at turn 200 and retained 15.40 MB for one conversation. The default capture mode is Safe,
so this is an automatic local storage and privacy cost rather than an opt-in diagnostic
cost.

Safe capture will instead retain a bounded diagnostic excerpt: the first system row, the
most recent user row, and the newest eight rows. All other message rows are replaced by one
content-free aggregate marker. Full capture remains the explicit, consent-gated mode for
exact semantic history within ADR-092's credential, binary, and size protections.

A versioned, atomic data migration applies the same compaction to existing Safe exchange
rows. This reclaims legacy growth without requiring users to discover a purge action.

## Goals

- Make each Safe request capture bounded by the current diagnostic excerpt rather than by
  the conversation's full age.
- Preserve enough context to identify the original system framing, current user request,
  and immediate assistant/tool loop around the provider call.
- Make elision explicit in both the captured request and `omitted_keys`.
- Apply the same policy to generic `messages_payload` and llama.cpp's literal
  `wire_payload.messages` capture.
- Reclaim already-persisted Safe captures automatically and transactionally.
- Leave Full capture semantics, capture scope, and explicit Full purge behavior unchanged.

## Non-goals

- Reconstructing omitted Safe history from references, deltas, hashes, other exchanges,
  transcripts, or messages.
- Adding timed retention, a background pruning service, or a new purge control.
- Changing provider requests, model context construction, transcripts, Trace events,
  message sync, or FTS ownership.
- Changing the Inspector layout or export-profile controls.
- Promising forensic erasure from SQLite free pages, WAL files, backups, or prior exports.

## Current ownership and failure

`ConsoleProviderGateway` builds semantic request captures at the provider boundary.
`build_request_capture` allowlists fields, canonicalizes endpoint identity, removes nested
credential fields, redacts automatic project-instruction bodies in Safe mode, stubs binary
content, and applies the shared capture budget. The resulting `ExchangeCapture` is compressed
and stored in local-only `message_exchanges` rows.

The generic path places the entire accumulated conversation in `messages_payload`. The
llama.cpp path separately copies the literal wire request under `wire_payload`, including its
own full `messages` list. Because every later call contains all earlier messages, retaining
those lists in every exchange produces quadratic cumulative storage.

Exchange bodies have one durable application owner: `message_exchanges`. They do not enter
the message transcript, sync log, FTS tables, or Trace trajectory ledger. Inspector export
projects the stored exchange and cannot recover data that capture omitted.

## Safe retention contract

### Retained rows

For a list with original zero-based positions, Safe capture retains the union of:

1. the first mapping row whose `role` is exactly `system`;
2. the last mapping row whose `role` is exactly `user`; and
3. the final eight physical rows in the list.

Overlapping selections are deduplicated. Retained rows remain in their original relative
order and keep their already-sanitized values. Non-mapping rows are eligible only through
the eight-row tail. If every row is retained, the list is unchanged and no history-elision
entry is added.

This contract preserves the three debugging facts Safe capture is expected to answer:

- what initial system framing was in effect;
- what current user request drove the call; and
- what recent assistant/tool interaction immediately preceded and followed that request.

It does not claim to preserve the entire causal history. Users who deliberately need that
evidence must choose Full before the send.

### One aggregate marker

When rows are omitted, the capture inserts exactly one internal marker at the position of
the first omitted row. Other retained rows remain ordered as above. Because the omitted
rows can be non-contiguous, the marker explicitly summarizes all omitted rows rather than
pretending to represent only one contiguous gap.

The marker has a versioned, capture-owned shape and contains only:

- an internal kind/version discriminator;
- the original row count;
- the omitted row count;
- normalized omitted-role counts for `system`, `user`, `assistant`, `tool`, and `other`;
- the original positions of retained rows.

Unknown, missing, or non-string roles contribute only to `other`; their raw values are not
retained. The marker contains no content, snippets, content lengths, hashes, IDs, timestamps,
or reconstructable references. In particular, Safe capture does not retain a digest that
could be used to test guesses about omitted private text.

The helper recognizes its own valid versioned marker and produces a fixed point when a
stored Safe request is projected through the capture builder again. An input marker never
disables compaction of surrounding rows. Malformed lookalikes are ordinary rows and remain
subject to the normal bounded selection.

### Ordering with existing protections

Safe request processing occurs in this order:

1. reject unknown top-level kwargs and inventory them in `omitted_keys`;
2. canonicalize credential-free endpoint identity;
3. redact tagged automatic project-instruction bodies;
4. remove nested structured credential fields and stub binary/base64 values;
5. compact the sanitized message history;
6. apply the existing shared uncompressed capture budget.

This ordering prevents the aggregate marker from describing raw secret or binary values and
keeps project-instruction bodies outside Safe storage. Existing project-instruction redaction
metadata remains visible even if its marker row is later among the omitted history.

Generic history elision adds the stable path `messages_payload.history` to `omitted_keys`.
llama.cpp history elision adds `wire_payload.messages.history`. The aggregate marker carries
the counts; the omission inventory carries only the stable path, so repeated projection does
not create duplicate or ever-changing strings.

### Full capture

The compactor is never invoked for `CaptureDetail.FULL`. Full continues to preserve semantic
message text exactly as ADR-092 defines while still enforcing its invariant protections:
unknown kwargs and structured credentials remain excluded, raw binary remains stubbed,
endpoints remain credential-free, and capture/decompression budgets remain bounded.

This task does not add automatic retention or pruning for Full. Full is deliberately enabled,
clearly consented, queryable by provenance, and already has conversation-scoped logical purge.

## llama.cpp parity

The existing llama.cpp capture closure owns a literal `wire_payload` outside the generic
allowlist builder. It must call the same pure list compactor after its existing Safe project-
instruction handling and sanitization. Both the streaming request and the stream-to-complete
fallback request use that helper and add `wire_payload.messages.history` when history is
elided. Full literal wire capture bypasses compaction.

No second retention algorithm or provider-specific tail size is introduced.

## Existing-data migration

ChaChaNotes advances from schema v52 to v53, subject to a final version-head recheck before
implementation lands. The step changes no table shape; its version records completion of a
one-time Safe capture data rewrite.

Within the migration chain's existing outer immediate transaction, the step:

1. keyset-pages `message_exchanges` rows whose queryable `capture_detail` is `safe` in bounded
   batches, without loading every row ID or blob into memory;
2. safely decodes each blob through the production decoder;
3. applies the same pure Safe request compactor to generic and llama.cpp stored shapes;
4. merges stable history-elision paths into the stored `omitted_keys` inventory;
5. rewrites only blobs whose decoded capture changed; and
6. advances the schema version atomically with the data changes.

Full rows, small Safe rows, and already-compacted Safe rows remain byte-identical. A recognized
oversize, corrupt, or unavailable blob is left byte-identical and counted as skipped; the
schema version may advance because retrying the same unreadable data cannot reclaim it. Only
the existing `CaptureUnavailableError`/`CaptureCorruptError` family is a per-row skip. Any
unexpected programming or SQLite error aborts the migration, rolls back all rewrites and the
version stamp, and is retryable on the next open.

Migration diagnostics report aggregate examined/changed/skipped counts and exception class at
most. They never log capture bodies, user text, blob bytes, message identifiers, or exception
values that may contain decoded content.

The synchronous step is intentionally simpler than a background retention worker. Its runtime
and peak-memory behavior will be measured on the historical 200-turn fixture. A separate
background design is warranted only if that evidence shows the bounded batched migration is
not acceptable.

## Runtime failure behavior

Compaction is a pure transformation over JSON-compatible capture values. Non-list message
fields keep the current defensive pass-through behavior. Unexpected capture-construction
failure remains best-effort: it can suppress capture for that provider call but must not block
or mutate the live provider request. Logs remain content-free.

## Inspector and export behavior

No new screen is required. The Exchange tab renders the stored request, so it naturally shows
the aggregate marker in place of omitted history. Its existing “Omitted by capture policy” line
shows the stable history path alongside credential, project-instruction, and truncation entries.

Safe Summary and Redacted Diagnostic export can expose only the compacted Safe representation.
Re-running the Safe builder is idempotent. Full Trace export of a Full capture remains unchanged;
Safe export cannot join other exchanges or transcripts to reconstruct omitted rows.

## Verification contract

Implementation follows test-driven development. Focused verification must include:

- pure helper selection, marker contents, normalized role counts, ordering, and fixed-point
  idempotency;
- first-system and latest-user retention when each falls outside the final eight rows;
- long assistant/tool loops and malformed/non-list inputs;
- absence of omitted body fragments, body lengths, hashes, raw unknown roles, and credentials;
- unchanged Full capture behavior;
- identical Safe behavior for generic and both llama.cpp capture paths;
- a production-shaped `ConsoleProviderGateway.stream_chat` probe across 200 growing turns,
  demonstrating that per-call retained history plateaus and cumulative capture growth is linear,
  with the measured sizes recorded in TASK-23026 implementation notes;
- a real historical v52 fixture proving automatic rewrite, unchanged Full/small/already-compact
  blobs, corrupt-blob skip, re-entry, rollback on unexpected failure, bounded batching, and
  `PRAGMA integrity_check`;
- privacy assertions across durable DB blobs, loaded in-memory captures, Inspector/export
  projections, and captured logs; and
- the complete ChaChaNotes migration test suite because the schema head changes.

A bespoke SIGKILL harness is not required. Transaction rollback, reopen/re-entry, and integrity
tests directly prove the failure property without introducing process-control machinery.

## Documentation

The Console user guide will state:

- Safe keeps the initial system context, latest user request, and newest eight message rows, with
  a content-free aggregate marker for older context;
- Safe history omitted before capture cannot be recovered by Inspector export;
- Full is the explicit choice for exact semantic diagnostic history and retains ADR-092's
  privacy warning and scoped purge boundaries; and
- migration automatically compacts readable legacy Safe exchange rows but cannot rewrite
  corrupt/unavailable records or erase prior backups and exports.

## Alternatives considered

### One fingerprint row per omitted message

Rejected. It still grows one metadata row per historical message, remains quadratic across all
turn captures, and content digests create a guess-verification oracle for omitted private text.

### Store deltas or references between exchanges

Rejected for Safe. It adds reference integrity, deletion, export, corruption, and reconstruction
semantics to preserve exact history that the explicit Full mode already provides.

### Keep only the newest fixed tail

Rejected. Long tool loops can push the current user request outside a small tail, and the initial
system framing remains valuable diagnostic context.

### Timed deletion or background pruning

Deferred. The immediate defect is repeated full-history storage. A bounded write-time contract
plus one transactional rewrite fixes both future and existing Safe growth without a scheduler or
new user-facing retention policy.
