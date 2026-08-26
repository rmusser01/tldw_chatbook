# ADR-089: Scoped Full semantic capture policy

Status: Proposed

Date: 2026-08-26

Related Task: [TASK-22507](../tasks/task-22507%20-%20Enable-scoped-Full-semantic-capture-in-Conversation-Inspector.md)

Related Spec: [Console Full Semantic Capture](../../Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md)

Amends: ADR-069's unconditional exclusion of project-instruction bodies from durable
exchange captures; all other ADR-069 boundaries remain accepted.

Clarifies: ADR-066's Trace ledger remains read-only. A live Trace screen may host a
control for future exchange-capture policy without mutating trajectory events; imported
Traces remain fully read-only.

## Context

The Console Conversation Inspector persists allowlisted provider exchanges at the
provider-adapter boundary. ADR-069 subsequently required automatic project-instruction
bodies to be removed from durable exchange captures, logs, transcripts, summaries, and
tool results. This is a safe default, but it also means the historical Inspector cannot
show all semantic context that caused a provider response.

The owner requires deliberate Full capture at three scopes: the next eligible send,
one inspected conversation, and the global Console default. Full capture must include
Anthropic system content, injected project/workspace/RAG context, and tool traffic;
remain bounded and credential-safe; expose honest retention; and support deletion of
Full records without deleting the conversation.

This changes privacy, provider/runtime boundaries, local schema, deletion semantics,
and the long-lived Inspector contract, so it requires a new ADR rather than an edit to
accepted ADR-069.

## Decision

1. **Safe remains the application default.** Capture detail is `safe` or `full`. The
   existing `[console] exchange_capture` Boolean remains the authoritative kill switch.
2. **Use scoped precedence.** An admitted human-authored manual or authorized queued
   turn resolves next-send override, conversation override, global default, then Safe.
   Autonomous wakeups do not consume or use the one-shot override.
3. **Freeze at admission.** The resolved detail is stored on the run's existing
   `ConsoleProviderStreamSignals`. Every provider call, retry, tool loop, and surviving
   fleet call on that run uses the same detail despite later settings changes.
4. **Capture semantic adapter content, not generic wire bytes.** Full retains
   allowlisted content handed to provider adapters and observable responses, including
   Anthropic system/messages/tools, project/workspace instructions, RAG context, tool
   schemas/calls/results, and response content. Provider-internal framing remains out
   of scope except where Chatbook already owns a literal payload such as llama.cpp.
5. **Structural protections are invariant.** Full never admits structured provider
   credentials or unknown kwargs, never persists raw binary/base64 bodies, never
   bypasses bounded request/stream accumulation, serialization, compression, or
   decompression limits, and never allows capture bodies into logs or raw exception
   diagnostics. Request and response tool payloads share the same binary-stubbing
   boundary. Endpoint fields reuse the existing canonical credential-free destination
   identity, excluding URL userinfo, query, and fragment. Secrets embedded inside
   allowed semantic text may be stored and are named in consent copy.
6. **Amend ADR-069 narrowly.** Automatic project-instruction bodies may enter only an
   explicitly Full durable exchange capture. They remain excluded from Safe captures,
   transcripts, agent steps, tool results, compaction summaries, metadata surfaces,
   ordinary logs, and permission/authority decisions. Full capture grants no workspace
   or tool permission.
7. **Keep ownership local and existing-pipeline based.** Global detail uses canonical
   Console config. Sparse conversation detail uses an optional local-only policy row;
   absence means inherit and a present row is checked Safe/Full. One-shot detail is
   process memory. Exchange content remains in
   `message_exchanges`; no new Trace event store or sync payload is added.
8. **Persist queryable provenance.** Each exchange stores `capture_detail` in its
   serialized capture and a checked queryable `message_exchanges` column, both derived
   from one immutable capture. A mismatch is corrupt, not guessed. Historical records
   default to Safe because the prior builder always applied Safe redaction.
9. **Separate capture from export.** Stored detail is Safe/Full. Export profiles are
   the existing `TraceExportProfile` values Safe summary, Redacted diagnostic, and Full
   trace; exchange-specific projection is separate from trajectory projection. Export
   cannot reconstruct content omitted at capture time, and every Full clipboard/file
   disclosure is confirmed.
10. **Provide scoped erasure under capture quiescence.** A conversation may delete only
    its Full exchange rows after the controller holds a lease preventing new admission
    and flush, with no active primary run, surviving/unsettled fleet writer, retained
    run signals, or exchange flush. Replacement in-memory/cache collections are staged
    before the SQLite transaction and swapped by reference immediately after commit,
    before releasing the lease, so a later flush cannot recreate deleted rows. Scope is
    the immutable conversation ID across active, off-path, abandoned, and soft-deleted
    messages, never the mounted transcript. A capture revision blocks stale Inspector
    expansion/export after purge. Purge does not change capture policy or delete Safe
    captures, messages, usage, exports, or backups.
11. **Classify mutations by resulting disclosure.** Each Apply changes one scope.
    Inherit or disarm is an escalation when it reveals an underlying Full policy, and
    global Full is always an escalation. Any Full-enabling result activates only after
    required confirmation and persistence succeeds. A result that remains or becomes
    Safe takes effect in memory even if its durable write fails and remains visibly
    unsaved. Unknown values resolve Safe. Any purge failure before SQLite commit leaves
    both durable and in-memory owners unchanged; post-commit UI refresh cannot recreate
    deleted records.
12. **Inspector and live Trace controls target immutable identity.** The shared policy
    surface acts on the conversation/session captured when its parent screen opened,
    carries process-local revisions against stale modals, and distinguishes an active
    run's frozen detail from future policy. Imported/shared Traces remain read-only and
    expose no live capture control.
13. **Use the canonical runtime config projection and structured write result.** Safe
    global changes publish to the app's shared config projection before persistence;
    Full publishes only after atomic file replacement. A successful replacement with a
    failed general cache refresh remains honestly saved and active in that projection.
    F9 Settings, the Inspector, and the live Trace screen share this path, confirmation,
    and config-generation fence; none owns a shadow value.
14. **Keep Off dormant and make resume explicit.** The kill switch disarms live
    one-shots but does not silently rewrite persistent conversation/global detail.
    Status names dormant Full policy, and a first-party Off-to-On change warns before
    any Full policy resumes. Full-enabling policy edits are unavailable while Off.

## Alternatives considered

### Keep Safe-only durable capture

Rejected. It preserves the strongest default but cannot diagnose behavior caused by
automatic instructions, retrieval context, or tool traffic that the provider actually
received.

### Capture literal HTTP requests and responses

Rejected. Provider adapters own different transport layers, credentials and headers
would create a larger secret boundary, streaming framing is provider-specific, and the
existing semantic gateway seam already captures the user-relevant request contract.

### Add a separate Full trace database or event stream

Rejected. It duplicates exchange ownership, adds dual-write/recovery drift, and is
unnecessary because `message_exchanges` already owns per-call request/response history.

### Use one global Full toggle

Rejected. It makes a high-risk diagnostic mode easy to leave enabled and cannot support
the approved least-duration next-send and conversation use cases.

### Store detail only inside compressed blobs and scan them for purge

Rejected. Counting/deleting would require decoding every blob, corrupt records would be
unclassifiable, and a privacy erasure action could silently leave content behind.

### Make purge also set policy to Safe

Rejected by owner choice. Deletion and future recording are separate actions; combining
them would make a destructive action silently mutate configuration.

### Claim scoped purge is forensic secure erasure

Rejected. SQLite row deletion cannot promise immediate removal from WAL/free pages,
filesystem snapshots, exports, or backups. The scoped action is honest logical record
deletion; a whole-database sanitization workflow has different availability and
data-loss consequences and requires a separate decision.

## Consequences

- Full records may contain highly sensitive local semantic text and can propagate into
  user-created exports and backups. Capture blobs are compressed, not additionally
  encrypted, and scoped purge is not forensic erasure of WAL/free pages. Consent and
  documentation must state those boundaries plainly.
- ChaChaNotes advances from the implementation-time current version through a migration
  adding queryable exchange detail and local conversation capture policy. At design
  time the provisional bump is v49 to v50; implementation must recheck.
- Existing records remain readable and are classified Safe without blob rewriting.
- The Inspector gains a compact status/control flow, per-call provenance, and scoped
  erasure while retaining Costs, Exchange, and Next Send ownership; the live Trace
  screen reuses the same future-capture control without changing Trace projection or
  export ownership.
- The canonical F9 Settings screen and Inspector edit one global source of truth.
- Multiple app processes sharing one profile remain unsupported; within one process,
  revision checks prevent stale Inspector writes.
- ADR-069 remains authoritative for instruction discovery, authority, permission,
  delivery, transcript, metadata, summary, tool-result, and logging boundaries except
  for the explicit Full exchange-capture allowance recorded here.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md)
- [TASK-22507](../tasks/task-22507%20-%20Enable-scoped-Full-semantic-capture-in-Conversation-Inspector.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-066](066-console-trajectory-view-and-trace-metadata.md)
- [Original Conversation Inspector design](../../Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md)
