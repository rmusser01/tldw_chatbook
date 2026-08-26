# Console Full Semantic Capture

**Date:** 2026-08-26

**Status:** Owner-approved; implementation planning complete; implementation not started

**Task:** [TASK-22507](../../../backlog/tasks/task-22507%20-%20Enable-scoped-Full-semantic-capture-in-Conversation-Inspector.md)

**ADR:** [ADR-089](../../../backlog/decisions/089-console-full-semantic-capture-policy.md)

**Builds on:** [Console Conversation Inspector](2026-08-18-console-conversation-inspector-design.md)

**Integrates with:** [Console Trajectory View](2026-08-14-console-trajectory-view-design.md)

**Amends:** the durable exchange-capture exclusion in [ADR-069](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md)

## Summary

The Conversation Inspector currently persists a useful but deliberately incomplete
provider exchange. In particular, automatic project-instruction bodies are removed
before storage. Users therefore cannot always answer why a provider behaved as it did,
even though the Inspector appears to show the request.

Add an explicit, privacy-sensitive **Full semantic capture** policy. A user may apply
Safe or Full detail to the next eligible send, the inspected conversation, or the
global Console default. The default remains Safe. Full capture retains semantic
content actually handed to the provider adapter, including Anthropic system content,
project/workspace instructions, retrieval context, tool schemas, tool calls, and tool
results. It is not generally a byte-literal HTTP recording.

The existing provider gateway, run signals, and exchange store retain data ownership;
the Inspector and live Trace screen project and control that state. This design does
not add a second tracing pipeline or database.

## Design decisions

1. Capture detail has two values: `safe` and `full`.
2. The existing `[console] exchange_capture` Boolean remains the authoritative global
   kill switch. A new global detail default is Safe.
3. Precedence is: next eligible send override, conversation override, global default,
   then application Safe.
4. The one-shot override belongs to one inspected Console session and is consumed only
   by an admitted human-authored manual or authorized queued turn. Autonomous agent
   wakeups, rejected sends, local commands, and cancelled pre-admission work do not
   consume it.
5. Capture detail freezes when the turn is admitted. Every provider call, retry, tool
   loop, and surviving fleet call using that run's signals retains the same detail.
6. Full capture keeps semantic text but still structurally excludes provider
   credentials, stubs binary/base64 payloads, and enforces bounded in-memory,
   serialized, compressed, and decompression limits.
7. Policy controls are available from the Conversation Inspector and live Trace screen
   through one shared, focused change flow rather than persistent scope buttons.
8. Capture detail and export profile are distinct concepts.
9. Users can delete stored Full captures for one quiescent conversation. Purge covers
   every branch and soft-deleted message in that conversation, does not change capture
   policy, and does not delete Safe captures, messages, usage, exported files, or
   backups.
10. Policy changes and purge always target the immutable conversation/session captured
    when the Inspector or live Trace screen opened, never whichever tab is active when
    the user confirms.

## Goals

- Make the Inspector honest about which semantic provider content was retained.
- Let users escalate diagnostic detail for the smallest useful duration.
- Capture injected project/workspace/RAG/tool context for Anthropic and other provider
  paths without admitting structured credentials.
- Keep policy deterministic across queues, autonomous work, retries, and tool loops.
- Provide a scoped erasure path for the new sensitive records.
- Preserve Safe behavior and backward compatibility for existing conversations.

## Non-goals

- Hidden chain-of-thought or provider-internal activity Chatbook cannot observe.
- Literal HTTP headers, provider credentials, transport framing, or TLS traffic.
- Capturing arbitrary auxiliary calls that are not currently owned by conversation
  exchange capture.
- Automatic secret detection inside user/project/tool text. Full mode warns that text
  may itself contain secrets.
- Cross-capture deduplication, retention schedules, automatic pruning, or backup
  deletion.
- Forensic secure erasure of SQLite free pages, WAL files, filesystem snapshots, or
  storage-device remnants. Scoped purge is logical record deletion; whole-database
  sanitization has different availability and data-loss trade-offs.
- A new Trace event store, settings subsystem, permission system, or dependency.
- Editing or replaying trajectory events. The live Trace ledger remains read-only; its
  capture control changes only future exchange policy owned outside the ledger.
- Repairing unrelated historical capture gaps unless a touched provider/retry path
  must carry the frozen detail correctly. Known gaps remain separately tracked.

## Terminology

- **Capture enabled:** whether the existing kill switch permits exchange capture at
  all.
- **Capture detail:** Safe or Full; determines which semantic bodies are retained.
- **Policy source:** next eligible send, conversation, global, or application default.
- **Eligible send:** an admitted manual or authorized queued human-authored turn.
- **Export profile:** Safe summary, Redacted diagnostic, or Full trace; determines what
  leaves the app from an already stored capture.
- **Full semantic capture:** the allowlisted provider-adapter request plus observable
  response content, not necessarily the final provider wire representation.

## Capture policy

### Stored and runtime state

The capture module owns a string enum `CaptureDetail(SAFE, FULL)` and a pure resolver.
It stays beside `ExchangeCapture` in `Chat/console_exchange_capture.py`; a separate
policy framework would add indirection without another consumer.

State is intentionally sparse:

- **Application default:** Safe, constant.
- **Global default:** `[console] exchange_capture_detail = "safe" | "full"` on disk,
  projected into the app's existing shared runtime config. The Inspector does not keep
  a second global shadow value.
- **Conversation override:** optional local-only Safe/Full record keyed by persisted
  conversation ID. No row means inherit global; a stored row is never nullable.
  Ephemeral sessions hold the optional override in memory and flush it if the
  conversation is later promoted. Promotion uses the same mutation failure rules as an
  explicit policy write: a failed Safe persistence over inherited Full remains Safe
  for the process, visibly unsaved, and retryable rather than silently becoming Full
  after promotion.
- **Next eligible send override:** nullable Safe/Full value held only in the live
  session. It is discarded on consumption, explicit disarm, session deletion/close,
  app restart, or observation that the capture kill switch has been turned off.
- **Process-local policy revision:** incremented for every first-party policy mutation.
  Inspector writes carry the revision they observed and reject stale updates from a
  second modal. Global writes also carry the existing atomic config generation so an
  Advanced Settings or F9 write cannot be overwritten by an older Inspector snapshot.
- **Process-local capture revision:** incremented when stored/in-memory exchanges are
  purged. Capture loaders and export actions carry the revision they observed so stale
  Inspector projections cannot disclose a capture after deletion.

Conversation persistence follows the existing sparse local policy pattern but uses a
small capture-policy repository/table rather than adding capture semantics to context
compaction policy. It has no sync columns or triggers.

### Resolution and admission

The controller resolves capture once at the accepted turn-admission boundary. Global
detail is read from the app's shared runtime config projection, not a modal-owned copy:

```text
if capture kill switch is off: disabled
else next eligible override ?? conversation override ?? global default ?? Safe
```

Resolution and one-shot consumption occur under the session's existing admission
serialization. The override is consumed only after the turn has an admitted owner and
frozen provider resolution. A send rejected for readiness, validation, permissions,
queue authority, cancellation, or a local command leaves it armed.

Cancellation after admission does consume the override: the admitted run owned and may
have used the frozen capture policy even if it produced no provider call. Every
rejection or cancellation before admission leaves it armed.

If a queued human turn is already next, the UI says that the next queued send will
consume the override. Queue cancellation does not consume it. An autonomous wakeup
may run first but cannot consume or use the one-shot; it resolves from conversation,
global, and application state.

`ConsoleProviderStreamSignals` carries both `exchange_capture_enabled` and the frozen
`capture_detail`. All call-scoped signal views inherit those values. Changing policy
while a run is active affects only a later admitted run.

### Mutation failure behavior

- Each Apply changes exactly one scope. The dialog displays all three scopes, but it
  does not batch global config and conversation storage into a false cross-store
  transaction.
- Classification uses the **resulting effective detail across the affected scope**, not
  the selected label. Removing a Safe override to Inherit while the inherited value is
  Full is an escalation. Disarming a one-shot Safe override over a Full conversation is
  also an escalation. Global Full is always an escalation because it can affect other
  conversations even when the inspected conversation has a Safe override.
- **Any result that enables Full:** show the confirmation required by that scope,
  persist first where persistence is required, then publish the runtime policy. Failure
  before the required durable replacement leaves the previous detail active and shows
  `Failed — previous policy retained` with Retry. A post-replacement config-cache
  refresh problem follows the explicit global partial-success rule below.
- **Any result that remains or becomes Safe:** publish the privacy-safe in-memory
  result immediately. If the durable write fails, the UI shows
  `Safe for this app session — save failed` and keeps Retry visible. It never claims
  the reduction will survive restart.
- **Global settings:** use the canonical atomic settings mutation and its structured
  result rather than collapsing it to a Boolean. A Safe result is published to the
  shared app-config projection before the disk attempt. A Full result is published only
  after `config.toml` was atomically replaced. If replacement succeeds but the general
  config-cache reload fails, the confirmed value still becomes the app's capture-policy
  runtime projection and the UI reports `Saved and active — settings cache refresh
  degraded`; it never claims that the previous value survived or that restart will
  differ. The Inspector, live Trace screen, and F9 Settings use this same mutation path
  and confirmation contract.
- **Unknown persisted/config values:** resolve to Safe and surface a content-free
  diagnostic. They never enable Full.

### Kill-switch lifecycle

Capture Off makes every detail policy dormant rather than deleting persistent
conversation/global choices. The UI states the dormant effective detail explicitly,
for example `Capture Off · Full resumes if capture is enabled`.

Turning the kill switch off through a first-party surface disarms every live one-shot
under admission serialization. Turning it back on is always confirmed when any Full
policy could resume; the warning need not enumerate sensitive titles or bodies. A
hand-edited config remains authoritative on reload/restart, and any refresh that
observes it must show the resulting active detail rather than silently presenting Safe.
Full-enabling policy changes are disabled while capture is Off; users may first enable
capture through the canonical setting flow.

## Capture content contract

### Safe

Safe preserves the incumbent exchange-capture contract. Automatic
project-instruction rows remain tagged and their bodies are replaced with the existing
omission marker before persistence. Existing allowlisting, omitted-key inventory,
binary stubs, and size cap remain. Response-side binary stubbing and the bounded
accumulation/decompression hardening below apply to both Safe and Full; they reduce
retention risk without admitting any new Safe content.

### Full

Full retains allowlisted semantic bodies that Safe omits, including:

- provider system content, including the Anthropic `system` content supplied at the
  adapter boundary;
- user and assistant message payloads;
- project/workspace instruction riders and lazily activated instruction content;
- staged RAG/retrieval context and source snippets present in the request;
- tool schemas, tool calls, tool-result messages, and observable response tool calls;
- sampling, reasoning, routing, usage, status, and observable response content already
  captured by the gateway.

Full does **not** weaken structural protections:

- request kwargs remain allowlisted; `api_key`, resolved credentials, authorization
  headers, and unknown kwargs are excluded by construction;
- `api_base_url`, URL-shaped `api_endpoint`, and `ExchangeCapture.endpoint` pass through
  the existing canonical provider-endpoint identity used by project-instruction
  destination fingerprints. Userinfo, query, and fragment never persist; an invalid
  endpoint becomes a content-free invalid marker rather than the raw value;
- binary/data-URI/base64 values in both requests and responses, including nested tool
  arguments/results, become deterministic size/hash stubs before an immutable capture
  is published;
- request construction and streaming response accumulation share a per-call 64 MiB
  uncompressed UTF-8 JSON budget. Once reached, capture records an explicit truncation
  inventory and stops retaining additional diagnostic bytes while the provider run
  continues normally;
- the existing compressed cap remains a second guard, and decoding rejects a blob whose
  decompressed JSON exceeds 64 MiB rather than allocating it without a bound. A legacy
  row beyond that safety ceiling is reported unavailable with a content-free reason;
- logs and exception surfaces never carry capture bodies or raw exception values from
  frames containing them.

Chatbook cannot reliably identify secrets typed inside ordinary semantic text. Every
Full confirmation names that limitation.

`capture_blob` compression is not encryption. Full confirmation and documentation say
that allowed semantic bodies are stored in the local ChaChaNotes database and inherit
only the database/filesystem protections already configured by the user.

### Provider boundary

The generic gateway records the prepared semantic kwargs immediately before invoking
the provider adapter and records content/tool calls observed on return. Anthropic's
system/message/tool inputs are therefore included in Full mode even though the adapter
may later transform them into Anthropic-specific wire blocks. Provider-internal prompt
caching markers, headers, or framing remain outside the generic contract. The
llama.cpp branch remains the documented exception where Chatbook owns the literal
payload.

## Persistence and migration

Implementation must re-read `_CURRENT_SCHEMA_VERSION`; it is 49 at design time, so
the provisional migration is v49 to v50.

1. `ExchangeCapture` gains `capture_detail` with a backward-compatible Safe default.
2. `message_exchanges` gains a non-null checked `capture_detail` column defaulting to
   `safe`. New writes derive both the blob and queryable column from the same immutable
   `ExchangeCapture`; reads reject a column/blob detail mismatch as corrupt rather than
   guessing. Existing rows are accurately Safe because the pre-feature builder always
   applied Safe project-instruction redaction.
3. A local-only `console_conversation_capture_policy` table stores a non-null checked
   Safe/Full override by conversation ID and cascades on conversation deletion. Absence
   means inherit; applying Inherit removes the row.
4. No capture or policy row enters sync, FTS, server payloads, conversation metadata,
   or Trace projection by default.

The queryable column is deliberate. Scanning and decoding every opaque capture blob
to count or purge Full records would make deletion slow, make corrupt blobs
unclassifiable, and risk leaving sensitive content behind.

## Scoped purge

The policy dialog shows `Stored Full captures: N` and `Delete Full captures…` for the
inspected conversation.

- Purge acquires a conversation-scoped **capture-quiescence lease** through the existing
  admission/controller serialization. The lease prevents new admissions and exchange
  flushes until purge finishes.
- The action is disabled unless every possible writer is quiescent: no admitted primary
  run, no surviving/unsettled fleet child or retained run signals capable of later
  attachment, and no exchange flush in flight. Its visible reason names the remaining
  owner rather than claiming the conversation is idle too early.
- Confirmation names the conversation, count, irreversibility, unaffected data, and
  the capture policy that will remain active. It calls the operation logical record
  deletion and states that SQLite WAL/free pages, exports, backups, and filesystem
  snapshots are outside the secure-erasure boundary.
- The count/delete query joins by immutable conversation ID and includes exchanges on
  the active path, off-path branches, abandoned regenerations, and soft-deleted
  messages. It never derives scope from the Inspector's currently mounted turn list.
- While holding the lease, the store precomputes replacement message exchange tuples,
  serialized-blob caches, and abandoned-run bookkeeping across the session's complete
  node graph without mutating live state.
- The database then deletes only `message_exchanges.capture_detail = 'full'` rows
  belonging to messages in that conversation, in one ChaChaNotes transaction.
- After the SQLite commit, the store swaps the already-built replacement collections
  by reference and bumps the capture revision before releasing the lease. No decoding,
  allocation, callback, or other fallible work is permitted between durable commit and
  those authoritative swaps. A process crash after commit discards the old memory and
  restarts from the deleted database state.
- Ephemeral sessions use the same lease and staged swap without a database write.
- Any failure before commit leaves durable and in-memory owners unchanged and offers
  Retry. Inspector repaint happens after the authoritative purge; if repaint fails,
  deletion remains successful and the UI offers Refresh rather than pretending the
  records were restored.
- The initiating Inspector clears its loaded call map and mounted Full bodies on
  success. Any other stale Inspector must revalidate the capture revision before body
  expansion, Copy, or Save; a mismatch clears its cached calls and requires Refresh.
  Thus a stale modal cannot export a purged capture even if it missed repaint.
- Exports and backups are outside this deletion boundary and are named in the
  confirmation.

Purge does not mutate next-send, conversation, global, or kill-switch policy.

## Conversation Inspector and live Trace UX

The Inspector receives immutable target identity plus injected policy read/write/count/
purge callbacks. It never resolves `active_session_id` when an action is pressed.

A pinned two-line status region sits below `Conversation Inspector` and above the
existing tabs:

```text
Capture · Safe · “Conversation title” · future calls
Next eligible send: Full (armed) · c Change…
```

Only relevant facts appear. If an active run differs from future policy, the second
line instead says, for example, `Active run continues with Safe · next run uses Full`.
Capture Off, temporary-session scope, applying, failed, and queued-consumer states use
persistent text rather than toast or color alone.

The live `TrajectoryScreen` receives the same immutable target and callbacks. Its title
area adds one compact line, `Future exchange capture: Safe · c Change…`; `c` opens the
same policy modal and its footer advertises the binding. At 80x24 the line remains
visible while the ledger scrolls. An imported/shared Trace has no live conversation
owner, renders `Capture policy unavailable for imported Trace`, and exposes neither the
binding nor a misleading editable control.

`c` opens a scrollable policy modal with fixed Cancel/Apply actions and vertical rows:

1. Next eligible send: Inherit / Safe / Full.
2. This conversation: Inherit / Safe / Full.
3. Global default: Safe / Full.
4. Stored Full captures: count and scoped delete action.

The user selects one scope and value per Apply. The UI previews the resulting effective
detail and whether the change is an escalation. Next-send Full needs no secondary
confirmation. Any conversation change resulting in Full uses a warning that states
target and persistence. Global Full uses a stronger confirmation with an explicit
acknowledgement that it applies to all Console conversations in the current app
configuration and survives restart. Changes whose resulting detail is Safe need no
warning. Duplicate actions are disabled while Applying. Escape safely cancels, and
focus returns to the status control.

F9 Settings presents the same global detail and kill-switch owners, not independent
copies. It uses the same Full and re-enable confirmations, config-generation fence, and
structured partial-write result as the Inspector flow.

At 80x24, policy content scrolls while the status and Cancel/Apply controls remain
reachable. Disabled reasons are visible text, not tooltip-only. Styling uses semantic
tokens and literal Safe/Full/Off text labels.

Each Exchange call title gains compact stored provenance such as `capture: Full`.
Current settings never relabel historical calls.

## Export profiles

Capture detail describes storage; export profile describes disclosure.

Reuse the existing `TraceExportProfile` enum, labels, and confirmation primitive for
Safe summary / Redacted diagnostic / Full trace. Do not add a second near-identical
profile enum. The exchange exporter remains a small source-specific governor because
the trajectory exporter consumes `TrajectorySnapshot` fields and does not understand
project-instruction origin tags or `capture_detail` availability.

These profiles apply to one selected Exchange call, matching the incumbent per-call
Copy/Save ownership. Conversation-wide Trace export remains owned by the Trace screen
and is outside this task. The Trace-screen capture control affects future Exchange
captures; it does not relabel or synthesize missing fields in the current ledger. The
Next Send tab retains its existing separate preview and export behavior.

- **Safe summary:** provider/model/status/usage/provenance and omission/truncation
  inventory, without semantic request/response bodies.
- **Redacted diagnostic (default):** the incumbent Safe semantic capture shape,
  including ordinary message bodies but omitting automatically injected project
  instruction bodies and structurally excluded credentials.
- **Full trace:** every allowed semantic body actually stored by Full capture.

Full trace is disabled with a visible reason for a Safe capture; export cannot recover
content that was never stored. Every profile still excludes structured credentials and
stubs binaries. Full clipboard or filesystem export requires confirmation every time.
One profile/destination flow replaces multiplying Copy/Save buttons. Expansion, Copy,
and Save all revalidate the capture revision immediately before reading the selected
capture.

## Error handling and observability

- Capture remains best-effort and must never fail a model run.
- Capture builders, gateway bookkeeping, serialization, and flush paths log only
  content-free categories and identifiers already permitted by current policy.
- Policy escalation and destructive purge are different: a failed Full-enabling change
  retains the prior policy, while a failed Safe write stays Safe in memory and visibly
  unsaved. Purge failure before SQLite commit preserves records; post-commit Inspector
  refresh failure cannot recreate them and is reported as a refresh problem.
- A stale Inspector or live Trace revision refreshes the current policy and asks the
  user to apply again. It never overwrites a newer decision.
- A missing/deleted target disables conversation and one-shot controls; global policy
  remains available.
- Turning capture off affects future calls, disarms live one-shot overrides, and leaves
  persistent detail policies dormant. Re-enabling warns before any Full policy resumes.
  Existing records remain until scoped purge or conversation deletion.

## Implementation decomposition

This specification crosses capture construction, run admission, local schema, purge,
and three UI entry points. TASK-22507 is therefore the architectural tracking task, not
a single oversized implementation PR. Before code begins, the implementation plan must
create atomic Backlog children in dependency order:

1. Capture detail/provenance, shared request/response hardening, migration, and policy
   repository, with no user-visible Full activation.
2. Admission-time policy resolution and one-shot consumption, frozen run signals, and
   Anthropic/generic/llama.cpp gateway behavior.
3. Conversation-wide count/logical-purge operations, capture quiescence, staged cache
   invalidation, and stale-export fences.
4. Shared policy modal and status projections in Inspector/live Trace/F9 Settings,
   per-call export profiles, documentation, and production-shaped verification.

Each child owns its targeted tests and leaves the application in a Safe-default usable
state. The user-visible Full entry points land only after their persistence,
confirmation, provenance, and purge dependencies exist.

## Testing and verification

Targeted automated coverage must include:

1. Pure precedence matrices for kill switch, one-shot, conversation, global, and Safe
   default; invalid values fail Safe.
2. Exact one-shot consumption for admitted manual/queued turns, including cancellation
   before versus after admission, queue cancellation, readiness rejection, local
   commands, agent wakeups, and active-run ordering.
3. Frozen detail across direct calls, agent tool loops, retries, and surviving fleet
   calls.
4. Safe-versus-Full request and response builders with tagged project instructions,
   RAG context, tool schemas/calls/results, ordinary semantic secrets, credential
   kwargs, credential-bearing endpoint URLs, nested response binary stubs, streaming
   uncompressed-budget truncation, compressed-cap fallback, and bounded decompression.
5. Anthropic adapter-boundary tests proving Full captures system/message/tool content
   while Safe applies the incumbent project-instruction omission.
6. Genuine historical migration fixtures plus current-schema and round-trip coverage
   for the queryable detail column and conversation policy table.
7. Scoped purge count/delete across active, off-path, abandoned, and soft-deleted
   messages; transaction rollback; quiescence across primary runs, surviving fleet
   signals and in-flight flushes; ephemeral staged swaps; stale-Inspector export
   rejection; in-memory/cache invalidation; and a mutation test proving a later flush
   cannot reinsert purged rows.
8. Export profile availability, redaction, confirmation, clipboard/file paths, and the
   guarantee that Safe capture cannot produce omitted Full bodies.
9. Two-Inspector stale policy/capture revisions, Inherit/disarm changes that reveal
   Full, single-scope Apply, config replacement with cache-reload failure, failed
   escalation/reduction and promotion writes, deleted targets, Capture Off/resume
   confirmation, long names, visible disabled reasons, focus restoration, and keyboard
   operation.
10. Production-shaped Textual geometry/compositor coverage at 80x24 using
    `ConsolidatedCSSApp` for both Inspector and live/imported Trace states, plus CSS
    bundle regeneration/integrity checks.
11. Privacy assertions over every durable owner reached by the real seam: decoded
    ChaChaNotes rows, in-memory/blob caches, exports, and configured filesystem logs.

Because this changes ChaChaNotes schema, the complete DB migration suite is required.
The repository-wide full test sweep still requires explicit owner approval under the
project testing policy.

## Documentation changes at implementation

- Amend the Console context/RAG guide to describe Safe and Full, what Full includes,
  and what it never includes.
- Amend project-instruction documentation to replace the unconditional durable-capture
  exclusion with the explicit Full opt-in exception.
- Document global/config persistence, one-shot expiry, conversation inheritance,
  provider-boundary caveats, compression-not-encryption, export profiles, logical
  purge, WAL/free-page limits, and backup limitations.
- Link ADR-089 from TASK-22507, the implementation plan, and closeout notes.

## Deferred work

- Timed retention, automatic pruning, deduplication, and backup management.
- Forensic/whole-database secure-erasure workflow.
- Raw provider-wire capture for adapters Chatbook does not own.
- Auxiliary-call capture and automatic secret classification inside semantic text.
- Cross-process policy coordination; the Console remains single-app-process owned.
