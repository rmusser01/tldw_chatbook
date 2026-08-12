# Durable Provider Tool-Continuation Checkpoints Design

Date: 2026-08-12
Status: Approved; written-spec review complete
Backlog task: [TASK-15675](../../../backlog/tasks/task-15675%20-%20Add-durable-provider-tool-continuation-checkpoints.md)
Architecture decision: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)

## Purpose

Make provider-required private reasoning and function-tool history portable
across restart, sync, and import without showing it in the transcript,
re-executing completed tools, or introducing provider-owned agent loops.

This is a provider-neutral persistence/runtime boundary. Moonshot/Kimi and
Z.ai/GLM opt in under TASK-15676; DeepSeek opts in under TASK-15677. Providers
that do not declare an exact continuation policy remain byte-for-byte
unchanged.

## Problem

The native agent runtime keeps assistant tool calls and tool results in an
in-memory message list. Some providers also require their private reasoning to
be echoed with those calls. Today an app restart loses that private history.
The visible final assistant row cannot be used as a late-only store because a
tool call can arrive with empty visible content and execute before the final
assistant answer exists durably.

Existing storage does not already solve this:

- `messages.metadata_json` is explicitly local-only and excluded from both
  sync paths;
- AgentRunsDB/run logs are local diagnostic records, not conversation history,
  and do not contain the provider's exact private reasoning contract;
- tool marker rows are display projections, not canonical provider history;
- provider conversation IDs are unavailable or deliberately not owned by
  Chatbook.

Arbitrary tools also prevent a true exactly-once claim. A process can stop
after an external side effect starts but before its result is committed. The
design must represent that ambiguity instead of silently retrying it.

## Scope

### Goals

- Give each assistant generation/variant that owns provider-required private
  continuation one durable owner.
- Commit the assistant call batch before tool dispatch and each provider-bound
  result before the next model request.
- Resume pending work only after an explicit user action and fresh approvals.
- Never automatically re-execute a completed, failed, or ambiguous call.
- Carry continuation through existing message sync, conflicts, branches,
  `.chatbook`, and ordinary JSON exports.
- Exclude continuation from visible/searchable/human-readable surfaces.
- Let provider adapters decide exactly which private history is replayed and
  for how long.
- Count the private history against the ordinary request context budget.

### Non-goals

- No distributed lock or exactly-once guarantee across concurrently active
  devices.
- No provider server-side session ownership or continuation ID.
- No general arbitrary provider-metadata store.
- No persistence of raw responses, SSE events, uncapped tool output, secrets,
  approval stamps, or provider credentials.
- No automatic tool execution on open, sync, restore, or import.
- No new conversation graph, agent loop, tool executor, or permission system.
- No private-reasoning display, search, summarization, or analytics feature.
- No separate application-level encryption beyond the protection already
  applied to message content.

## Durable Owner

Schema migration adds one nullable `provider_continuation_json` column to
`messages`. It belongs to the assistant message row for one generation or
regenerated variant.

The assistant row is normally still created lazily. A plain text request does
not gain a durable blank placeholder. When, and only when, a complete valid
assistant function-call batch arrives, the store force-creates that assistant
row with its preallocated stable message ID, empty visible content, and the
first continuation checkpoint in one transaction. Tool dispatch is blocked
until that transaction succeeds. A Kimi K3 tool-free answer is different: its
visible final content already requires the normal assistant-row write, so its
provider-required preserved reasoning is committed on that same row in the
same final-content transaction; it never creates a speculative blank row.

This ownership gives the checkpoint existing semantics for:

- parent/child conversation branches;
- regenerated assistant siblings/variants;
- soft deletion and active-path selection;
- message versions and optimistic conflicts;
- final content replacement on the same assistant row;
- export ordering and import identity.

A restored row whose content is empty and continuation is nonterminal is not
rendered as a blank answer. Console renders a bounded “Interrupted tool run”
state with explicit Resume and Discard actions. The private payload itself is
never rendered.

## Canonical V1 Payload

The JSON is deterministic, versioned, and validated before every write/read.
Its canonical shape is:

```json
{
  "schema_version": 1,
  "checkpoint_revision": 4,
  "provider": "deepseek",
  "protocol": "responses",
  "model": "deepseek-v4-flash",
  "api_base_url": "https://api.deepseek.com",
  "state": "active",
  "rounds": [
    {
      "assistant_content": "",
      "reasoning_blocks": ["exact private provider reasoning"],
      "calls": [
        {
          "call_id": "call_1",
          "name": "calculator",
          "arguments": "{\"expression\":\"2+2\"}",
          "state": "completed",
          "result": "4"
        }
      ]
    }
  ]
}
```

Allowed top-level keys are exact. `provider` is limited to providers with an
approved replay policy. `protocol` is `chat_completions` or `responses` and
must be allowed for that provider. Model and normalized base are nonblank,
bounded strings. The URL has already passed the provider's structural
normalizer and cannot contain userinfo, query, fragment, or credentials.
`state` is exactly `active` or `complete`; discarded continuation is removed,
not retained as a third state.

Each round represents one provider request's assistant output:

- `assistant_content` is the exact bounded assistant content paired with the
  call batch, including an empty string;
- `reasoning_blocks` preserves provider order and exact string bytes after
  strict UTF-8/JSON decoding;
- `calls` preserves assistant call order and may be empty only for a Kimi K3
  final reasoning-only round in a complete checkpoint;
- `call_id` is nonblank and unique across the checkpoint; `name` is nonblank,
  valid, and may repeat because one function can be called more than once;
- `arguments` is the exact validated JSON-object string returned by the
  provider, with finite JSON values only;
- `state` is `pending`, `executing`, `completed`, or `failed`;
- `result` is absent for pending/executing and is the exact capped string sent
  to the provider for completed/failed calls.

Provider builders additionally reject a call-ID collision anywhere in the
expanded outbound history. No provider output item ID, response ID,
conversation ID, timestamp, process ID, approval decision, or opaque vendor
object is accepted. Protocol builders
derive their external assistant/function-call/function-output shapes from this
canonical record.

### Bounds

- serialized UTF-8 payload: at most 8 MiB per assistant message;
- rounds: at most 128;
- total calls: at most 128;
- call ID/base/model: at most 4 KiB each;
- function name: the provider-approved maximum, never more than 64 characters;
- arguments: at most 1 MiB per call;
- reasoning blocks together: at most 4 MiB;
- result: never larger than the existing provider-bound tool-result cap;
- JSON depth: at most 32; JSON nodes: at most 100,000.

The serialized total is authoritative even when individual limits leave more
theoretical room. Parsing and validation are iterative/bounded so a malicious
import cannot escape raw recursion or allocation errors.

## Checkpoint State Machine

For a complete assistant call batch:

1. Normalize all calls and private reasoning.
2. Create/update the assistant row with the full batch and every call
   `pending`.
3. Run the existing batch review/approval flow.
4. Immediately before dispatching one approved call, persist it as
   `executing`.
5. Invoke the existing tool executor.
6. Persist the exact capped provider-bound result as `completed` or `failed`.
7. Only after every result write succeeds may the runtime issue the next
   provider request.
8. Add later assistant call batches as additional rounds under the same row.
9. When the provider returns a final tool-free answer, update visible content
   and mark the checkpoint `complete` in one transaction. For Kimi K3 this
   transaction first appends the final response as a reasoning-only round whose
   `assistant_content` exactly equals the visible final content. That private
   round is the provider-history representation of the same assistant message,
   not a second visible assistant row.

For a Kimi K3 turn with no calls, the same rule creates the sole reasoning-only
round and commits it with final visible content. For a K3 turn with earlier
tool rounds, it appends the post-tool final reasoning-only round after them.
No call state or Resume action exists for a reasoning-only round; it is retained
solely for K3's required historical Preserved Thinking replay. A mismatch
between its assistant content and the row's visible content fails the atomic
write.

Review refusal is stored as a failed/result-bearing call because that exact
string is sent back to the provider. Cancellation before dispatch leaves a
call pending. Cancellation after `executing` but before result commit leaves an
ambiguous call.

On a persistent conversation, any required checkpoint failure stops the run
before the next side effect or provider request and surfaces a safe persistence
error. It does not downgrade to in-memory continuation. An explicitly
ephemeral conversation may continue in memory because the user selected a
non-durable session; UI copy labels it non-resumable.

## Restore And Resume

Loading data never executes tools. A nonterminal checkpoint from another
process/device is presented as interrupted.

Resume requires:

- explicit user action;
- the same provider, protocol/API mode, model, and normalized base stored in
  the checkpoint;
- a currently valid credential resolved through normal configuration;
- the required tool still present and disclosed;
- fresh review/approval for every pending call;
- no unresolved sync conflict;
- no call in `executing` state.

Completed and failed results are replayed verbatim and never dispatched.
Pending calls may execute after fresh approval. An `executing` call is
ambiguous: automatic resume is blocked because the external side effect may
already have happened. The minimal recovery is to discard the interrupted run
and start a new turn; manually asserting a result is a separate future feature.

Changing Settings does not mutate the checkpoint. A provider/model/mode/base
mismatch offers Restore pinned settings or Discard, not silent translation.
Credentials are intentionally current rather than pinned so key rotation does
not make otherwise valid history unresumable.

Discard is a durable, optimistic whole-message transition and never executes a
tool. If the checkpoint-created assistant row has no visible content, Discard
atomically clears the continuation and tombstones that assistant generation
under the existing branch operation. If visible content exists, Discard keeps
that content but atomically clears the continuation, making the row
non-resumable. Both forms bump message version/hash and record the durable sync
intent in the same transaction; stale-version conflict leaves the row and
checkpoint unchanged.

Sync is portability, not execution coordination. A remote active checkpoint
requires explicit Take over and the same validation as local Resume. The UI
warns that another device must not still be running the turn. This design does
not claim a distributed lease or exactly-once cross-device tool execution.

## Provider Replay Policies

The store/runtime exposes validated rounds; provider builders own replay:

- **Kimi K3:** Preserved Thinking is always enabled. Replay exact bounded
  reasoning for every retained K3 assistant turn, plus calls/results for tool
  rounds, on later K3 requests while each owning visible turn remains in
  context. Other curated Kimi families use only their explicitly documented
  active/restored tool-run policy.
- **GLM:** replay exact reasoning/calls/results for the active or restored tool
  run with `clear_thinking=false`. Ordinary completed turns use
  `clear_thinking=true` and omit private reasoning.
- **DeepSeek:** replay every retained completed tool-bearing round, including
  its exact reasoning and paired results, on later DeepSeek requests because
  the provider requires tool-call reasoning in subsequent requests.

The provider/protocol pair is closed and versioned. Adding another provider
requires an ADR check, an exact validator/translator, joined resume tests, and
privacy/budget coverage; it is not a string added to a generic passthrough map.

## History Budget And Compaction

Private history is counted before request construction. Counts include
reasoning blocks, assistant content, function names/arguments/IDs, tool result
content, and protocol framing.

For DeepSeek and Kimi K3, one visible user/assistant turn and all private rounds
owned by that assistant are one eviction unit. If the unit cannot fit, history
budgeting evicts the entire unit according to the existing oldest-first branch
policy; it never keeps the visible final answer while silently dropping only
the provider-required private portion. Active/incomplete checkpoints are not
automatically compacted and block send when they cannot fit safely.

Non-K3 Kimi and GLM completed private rounds are not sent on ordinary later
turns, but their persisted bytes remain subject to the storage cap and are
counted when resuming that exact checkpoint.

Summaries are derived only from visible conversation content. They never
contain private continuation and cannot replace it for provider replay.

## Sync And Conflict Contract

`provider_continuation_json` becomes part of the message's durable version and
payload hash. The transaction owner for portable intent is the ChaChaNotes
database, not the separate Sync-v2 state database:

- legacy message sync triggers include it in create/update/undelete payloads
  and watch it in the update predicate;
- Sync v2 includes it inside the encrypted message payload, never routing
  metadata;
- the existing message sync trigger writes a complete versioned `sync_log` row
  in the same SQLite transaction as each checkpoint/content/call-state change;
  that ChaChaNotes row is the durable local intent source of truth;
- when Sync v2 is configured, an idempotent reconciler projects each unbridged
  message intent into the separate Sync-v2 outbox, keyed by message ID,
  version, and payload hash; a durable bridge cursor/receipt may live in the
  Sync-v2 repository, but the source `sync_log` row is retained until the
  outbox write is durable;
- dispatch never waits for remote acknowledgement, but before a provider tool
  side effect it verifies the same-transaction `sync_log` intent exists and,
  when Sync v2 is enabled, that its idempotent projection has succeeded into a
  durable Sync-v2 outbox; an unavailable or in-memory-only Sync-v2 repository
  blocks start/resume with safe recovery copy;
- a post-commit notifier/wakeup may fail or the process may crash, and startup/
  ordinary reconciliation still discovers the ChaChaNotes intent and enqueues
  it exactly once by the idempotency key;
- restore applies visible content and continuation atomically;
- field-level or round-level merge is forbidden.

If two devices change the same continuation lineage, the ordinary message
conflict is raised even when visible content matches. Resume is blocked until
one complete message version wins. Call arrays/results from different versions
are never combined.

Soft-deleted/off-branch assistant rows retain their sidecar only as the parent
message record already does and are not eligible for resume. Regeneration
creates/uses a distinct assistant variant row; it never overwrites another
variant's checkpoint. Editing an ancestor invalidates/tombstones descendant
checkpoints through existing branch operations.

## Export And Import

Versioned `.chatbook` and ordinary conversation JSON include:

```json
{
  "role": "assistant",
  "content": "visible final answer",
  "_private": {
    "provider_continuation": {"schema_version": 1}
  }
}
```

Export UI/copy warns that these formats can contain private model reasoning,
tool arguments, and provider-bound tool results. The field receives the same
file protection as visible content; no extra encryption is introduced.

For `.chatbook` to call the result resumable/lossless, its next format version
must export every included message with an export-local stable ID, parent ID,
variant identity/order, selected/active-leaf ownership, role/content, deletion
eligibility, and private continuation. Import first validates and remaps the
complete graph, then attaches each checkpoint to the remapped assistant owner.
If an older/linear package cannot reconstruct that ownership, it imports the
visible messages but discards the private field with the safe warning below.

Ordinary linear JSON covers the active exported path only. Each assistant item
uses an explicit projection of role, visible content, safe public fields, and
the `_private.provider_continuation` field; it never serializes a database row
or arbitrary message dictionary wholesale. Import assigns a new local message
identity and attaches the validated private field only to that same assistant
item. TASK-15675 audits and updates the canonical `.chatbook` creator/importer
and every conversation JSON exporter/importer, including the current
`Chat_Functions.py` JSON path.

Text, Markdown, rendered transcripts, clipboard-visible message copy, FTS,
summaries, logs, errors, usage snapshots, and ordinary telemetry exclude the
field. Debug code must not log validation failures with the raw payload.

Import accepts the private field only after the full V1 validator passes and
the owner message identity/role is valid. Invalid, unsupported, oversized, or
contradictory private data is discarded while visible messages import. The
import report says exact tool continuation was discarded and identifies the
message by safe ordinal/ID only; it never includes private content. Import does
not execute or mark a checkpoint resumable until ordinary resolution and
permission validation succeeds.

## Error And Privacy Contract

- Persistence, validation, conflict, and resume errors contain provider,
  protocol, safe state, and recovery action only.
- No credentials, full sensitive URLs, reasoning, arguments, results, prompts,
  raw JSON, or response bodies enter logs/exceptions.
- Raw parser/JSON/SQLite/recursion/cleanup exceptions do not escape public
  boundaries with private data attached as cause/context.
- Cleanup is best-effort and cannot mask the primary provider, persistence, or
  cancellation outcome.
- Private data is immutable while handed across thread boundaries; all public
  snapshots are copies.

## Testing Strategy

### Schema and persistence

- V1 valid/invalid matrices, exact key sets, finite JSON, depth/node/byte caps;
- repeated function names with unique call IDs are valid; duplicate call IDs
  within a checkpoint or expanded outbound history are rejected;
- first tool batch creates stable empty assistant row + checkpoint atomically;
- failed create/update prevents dispatch/provider continuation;
- call-state crash points before/after dispatch and result commit;
- final content + complete state atomicity;
- regenerated variants, branches, soft delete, ancestor edit, and active path;
- no use of local-only `metadata_json`.

### Runtime and safety

- full batch persisted before the first invocation;
- each call is `executing` before invocation and result-bearing before the next
  model call;
- completed/failed never re-execute; executing never auto-resumes;
- pending requires fresh approval;
- provider/model/mode/base mismatch blocks; current rotated credential works;
- open/import/sync are execution-free;
- persistent write failure stops, explicitly ephemeral session remains
  in-memory only;
- cancellation/timeout/late tool completion cannot create a second dispatch.

### Sync, export, budget, and privacy

- legacy and Sync v2 round trips, encrypted-payload placement, hash/version
  change, base-version conflict, and no field merge;
- `.chatbook` graph/variant ID remap and active-path JSON round trips with
  explicit `_private` projection plus warning;
- text/Markdown/FTS/render/log/error/usage/summaries contain no canary;
- malformed private import preserves visible message and emits redacted
  warning;
- DeepSeek replay expands visible + private history and evicts it atomically;
- Kimi K3 later turns include every retained reasoning owner atomically;
- non-K3 Kimi/GLM ordinary later turns exclude completed private reasoning;
- Discard clears/tombstones with version/hash/outbox update and no execution;
- crash after local commit but before notifier still leaves discoverable sync
  intent;
- Sync-v2 in-memory-only fallback blocks pre-execution start/resume, while
  durable reconciliation is idempotent across crashes before/after outbox
  insertion and bridge-cursor update;
- configured Sync-v2 projection failure after the ChaChaNotes commit blocks
  tool dispatch; retry/restart reconciles the same intent without duplicating
  the outbox envelope;
- mutation checks remove the pre-dispatch write, result barrier, conflict hash,
  or private-history token count and must fail.

No default test performs a provider request. Provider-specific joined and paid
coverage belongs to the dependent provider tasks.

## Delivery Boundary

TASK-15675 owns only schema, canonical validator/model, persistence/sync,
export/import, history-budget integration, explicit interrupted-run UX, and the
narrow runtime checkpoint seam. It proves the seam with deterministic provider
fixtures.

TASK-15676 opts Moonshot/Kimi and Z.ai/GLM into that contract while hardening
their Chat-Completions providers. TASK-15677 adds DeepSeek dual mode and its
stricter later-turn replay. Neither dependent task may duplicate storage or
resume ownership inside an adapter.
