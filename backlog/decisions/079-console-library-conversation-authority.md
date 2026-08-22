# ADR-079: Per-conversation Console Library authority and activity

Status: Accepted
Date: 2026-08-22
Related Task: [TASK-19900 - Make Console Library controls explicit per conversation](../tasks/task-19900%20-%20Make-Console-Library-controls-explicit-per-conversation.md)
Amends: ADR-003's Settings ownership for future-session defaults, ADR-030's
Console provider availability, and ADR-063's initial-dispatch-to-continuation
ownership handoff; preserves ADR-030's Direct-versus-RAG selector

## Decision

Console will treat manual Library search, application-initiated pre-send
retrieval, and assistant-initiated Library tools as three different mechanisms
with different authorities.

Manual **Search Library** remains a user action available in every
conversation. Two independent per-conversation controls govern the other
mechanisms:

- `auto_retrieve_on_send`: Never or Automatic.
- `assistant_library_access`: Blocked or Allowed.

The controls are private device policy. Persist them in a dedicated
`console_conversation_library_policy` table with one row per locally governed
conversation, a row schema version, optimistic `policy_revision`, update time,
and no sync projection. Do not write them into synced conversation metadata,
message payloads, Chatbook exports, or server state.

Shipped global defaults are Never and Blocked and seed only a newly created
local Console session. Once captured, a session/conversation does not inherit
later global-default changes; its captured policy is inserted atomically when
that new local conversation is first persisted. A conversation first observed
through sync or import without a local policy row resolves to Never and
Blocked rather than inheriting a global value, and remains write-free until an
explicit local policy save. A missing row or read error is never permission.

Every production opener supplies the v44→v45 migration one sanitized effective
legacy automatic-retrieval boolean. The database module does not read config.
The migration-capable initializer acquires `BEGIN IMMEDIATE` before its first
schema-version read rather than trying to upgrade a deferred outer transaction;
inside that transaction it requires the seed, creates the policy table plus a
dedicated `console_dispatch_checkpoints` table/index, inserts final policy rows
for active and soft-deleted conversations present in the migration transaction,
adds nullable `messages.assistant_generation_state`, and advances the version.
The same v45 transaction replaces all four final message Sync-v1
create/update/delete/undelete triggers so payloads serialize the new field and
updates watch it. The historical v4 bootstrap schema stays unchanged: fresh
databases traverse the migrations and receive the final trigger definitions in
v45 only after the column exists. No later task relies on rerunning v45 to
repair triggers.
Because Sync-v2 source proof compares exact message payloads, the same
foundation delivery also updates its record, source-reader, and envelope
normalization before the new Sync-v1 payload can reach the outbox. New payloads
carry the field explicitly; an older payload missing only this field normalizes
it to NULL without weakening rejection of unknown, malformed, or mismatched
data. The automatic-send task consumes this compatibility seam rather than
introducing it later.
Those policy rows use the seeded automatic value and
assistant Allowed to preserve the previously always-advertised built-in Library
provider. Missing seed or failure rolls the whole migration back rather than
guessing. Conversations inserted later cannot be swept into a backfill because
there is no later initializer.

An app/store-owned coordinator publishes every committed CAS result to all
live holders for the same conversation and re-reads durable policy at the
execution linearization point. A stale in-memory Allowed value is never
authority after another process has committed Blocked. A commit after capture
affects the next turn; the current turn remains immutable.

At actual turn execution, Console first captures immutable Library authority
after that durable read. After the provider gateway applies endpoint fallback,
normalization, and readiness, it freezes a separate resolved destination record
classified as on-device, private network, public network, or external/unknown.
Unknown/custom destinations never default to on-device. The final execution
context combines both records. Queued sends capture after dequeue, not when
typed. The primary agent and all subagents share that context. A later policy,
selector, or provider change applies only to later executed turns.

When assistant access is Blocked, the built-in Library provider is absent.
The complete built-in Library namespace remains statically reserved in every
mode, derived from ADR-030's descriptor registry plus the RAG tool-name
constant: all 18 direct names and `search_library_rag`. Skills and MCP
profiles cannot claim a name simply because the current conversation blocks
or selects the other provider. This policy governs the built-in local Library
capability only; MCP and workspace/file tools retain their own ADR-032
permission principals and disclosures.

Temporary sessions do not broadly trust the catalog source string `library`.
When Allowed, their ephemeral gate admits only an authenticated built-in
Library provider and exact names from that audited read-only reserved set;
unknown names and source spoofing fail closed.

When assistant access is Allowed, ADR-030's existing global
`direct_library_tools` value remains a selector, not an enable switch:
`true` composes the six-category, 18-tool Direct provider; `false` composes
the bounded RAG provider over Notes, Media, and Conversations. Provider/model
changes do not silently clear the conversation policy. If either Automatic or
Assistant Allowed can place Library data in the request, moving from on-device
inference to a private/public/unknown external destination updates the expanded
runtime detail from the gateway-resolved record and shows a persistent
non-blocking inline disclosure before dispatch while preserving the stored
choice. This includes Automatic + Blocked.

Automatic retrieval uses the executed draft and a fixed source-category set:
Notes, Media, and Conversations. It never inherits the source toggles from a
manual search. The existing conversation/workspace item scope still narrows
eligible Note and Media items; an active item scope excludes Conversations
under the established scope semantics. Explicitly staged evidence suppresses
automatic retrieval for that send.

Automatic retrieval is a pre-dispatch gate backed by a store-owned
`ConsoleTurnPreparation` state machine. Draft, attachments, staged evidence,
prefill, optimistic echo, queue claim, authority, and resolved destination
remain preparation-owned until one compare-and-set commit. Timeout or service
failure pauses before provider dispatch and offers Retry, Send once without
Library, or Cancel. Manual cancel restores composer state; queued cancel
releases the exact entry back to pending and pauses later entries. Navigation
does not orphan a paused preparation. Retry keeps the same authority and
provider intent, creates a new retrieval attempt, and refuses a silently
changed resolved destination.

Ordinary-session persistence identity and auto-title are also preparation
state. A new durable conversation ID or computed title is published to the
session only after the transaction containing the conversation, policy, USER
turn, and preparation disclosure commits. A pre-commit failure restores the
exact prior session identity/title and moves committing back to a recoverable
paused state; post-commit failure follows interrupted-turn recovery and is
never automatically replayed.

For a durable manual or queued user-text turn, that transaction also creates the
empty assistant recovery owner and a row in the dedicated device-local
`console_dispatch_checkpoints` table. Stored state and revision columns provide
conditional CAS; strict bounded JSON columns contain authority,
credential-free destination, origin/queue, and reconstructability metadata but
no draft, prefill text, evidence body, attachment bytes, credential, or provider
request. The table is not `message_trajectory_metadata`, has no sync/import or
ADR-067 export mapping, and is removed after the assistant reaches a durable
terminal state.

The assistant owner carries a closed nullable
`messages.assistant_generation_state`. Historical `NULL` assistant state means
complete unless canonical `provider_continuation_json` is active. A valid active
ADR-063 continuation is authoritative regardless of NULL or a stale new-state
value. Its recovery surface may be exposed first, but actions remain disabled
while the loader lazily normalizes `continuation_active` under
message-version/deletion CAS. Success returns the committed version/hash and
rebinds the recovery handle before Resume/Discard is enabled. CAS conflict
forces a fresh ownership read: changed, invalid, missing, or deleted owners lose
their stale actions and are quarantined; the same valid continuation at a newer
version is rebound/retried from that observed row. A known rolled-back write
failure preserves recovery only after a fresh read confirms the original valid
owner and version. New owners progress through `accepted`, `dispatch_started`,
`continuation_active`, or one of `complete`/`stopped`/`failed`/`discarded`. The
field participates in
the message version/hash, Sync v1/v2, `.chatbook`, JSON, deletion, and conflict
contracts. A remote/imported unresolved owner is inert and visibly attributed
to the source device rather than rendered as a blank assistant or given access
to the device-local checkpoint.

This deliberately amends ADR-063's rejection of persisting every ordinary
empty assistant before the first provider request. That rejection assumed no
durable pre-dispatch acceptance owner was needed. Here the assistant is not an
unexplained blank: its synchronized closed state is the portable inert owner,
while the local checkpoint supplies source-device recovery and atomic handoff
to ADR-063 if a tool batch arrives.

An `accepted` checkpoint proves provider invocation has not started and offers
Retry response or Discard. A `dispatch_started` checkpoint makes provider
delivery indeterminate and offers warned Retry anyway or Discard; neither state
is auto-replayed. Retry reuses the same durable USER/assistant owners and frozen
authority, and is unavailable with a reason when transient inputs cannot be
reconstructed. Discard keeps the USER turn, marks the assistant interrupted,
and settles an accepted queued entry without returning it to pending; later
queue work stays paused until recovery settles.

Terminal completion and Discard use one expected-revision transaction to write
the assistant terminal state/content, guard the frozen USER/assistant
`messages.version` values and both `deleted = 0`, record sync intent, and delete
its checkpoint. Loader reconciliation treats
a terminal assistant as authoritative over a stale checkpoint, hydrates valid
nonterminal pairs, and quarantines missing/cross-conversation/wrong-role
ownership without invoking a provider. Ephemeral turns use only the
runtime-owned in-memory analogue and cannot be promoted while preparation or
dispatch recovery remains unresolved.

If a supported provider emits a tool batch, one transaction writes ADR-063's
validated `provider_continuation_json`, sets continuation-active, bumps message
version/hash and sync intent, and deletes the dispatch checkpoint before any
tool executes. ADR-063 then becomes the only recovery owner. If both owners are
observed after corruption or an older partial implementation, the active
provider continuation wins and the stale dispatch checkpoint is removed; two
Retry/Discard surfaces are never composed.

Zero-match and one-shot-bypass sends atomically store a bounded device-local
`library_preparation` sidecar event with the USER turn. That event contains no
query or source identity and owns the persistent sent-turn disclosure. A
cancelled preparation persists neither the transient echo nor the sidecar. The
one-shot bypass never changes future policy.

Sidecar contributions participate through an insert-only
`ConsoleTransactionWriter`, not a raw `sqlite3.Cursor`. The persistence owner keeps
the cursor private and supplies only parameterized single-row/batch INSERT methods;
the capability exposes no connection, authorizer, transaction/savepoint or
ATTACH/DETACH control, commit/rollback, connection factory, or publication/session
state. Contribution exceptions propagate through the same caller-owned
`BEGIN IMMEDIATE` transaction. This is a public API capability boundary for trusted
in-process components, not a claim that Python code is a hostile-code sandbox.

Assistant Library use is reviewable but is not evidence staging. Capture a
bounded `library_activity` event in the existing device-local
`message_trajectory_metadata` sidecar at the built-in Library provider result
seam before result truncation or delivery to the model. Anchor it to the
durable turn opener and identify attempt/run, primary or subagent actor,
provider identity, Direct/RAG mode, operation, status, result count, and
bounded source references. Store only a bounded query preview, opaque IDs,
bounded titles, and scrubbed errors—never source bodies, excerpts, local
paths, credentials, or unbounded tool output.

The app/store-owned in-memory activity sink must accept the record before a
Library result is released to the model; failure to capture withholds the
result. Durable persistence may retry after the model-visible step. A failed
write remains in the store-owned buffer across navigation with an explicit
“not saved in this session” warning and a final bounded flush on close,
promotion, or shutdown. Process crash may lose an unsaved event. This provides
trustworthy ordinary review without claiming audit-grade availability coupling.

`library_activity` is an event about its anchor, not the anchor message's own
trajectory row. The generic trajectory projection must explicitly exclude it
from message-row ownership and ordinary tool nesting so it cannot displace
timing or appear twice. A separate pure projection supplies the Console's
Selected turn Inspector group. It never enters Sources, staged context,
prompts, provider history, or the next send. Default trajectory export redacts
its query and source-reference details; full export remains an explicit user
opt-in under ADR-067.

Device-local means never synchronized automatically, not a promise that
explicit trajectory export cannot copy bounded/redacted data out of the device.
Soft deletion and synchronized tombstones retain policy and sidecars in an
inert state so restore resumes the same local authority. Only permanent local
deletion cascades them.

The always-visible Console status uses one fixed-order two-axis chip:
`Library · Auto {off|on} · Agent {blocked|allowed}`. Runtime readiness and
provider destination are separate expanded details, not additional chip axes.
The chip opens a Library Access policy modal with explicit Save/Cancel and
revision-conflict handling. Manual Search Library uses a separate search
surface, prefilled directly from the composer and labeled so its source
filters apply to that search only. Staged/cited evidence and assistant
activity are separate review concepts; activity belongs under a Selected turn
Inspector group rather than in Sources.

## Context

Console currently exposes three ways to consult a user's Library but presents
them as one loosely named RAG feature. A user can run Search Library, a global
`rag_auto_retrieve_on_send` switch can retrieve before every text send, and an
agent can receive either ADR-030 Direct tools or the RAG fallback whenever the
agent runtime is active. The current status chip reports whether evidence is
staged, not who is authorized to retrieve it.

The inherited PR proposed one Off/Manual/Auto mode in synced
`conversations.metadata` and interpreted `direct_library_tools=false` as no
assistant access. That model conflates the user, application, and assistant;
it cannot express automatic retrieval with an assistant blocked, or manual
only with an assistant allowed. It also contradicts ADR-030, where `false`
selects RAG rather than disabling the provider.

This decision changes storage and migration, sync ownership, assistant
permission and runtime composition, per-turn configuration, data minimization,
and long-lived Console disclosure. A new ADR is required rather than editing
accepted ADR-003, ADR-030, ADR-032, ADR-066, or ADR-067 in place.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| One Off/Manual/Auto conversation mode | It merges automatic retrieval and assistant authorization and cannot represent the four required policy combinations. Manual search also remains available in every state, so “Manual” is not an exclusive mode. |
| Store policy in `conversations.metadata` | That metadata participates in sync and import, spreading a device's local model-access decision to other devices and reusing a corruption-prone merge seam for a privacy control. |
| Let a missing row inherit current global settings | A synced/imported or unreadable conversation could silently acquire Library access because of unrelated device defaults. Missing authority must fail closed. |
| Create pending rows in migration and read config later | A crash or competing process could change which value is preserved. Supplying one sanitized seed to the schema transaction makes the upgrade deterministic. |
| Enter `BEGIN IMMEDIATE` inside a nested v44 migration step | The existing outer deferred transaction has already read the version and cannot be safely upgraded under competing openers. The migration runner must acquire the write lock before that first read. |
| Trust a session's cached policy at execution | Another tab or process could revoke access while the stale holder continued to expose Library tools. Durable execution-time re-read provides a clear linearization point. |
| Classify egress from provider name/API-key presence | Custom endpoints and provider fallbacks make those signals unreliable. Only the gateway-resolved credential-free endpoint can support conservative classification. |
| Interpret `direct_library_tools=false` as disabled | ADR-030 defines it as the RAG fallback selector; changing that meaning would break existing Settings, tests, and user expectations. |
| Reserve only names advertised in the current mode | A Skill or MCP tool could occupy a dormant built-in name, then shadow or break the trusted provider when policy or selector changes. |
| Add `library` to the temporary-session audited-source whitelist | A future write tool or source-spoofing provider would become eligible automatically. Temporary execution instead requires the authenticated built-in provider and exact audited name. |
| Keep retrieval pause state in the mounted Screen | Navigation, shutdown, and queued background work could orphan a claim or lose staged state. Store ownership plus compare-and-set transitions makes recovery lifecycle-safe. |
| Publish first-persistence ID/title before commit | A rolled-back SQLite transaction could leave the session pointing at a nonexistent conversation or carrying a title for a send that never happened. Publication is a post-commit effect. |
| Create the assistant/recovery owner after USER commit | A crash between those writes would leave an accepted USER with no durable owner capable of recovery. The assistant placeholder and dispatch checkpoint belong in the same transaction. |
| Pass the raw SQLite cursor to a generic contribution and try to police it with a temporary authorizer | The supplied cursor exposes its connection and authorizer mutator, so a contribution can clear the guard and commit irreversibly before a postcondition runs. An insert-only writer capability keeps transaction control out of the public contribution API. |
| Store dispatch recovery in `message_trajectory_metadata` | That event ledger has no checkpoint revision column and ADR-067 exports its rows. A dedicated temporary table gives CAS, atomic settlement, and explicit non-export ownership. |
| Update the terminal assistant and delete its checkpoint separately | A crash could hydrate Retry anyway beside a completed response or lose a Discard halfway through. One expected-revision settlement transaction makes the pair indivisible. |
| Keep assistant generation state in memory while syncing the empty owner | Another device or export would receive a blank checkpoint-free assistant. A closed message field makes unresolved and empty-terminal rows inert and truthful across every projection. |
| Keep dispatch recovery beside an ADR-063 continuation | Two recovery owners could replay the provider request or its tool batch. The tool-batch transaction must hand ownership to ADR-063 before tools execute. |
| Let assistant activity appear in Sources | Sources govern evidence staged/sent/cited by the application. Agent tool reads are historical activity and must not be staged into a later prompt or misrepresented as cited evidence. |
| Add a second Library-activity table | The existing device-local sidecar already owns turn-attributed event metadata and supports new unconstrained event kinds; another ledger would duplicate sequencing, ownership, export, and deletion rules. |
| Infer activity by parsing tool-marker text or provider capture | Tool markers are deliberately lossy/session-only and provider captures occur after transformations; neither is an authoritative minimized local event. |
| Store complete Library tool results for review | Full Notes, Media, Conversations, Prompts, Skills, or Collections would duplicate private bodies, increase local retention, and make review/export a larger privacy surface. |
| Make activity persistence a hard prerequisite for the entire model turn | This would provide audit-grade coupling at the cost of failing useful turns for a local review-sidecar outage. Capture-before-release plus visible persistence failure is the proportional boundary. |
| Let automatic retrieval inherit manual source filters | A one-off manual filter would silently change future sends. Fixed Notes/Media/Conversations behavior is predictable and matches the bounded RAG provider. |
| Proceed automatically after retrieval failure | A user who selected Automatic has asked for Library preparation; silently dispatching without it makes the control untruthful. A one-shot, explicit bypass preserves agency without changing policy. |

## Consequences

### Benefits

- Users can independently control what the application does before a send and
  what the assistant may initiate during a turn.
- New or unreadable local policy fails closed without removing manual user
  access to Search Library.
- Existing conversations retain their effective behavior once, while later
  global changes cannot rewrite them.
- ADR-030's Direct/RAG capability distinction stays intact behind an explicit
  authorization gate.
- Assistant Library reads are attributable and reviewable without becoming
  prompt context or a second copy of Library content.
- The fixed chip grammar and separate policy/search surfaces make authority
  visible without turning the status strip into a workflow form.

### Accepted trade-offs

- The main conversation database gains a device-local policy table and a required
  sanitized migration seed for v44→v45; production database openers must pass it.
- Existing conversations are intentionally backfilled to Allowed and their
  current automatic default, so privacy-tight defaults apply prospectively
  rather than silently changing established behavior.
- Global policy defaults apply only at local session creation, which means two
  devices may intentionally hold different policy for the same synced
  conversation.
- Failing closed can make an existing conversation temporarily behave more
  restrictively when its policy cannot be read.
- The activity record is sufficient for ordinary review but not a complete
  audit log and not a retained copy of the model's full tool result.
- The auto-retrieval pause adds latency and an explicit decision on failure;
  users can bypass it once without weakening future turns.
- Accepted durable sends briefly retain a row in the device-local operational
  dispatch-checkpoint table; this bounded state is required to recover
  post-commit failures without persisting a complete provider request.
- Message sync/export gains a closed assistant-generation state so another
  device can render unresolved owners inertly without receiving local recovery
  authority.
- Trajectory projection/export redaction must understand `library_activity`
  and `library_preparation` even though no sidecar schema migration is required
  for either event kind.

## Rollback

- Disable policy editing and automatic retrieval while continuing to read
  stored rows as Never/Blocked; do not reinterpret them through globals.
- Omit the built-in Library provider if policy or activity capture is
  unavailable.
- Retain the v45 column/tables and sidecar events during rollback; do not
  down-migrate or copy device-local state into synchronized metadata. Keep the
  checkpoint loader, ADR-063 precedence handoff, inert remote projection, and
  atomic Discard drain available until no dispatch checkpoints remain; a build
  without that drain must refuse v45 writes.
- The manual Search Library action remains available throughout rollback.

## Links

- [TASK-19900](../tasks/task-19900%20-%20Make-Console-Library-controls-explicit-per-conversation.md)
- [Design specification](../../Docs/superpowers/specs/2026-08-22-console-library-controls-design.md)
- [ADR-003: Settings Library/RAG Defaults Boundary](003-settings-library-rag-defaults.md)
- [ADR-024: Canonical RAG Citation Provenance](024-rag-citation-provenance-and-source-resolution.md)
- [ADR-030: Direct Local Library Tool Boundary](030-local-library-agent-tool-boundary.md)
- [ADR-031: TUI Keybinding and Footer-Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-052: Console Conversation Memory and Compaction Policy](052-console-conversation-memory-and-compaction-policy.md)
- [ADR-063: Hosted Provider Wire and Durable Tool Continuation](063-hosted-provider-wire-and-durable-tool-continuation.md)
- [ADR-066: Console Trajectory View](066-console-trajectory-view-and-trace-metadata.md)
- [ADR-067: Trajectory Export Format](067-trajectory-export-format.md)
