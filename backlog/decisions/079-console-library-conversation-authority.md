# ADR-079: Per-conversation Console Library authority and activity

Status: Accepted
Date: 2026-08-22
Related Task: [TASK-19900 - Make Console Library controls explicit per conversation](../tasks/task-19900%20-%20Make-Console-Library-controls-explicit-per-conversation.md)
Amends: ADR-003 and ADR-030; supersedes their implication that a global
Console setting alone determines automatic or assistant Library availability

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
and a one-time legacy-initialization marker. Do not write them into synced
conversation metadata, message payloads, exports, or server state.

Shipped global defaults are Never and Blocked and seed only a newly created
local Console session. Once captured, a session/conversation does not inherit
later global-default changes; its captured policy is inserted atomically when
that new local conversation is first persisted. A conversation first observed
through sync or import without a local policy row resolves to Never and
Blocked rather than inheriting a global value, and remains write-free until an
explicit local policy save. A missing row or read error is never permission.

The schema migration inserts legacy-marked rows only for conversations that
already exist in the migration transaction. Before Console becomes
interactive, an idempotent initializer snapshots the then-current global
automatic-retrieval value into those rows, sets assistant access Allowed to
preserve the previously always-advertised built-in Library provider, and
clears the marker. Conversations inserted later cannot be swept into that
backfill. An initialization or read failure leaves Library-sensitive behavior
fail-closed and visibly unavailable until retry.

At actual turn execution, Console captures one immutable library-policy
snapshot with the other `ConsoleTurnExecutionContext` facts. Queued sends
capture after dequeue, not when typed. The primary agent and all subagents for
that turn share the same snapshot. A policy or global Direct/RAG-selector
change during a running turn applies only to later executed turns.

When assistant access is Blocked, the built-in Library provider is absent.
The complete built-in Library namespace remains statically reserved in every
mode: all 18 ADR-030 direct names and `search_library_rag`. Skills and MCP
profiles cannot claim a name simply because the current conversation blocks
or selects the other provider. This policy governs the built-in local Library
capability only; MCP and workspace/file tools retain their own ADR-032
permission principals and disclosures.

When assistant access is Allowed, ADR-030's existing global
`direct_library_tools` value remains a selector, not an enable switch:
`true` composes the six-category, 18-tool Direct provider; `false` composes
the bounded RAG provider over Notes, Media, and Conversations. Provider/model
changes do not silently clear the conversation policy. Moving an allowed
conversation from local inference to a cloud destination updates the expanded
runtime detail and shows a persistent non-blocking inline disclosure before
the next send while preserving the stored choice.

Automatic retrieval uses the executed draft and a fixed source-category set:
Notes, Media, and Conversations. It never inherits the source toggles from a
manual search. The existing conversation/workspace item scope still narrows
eligible Note and Media items; an active item scope excludes Conversations
under the established scope semantics. Explicitly staged evidence suppresses
automatic retrieval for that send.

Automatic retrieval is a pre-dispatch gate. Console shows preparation with a
cancel affordance. Timeout or service failure pauses before provider dispatch
and offers Retry, Send once without Library, or Cancel, preserving the draft
and policy. The one-shot bypass does not change future behavior. A successful
zero-match retrieval may proceed, but the sent turn retains a visible
disclosure that it used zero Library matches and was sent without Library
evidence.

Assistant Library use is reviewable but is not evidence staging. Capture a
bounded `library_activity` event in the existing local-only
`message_trajectory_metadata` sidecar at the built-in Library provider result
seam before result truncation or delivery to the model. Anchor it to the
durable turn opener and identify attempt/run, primary or subagent actor,
provider identity, Direct/RAG mode, operation, status, result count, and
bounded source references. Store only a bounded query preview, opaque IDs,
bounded titles, and scrubbed errors—never source bodies, excerpts, local
paths, credentials, or unbounded tool output.

The in-memory activity sink must accept the record before a Library result is
released to the model; failure to capture withholds the result. Durable
persistence may retry after the model-visible step, but exhaustion is shown as
an explicit local “not saved” warning. This provides trustworthy ordinary
review without claiming audit-grade availability coupling.

`library_activity` is an event about its anchor, not the anchor message's own
trajectory row. The generic trajectory projection must explicitly exclude it
from message-row ownership and ordinary tool nesting so it cannot displace
timing or appear twice. A separate pure projection supplies the Console's
Selected turn Inspector group. It never enters Sources, staged context,
prompts, provider history, or the next send. Default trajectory export redacts
its query and source-reference details; full export remains an explicit user
opt-in under ADR-067.

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
| Interpret `direct_library_tools=false` as disabled | ADR-030 defines it as the RAG fallback selector; changing that meaning would break existing Settings, tests, and user expectations. |
| Reserve only names advertised in the current mode | A Skill or MCP tool could occupy a dormant built-in name, then shadow or break the trusted provider when policy or selector changes. |
| Let assistant activity appear in Sources | Sources govern evidence staged/sent/cited by the application. Agent tool reads are historical activity and must not be staged into a later prompt or misrepresented as cited evidence. |
| Add a second Library-activity table | The existing local sidecar already owns turn-attributed event metadata and supports new unconstrained event kinds; another ledger would duplicate sequencing, ownership, export, and deletion rules. |
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

- The main conversation database gains a local-only policy table and upgrade
  initialization step; current schema v44 is expected to advance to v45.
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
- Trajectory export redaction must understand the new event kind even though
  no sidecar schema migration is required for the kind itself.

## Rollback

- Disable policy editing and automatic retrieval while continuing to read
  stored rows as Never/Blocked; do not reinterpret them through globals.
- Omit the built-in Library provider if policy or activity capture is
  unavailable.
- Retain the local table and sidecar events during rollback; do not down-migrate
  or copy them into synchronized metadata.
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
- [ADR-066: Console Trajectory View](066-console-trajectory-view-and-trace-metadata.md)
- [ADR-067: Trajectory Export Format](067-trajectory-export-format.md)
