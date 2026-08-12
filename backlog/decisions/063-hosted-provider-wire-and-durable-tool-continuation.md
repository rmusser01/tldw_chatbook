# ADR-063: Use a neutral hosted wire boundary with durable tool continuation

Status: Accepted
Date: 2026-08-12
Related Tasks: [TASK-15675](../tasks/task-15675%20-%20Add-durable-provider-tool-continuation-checkpoints.md), [TASK-15676](../tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Related Specs: [Durable Provider Tool Continuation](../../Docs/superpowers/specs/2026-08-12-durable-provider-tool-continuation-design.md), [Moonshot/Kimi and Z.ai/GLM Hosted Chat](../../Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md)
Supersedes: ADR-062

## Decision

Chatbook will retain ADR-062's provider-neutral hosted Chat-Completions wire
boundary and replace its ephemeral-only reasoning handoff with a durable,
versioned provider-continuation checkpoint owned by the assistant generation
that initiated the tool run.

The neutral hosted wire layer owns HTTP/SSE mechanics, limits, retries,
normalization, resource closure, and redacted typed errors. Provider builders
continue to own model policy, payload allowlists, thinking/reasoning controls,
finish classification, and history translation. The shared layer does not read
configuration, execute tools, or decide which private history a provider must
replay.

The durable owner is a nullable `provider_continuation_json` field on the
assistant message row. It is not a generic metadata bag. A versioned validator
admits only canonical assistant reasoning/text, complete function calls, paired
provider-bound tool results, call state, and non-secret pinned resolution data
for explicitly supported providers and protocols. Credentials, raw HTTP
responses, arbitrary vendor metadata, and full uncapped tool output are never
stored there.

When a complete assistant tool-call batch first arrives, Chatbook creates the
otherwise-empty assistant row with a stable ID and its checkpoint in one local
transaction before any tool executes. Each call transitions durably through
`pending`, `executing`, and `completed` or `failed`; the exact capped result sent
back to the provider is committed before another provider request. A persistent
conversation fails closed if the required checkpoint write fails. Explicitly
ephemeral conversations may continue in memory and are disclosed as
non-resumable.

Kimi K3 also requires preserved reasoning for ordinary historical assistant
turns. Those reasoning-only complete checkpoints are committed with the normal
final visible assistant row and never create a blank row or resumable tool
state.

Restoration never executes work automatically. A user must choose Resume (or
an explicit cross-device takeover), current permissions are evaluated again,
and the checkpoint's provider, model, API mode, and normalized base must still
match. Current credentials are resolved through the ordinary secret boundary
and are never checkpointed. A restored `executing` call is ambiguous and cannot
be re-run automatically; completed/failed calls are replayed as recorded and
pending calls require fresh approval.

The continuation field follows its assistant branch/variant and participates
in the same message version, hash, sync, deletion, and whole-record conflict
contract as visible content. The ChaChaNotes message transaction and its
trigger-written `sync_log` row are the atomic durable local intent. An
idempotent reconciler projects that row into the separate Sync-v2 outbox; the
two databases are not described as one transaction. When Sync v2 is configured,
tool execution waits for that durable idempotent outbox projection and fails
closed if only an in-memory repository is available, but remote acknowledgement
is not required. Conflict resolution never merges continuation
subfields or tool-call arrays. Sync provides portable state, not a distributed
execution lock; remote takeover is explicit and never claims exactly-once
execution across concurrently active devices.

The field uses the same storage protection as message content. It is encrypted
where Sync v2 encrypts message content and otherwise receives no additional
application-level encryption. Versioned `.chatbook` preserves/remaps message
graph and variant ownership before attaching it; ordinary active-path JSON uses
an explicit private projection. Both warn that model reasoning, tool arguments,
and provider-bound results are present. Rendering, FTS, text/Markdown export,
summaries, logs, errors, and usage exclude it.

Provider replay remains explicit policy. Kimi K3's always-on Preserved Thinking
replays every retained reasoning owner on later K3 requests; other Kimi
families and GLM use only their documented active/restored policy. DeepSeek
replays reasoning and tool history from completed tool-bearing turns on later
same-provider requests while the owning visible turn remains in context. The
history budget counts private continuation and retains or evicts the owning
visible turn plus its private rounds atomically.

Discard never executes work. It atomically clears continuation on an assistant
row with visible content, or clears and tombstones a checkpoint-created blank
assistant generation, while bumping version/hash and recording sync intent.

## Context

Kimi, GLM, and DeepSeek return private reasoning that can be required to
continue function tools correctly. An in-memory field is insufficient: a
process restart, sync restore, or lossless import can otherwise leave a valid
conversation unable to continue. Attaching state only after the final answer is
also too late because Console intentionally defers persistence for empty
assistant placeholders, while tool execution begins before a final answer
exists.

The repository already owns conversation branches and variants through message
rows. Persisting a separate conversation graph or provider session would add a
second ownership system. Creating the assistant row only when a complete tool
batch makes durability necessary preserves the existing no-empty-row behavior
for ordinary text turns while giving active tool work a stable owner.

Arbitrary tools cannot provide transactional exactly-once execution. A process
can die after an external side effect begins and before its result is recorded.
Durable pre-execution state therefore provides honest ambiguity rather than a
false automatic-retry guarantee.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep reasoning in memory | Breaks restart, sync, and lossless-import resume. |
| Put continuation in local-only `metadata_json` | That column is intentionally excluded from sync and already has unrelated closed schemas. |
| Add a separate continuation table/graph | Duplicates message branch, variant, deletion, conflict, and export ownership without a demonstrated one-to-many need. |
| Attach state to the initiating user row | Regenerated assistant variants under one user row can require different provider histories and would overwrite each other. |
| Persist every empty assistant placeholder before the first request | Creates durable blank rows for ordinary text calls and expands cleanup behavior unnecessarily. |
| Automatically replay `executing` calls after restart | Can repeat an irreversible external side effect. |
| Require remote sync acknowledgement before each tool | Makes local execution depend on network availability and promises stronger coordination than existing message sync provides. |
| Encrypt continuation with a separate application key | Adds cross-device key ownership while visible conversation content remains under the ordinary message protection boundary. |

## Consequences

- A schema migration, sync payload update, import/export update, and explicit
  interrupted-run UX are required before providers opt in.
- Tool-bearing assistant rows may exist with empty visible content while a run
  is interrupted; loaders render an interrupted/resume state rather than a
  blank assistant message.
- Private provider context is portable and more sensitive than ordinary final
  text, so JSON and `.chatbook` exports must disclose its inclusion.
- DeepSeek histories can cost more because required private tool rounds count
  toward future context; budgeting and eviction must include them.
- The runtime gains one narrow checkpoint callback at the model/tool boundary,
  not a new agent loop or provider metadata dictionary.
- Providers opt in only after joined crash/restore and privacy tests prove their
  exact translation policy.

## Links

- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-045: QwenCloud Dual-API Provider Boundary](045-qwencloud-dual-api-provider-boundary.md)
- [Superseded ADR-062](062-hosted-chat-completions-provider-boundary.md)
