# ADR-028: Structured Prompt Artifacts and Auxiliary Improvement Calls

Status: Accepted
Date: 2026-08-01
Related Task: N/A
Supersedes: N/A

## Decision

Store editable structured Prompts and Recipes in the existing Prompts data
model as namespaced, versioned block documents, with deterministic compiled
System and User text for compatibility. Route Console prompt-improvement work
through a typed, side-effect-free auxiliary-completion method on the existing
Console provider gateway.

Structured records use:

- `prompt_format = "structured"`;
- `prompt_schema_version = 1`;
- a `prompt_definition` whose kind is `block_prompt` or `block_recipe`;
- compiled `system_prompt` and `user_prompt` fields for legacy consumers.

The structured definition is canonical. The compiled fields are regenerated
from it on each structured save. Existing legacy records remain legacy unless
the user explicitly saves a structured working copy.

Both Prompts and Recipes remain first-class records in Library > Prompts. A
Recipe creates an unsaved Prompt working copy and cannot be directly applied.
No parallel recipe table, generic workflow schema, or implicit record upgrade
is introduced.

The Console owns one `ConsolePromptsModal` shell and its ephemeral working-copy
navigation. A Textual-independent block codec owns parse, validation, compile,
and round-trip behavior. Prompt Library services own source authority and
optimistic version checks. A `PromptImprovementService` owns request and
response semantics but depends on an injected auxiliary provider port.

The auxiliary provider operation reuses the active provider resolution and
credentials. It is non-streaming, text-only, tools-disabled, absent from the
chat transcript, and unable to inherit conversation history, RAG context,
attachments, staged sources, or chat stop sequences. It returns typed outcomes
and never converts generic provider fallback copy into successful prompt
content.

## Context

Console currently has separate prompt picking, system-prompt editing, context
inspection, and normal chat-generation paths. Reusing those widgets directly
would either chain modal screens, expose the full conversation payload, or
couple prompt improvement to transcript side effects.

The Prompts database already has fields intended for structured definitions,
but definitions in the repository are heterogeneous. A schema version alone
cannot distinguish a prompt block document from an unrelated structured
payload. A required `kind` provides that namespace while allowing future
versioned readers to fail closed.

Structured artifacts also need two representations. Users and new editors need
lossless editable blocks, while existing consumers expect plain System and
User strings. Choosing the definition as canonical and compiled strings as a
compatibility projection avoids parallel editable authorities.

Prompt improvement must use the same provider/model setup as normal Console
work without becoming a normal chat turn. A typed gateway seam preserves the
provider-resolution boundary while allowing the operation to disable tools,
streaming, history, and side effects deliberately.

## Required Boundaries

- `ConsolePromptsModal` owns presentation, focus, and ephemeral navigation. It
  does not call provider adapters or prompt databases directly.
- The block codec imports no Textual widget, provider, database, or application
  state.
- Structured readers require a known kind and supported schema version.
  Unknown documents fall back to legacy compiled text rather than partial
  interpretation.
- A structured save validates and writes definition plus compiled text as one
  prompt-record update.
- Conditional Update requires a real expected-version check at the source
  boundary. Fetch-then-unconditional-write is not accepted as optimistic
  locking.
- The auxiliary provider port resolves through `ConsoleProviderGateway`; the
  feature does not call provider modules or `chat_api_call` directly.
- Auxiliary requests contain only trusted optimizer instructions, the current
  unsent draft, optional current system prompt, and optional Recipe definition.
- Tools, transcript writes, conversation history, RAG, attachments, staged
  sources, and prompt-content logging are excluded from auxiliary calls.
- There is no hidden model repair request. Retry is explicit user intent.
- Auto application is gated by session and content fingerprints plus
  deterministic preservation checks.
- Applying a System lane is separately authorized and defaults off.
- Live-session application and durable conversation persistence are reported
  honestly as separate outcomes; the UI does not claim a cross-store atomic
  transaction it cannot guarantee.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Store only compiled System/User text | Cannot losslessly reopen user-defined block structure or distinguish Prompts from Recipes. |
| Add a separate Recipes table | Duplicates Prompt Library ownership, search, versioning, import/export, and local/server policy for no independent lifecycle benefit. |
| Treat every structured definition as one schema | Existing definitions are heterogeneous; version numbers do not identify document semantics. |
| Make compiled text and blocks independently editable | Creates two writable authorities that drift and makes conflict resolution ambiguous. |
| Reuse normal Console chat submit | Pulls in transcript/history behavior, streaming fallbacks, tools, and unrelated next-send context. |
| Call provider integrations directly from the feature | Duplicates provider/model/credential resolution and violates the Console provider boundary. |
| Chain picker, review, and recipe modals | Makes Back/Escape, dirty state, focus restoration, and request cancellation harder to reason about. |
| Automatically repair malformed output with another model call | Adds hidden cost and latency and makes one click produce an undisclosed number of requests. |

## Consequences

### Benefits

- Structured Prompts and Recipes round-trip without breaking legacy consumers.
- One canonical representation prevents raw-text and block-editor drift.
- Prompt improvement uses the current provider setup without becoming a chat
  turn or side-effecting the session before Apply.
- The block codec, provider seam, and UI state can be tested independently.
- Local and server Prompt Library ownership remains visible and version-aware.

### Accepted trade-offs

- Every structured save performs deterministic compilation in addition to
  storing the block definition.
- Server Update and search remain unavailable when the connected backend
  cannot provide the required capability honestly.
- Synchronous provider work may be impossible to abort after dispatch; the UI
  detaches and discards late results.
- Deterministic validators reduce but cannot prove qualitative prompt quality,
  so a reviewed quality corpus remains necessary.
- The Console action row must become adaptively two-row at narrow widths,
  superseding the prior unconditional one-row goal.

## Links

- [Console Prompt Workbench and Improvement Design](../../Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md)
- [ADR-005: Console Workspace Server Readiness](005-console-workspace-server-readiness.md)
- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
