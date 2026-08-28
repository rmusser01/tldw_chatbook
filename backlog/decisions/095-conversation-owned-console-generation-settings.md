# ADR-095: Persist Console generation settings with the conversation

Status: Accepted
Date: 2026-08-27
Related Task: TASK-22515
Extends: ADR-006, ADR-033, ADR-052, and ADR-092

## Decision

An explicit Apply from either the Console Provider/Model popover or the full
Console Settings modal updates the exact originating Console conversation's
generation settings immediately. Any consumer that resolves its execution
settings after that commit observes the new values. Work that already captured an
immutable execution context continues with that captured context.

Persisted conversations store a versioned `console_generation_settings` object in
`conversations.metadata`. The object is an allowlisted snapshot of safe,
user-editable generation fields:

- `provider`
- `model`
- `temperature`
- `top_p`
- `min_p`
- `top_k`
- `max_tokens`
- `seed`
- `presence_penalty`
- `frequency_penalty`
- `reasoning_effort`
- `reasoning_summary`
- `verbosity`
- `thinking_effort`
- `thinking_budget_tokens`
- `streaming`

The snapshot never contains `base_url`, credentials, credential references,
`source`, character identity, system prompt, or pinned prefill. Endpoint and
credential resolution remains provider-configuration-owned. System prompt and
pinned prefill keep their existing conversation-owned persistence paths.

Both Console settings surfaces call one conversation-settings commit seam. Each
surface submits its validated target values plus the fields it exposes; neither
surface rebases settings itself. The session controller is the sole rebase and
commit owner. It serializes the resulting complete safe allowlist so a provider
change cannot leave stale provider-specific fields in the durable overlay.

When the provider is unchanged, the controller preserves compatible current values
and overlays the fields exposed by the submitting surface. On a provider change,
it starts from defaults for the selected provider/model, resolves that provider's
endpoint afresh, overlays only fields exposed by the surface and supported by the
target, and clears unsupported reasoning/thinking fields. The quick popover never
submits an endpoint. The full modal may submit a session-only endpoint only when
the endpoint draft is explicitly bound to the selected provider; otherwise the
controller uses the configured endpoint. Conversation resume performs the same
provider-first rebase before applying the saved safe snapshot.

Each live Apply increments a process-local settings revision on the exact session.
Metadata persistence carries that revision plus the captured conversation identity.
A retry proceeds only while both still match and the owned metadata value has not
been superseded; sibling-only version conflicts may reload, merge, and retry within
the existing bounded policy. Missing or malformed objects fail closed to current
defaults. A future unsupported version is preserved and is never overwritten by an
older client without an explicit user reset.

An unsaved ordinary session stages its live generation settings and writes the
safe snapshot when the conversation is first persisted. A temporary conversation
remains non-durable; promotion writes its current safe snapshot into the promoted
conversation. The serializer and provider-rebase helper remain reusable by the
future fork implementation governed by ADR-092, but TASK-22515 does not implement
or test a conversation-fork flow.

The Provider/Model popover contains and returns no compaction controls or values.
Compaction remains in the full Settings Context view with its existing owner,
storage, and behavior. It cannot delay, veto, qualify, or otherwise affect
Provider/Model Apply or modal dismissal.

## Context

The Provider/Model popover currently creates a new in-memory
`ConsoleSessionSettings` value and dismisses it to `ChatScreen`. The screen updates
the active session, but provider/model generation settings are not restored when a
conversation is reopened after restart. Resume currently restores only the system
prompt and pinned prefill.

The mouse path can also fail before the semantic Apply handler because an Input may
retain Textual mouse capture. The full Settings modal already contains the required
input-release and redirected-click pattern. The quick popover does not. Live
testing also exposed a deferred `_sync_fold_hint` callback that can query the
dismissed popover and raise `NoMatches`.

The current quick Apply uses `dataclasses.replace`, which preserves `base_url` when
the provider changes. That can route a newly selected provider through the previous
provider's endpoint. Persisting the full settings dataclass would make the problem
worse and could copy configuration-owned or sensitive values into conversation
data.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep settings in process memory | Fails the required reopen-after-restart behavior and continues to make Apply appear ineffective. |
| Persist a global default | Mutates unrelated and future conversations; contradicts exact Console-session ownership in ADR-033. |
| Add a dedicated conversation-settings table | Adds a migration and repository surface without improving the required single-conversation lookup; versioned conversation metadata already provides the needed ownership and merge rules. |
| Store the full `ConsoleSessionSettings` dataclass | Would persist endpoints and runtime-only fields and would couple the storage schema to an internal Python type. |
| Put generation settings in the compaction policy store | Mixes independent owners and makes an unrelated context policy capable of blocking Provider/Model Apply. |

## Consequences

- Provider/Model Apply becomes durable for persisted conversations without a
  schema migration.
- Quick and full Console Settings can no longer disagree about whether their safe
  generation fields survive restart.
- Provider endpoints remain configuration-owned. A custom session endpoint still
  requires the existing Save-as-default path to survive restart.
- Apply remains successful in live session state even if metadata persistence
  fails; the user receives a precise warning that the change could not be saved
  for restart. Compaction is absent from the popover and the outcome.
- A per-session settings revision prevents an older persistence completion or
  retry from overwriting a newer Apply or a rebound conversation identity.
- Metadata parsing and serialization require a small versioned helper with an
  explicit allowlist and sibling-preserving merge.
- Provider changes must use one provider-aware rebase path from both settings
  surfaces and conversation hydration.
- The popover interaction needs captured-click recovery and teardown-safe deferred
  callbacks, using the existing full-modal pattern rather than a new event system.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md)
- [TASK-22515](../tasks/task-22515%20-%20Make-Console-provider-Apply-update-and-persist-conversation-settings.md)
- [ADR-006: Provider-aware generation settings](006-provider-aware-generation-settings.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-052: Conversation memory and compaction policy](052-conversation-memory-and-compaction-policy.md)
- [ADR-092: Console chat fork copy and authority boundary](092-console-chat-fork-copy-and-authority-boundary.md)
