# ADR-095: Own Console conversation settings and explicit defaults separately

Status: Accepted
Date: 2026-08-27
Related Task: TASK-22515
Extends: ADR-006, ADR-033, and ADR-052

## Decision

An explicit `Apply to this chat` from either the Console Provider/Model popover or
the full Console Settings modal updates the exact originating Console conversation's
generation settings immediately. Any consumer that resolves its execution settings
after that commit observes the new values. Work that already captured an immutable
execution context continues with that captured context.

The settings surfaces also expose two explicit default actions. `Save as model
default` first performs the same exact-origin live Apply, then field-masked-patches
`api_settings.<provider>.model_defaults[<exact model id>]`. `Make default for new
chats` performs that model-profile patch and atomically changes
`chat_defaults.provider` plus `chat_defaults.model`. Ordinary Apply never mutates
configuration, and compaction is excluded from both default actions.

Every blank new-chat creation path without an explicit source-settings intent uses
the saved global provider/model and its exact model profile. This includes Ctrl+T,
temporary chats, workspace-created blank chats, and the initial pristine Console
chat after startup. Existing/open conversations do not rebase. Duplicate, Branch,
Continue, and handoff operations carrying explicit settings remain source-owned and
are not eligible blank-chat creation.

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

Both Console settings surfaces call one conversation-settings Apply orchestration.
Each surface submits its validated target values, discriminated action, and field
mask;
neither surface rebases settings itself. The session controller is the sole
provider/model-rebase owner. It serializes the resulting complete safe allowlist so
a provider or model change cannot leave stale values in the durable overlay.

Whenever provider or model changes, the controller starts from the selected
provider/model's established default chain, resolves that provider's endpoint
afresh, overlays only deliberately edited fields supported by the target, and clears
unsupported reasoning/thinking fields. Untouched fields therefore load the target
model profile; deliberate edits remain visibly marked. An open settings transaction
keeps a process-local draft map keyed by canonical provider plus literal model ID so
A → B → A restores A's unfinished edits. `Full settings…` transfers that draft map,
field provenance, compaction draft, and exact origin into the full Model view without
applying or discarding it. Conversation resume performs the same provider-first
rebase before applying the saved safe snapshot.

The quick model-profile field mask is temperature and streaming. The full Model
mask is every supported sampler, reasoning/thinking, token-limit, and streaming
field it exposes. Blank profile values delete the exact override so lower-precedence
defaults apply; the conversation still stores the effective value resolved at Apply
time as a complete snapshot. Full Settings therefore represents streaming as
Inherit, On, or Off.

Default mutation rereads config under the existing lock and patches only the exact
masked profile fields, global provider/model when applicable, and an eligible
endpoint. It preserves sibling profiles, unexposed fields, unrelated concurrent
edits, and literal model IDs containing punctuation. Only full Settings `Make
default for new chats` may include an endpoint, and only when it was explicitly
edited and the user left the scoped checkbox checked. Its UI and logs contain only
a sanitized host plus a syntactic Local/LAN/Remote-or-unknown classification; no DNS
lookup, credential, userinfo, path, query, or fragment is used.

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
conversation.

Apply from the quick popover includes compaction mode. Compaction remains a sparse
`ConsoleContextPolicyOverrides` value in its existing
`console_conversation_context_policy` owner; it is never copied into
`console_generation_settings`. The Apply orchestration commits generation settings
and the complete context-policy snapshot live to the exact origin before yielding,
then persists generation metadata and that complete policy snapshot through their
respective existing owners. One durable failure cannot roll back or veto the other
live component or modal dismissal.

Each session retains a bounded process-local durability record whose component keys
are `generation_settings` and `context_policy`. The collapsed Model section shows a
warning badge; its expanded rail displays the failed components and a `Retry save`
action until the still-current component snapshots save successfully, a newer change
supersedes them, or the session closes. A quick-popover context-policy failure may be
labeled `compaction`; a full-modal failure is labeled `context settings` when it
contains other policy edits. Retrying context policy writes the complete still-current
policy snapshot and is guarded by its policy revision plus the captured conversation
identity, so it cannot overwrite a newer non-compaction policy change or clear a
newer failure.

Unsaved ordinary sessions stage both components without displaying a failure.
First persistence includes generation metadata with conversation creation and uses
the existing context-policy post-create flush; a failed component enters the same
visible durability state. Temporary sessions remain non-durable until promotion.
Promotion includes generation metadata in the conversation bundle and persists
compaction through its existing owner afterward; a compaction failure leaves the
successful promotion intact and visible as `compaction` not saved.

Default configuration failures use a separate bounded app-level record because the
failed owner affects future chats rather than one conversation. Failure before file
replacement reads `Not written to disk` and offers `Retry default save` plus
`Discard retry`. Failure after successful file replacement reads `Saved on disk;
running app refresh failed` and offers cache-only `Refresh running app` plus
`Dismiss`; it never repeats the disk mutation, and restart loads the saved values.
Discard and Dismiss acknowledge the pending recovery state and never imply rollback
of live or already-durable values. A newer explicit default action supersedes an
older failed intent, while locked reread and exact-field patching prevent stale retry
from overwriting unrelated or newer config edits.

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

The existing per-model profile schema already has the required precedence and full
Settings support, but current Console default saving writes broader provider/global
sections instead of the exact profile. Current Ctrl+T blank-chat creation also clones
the active session, so a global default can be saved successfully without controlling
the next obvious new chat. This decision makes blank-chat creation resolve the saved
default and reserves source cloning for explicitly source-owned flows.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep settings in process memory | Fails the required reopen-after-restart behavior and continues to make Apply appear ineffective. |
| Make ordinary Apply also mutate the global default | Silently changes future chats and contradicts the explicit exact-conversation scope; global mutation is allowed only through `Make default for new chats`. |
| Add a new preset/profile schema | Duplicates the existing exact-model profile owner and creates another precedence layer without user value. |
| Keep Ctrl+T cloning the active session | Makes `Make default for new chats` false for the primary blank-chat path; explicit Duplicate/Branch/Continue already cover intentional carryover. |
| Add a dedicated conversation-settings table | Adds a migration and repository surface without improving the required single-conversation lookup; versioned conversation metadata already provides the needed ownership and merge rules. |
| Store the full `ConsoleSessionSettings` dataclass | Would persist endpoints and runtime-only fields and would couple the storage schema to an internal Python type. |
| Put generation settings in the compaction policy store | Mixes independent owners and makes an unrelated context policy capable of blocking Provider/Model Apply. |

## Consequences

- Provider/Model Apply becomes durable for persisted conversations without a
  schema migration.
- Quick and full Console Settings can no longer disagree about whether their safe
  generation fields survive restart.
- Exact-model defaults reuse the existing model-profile schema; no new preset owner
  or precedence layer is introduced.
- `Make default for new chats` controls every eligible blank-chat path in the current
  process after runtime publication and across reboot. Existing/open and explicitly
  source-owned conversations remain unchanged.
- Provider endpoints remain configuration-owned. Only full Settings `Make default
  for new chats` may persist an explicitly edited, checked endpoint.
- Apply remains successful in live session state even when either durable write
  fails. The user sees the exact failed components persistently after the popover
  closes and can retry their still-current generation or complete context-policy
  snapshot.
- A per-session settings revision prevents an older persistence completion or
  retry from overwriting a newer Apply or a rebound conversation identity.
- Metadata parsing and serialization require a small versioned helper with an
  explicit allowlist and sibling-preserving merge.
- Provider changes must use one provider-aware rebase path from both settings
  surfaces and conversation hydration; model changes use the same path and draft
  provenance rules.
- Default config failure is app-global and distinguishes not-written state from
  already-written/runtime-stale state, while conversation durability remains local
  to the owning session.
- The popover interaction needs captured-click recovery and teardown-safe deferred
  callbacks, using the existing full-modal pattern rather than a new event system.
- Compaction keeps its existing schema and repository, while the shared Apply
  orchestration and session-local durability record coordinate honest outcomes
  across the two owners.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md)
- [TASK-22515](../tasks/task-22515%20-%20Make-Console-provider-Apply-update-and-persist-conversation-settings.md)
- [ADR-006: Provider-aware generation settings](006-provider-aware-generation-settings.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-052: Conversation memory and compaction policy](052-conversation-memory-and-compaction-policy.md)
