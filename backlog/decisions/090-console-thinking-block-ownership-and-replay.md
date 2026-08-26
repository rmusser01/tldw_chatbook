# ADR-090: Store displayable thinking separately from private provider continuation

Status: Accepted
Date: 2026-08-26
Related Task: [TASK-18932](../tasks/task-18932%20-%20Console-toggleable-live-reasoning-display.md)
Related Spec: [Console Thinking Blocks](../../Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md)
Extends: ADR-063 and ADR-066

## Context

Console currently strips or privately retains several provider reasoning shapes while
the transcript can show only a safe, session-only intermediate model preamble. This
keeps hidden chain-of-thought out of the UI, but it also discards reasoning that local
and some hosted providers intentionally expose to users. Local llama.cpp and vLLM
models may additionally expect their visible thinking blocks in later history.

ADR-063 stores provider-private reasoning and tool continuation in
`provider_continuation_json`. That state is authoritative for mandatory replay and is
excluded from transcript, search, and ordinary export. Reusing it for displayable
thinking would mix two privacy classes, make local non-continuation reasoning awkward,
and risk rendering data whose provider contract explicitly keeps private.

The feature also needs honest absence semantics. Provider capability or a
thinking-enabled request does not prove that one turn produced reasoning. Proprietary
UI copy is justified only by an actual response event.

## Decision

Chatbook will store user-displayable model thinking in a dedicated, nullable,
versioned `thinking_blocks_json` field owned by the selected assistant generation.
It is separate from visible answer `content`, local-only metadata, agent TOOL markers,
and ADR-063 private provider continuation.

The accepted boundary is:

1. Provider adapters, not the shared gateway, classify current-turn reasoning as
   displayable, proprietary evidence, or ignored. A generic field name is not enough.
2. The stream adds a displayable delta event and a content-free proprietary-evidence
   event. No event means no Thinking block, even for a capable model.
3. Displayable blocks store exact bounded text plus stable ID, round, provider/model/
   protocol provenance, source encoding, and complete/stopped/failed status.
   Proprietary blocks are structurally text-free.
4. Proprietary evidence renders only the application constant
   `Proprietary thinking obfuscated - not available`. The constant is not provider
   data and never enters provider history. Its disclosure follows the same expanded-
   live, one-time auto-collapse, and manual-override lifecycle as displayable thinking;
   terminal-only evidence that arrives after the collapse boundary settles collapsed.
5. Every live Console text variant owns answer, thinking, usage, and its separately
   governed continuation/provenance as one generation envelope. Persistence and sync
   project the currently selected variant atomically, matching the existing selected-
   content durability boundary. This ADR does not make all session-only superseded
   text variants durable across restart.
6. Provider continuation remains authoritative for mandatory replay. Displayable
   thinking supplies only optional replay through a dedicated nullable conversation
   policy field whose stored values are Auto/Include/Exclude and whose null legacy
   value means Auto. The field participates in conversation sync and round-trip export.
   An effective read-only Required state reflects continuation rules without
   overwriting the saved optional preference.
7. One provider-resolved history projection supplies serialization and context
   budgeting. It prevents duplicate replay/counting and retains or evicts answer,
   optional thinking, and mandatory continuation as one owner group.
8. Adapters serialize only compatible, replay-eligible blocks in their exact source
   encoding. Chatbook never injects thinking as ordinary assistant text or translates
   it generically across providers. Include fails before send when an eligible block
   cannot be serialized safely.
9. `Show model thinking` is a device-local persisted presentation setting, excluded
   from conversation sync/export and default On. Turning it off hides displayable and
   proprietary Thinking rows without changing capture, persistence, continuation, or
   replay policy.
10. Importable conversation formats and Sync v2 preserve thinking and conversation
    replay policy under message-content protection. Human-readable exports, FTS,
    search, summaries, titles, logs, errors, usage, and speech exclude model thinking.
11. Existing rows are not backfilled. Missing storage means no recorded evidence. An
    unknown version already present durably is preserved by unrelated writes but is
    neither rendered nor replayed; imports and incoming sync reject unsupported
    versions before mutation.
12. Persistent backends must advertise round-trip support for the envelope/policy
    version. Unsupported server-backed conversations fail before provider contact
    rather than silently making a persistent turn lossy.

TASK-19426's safe intermediate model preamble remains a distinct session-only
`Planning` activity. It does not become durable thinking and is suppressed only when
an actual Thinking block already represents that model round.

Final answer content and its terminal thinking envelope commit in the same
ChaChaNotes transaction. ADR-063 keeps its earlier write-ahead timing for mandatory
tool continuation. The trigger-written local sync intent remains authoritative; Sync
v2 outbox projection is idempotent post-commit reconciliation, not a cross-database
transaction claim.

Assistant regeneration snapshots/restores the complete live generation envelope.
Assistant edit remains edit-in-place but explicitly clears thinking and continuation
for the selected generation after confirmation. Ordinary continuation discard leaves
a content-free proprietary-evidence marker intact; edit or deletion removes it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Put displayable thinking in `provider_continuation_json` | Mixes private mandatory state with user-visible optional history, risks disclosure, and does not fit local models without continuation. |
| Store thinking as ordinary TOOL or transcript messages | Pollutes the conversation tree, FTS, exports, and provider history while duplicating Assistant-turn ownership. |
| Append thinking to assistant `content` | Makes privacy filtering and compatible replay impossible and changes the visible answer sent to every provider. |
| Keep thinking session-only | Breaks restart, sync, import, and local-model history requirements. |
| Show a placeholder whenever a model supports thinking | Falsely implies hidden content was produced on turns with no actual evidence. |
| Translate every block to generic `<think>` text | Can corrupt provider history and crosses incompatible protocol/privacy boundaries. |
| Persist each streamed delta | Creates write amplification and durable blank/partial rows for optional display state; terminal persistence is sufficient outside ADR-063 mandatory checkpoints. |
| Make every superseded Console text variant durable in this task | Expands the feature into a separate variant-persistence redesign not required to preserve the currently selected conversation history. |

## Consequences

- ChaChaNotes, conversation policy storage, Sync v2, server persistence, import/export,
  and conflict/hash payloads require versioned updates.
- Provider adapters gain an explicit reasoning-disposition contract; opt-in requires
  joined positive and no-evidence control tests.
- The Assistant-turn activity status vocabulary expands to live, stopped, failed, and
  unavailable, with lazy historical detail and stable owner mapping.
- Displayable reasoning becomes sensitive conversation data and is included in
  round-trip exports with a warning, but receives no additional application-level key
  beyond the message-content protection regime.
- Server-backed chats on older contracts may be temporarily read-only for sends until
  upgraded; this is preferable to silent loss.
- A process crash can lose an unfinished optional local thinking block because the
  design persists at terminal settlement. Mandatory continuation keeps ADR-063's
  stronger checkpoint durability.
- Migration and UI live checks must use isolated data directories so an additive
  schema bump does not strand other worktrees on the shared developer profile.

## Links

- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-031: TUI Keybinding and Footer Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-063: Hosted Wire and Durable Tool Continuation](063-hosted-provider-wire-and-durable-tool-continuation.md)
- [ADR-064: DeepSeek Dual API Provider Boundary](064-deepseek-dual-api-provider-boundary.md)
- [ADR-066: Local Provider Thinking Controls](066-local-provider-thinking-controls.md)
- [Console Assistant Turn Grouping](../../Docs/superpowers/specs/2026-08-21-console-assistant-turn-grouping-design.md)
