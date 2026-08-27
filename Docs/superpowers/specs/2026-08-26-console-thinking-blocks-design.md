# Console Thinking Blocks — Design

Date: 2026-08-26
Status: Approved
Related Task: [TASK-18932](../../../backlog/tasks/task-18932%20-%20Console-toggleable-live-reasoning-display.md)
Related ADR: [ADR-090](../../../backlog/decisions/090-console-thinking-block-ownership-and-replay.md)
Extends: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md), [ADR-066](../../../backlog/decisions/066-local-provider-thinking-controls.md), and the [Assistant Turn Grouping design](2026-08-21-console-assistant-turn-grouping-design.md)

## Goal

Show model-provided thinking in the Console as an honest, collapsible part of its
owning Assistant turn. Displayable thinking streams expanded while it is live, then
collapses automatically like tool activity. Thinking remains available after restart
and may be replayed to compatible local or hosted models under an explicit
per-conversation policy.

The feature must distinguish three facts that are currently easy to conflate:

1. Displayable reasoning text that a provider intentionally exposes to the user.
2. Actual evidence that a provider produced proprietary reasoning whose text Chatbook
   is not allowed to expose.
3. A safe agent-step preamble that describes planning but is not model reasoning.

Only the first two create model Thinking blocks. Capability, model family, or a
thinking-enabled setting alone is never evidence that a particular turn produced
thinking.

## Approved User Experience

An Assistant turn may contain ordered activity from several model rounds:

```text
Assistant
  ▾ Thinking                                      live
     <displayable provider reasoning streams here>
  ▸ fs_read · configuration                       success
  ▸ Thinking                                      done
  ▸ apply_patch · settings                        success

  <final answer>
```

Displayable thinking starts expanded when its first delta arrives. The first visible
answer delta or tool event collapses it once. If neither arrives, terminal settlement
collapses it. A manual expand or collapse cancels the pending automatic transition;
the application never fights the user's choice.

For actual proprietary evidence, the header is `Thinking · unavailable`. Expanding it
shows exactly:

```text
Proprietary thinking obfuscated - not available
```

That notice is application copy, not stored provider text. A turn with no actual
reasoning evidence shows no Thinking block and no notice.

Proprietary evidence follows the same disclosure lifecycle: it appears expanded when
the event first arrives, then auto-collapses once at the answer/tool boundary or
terminal fallback. If that boundary already passed before a terminal-only evidence
event arrives, it settles collapsed rather than flashing open. Manual interaction still
wins.

Historical model Thinking blocks restore collapsed. Each block expands independently
with the same mouse, Enter, Space, and existing `o` action semantics as tool
disclosures. No new global key binding or footer hint is introduced.

## Scope

### In scope

- Console model streams that expose displayable thinking through an adapter-approved
  reasoning field or a start-anchored `<think>...</think>` section.
- Content-free proprietary-evidence events emitted by provider adapters for the
  current turn.
- Durable, variant-owned thinking envelopes and per-conversation replay policy.
- Global presentation control in canonical F9 Settings, default on.
- Context/Memory control for optional replay: Auto, Include, and Exclude.
- Importable conversation exchange, synchronization, search/export privacy, and
  local/server persistence compatibility.
- Reuse of the Assistant-turn disclosure hierarchy shipped by TASK-19426.

### Non-goals

- Exposing private continuation content governed by ADR-063.
- Inferring chain-of-thought from model capability, settings, timing, token counts,
  empty assistant content, or provider name.
- Translating arbitrary reasoning between incompatible provider protocols.
- Sending proprietary markers to a model.
- Persisting every superseded Console text variant across restart. Today those
  variants are session-resident and only the selected answer is projected durably;
  this feature preserves that boundary while making each live variant own matching
  thinking state.
- Adding Thinking to human-readable transcript, Markdown, text, summary, title,
  search, usage, or logging surfaces.
- Changing legacy Chat-window transcript behavior.

## Terms and Truth Rules

### Displayable thinking

Reasoning text that the owning provider adapter explicitly classifies as safe for the
user-visible transcript. The adapter decision is authoritative; a generic field named
`reasoning_content` is not automatically displayable.

### Proprietary evidence

A content-free fact emitted by an adapter because the current provider response
actually contained or signaled private reasoning. The event proves occurrence only.
It carries no reasoning text, hashes, excerpts, lengths, or token-derived surrogate.

### Planning

The existing privacy-safe intermediate `STEP_MODEL.summary` presentation from
TASK-19426. It is renamed from Thinking to Planning when no actual model-thinking event
owns that position. Planning remains session-only agent activity and is unaffected by
the global model-thinking visibility setting.

### No evidence

No displayable delta and no proprietary-evidence event occurred in the current turn.
The UI renders nothing. Historical rows without a thinking envelope also mean no
recorded evidence; migrations never synthesize one.

## Provider Stream Boundary

The visible content stream remains plain strings. The provider gateway adds two narrow
event types beside the existing terminal tool-call item:

- `ProviderThinkingDelta(text, source_format)` carries displayable text.
- `ProviderProprietaryThinkingEvidence(source_format)` carries occurrence only.

Events also inherit the frozen provider, model, and protocol resolution for the run.
Their representations and logs are content-free; raw displayable text travels only in
the value that the turn accumulator owns.

Provider adapters decide disposition:

- Direct/local llama.cpp and compatible vLLM paths may classify split reasoning or a
  start-anchored think section as displayable.
- Hosted adapters classify each documented field as displayable, proprietary, or
  ignored. Kimi preserved thinking remains proprietary under ADR-063.
- A provider that returns no actual reasoning event emits no block even when thinking
  was enabled in the request.
- Unsupported or ambiguous fields fail closed into ignored or proprietary policy;
  shared gateway code never guesses from a vendor-neutral key.

### Streaming `<think>` splitter

ADR-066's start-anchored stripping becomes a reusable streaming splitter. It must:

- recognize an opening tag split across arbitrary chunks;
- tolerate leading whitespace and the empty think prefix produced by some templates;
- emit reasoning and visible answer through separate channels;
- preserve literal mid-reply `<think>` text;
- never emit a partial opening tag;
- settle an unclosed start-anchored block as failed rather than reclassifying it as a
  complete replayable block; and
- keep raw values out of diagnostics.

The non-streaming path applies the same classification to the complete response.

## Thinking Envelope

Assistant generation state gains a nullable `thinking_blocks_json` field. The field is
separate from visible `content`, local-only `metadata_json`, and private
`provider_continuation_json`.

Version 1 is a bounded object with an ordered `blocks` array. Each block contains:

- stable block ID;
- model-round ordinal;
- visibility: `displayable` or `proprietary`;
- frozen provider key, model identifier, and protocol/API mode;
- source format needed for exact compatible replay;
- terminal status: `complete`, `stopped`, or `failed`; and
- exact text only when visibility is `displayable`.

The envelope stores no rendered header, proprietary notice, disclosure state, token
estimate, summary, HTML, terminal markup, or inferred capability. Proprietary blocks
are structurally unable to carry text.

Raw displayable text is stored exactly after boundary validation. Terminal rendering
uses the existing terminal-safe projection without modifying stored text. Block
objects use content-free `repr` and error messages.

Parsers enforce schema version, allowed keys and enum values, provider/model/protocol
bounds, block count, per-block size, and total size. An unknown version already present
in durable state remains opaque and is preserved by unrelated writes; it is not
rendered or replayed and generation-mutating actions fail closed. Imports and incoming
sync writes reject unsupported versions before mutation. Malformed known versions
produce a content-free warning and are never partially accepted.

## Ownership, Variants, and Persistence

Each live `ConsoleVariant` owns a generation envelope: answer content, thinking
envelope, usage, and the separately governed continuation/provenance associated with
that generation. `_VariantStreamBase` snapshots the same state before regeneration.
Selecting a variant swaps the complete generation envelope together, so the answer
cannot be paired with thinking or continuation from a different attempt.

The existing durability boundary remains explicit:

- Console text variants are currently session-resident.
- The selected variant's answer is the durable message-row projection.
- This task makes its thinking and continuation projection atomic with that selected
  answer.
- Earlier live variants retain their own envelopes during the session. Persisting all
  superseded text variants across restart is a separate feature.

Normal generation finalization writes selected answer content and
`thinking_blocks_json` in the same ChaChaNotes message transaction. Provider
continuation keeps ADR-063's earlier checkpoint timing and authoritative replay role;
it is never copied into the thinking field. The transaction's trigger-written sync
intent is authoritative local state, while Sync v2 projection remains an idempotent
post-commit reconciliation rather than a claimed cross-database transaction.

All persistence seams must round-trip the field: ordinary creation/update, stopped or
failed settlement, dispatch recovery, continuation-created owners, local and server
adapters, active-path hydration, selected variant changes, sync, conflict handling,
and import/export. Generic content writes preserve even unknown thinking envelopes by
default. Only an explicit generation replacement, confirmed assistant edit, discard,
or deletion may clear them.

Whole-record conflict policy applies to selected answer content, thinking, and
continuation projection. Blocks are never merged individually. Deletion/tombstoning
removes durable thinking. Transcript pruning only unmounts a whole rendered Assistant
turn. Context-budget eviction only omits an owner group from one request and never
mutates persistence.

### Regeneration and editing

Regeneration creates a new live variant with a fresh thinking envelope. Earlier live
variants keep their own blocks. Abandoned regeneration restores answer, thinking,
usage, and continuation from `_VariantStreamBase`; it cannot leave the failed
attempt's reasoning attached to the restored answer.

Assistant editing keeps the current edit-in-place behavior. Before applying an edit,
the UI explains that generation provenance for the selected variant will be cleared.
Confirmation clears its thinking and provider continuation while leaving other live
variants unchanged. Editing never fabricates a user-owned reasoning record.

A proprietary-evidence block survives ordinary provider-continuation lifecycle
discard because it records an actual historical occurrence without retaining private
text. Editing or deleting the owning generation removes it.

## History Replay Policy

Every conversation owns a dedicated nullable `thinking_history_policy` field, included
in conversation sync and round-trip export. Null and missing legacy values mean Auto;
stored values are:

- `auto` — default. Include a compatible optional displayable block only when the
  target adapter says the fully resolved request expects it.
- `include` — include every complete, replay-eligible displayable block compatible
  with the frozen target protocol. If an otherwise eligible block cannot be serialized
  safely, block the request before contacting the provider.
- `exclude` — omit all optional displayable thinking blocks.

`Required` is a read-only effective state, not a stored fourth value. It appears when
ADR-063 continuation requires replay for the current target. The user's saved optional
preference remains intact for a later model switch.

The Context/Memory surface edits the conversation value and offers “Save as default
for new conversations.” The global default is `auto`; saving another value affects
only subsequently created conversations. Existing conversations with no value resolve
to Auto. Unknown imported values resolve to Auto with a content-free warning.

The provider adapter resolves replay against the complete prepared target, not model
name heuristics or raw settings. Source encoding is retained so a compatible adapter
can restore its exact protocol representation. Chatbook never injects thinking as
ordinary assistant text and never performs generic cross-provider translation.

One resolved history projection feeds both serialization and context token budgeting.
It combines visible turns, optional compatible thinking, and mandatory private
continuation without duplicate send or duplicate count. Owner groups are retained or
evicted atomically. Proprietary, failed, malformed, unknown-version, and otherwise
ineligible blocks are not candidates for optional replay.

## Settings and Presentation State

Canonical F9 Settings gains `Show model thinking` under Console Behavior. It is a
device-local persisted configuration value, excluded from conversation sync/export,
and defaults On. Changing it applies immediately to mounted history and live streams
without restarting.

The setting is presentation-only:

- Off hides displayable and proprietary model Thinking disclosures.
- Capture, persistence, provider continuation, and replay policy are unchanged.
- Planning and tool activity remain visible.
- Re-enabling reconstructs historical rows from stored envelopes.

The refresh is targeted to affected Assistant turns. It preserves transcript scroll,
answer focus, tool state, and per-session disclosure state where the same block remains
available. Hiding a selected Thinking row clears that selection and navigation skips
hidden rows.

Disclosure expansion is session presentation state keyed by trusted assistant owner,
session identity, and an internal hash/ordinal derived from the stable block ID.
Imported IDs never become raw Textual DOM IDs. Model block IDs map explicitly back to
their assistant owner for Inspector, copy, selection, navigation, and pruning.

Historical collapsed details are genuinely lazy: the mounted disclosure receives an
empty/bounded projection while collapsed and resolves the full sidecar only for
expansion, copy, or Inspector. Live deltas update the existing disclosure in place;
they do not remount the turn or reset selection/scroll. Any manual disclosure action
cancels its pending auto-collapse.

Statuses describe reasoning capture, not later tool outcomes:

- `live` while receiving displayable deltas;
- `done` after a complete block;
- `stopped` after user cancellation;
- `failed` after malformed/incomplete capture or generation failure; and
- `unavailable` for proprietary evidence.

The activity header layout and enum expand to fit these literal statuses at supported
terminal widths. Expanded bodies remain wrapped, dim, literal, and terminal-safe.

## Planning Compatibility

TASK-19426's safe intermediate marker remains useful, but it must not impersonate raw
reasoning. When no actual model-thinking event owns that model round, the marker label
is `Planning`. When an actual Thinking block exists, no duplicate Planning marker is
created for the same round.

The existing conservative `safe_intermediate_thinking_summary` rules continue to
reject provider-private and think-tag shapes. Renaming the presentation does not widen
what safe step summaries may disclose or persist them.

## Imports, Exports, and Privacy

Every format is classified by purpose:

- Importable/round-trip conversation formats include the selected variant's thinking
  envelope, conversation replay policy, and the same sensitivity warning used for
  private provider continuation. They preserve every variant the format already owns;
  this task does not add session-only superseded variants to the export contract.
- Human-readable, answer-oriented, text, Markdown, summary, title, clipboard-answer,
  and ordinary transcript exports omit model Thinking blocks and the proprietary
  notice by default.

Import preflight validates every thinking envelope and policy for one conversation
before mutating that conversation. Unsupported versions, invalid bounds, provenance,
or structure exclude and report the conversation without a partially attached
envelope. The design does not claim that unrelated conversations share one importer
transaction.

Displayable thinking uses the same at-rest and Sync v2 protection regime as message
content; no second application key is introduced. Proprietary raw text remains solely
inside ADR-063's private continuation field where required. Export warnings appear if
either private continuation or displayable thinking is present.

FTS, search, titles, summaries, diagnostics, logs, exceptions, usage displays, speech,
and answer-only actions consume explicit visible-content projections. Tests inspect
every default durable owner reached by the real persistence path, including sidecars,
sync payloads, and exports; merely proving the primary message table safe is
insufficient.

## Migration and Compatibility

The implementation rechecks the current ChaChaNotes schema number immediately before
adding the migration; this design does not pin a provisional version. Migration
fixtures start from a genuine historical schema and assert the current version
relatively.

Existing message rows receive a nullable field with no backfill. Missing means no
recorded evidence. Existing conversations receive Auto replay behavior unless they
later save another policy.

Persistent local and server adapters expose versioned thinking round-trip capability.
A server-backed conversation cannot send while its backend cannot preserve the
current envelope/policy version. The refusal occurs before contacting the model and
offers an upgrade-oriented explanation; persistent conversations never silently
degrade to session-only reasoning. Older sync peers must reject affected writes or
operate read-only rather than down-converting unknown data.

Schema-migration verification uses isolated temporary data directories only. A live
app launch must not migrate the developer's shared profile while other worktrees still
run older schema code.

## Failure and Resource Handling

Assistant content and its terminal thinking envelope commit atomically. If that write
fails, the turn remains visibly unsaved and retryable; Chatbook does not claim the
answer was durably saved. Earlier mandatory continuation checkpoints retain their
ADR-063 recovery behavior.

Block count and byte limits bound live memory, persistence, sync, and import. Exceeding
a limit stops the request/capture, settles the partial block as failed where durable
settlement is possible, and excludes it from ordinary optional replay. It never
silently truncates a block while labelling it complete.

The feature persists on normal completion, user stop, and handled failure, not per
token. A process crash during an otherwise optional local reasoning stream may lose
that unfinished partial block; periodic per-token checkpoints and durable blank rows
are intentionally rejected. Mandatory provider continuation keeps its independent
write-ahead durability contract.

## Verification Strategy

### Provider and stream tests

- Displayable and proprietary event classification for each opted-in adapter.
- No-event control legs for thinking-capable providers.
- Complete event-sequence tests, including repeated, interleaved, conflicting, and
  misplaced reasoning/tool/terminal events.
- Split reasoning and `<think>` parsing across every chunk boundary, empty prefixes,
  leading whitespace, unclosed blocks, cancellation, and literal mid-answer tags.
- Raw proprietary values absent from events, representations, diagnostics, and logs.

### History and persistence tests

- Auto, Include, Exclude, and effective Required resolution.
- Exact compatible encoding, preflight refusal for incompatible eligible blocks, and
  no generic translation.
- One projection used for serializer and budget, with owner-atomic eviction and no
  duplicate send/count.
- New, complete, stopped, failed, recovered, and continuation-created message paths.
- Regeneration restore/finalize, live variant selection, confirmed assistant edit,
  deletion, transcript pruning, and context eviction.
- Genuine historical migration, restart hydration, sync/hash/conflict behavior,
  server capability refusal, import preflight, and round-trip export.

### UI tests

- Expanded-live to one-time auto-collapse at first visible/tool activity and terminal
  fallback.
- Manual override, restored collapsed state, independent expansion, show/hide, focus,
  selection, navigation, pruning, and lazy detail mounting.
- In-place live updates retain widget identity and scroll position.
- Wide/narrow painted layout for all five statuses using the production stylesheet.
- Copy and Inspector resolve full displayable blocks while proprietary blocks expose
  only the fixed notice.

### Privacy and negative controls

- No marker for capability-only or no-evidence turns.
- The exact proprietary notice appears only after an actual evidence event.
- No raw proprietary content enters thinking storage, transcript, search, title,
  summary, usage, log, error, speech, or export surfaces.
- Human-readable exports omit model thinking; importable exports retain it.
- Decoded inspection of every default durable owner, not only the main database.
- Mutation/negative controls prove each evidence gate, visibility filter, and
  persistence preservation assertion can fail.

Targeted tests cover the modified and behavior-reachable suites. Per repository
policy, a full test sweep requires explicit user opt-in. Live TUI verification uses an
isolated scratch profile and waits for authoritative state, DOM, and paint readiness.
Provider contract checks trace only event structure and use real services where
available without logging raw reasoning.

## Delivery Decomposition

TASK-18932 becomes the feature parent. After this written specification is approved,
the implementation plan will create dependency-ordered atomic child tasks for:

1. ADR/schema, envelope validation, selected-variant persistence, sync, and backend
   capability.
2. Provider thinking events, local splitter, and resolved history projection.
3. Assistant-turn disclosure UI, settings, and Planning compatibility.
4. Import/export privacy, compatibility gates, documentation, and joined integration
   coverage.

No child implementation begins until the plan identifies exact ownership seams and
targeted verification commands.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/090-console-thinking-block-ownership-and-replay.md`

Reason: this feature changes message/variant storage, schema and migration behavior,
sync/conflict ownership, provider streaming contracts, optional history replay,
privacy/export boundaries, and long-lived Console interaction structure. ADR-090
extends ADR-063 and ADR-066 without exposing or duplicating private continuation.
