# Console Native Chat Core Design

Date: 2026-05-21
Status: User-approved design direction, pending written-spec review
Primary Repo: `tldw_chatbook`
Scope: Shared native chat core, wired through Console first

## Summary

The Console should become a first-class chat and agentic control surface rather than a native shell wrapped around legacy chat widgets. This design defines a shared native chat core that Console uses first, with legacy chat screens left in place until they can be migrated later.

The priority is long-term stability and maintainability over short-term compatibility shortcuts. Console should own its visible composer, transcript, selected-message action model, provider streaming state, and recovery states through explicit services and state contracts. Legacy chat widget internals should no longer be the hidden source of truth for Console behavior.

## Goals

- Make Console chat send, streaming, stop, blocked-send recovery, and rendered transcript behavior reliable without depending on hidden legacy widgets.
- Create reusable service boundaries that legacy chat screens can migrate onto later.
- Validate a real local llama.cpp OpenAI-compatible streaming loop at `http://127.0.0.1:9099`.
- Preserve the approved terminal-native Console shell, three-column layout, compact density, keyboard-first interaction model, and bottom status bar.
- Render transcript messages with clear terminal rules and compact default spacing.
- Support selectable messages with a contextual action row.
- Show unavailable or future paths as explicit `WIP` or placeholder states rather than silently disabling or hiding them.
- Preserve large-paste collapse behavior in the Console composer.

## Non-Goals

- Do not rewrite every legacy chat screen in this slice.
- Do not remove `ChatWindowEnhanced` from the codebase in this slice.
- Do not implement full Library Search/RAG, MCP runtime management, ACP task execution, or workspace sync in this slice.
- Do not treat screenshots, SVG mockups, or code-generated layouts as approval artifacts. Visible Console states require actual rendered CDP/textual-web screenshots before implementation signoff.
- Do not silently fall back from streaming to non-streaming when the target provider cannot stream. If fallback is offered, the UI must state what happened and require an explicit retry/fallback action.

## Approved Direction

The selected approach is a shared native chat core, wired through Console first.

Rejected alternatives:

- Console-only compatibility adapter around legacy send handlers. This is faster but keeps legacy widgets as the hidden source of truth.
- Big-bang replacement of Console and all legacy chat screens. This has too much regression risk for one slice.

The shared-core approach gives Console a durable architecture now while preserving a staged migration path for older chat surfaces.

## Architecture

The core should be independent of Textual widgets. Widgets render state and emit user intent; services own chat behavior.

```text
Console UI
  -> ConsoleChatController
    -> ChatSessionStore
    -> ChatMessageStore
    -> ProviderGateway
    -> MessageActionService
    -> ChatRunState
```

### ConsoleChatController

Owns Console behavior:

- send draft
- stop active run
- retry blocked or failed run
- select transcript message
- invoke message action
- switch chat tab or session
- surface visible run state to Console widgets

It should coordinate services, not query Textual widgets for business state.

### ChatSessionStore

Owns session and thread state:

- active Console session
- session list
- conversation/thread persistence
- selected variant per assistant turn
- session metadata needed for later workspace handoff

Existing DB adapters should be reused where practical, but accessed behind this service boundary.

### ChatMessageStore

Owns message persistence and transcript state:

- user, assistant, system, recovery, and tool messages
- stable message IDs
- optimistic updates while streaming
- streaming chunk append/finalize
- stopped or failed partial responses
- regenerated variants tied to one assistant turn

Widget mutation must not be the persistence mechanism.

### ProviderGateway

Owns provider execution:

- provider/model resolution
- provider readiness
- local llama.cpp URL and model discovery
- streaming request lifecycle
- stop/cancel support
- error normalization

The first hard target is a local llama.cpp OpenAI-compatible server at:

```text
http://127.0.0.1:9099
```

When possible, model discovery should read:

```text
GET /v1/models
```

### MessageActionService

Owns transcript action behavior:

- Copy
- Edit
- Save as...
- Regenerate
- Continue/extend from message
- Feedback thumbs up/down
- Delete
- Variant navigation with `<` and `>`

Any unavailable destination or operation must return an explicit unavailable result containing visible user-facing copy.

### ChatRunState

Owns visible run-state transitions:

```text
idle -> validating -> streaming -> completed
idle -> validating -> blocked
streaming -> stopped
streaming -> failed
failed -> retrying -> streaming
```

The Console should render these states in the transcript, inspector, composer action row, and bottom status bar as appropriate.

## Console Send Flow

```text
Composer submit
  -> validate non-empty draft
  -> resolve active workspace and session
  -> resolve provider and model
  -> create persisted user message
  -> create pending assistant message
  -> start provider stream
  -> append streamed chunks
  -> finalize, stop, or fail
```

Required behavior:

- Composer text clears only after the user message is accepted into the transcript.
- Blocked sends preserve composer text.
- Blocked sends create a visible recovery state, not a silent notification-only failure.
- Streaming updates the assistant message in place.
- Stop leaves the partial assistant message visible and marked as stopped.
- Failed runs leave a visible inline error with retry/recovery options.
- Completed runs unlock the selected-message actions.

## llama.cpp Streaming Target

The native core must support the local llama.cpp target as the first end-to-end provider QA path.

Provider assumptions:

- OpenAI-compatible HTTP API.
- Base URL: `http://127.0.0.1:9099`.
- Chat completions path: `/v1/chat/completions`.
- Streaming response format: OpenAI-compatible SSE chunks.
- Models path, when available: `/v1/models`.

Readiness behavior:

- If the server is reachable and a model can be resolved, provider state is ready.
- If the server is unreachable, Console shows `Provider blocked: llama.cpp server is not reachable at 127.0.0.1:9099`.
- If no model can be resolved, Console shows `Provider blocked: select or configure a llama.cpp model`.
- If streaming fails after dispatch, Console shows a failed assistant turn with the error summary and retry/fallback choices.

Streaming is required for this slice. A non-streaming fallback may exist later, but it must be explicit and user-visible.

## Transcript Rendering

The transcript is a navigable message list, not a passive log.

Default compact rendering:

```text
────────────────────────────────────────────────────────────
User
What changed in the workspace since yesterday?
────────────────────────────────────────────────────────────
Assistant
Here are the relevant changes...
────────────────────────────────────────────────────────────
```

Rules:

- Use full-width terminal rules that match the existing Console grid styling.
- Keep unselected messages vertically compact.
- Do not show action chrome on unselected messages.
- Preserve readable wrapping inside the transcript panel.
- Render system/recovery messages in the same flow with explicit labels.
- Tool messages and approval messages can use specialized labels, but should follow the same selection and rule grammar.

## Message Selection And Actions

When a message is selected, the transcript shows a contextual action row.

Standard selected action row:

```text
Copy | Edit | Save as... | ♻ | ---> | 👍/👎                 🗑
```

Selected action row when regenerated variants exist:

```text
Copy | Edit | Save as... | < | > | ♻ | ---> | 👍/👎          🗑
```

Meaning:

- `Copy`: copy message content.
- `Edit`: edit the selected message where permitted.
- `Save as...`: open a modal with available export/conversion destinations.
- `♻`: regenerate the assistant response.
- `<` / `>`: cycle regenerated variants.
- `--->`: continue or extend from the selected message.
- `👍/👎`: record feedback where supported.
- `🗑`: delete the selected message where permitted.

If a terminal or Textual path cannot render emoji reliably, the implementation may use stable textual fallbacks while preserving the same action order and intent:

```text
Copy | Edit | Save as... | Regen | ---> | Up/Down           Delete
```

Unavailable actions:

- Must remain visible if they are part of the expected action set.
- Must be marked disabled or `WIP`.
- Must expose why they are unavailable, for example `Save as Note (WIP: notes adapter unavailable)`.
- Must not silently no-op.

## Keyboard And Mouse Interaction

Major Console areas must be reachable without a mouse.

Required keyboard model:

- `Tab` moves through major screen areas: left rail, transcript, inspector, composer, footer/status.
- Transcript focus supports `Up`/`Down` and `j`/`k` for message selection.
- `Enter` on a selected message reveals or focuses the action row.
- `Enter` on a focused action invokes the same behavior as click.
- `Esc` collapses the action row and returns to compact transcript mode.
- Composer shortcuts must not conflict with transcript navigation while the composer is focused.

Mouse model:

- Clicking a message selects it.
- Clicking an action invokes it.
- Clicking away from the selected message may collapse the action row if focus leaves the transcript.

## Composer Behavior

The native Console composer remains the visible source of truth for draft text.

Rules:

- Normal typing stays literal.
- Large pasted chunks over 50 characters collapse according to the existing Console large-paste design.
- `draft_text()` must return the exact send payload, including collapsed paste contents.
- Blocked send preserves the draft.
- Accepted send clears the draft only after the user message is recorded.
- While streaming, the primary send control should visibly become or defer to Stop.
- The composer must remain readable while focused, including long input. It may expand, push content, or use a bounded overlay, but it must not hide typed text.

## Save As Modal

`Save as...` should be a single entry point for converting or exporting a message.

Initial destinations can include:

- Note
- Chatbook artifact
- File/export path
- Library item or source, if available

Unavailable destinations must appear as disabled `WIP` options with explicit copy. This is required because users need to understand whether a missing path is unsupported, not configured, or simply not implemented yet.

## Regeneration And Variants

Regeneration creates a variant under the same assistant turn rather than creating an unrelated message.

Rules:

- The selected assistant message keeps a stable turn ID.
- Each regenerated response becomes a variant.
- `<` and `>` cycle variants when multiple variants exist.
- The currently selected variant is the one displayed in the transcript.
- `--->` continues from the currently selected variant.
- Deleting a variant should not delete the whole turn unless it is the only variant and the delete action is confirmed or clearly scoped.

## Workspace And Source Context

This slice should respect the current workspace context without implementing full workspace sync.

Rules:

- The active workspace is part of the session/run context.
- Workspace switching should not hide Library or Note items globally.
- Sources from other workspaces may be visible elsewhere in the app, but Console can only stage/manipulate sources valid for the current workspace.
- If a selected source or action is blocked by workspace policy, Console must show an explicit recovery message.

## Migration Strategy

The migration should be staged.

1. Introduce pure state and service contracts with tests.
2. Wire Console send/streaming/blocked-state behavior to the native core.
3. Replace transcript rendering with native message list state.
4. Add selectable message actions and `Save as...` modal routing.
5. Validate llama.cpp streaming and stop behavior through CDP/textual-web.
6. Leave legacy chat screens untouched except for shared service compatibility.
7. Later, migrate legacy chat screens onto the shared core one surface at a time.

Each implementation PR should leave the Console usable and screenshot-verifiable.

## Testing Requirements

Automated tests should be written before implementation.

Required unit coverage:

- Provider/model resolution for llama.cpp.
- Provider blocked states for unreachable server and missing model.
- Run-state transitions.
- Composer draft preservation on blocked send.
- Message append/finalize/stop/fail behavior.
- Variant creation, selected variant switching, and continuation target selection.
- Action availability and explicit `WIP` labeling.

Required mounted Textual coverage:

- Console composer submits through native state.
- Blocked send preserves the draft and renders recovery copy.
- Transcript renders compact message separators.
- Clicking a message selects it and shows the action row.
- `Tab` reaches transcript and composer.
- `Enter` activates selected message/action behavior.
- Stop state is visible while streaming.
- `Save as...` modal shows available and WIP destinations.

Required live/CDP verification:

- Idle Console screenshot.
- Typed composer screenshot.
- llama.cpp streaming screenshot.
- Completed response screenshot.
- Selected message with action row screenshot.
- Regenerated message with `<`/`>` variant controls screenshot.
- Blocked provider recovery screenshot.
- WIP/unavailable action screenshot.

The user must approve actual rendered screenshots before a visible Console state is considered approved.

## Risks And Mitigations

Risk: the shared core grows too large and becomes a second monolith.

Mitigation: keep services small and independently testable: controller, provider gateway, stores, action service, run state.

Risk: migrating Console off legacy widgets breaks existing DB or provider assumptions.

Mitigation: reuse existing persistence and provider adapters behind new service boundaries, and add regression tests before removing any compatibility path.

Risk: local llama.cpp behavior differs across server builds.

Mitigation: treat `/v1/models` and streaming support as probed capabilities and show explicit blocked/fallback states when a capability is missing.

Risk: action parity becomes too broad for one PR.

Mitigation: implement action availability plumbing first. Actions that cannot be completed safely must be visible as `WIP` with a reason rather than hidden.

Risk: keyboard navigation conflicts with composer editing.

Mitigation: route transcript navigation only while transcript has focus and keep composer key handling local to the composer.

## Acceptance Criteria

- Console send/stream/stop/recovery no longer depends on hidden legacy chat widgets as the source of truth.
- Console can complete a streaming chat loop against local llama.cpp at `127.0.0.1:9099`.
- Blocked sends preserve draft text and show visible recovery copy.
- Transcript messages render with full-width terminal separators and compact default spacing.
- Users can select messages by mouse and keyboard.
- Selected messages show the approved contextual action row.
- `Enter` and click invoke the same action behavior.
- Regenerated assistant turns expose `<` and `>` variant navigation.
- `--->` continues or extends from the selected message or selected variant.
- `Save as...` centralizes export/conversion choices.
- Unavailable actions and destinations are explicitly marked as `WIP` or placeholders with user-facing reasons.
- Automated tests cover the state contracts, mounted UI behavior, and action availability.
- Actual CDP/textual-web screenshots are captured and user-approved for the required Console states.
