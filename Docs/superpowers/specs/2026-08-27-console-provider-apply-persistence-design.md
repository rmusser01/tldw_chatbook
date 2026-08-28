# Console Provider Apply and Conversation Persistence Design

**Status:** Approved
**Date:** 2026-08-27
**Task:** TASK-22515
**ADR:** ADR-095

## Goal

Make Apply in the Console Provider/Model popover reliably close the popover,
update the exact conversation immediately, and preserve the conversation's safe
generation settings across restart. Give mouse and keyboard activation the same
semantic path. Use the same durable conversation-settings contract from the full
Console Settings modal.

## User-Visible Contract

Apply means **apply to this conversation now**.

- The exact conversation that opened the modal is updated even if another tab
  becomes active before the result is handled.
- Every execution context resolved after Apply observes the new conversation
  settings. A request that already captured its execution context remains
  unchanged.
- The modal closes after a valid Apply from either mouse or keyboard.
- A persisted conversation restores its applied settings after the app restarts.
- An unsaved ordinary chat saves the settings with its first persistence.
- A temporary chat keeps the settings only for its temporary lifetime unless it
  is promoted.
- Invalid input keeps the modal open and identifies the field that must be fixed.
- A local metadata save failure does not roll back the live conversation setting;
  the notification states that the setting is active now but was not saved for
  restart.

The Provider/Model popover contains no compaction controls and returns no compaction
value. Compaction remains available in the full Settings Context view through its
existing independent owner and persistence path.

## Selected Architecture

The existing Console session remains the live owner. The existing conversation row
becomes the durable owner of an allowlisted generation-settings snapshot.

One session-controller commit accepts a typed intent containing:

- the stable originating session ID;
- the conversation identity captured when the surface opened, if one exists;
- the validated target provider/model and values submitted by the surface;
- the exact set of fields exposed by that surface;
- a full-modal endpoint only when its draft is bound to the target provider.

The controller is the only provider-rebase owner. The commit validates that the
origin still exists and still has the captured conversation identity, rebases the
settings, replaces that session's live settings, increments its monotonic settings
revision, synchronizes the Console summary/control bar immediately, and persists
the safe snapshot when the session has a conversation row. It never falls back to
the currently active session. If the origin closed or was rebound, it reports
`Chat closed; nothing applied`.

The quick popover and full modal both call this seam. No parallel quick-settings
service, new database table, or compaction transaction is introduced.

## Durable Data

`conversations.metadata` gains one owned namespace:

```json
{
  "console_generation_settings": {
    "version": 1,
    "provider": "OpenAI",
    "model": "gpt-5",
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": null,
    "top_k": null,
    "max_tokens": null,
    "seed": null,
    "presence_penalty": null,
    "frequency_penalty": null,
    "reasoning_effort": null,
    "reasoning_summary": null,
    "verbosity": null,
    "thinking_effort": null,
    "thinking_budget_tokens": null,
    "streaming": true
  }
}
```

Only the listed fields are serialized. `base_url`, credentials, credential
references, system prompt, pinned prefill, character identity, and runtime
provenance are excluded. Existing owners continue to persist system prompt and
pinned prefill.

The metadata helper follows the existing roleplay/speech patterns:

- parse only mappings with supported versions and valid field types;
- fail closed on absent, malformed, or unknown values;
- preserve every sibling metadata key during writes;
- use optimistic conversation versions and bounded conflict retry;
- refuse to overwrite a future unsupported version.

Persisting the complete rebased safe snapshot on either settings surface prevents a
prior provider's hidden tuning fields from surviving a provider switch.

## Provider Rebase

A provider change is not implemented with a field-only `replace(...)`. The session
controller owns this operation for both settings surfaces.

1. Normalize the selected provider and model.
2. If the provider is unchanged, preserve compatible current values and overlay the
   submitted surface fields.
3. If the provider changed, build provider/model defaults for the selected provider.
4. Resolve the selected provider's configured endpoint through the existing
   provider-resolution path.
5. Discard the previous provider's endpoint and incompatible provider-specific
   reasoning/thinking values.
6. Overlay only values exposed by the submitting surface and supported by the
   selected provider.
7. Accept a full-modal session endpoint only when the endpoint draft is explicitly
   bound to the selected provider; the quick popover never submits an endpoint.
8. Mark the resulting snapshot as user-authored and commit it to the exact session.

Conversation hydration performs the same ordering: parse the saved provider first,
build defaults for that provider, then overlay the remaining saved fields. A model
missing from the current catalog remains a valid explicit custom selection; an
unconfigured provider produces the existing blocked readiness state rather than a
silent fallback.

## Apply Flow

1. Opening either settings surface captures the origin session ID and current
   conversation identity before catalog loading or other asynchronous work.
2. Widget edits remain a local draft until Apply.
3. Apply validates provider, model, and numeric fields. Validation failure keeps the
   surface open and focuses the first invalid control.
4. The surface submits its typed intent to the session controller.
5. The controller performs provider-aware rebasing, commits the immutable settings
   snapshot to the origin session, and increments that session's settings revision.
6. Mounted summaries and provider/model controls synchronize immediately.
7. The modal dismisses.
8. If the session is already persisted, its safe metadata snapshot is written with
   sibling-preserving optimistic concurrency. A retry proceeds only while the
   settings revision and conversation identity still match and the owned metadata
   key has not been superseded. Sibling-only conflicts may be merged and retried.
   If the session is unsaved, the live snapshot is serialized during first
   conversation persistence.
9. Persistence failure leaves the live change applied and emits one precise warning:
   `Applied to this conversation, but could not save for restart.`

The execution boundary is objective: consumers that already hold an execution
context keep it; consumers resolving after step 5 read the new settings. The design
does not special-case user sends, retries, regenerations, queues, or agent work.

## Mouse and Keyboard Reliability

The quick popover reuses the full Settings modal's established Textual behavior:

- the temperature input releases mouse capture on click/blur;
- a click redirected to a captured input is hit-tested against visible buttons;
- a recovered Apply invokes `button.press()`;
- `Button.Pressed` remains the only semantic Apply event;
- redirected recovery cannot press the button twice;
- backdrop dismissal remains cancellation, never Apply.

Deferred UI callbacks tolerate a dismissed popover. `_sync_fold_hint` and equivalent
lookups handle `NoMatches` or verify mounting before querying descendants.

## Presentation and Validation

The compact surface keeps its existing purpose. It does not gain a new persistence
state machine or additional configuration sections.

- Title: `Provider / Model`
- Visible labels: Provider, Model, Temperature, Streaming
- Actions: `Full settings…`, `Cancel`, `Apply`
- Scope copy: `Applies to this conversation`
- Unsaved chat copy: `Saved with the conversation after its first message`
- Temporary chat copy: `Temporary until this chat is promoted`

The current compaction threshold, help, and mode controls are removed from this
popover. Compaction remains available in the full Settings Context view.

Temperature parsing no longer silently restores the old value. Invalid input shows
an inline error and keeps the draft intact. Success uses one concise notification;
there is no `Next send` label because the conversation setting changes immediately.

## Full Settings Alignment

Full Console Settings uses the same provider rebase and session commit. Its safe
generation fields therefore have the same restart behavior as the quick popover.

The full modal's endpoint remains configuration-owned. A custom endpoint can affect
the live session, but it survives restart only through the existing Save-as-default
path and warning. This task does not copy endpoints into conversation metadata.

System prompt and pinned prefill retain their existing storage paths; the shared
commit coordinates live `ConsoleSessionSettings` replacement without duplicating
their durable data in `console_generation_settings`.

## Resume and First Persistence

- **Resume:** Build defaults for the saved overlay provider, apply the validated
  safe snapshot, resolve endpoint/configuration afresh, and set `source="user"`.
- **Unsaved ordinary chat:** Keep the applied live snapshot on the session and
  include its serialized safe form when first creating the conversation row.
- **Temporary chat:** Do not create durability merely because Apply was clicked.
  Promotion writes the current safe snapshot with the new conversation.

The serializer and provider-rebase helper are reusable by the future fork flow under
ADR-092, but this task does not implement or test that unshipped flow.

## Error Handling

- Missing origin session: no retargeting; notify and discard the result.
- Invalid fields: keep the modal open with inline feedback.
- Unsupported saved metadata version: preserve it, warn once when relevant, and do
  not overwrite it automatically.
- Metadata version conflict: reload and retry only for sibling-only changes while
  the session settings revision, conversation identity, and owned metadata base
  still match. Abort an obsolete Apply rather than overwriting a newer one.
- Persistence failure: retain the live applied settings and report only the restart
  consequence. Do not mention compaction.
- Unconfigured saved provider: preserve the selection and expose the normal blocked
  readiness/recovery UI; never silently substitute another provider.
- Deferred callback after dismissal: treat the absent descendant as normal teardown.

## Testing Strategy

Focused tests cover:

1. A real routed Textual `Click` after the temperature Input owns mouse capture;
   Apply fires once and dismisses.
2. Keyboard focus plus Enter reaches the same `Button.Pressed` handler.
3. Dismissal before deferred fold-hint synchronization does not raise `NoMatches`.
4. Invalid temperature remains open and shows an inline error.
5. Applying while another session becomes active updates only the captured origin.
6. A consumer resolving settings after Apply receives the new values; an already
   captured execution context remains unchanged.
7. Quick and full Settings round-trip safe generation fields through the shared
   commit and restart hydration.
8. Provider switching begins with a non-null old endpoint and proves that endpoint
   is not retained.
9. Metadata writes preserve speech, roleplay, prefill, and unrelated sibling keys;
   bounded conflict retry cannot let an older Apply overwrite a newer one.
10. Corrupt and future-version overlays fail closed without destructive overwrite.
11. Unsaved-first-persistence and temporary-promotion flows preserve the defined
    durability boundary.
12. The quick popover contains and returns no compaction controls or value; full
    Settings Context compaction behavior and its existing tests remain unchanged.

Verification uses only affected pytest modules plus targeted lint/format and
`git diff --check`. A full-suite sweep requires explicit user approval under the
repository testing policy.

## Non-Goals

- No global provider/default mutation from Apply.
- No endpoint or credential persistence in conversation metadata.
- No new database table or schema migration.
- No compaction ownership, storage, transaction, error, or full-Settings UX changes;
  the unrelated controls are removed only from the Provider/Model popover.
- No conversation-fork implementation or integration test.
- No attempt to mutate a request that already captured its execution context.
- No live model-catalog refresh inside Apply.
- No generic Console settings refactor outside the shared commit and rebase seams.

## ADR Check

**ADR required:** yes
**ADR path:** `backlog/decisions/095-conversation-owned-console-generation-settings.md`
**Reason:** This changes durable conversation ownership, restart hydration,
provider-resolution precedence, and the cross-surface settings contract.

## Acceptance Mapping

- **AC1:** exact origin capture, routed mouse recovery, keyboard convergence, and
  teardown-safe dismissal.
- **AC2:** immediate session commit and immutable already-captured execution context.
- **AC3:** one commit seam and one safe snapshot for both settings surfaces.
- **AC4:** metadata serialization plus provider-first hydration.
- **AC5:** provider-aware rebase and endpoint exclusion regression.
- **AC6:** first-persistence, temporary, and promotion behavior.
- **AC7:** removal of compaction from the Provider/Model popover and preservation of
  the independent full-Settings owner.
- **AC8:** inline validation and deferred-callback regression.
- **AC9:** focused interaction, persistence, concurrency, hydration, and endpoint
  coverage.
