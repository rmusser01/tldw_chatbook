# Console Provider Apply and Conversation Persistence Design

**Status:** Draft — awaiting user review
**Date:** 2026-08-27
**Task:** TASK-22515
**ADR:** ADR-095

## Goal

Make Apply in the Console Provider/Model popover reliably close the popover,
update the exact conversation immediately, and preserve the conversation's safe
generation settings across restart. Give mouse and keyboard activation the same
semantic path. Keep compaction mode in the quick surface and apply it through its
existing independent persistence owner. Use the same conversation-settings Apply
orchestration from the full Console Settings modal.

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
- A durable save failure does not roll back either live component. After the modal
  closes, the Console keeps an explicit per-component failure visible until the
  current value is saved successfully, superseded by a newer Apply, or the session
  closes.

Apply includes provider, model, temperature, streaming, and compaction mode.
Compaction keeps its existing context-policy owner and storage; it is not serialized
inside the generation-settings metadata object. Generation-settings and
context-policy persistence outcomes are tracked independently. The quick surface may
label a context-policy failure as `compaction` because that is the only policy field
it edits.

## Selected Architecture

The existing Console session remains the live owner. The existing conversation row
becomes the durable owner of an allowlisted generation-settings snapshot.

One conversation-settings Apply orchestrator accepts a typed intent containing:

- the stable originating session ID;
- the conversation identity captured when the surface opened, if one exists;
- the validated target provider/model and values submitted by the surface;
- the submitted compaction mode or full-modal context-policy overrides;
- the exact set of fields exposed by that surface;
- a full-modal endpoint only when its draft is bound to the target provider.

The orchestrator validates that the origin still exists and still has the captured
conversation identity. It delegates provider rebasing to one controller seam and
compaction to the existing context-policy store seam, then applies both live values
to the exact session before yielding. It increments the relevant process-local
revisions, synchronizes the Console summary/control bar immediately, and starts the
two independent durable writes when the session has a conversation row. It never
falls back to the currently active session. If the origin closed or was rebound, it
reports `Chat closed; nothing applied`.

The quick popover and full modal both call this orchestration. No parallel
quick-settings service, new database table, combined persistence abstraction, or
cross-owner transaction is introduced.

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

Compaction remains a sparse `ConsoleContextPolicyOverrides` value in
`console_conversation_context_policy`. Apply changes only its `compaction_mode`
field from the quick surface, preserving the other context-policy overrides. Its
existing first-persistence and resume paths remain authoritative.

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
4. The surface submits one typed intent containing the generation draft and
   compaction/context-policy draft.
5. The Apply orchestrator verifies the captured origin, performs provider-aware
   rebasing, and derives the new sparse context-policy overrides.
6. It commits both live snapshots to the origin session and increments their
   process-local revisions before yielding. Mounted summaries and controls
   synchronize immediately, and the modal closes.
7. For an existing conversation, the generation snapshot is written to metadata
   with sibling-preserving optimistic concurrency while the complete still-current
   context-policy snapshot is written through the existing context-policy repository.
   Neither durable write gates or rolls back the other live component.
8. Each persistence attempt is bound to the captured conversation identity and the
   revision of that component. Sibling-only metadata conflicts may be merged and
   retried. A stale completion cannot clear or overwrite the outcome of a newer
   Apply.
9. The session records `generation_settings` and `context_policy` as the two possible
   failed components in a bounded process-local durability record. The collapsed
   Model section shows a warning badge; when expanded, its rail names each failed
   component and exposes `Retry save` after the modal closes. A quick-popover failure
   may read `Not saved: generation settings · compaction`; a full-modal policy
   failure reads `context settings` when fields beyond compaction were submitted.
   Retrying `context_policy` writes the complete still-current policy snapshot and
   carries its policy revision, never merely the old compaction field. A component
   clears on successful persistence or when any newer change supersedes its snapshot;
   the record disappears when no component remains failed.
10. If the session is unsaved, both snapshots are staged rather than marked failed.
    Generation metadata is included at first conversation creation and context
    policy uses its existing post-create flush. Failures discovered at first
    persistence enter the same per-component durability record.

The execution boundary is objective: consumers that already hold an execution
context keep it; consumers resolving after the live commit in step 6 read the new
settings. The design does not special-case user sends, retries, regenerations,
queues, or agent work.

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

- Title: `Conversation settings`
- Visible labels: Provider, Model, Temperature, Streaming, Compaction
- Actions: `Full settings…`, `Cancel`, `Apply`
- Scope copy: `Applies to this conversation`
- Unsaved chat copy: `Saved with the conversation after its first message`
- Temporary chat copy: `Temporary until this chat is promoted`

The current compaction threshold, help, and mode control remain in the quick
popover. Full Settings Context remains the deeper editor for the rest of the
context policy.

Temperature parsing no longer silently restores the old value. Invalid input shows
an inline error and keeps the draft intact. Apply emits at most one concise outcome
notification; persistent failures live in the Model rail rather than disappearing
with a toast. There is no `Next send` label because the conversation setting changes
immediately.

## Full Settings Alignment

Full Console Settings uses the same Apply orchestration, provider rebase, generation
metadata writer, complete context-policy snapshot owner, and per-component
durability reporting. Its safe generation fields therefore have the same restart
behavior as the quick popover. A full-modal policy Retry is revision-guarded against
the entire current context-policy snapshot, not only compaction mode.

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
  include its serialized safe form when first creating the conversation row. Flush
  staged context policy through its existing owner and surface either component's
  failure after first persistence.
- **Temporary chat:** Do not create durability merely because Apply was clicked.
  Promotion includes the current safe generation snapshot in the conversation
  bundle, then persists the current context policy through its existing owner. A
  post-promotion context-policy failure leaves promotion intact and enters the same
  visible compaction failure state.

## Error Handling

- Missing origin session: no retargeting; notify and discard the result.
- Invalid fields: keep the modal open with inline feedback.
- Unsupported saved metadata version: preserve it, warn once when relevant, and do
  not overwrite it automatically.
- Metadata version conflict: reload and retry only for sibling-only changes while
  the session settings revision, conversation identity, and owned metadata base
  still match. Abort an obsolete Apply rather than overwriting a newer one.
- Generation metadata failure: retain the live generation settings and record
  `generation settings` as not saved.
- Context-policy failure: retain the complete live context-policy snapshot and record
  `context_policy` as not saved. Display `compaction` only when the quick surface was
  the source and compaction was the sole policy edit.
- Partial failure: name only the failed component; never imply that the successful
  component was lost.
- Retry: use the still-current component snapshot, component revision, and captured
  conversation identity. A context-policy Retry writes the complete current policy;
  stale retry state cannot overwrite any newer policy edit or generation Apply.
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
   orchestration and restart hydration, while compaction round-trips through its
   existing context-policy owner.
8. Provider switching begins with a non-null old endpoint and proves that endpoint
   is not retained.
9. Metadata writes preserve speech, roleplay, prefill, and unrelated sibling keys;
   bounded conflict retry cannot let an older Apply overwrite a newer one.
10. Corrupt and future-version overlays fail closed without destructive overwrite.
11. Generation-settings-only, context-policy-only, and dual persistence failures
    close the modal, retain both live values, identify the failed components
    persistently, and clear only the successfully retried current components.
12. Unsaved-first-persistence and temporary-promotion flows stage both components
    and surface any later component-specific failure.
13. The quick popover retains and returns compaction mode; changing it preserves all
    other context-policy overrides and full Settings Context behavior.
14. A newer non-compaction context-policy edit supersedes a failed policy snapshot;
    Retry cannot restore any value from the obsolete snapshot.

Verification uses only affected pytest modules plus targeted lint/format and
`git diff --check`. A full-suite sweep requires explicit user approval under the
repository testing policy.

## Non-Goals

- No global provider/default mutation from Apply.
- No endpoint or credential persistence in conversation metadata.
- No new database table or schema migration.
- No compaction storage/schema change or combined generation/compaction transaction.
- No broader Context & memory redesign beyond sharing the Apply orchestration and
  durability outcome reporting.
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
- **AC3:** one Apply orchestration and one safe generation snapshot for both settings
  surfaces, with compaction delegated to its existing owner.
- **AC4:** metadata serialization plus provider-first hydration.
- **AC5:** provider-aware rebase and endpoint exclusion regression.
- **AC6:** first-persistence, temporary, and promotion behavior for both durable
  components.
- **AC7:** inline validation and deferred-callback regression.
- **AC8:** persistent per-component failure and current-snapshot retry behavior.
- **AC9:** retention of quick-popover compaction through its independent owner.
- **AC10:** focused interaction, persistence, concurrency, hydration, endpoint, and
  partial-failure coverage.
