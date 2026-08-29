# Console Provider Apply, Conversation Persistence, and Defaults Design

**Status:** Approved; implementation-plan review findings addressed
**Date:** 2026-08-27
**Task:** TASK-22515
**ADR:** ADR-095

## Goal

Make Apply in the Console Provider/Model popover reliably close the popover,
update the exact conversation immediately, and preserve the conversation's safe
generation settings across restart. Add explicit per-model and new-chat default
actions that reuse the existing model-profile and configuration owners. Every
eligible blank new chat uses the saved global provider/model and that exact model's
profile in the running app and after reboot. Give mouse and keyboard activation the
same semantic path. Keep compaction mode conversation-only through its existing
independent persistence owner. Use the same conversation-settings Apply orchestration
from the full Console Settings modal.

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
- `Save as model default` applies the draft to the originating conversation, then
  saves only the submitting surface's model-profile fields for the exact normalized
  provider and literal model ID.
- `Make default for new chats` performs the same live Apply and model-profile save,
  then changes the global provider/model used by every eligible blank new chat in
  the current process and after reboot.
- The two default actions close after the live conversation commit. A later config
  failure remains explicit and retryable without pretending that the live Apply
  failed.
- Ordinary `Apply to this chat` never changes a model profile, global default, or
  endpoint.
- Compaction is never part of either model or global defaults. It remains applied
  only to the originating conversation.

Apply includes provider, model, temperature, streaming, and compaction mode.
Compaction keeps its existing context-policy owner and storage; it is not serialized
inside the generation-settings metadata object. Generation-settings and
context-policy persistence outcomes are tracked independently. The quick surface may
label a context-policy failure as `compaction` because that is the only policy field
it edits.

## Selected Architecture

The existing Console session remains the live owner. The existing conversation row
becomes the durable owner of an allowlisted generation-settings snapshot. Existing
configuration sections remain the only owners of model profiles, the global
provider/model, and provider endpoints.

One conversation-settings Apply orchestrator accepts a typed intent containing:

- the stable originating session ID;
- the conversation identity captured when the surface opened, if one exists;
- the validated target provider/model and values submitted by the surface;
- the submitted compaction mode or full-modal context-policy overrides;
- the exact set of fields exposed by that surface;
- a full-modal endpoint only when its draft is bound to the target provider.
- a discriminated action: `apply_chat`, `save_model_default`, or
  `make_new_chat_default`;
- a field mask for the model-profile mutation when the action saves defaults;
- a full-modal endpoint-save flag only when the endpoint was explicitly edited and
  the user left the scoped checkbox enabled.

The orchestrator validates that the origin still exists and still has the captured
conversation identity. A chat that was unsaved when the surface opened may complete
its normal first-persistence transition from no conversation ID to its newly created
ID without being treated as rebound; every other identity change is rejected. The
session-lifecycle seam must distinguish that first-persistence bind from an explicit
rebind rather than accepting every `None` → ID transition. It delegates provider
rebasing to one controller seam and
compaction to the existing context-policy store seam, then applies both live values
to the exact session before yielding. It increments the relevant process-local
revisions, synchronizes the Console summary/control bar immediately, and starts the
two independent durable writes when the session has a conversation row. It never
falls back to the currently active session. If the origin closed or was rebound, it
reports `Chat closed; nothing applied`.

The quick popover and full modal both call this orchestration. No parallel
quick-settings service, new database table, combined persistence abstraction, or
cross-owner transaction is introduced.

Default configuration mutation begins only after the exact-origin live commit
succeeds. It runs as one locked, field-masked patch against the freshly reread config,
then publishes the new runtime configuration. Conversation metadata and context
policy keep their independent owners and outcomes. A bounded app-level default
durability record is separate from each session's conversation durability record.

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

### Existing configuration owners

The default actions reuse the established configuration schema:

- `api_settings.<provider>.model_defaults[<exact model id>]` owns per-model
  generation defaults. Model IDs are trimmed literal mapping keys; dots, slashes,
  colons, and other punctuation are never interpreted as config paths.
- `chat_defaults.provider` and `chat_defaults.model` own the global selection used
  by eligible blank new chats.
- A full-Settings `Make default for new chats` may also update the selected
  provider's existing endpoint field when the endpoint is explicitly dirty and its
  scoped checkbox remains checked.

The quick surface's profile mask is `temperature` plus `streaming`. The full Model
surface's mask is every supported exposed sampler, reasoning/thinking, token-limit,
and streaming field. The patch changes only those exact fields, preserves unexposed
profile fields and sibling model profiles, and removes an exact profile field when
the submitted value means inherit. Provider aliases are matched canonically while
the existing raw provider-section identity is preserved. Credentials and credential
references are never copied.

The mutation rereads the config while holding the existing config lock, applies the
model-profile patch plus any global provider/model and eligible endpoint changes as
one atomic file intent, replaces the file once, and then refreshes runtime config.
It returns the existing rich phase result rather than collapsing file replacement
and runtime publication into one boolean.

## Provider and Model Rebase

A provider or model change is not implemented with a field-only `replace(...)`. The
session controller owns this operation for both settings surfaces.

1. Normalize the selected provider and model.
2. Build the complete effective defaults for the selected provider/model whenever
   either selection changes.
3. Rebase every untouched draft field from the target model profile, saved Console
   provider defaults, chat defaults, and provider settings in the established
   precedence order.
4. Resolve the selected provider's configured endpoint through the existing
   provider-resolution path.
5. Discard the previous provider's endpoint and incompatible provider-specific
   reasoning/thinking values.
6. Overlay only deliberately edited draft fields exposed by the submitting surface
   and supported by the selected provider. Carried edits remain visibly marked
   `edited`; untouched values display the target model's effective values.
7. Accept a full-modal session endpoint only when the endpoint draft is explicitly
   bound to the selected provider; the quick popover never submits an endpoint.
8. Mark the resulting snapshot as user-authored and commit it to the exact session.

Each open settings transaction keeps only a process-local draft map keyed by
canonical provider plus literal model ID. Switching A → B → A restores A's unfinished
edits while rebasing untouched fields for B. Unsupported provider-specific fields are
removed from the target draft rather than hidden and later resurrected. This draft
map is discarded when the transaction ends and never becomes another persistence
owner.

`Full settings…` deepens the same transaction. It transfers the complete quick draft,
field provenance, compaction draft, exact origin identity, and keyed draft map to the
full modal, opens the Model view, and establishes predictable focus. It never applies
or discards quick edits merely because the user requested the deeper surface.

Conversation hydration performs the same ordering: parse the saved provider first,
build defaults for that provider, then overlay the remaining saved fields. A model
missing from the current catalog remains a valid explicit custom selection; an
unconfigured provider produces the existing blocked readiness state rather than a
silent fallback.

## Apply Flow

1. Opening either settings surface captures the origin session ID and current
   conversation identity before catalog loading or other asynchronous work.
2. Widget edits remain a local keyed draft until a committing action.
3. A committing action validates provider, model, and numeric fields. Validation
   failure keeps the surface open and focuses the first invalid control.
4. The surface submits one typed intent containing the generation draft,
   compaction/context-policy draft, discriminated action, and applicable default
   field mask.
5. The Apply orchestrator verifies the captured origin, performs provider/model-aware
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
11. For `save_model_default` or `make_new_chat_default`, the immutable config intent
    runs after step 6. A missing, closed, or rebound origin prevents both live Apply
    and default mutation; default saving never proceeds as a detached side effect.
12. `save_model_default` patches only the exact model profile. The quick field mask
    is temperature and streaming; the full field mask contains all supported fields
    exposed by its Model view.
13. `make_new_chat_default` atomically patches that model profile plus global
    provider/model. Only a full-modal, explicitly dirty, checked endpoint is included.
14. A successful runtime publication makes subsequent eligible blank-chat creation
    observe the new global provider/model and model profile immediately. A file that
    was replaced despite runtime-publication failure remains valid across restart;
    the current process continues to advertise its stale-runtime recovery state.
15. An ordinary `apply_chat` neither clears nor silently discards an earlier failed
    default intent. A newer explicit default action supersedes the earlier pending
    intent; revision and locked-config checks prevent stale retry from overwriting a
    newer edit.

The execution boundary is objective: consumers that already hold an execution
context keep it; consumers resolving after the live commit in step 6 read the new
settings. The design does not special-case user sends, retries, regenerations,
queues, or agent work.

## Eligible New Chats

After a successful `Make default for new chats`, every blank Console creation path
without an explicit source-settings intent resolves fresh settings through
`build_default_console_session_settings` and therefore observes the saved global
provider/model and that exact model's profile. This includes:

- Ctrl+T blank chats;
- new temporary chats;
- workspace-created blank chats; and
- the initial pristine Console chat after startup.

Existing and already-open conversations never rebase merely because a global
default changed. Deliberate Duplicate, Branch, Continue, or handoff operations that
carry explicit source settings are not blank-chat creation; their explicit intent
continues to win and the UI must not describe them as using the new-chat default.

The previous Ctrl+T behavior that cloned the active session is replaced for blank
chat creation. Tests enumerate each eligible path so `Make default for new chats`
cannot be technically successful while the next ordinary chat ignores it.

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

The compact surface keeps its existing purpose and progressively discloses default
actions. It does not gain additional configuration sections.

- Title: `Conversation settings`
- Visible labels: Provider, Model, Temperature, Streaming, Compaction
- Main actions: `Cancel`, `Full settings…`, `Defaults…`, `Apply to this chat`
- Scope copy: `Applies to this conversation`
- Unsaved chat copy: `Saved with the conversation after its first message`
- Temporary chat copy: `Temporary until this chat is promoted`

`Defaults…` replaces rather than expands the main footer. Its chooser keeps the
current draft and exact target visible:

- `Save as model default` — `Remember Temperature + Streaming for
  {provider}/{model}. New-chat provider/model unchanged.`
- `Make default for new chats` — `Save this model profile and start eligible new
  chats with {provider}/{model}.`
- `Back`
- Scope copy: `Compaction stays with this chat.`

The full Model view uses the same two intent labels but saves every supported field
it exposes. Blank values display `Inherit`; for the conversation Apply they resolve
the current lower-precedence value and freeze that effective value in the complete
conversation snapshot, while the model-profile mutation deletes that exact override.
Streaming is a three-state `Inherit` / `On` / `Off` control in the full view. The
quick surface shows effective temperature and streaming and marks deliberately
carried values `edited` after a provider/model switch.

The chooser is a real substate with no more than three visible actions. First Escape
or `Back` returns to the main footer without losing the draft; a second Escape from
the main state cancels the popover. Mouse recovery, Enter activation, and duplicate
activation guards apply to every committing action, not only ordinary Apply.

An unconfigured or readiness-blocked provider may remain an explicit conversation
selection, but `Make default for new chats` is disabled with a concise explanation
until the provider is configured. Successful default actions report two independent
receipts: `This chat updated` and the exact default scope saved.

The current compaction threshold, help, and mode control remain in the quick
popover. Full Settings Context remains the deeper editor for the rest of the
context policy.

Temperature parsing no longer silently restores the old value. Invalid input shows
an inline error and keeps the draft intact. Apply emits at most one concise outcome
notification; persistent failures live in the Model rail rather than disappearing
with a toast. There is no `Next send` label because the conversation setting changes
immediately.

## Full Settings Alignment

Full Console Settings uses the same Apply orchestration, provider/model rebase,
generation metadata writer, complete context-policy snapshot owner, and
per-component durability reporting. Its safe generation fields therefore have the
same restart behavior as the quick popover. A full-modal policy Retry is
revision-guarded against the entire current context-policy snapshot, not only
compaction mode.

The full modal's endpoint remains configuration-owned. A custom endpoint can affect
the live session, but it survives restart only through `Make default for new chats`,
only after that provider-bound endpoint was explicitly edited, and only while the
scoped checkbox remains enabled. `Save as model default`, ordinary Apply, and the
quick popover never persist an endpoint. This task does not copy endpoints into
conversation metadata.

The checked line shows only a sanitized host and conservative network class, for
example `Also save connection: 192.168.1.20:8080 · LAN`. Sanitization removes
userinfo, path, query, and fragment before presentation or logging. Classification
is syntactic and performs no DNS lookup:

- `Local`: localhost or a loopback literal;
- `LAN`: private/link-local literals and `.local` hostnames;
- `Remote`: public IP literals; and
- `Remote/unknown`: other hostnames whose address class is not known locally.

Credentials and credential references are never shown or copied by this action.

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
- Default mutation before file replacement: retain the live conversation Apply and
  show an app-level `Not written to disk` record with the exact action, provider,
  literal model, field scope, and sanitized optional endpoint. Offer
  `Retry default save` and `Discard retry`; Discard removes the pending retry and
  never implies that the live conversation change was rolled back.
- Runtime publication after successful file replacement: report `Saved on disk;
  running app refresh failed`. Offer `Refresh running app` and `Dismiss`. Refresh
  rereads and republishes the on-disk config without repeating the disk mutation;
  Dismiss acknowledges the running-process limitation and never claims to undo the
  durable default. Restart loads the already-saved values.
- Default failure scope: configuration failure is app-global and appears from every
  Console Model rail and full Settings until recovered, dismissed/discarded,
  superseded by a newer explicit default action, or the app closes. Conversation
  generation/context-policy failures remain session-local and process-local.
- Default retry concurrency: the immutable intent is applied only after rereading
  the locked config and preserves unrelated external edits. A stale intent cannot
  replace a newer explicit edit to the same owned fields.
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
6. An unsaved origin that receives its first conversation ID while the settings
   surface is open remains the valid origin, while an explicitly rebound session is
   rejected.
7. A consumer resolving settings after Apply receives the new values; an already
   captured execution context remains unchanged.
8. Quick and full Settings round-trip safe generation fields through the shared
   orchestration and restart hydration, while compaction round-trips through its
   existing context-policy owner.
9. Provider switching begins with a non-null old endpoint and proves that endpoint
   is not retained.
10. Metadata writes preserve speech, roleplay, prefill, and unrelated sibling keys;
   bounded conflict retry cannot let an older Apply overwrite a newer one.
11. Corrupt and future-version overlays fail closed without destructive overwrite.
12. Generation-settings-only, context-policy-only, and dual persistence failures
    close the modal, retain both live values, identify the failed components
    persistently, and clear only the successfully retried current components.
13. Unsaved-first-persistence and temporary-promotion flows stage both components
    and surface any later component-specific failure.
14. The quick popover retains and returns compaction mode; changing it preserves all
    other context-policy overrides and full Settings Context behavior.
15. A newer non-compaction context-policy edit supersedes a failed policy snapshot;
    Retry cannot restore any value from the obsolete snapshot.
16. Selecting a different model under the same provider rebases untouched fields
    from that exact model profile while preserving and marking deliberate edits.
17. Provider/model A → B → A restores A's unfinished keyed draft and never revives
    fields unsupported by B.
18. `Full settings…` transfers the complete quick draft, compaction draft, field
    provenance, and exact origin without applying or losing edits.
19. Quick `Save as model default` changes only temperature and streaming for the
    exact provider/model and preserves every sibling profile and advanced field.
20. Full `Save as model default` changes all supported exposed profile fields;
    blank deletes the exact override and conversation Apply freezes the resolved
    effective value. Streaming covers Inherit, On, and Off.
21. `Make default for new chats` atomically patches the exact model profile plus
    global provider/model, without changing compaction, credentials, or unrelated
    provider settings.
22. Ctrl+T, temporary, workspace-created, and initial pristine blank chats observe
    the saved global provider/model and exact model profile immediately after runtime
    publication and after restart.
23. Existing/open conversations remain unchanged, while Duplicate, Branch,
    Continue, and explicit handoff intents retain their source-specific behavior.
24. A blocked provider cannot become the new-chat default and explains why the
    action is unavailable.
25. Only a full, explicitly dirty, checked endpoint is included in `Make default for
    new chats`; quick actions and `Save as model default` never persist endpoints.
26. Endpoint previews remove credentials and URL details, classify loopback/private/
    public literals correctly, conservatively label other hostnames, and perform no
    DNS lookup.
27. Concurrent exact-model config patches preserve sibling models, unexposed fields,
    and newer unrelated edits, including literal model IDs with punctuation.
28. Before-replace failure exposes `Retry default save` / `Discard retry`; successful
    file replacement plus publication failure exposes cache-only `Refresh running
    app` / `Dismiss` and never repeats the disk write.
29. Ordinary conversation Apply preserves an earlier app-global default failure;
    a newer explicit default action supersedes it without stale overwrite.
30. The main footer and Defaults substate fit and retain complete keyboard focus/tab
    order at 60×24 and 72×24; mouse and Enter activate each committing action once,
    and hierarchical Escape preserves or cancels the draft as specified.

Verification uses only affected pytest modules plus targeted lint/format and
`git diff --check`. A full-suite sweep requires explicit user approval under the
repository testing policy.

## Non-Goals

- No global provider/default mutation from ordinary `Apply to this chat`; only the
  two explicit default actions mutate configuration.
- No endpoint or credential persistence in conversation metadata.
- No endpoint persistence from quick settings or `Save as model default`.
- No new database table or schema migration.
- No new preset/profile schema; reuse the existing exact-model profile mapping.
- No compaction storage/schema change or combined generation/compaction transaction.
- No broader Context & memory redesign beyond sharing the Apply orchestration and
  durability outcome reporting.
- No attempt to mutate a request that already captured its execution context.
- No live model-catalog refresh inside Apply.
- No generic Console settings refactor outside the shared commit and rebase seams.

## ADR Check

**ADR required:** yes
**ADR path:** `backlog/decisions/095-conversation-owned-console-generation-settings.md`
**Reason:** This changes durable conversation ownership, exact-model/global config
ownership, blank-chat creation, restart hydration, provider/model precedence, and
the cross-surface settings contract.

## Acceptance Mapping

- **AC1–AC4:** exact-origin interaction, immediate execution boundary, shared Apply
  orchestration, safe metadata, and restart hydration.
- **AC5–AC6:** provider/model draft rebasing, keyed provenance, and lossless
  quick-to-full transfer.
- **AC7–AC8:** progressive Defaults interaction and exact-model field-masked saving.
- **AC9–AC10:** atomic new-chat defaults and complete eligible/excluded creation
  behavior in-process and after reboot.
- **AC11–AC12:** full-only explicit endpoint persistence and conversation-only
  compaction ownership.
- **AC13–AC15:** staging/promotion, validation/teardown reliability, and
  revision-guarded conversation durability recovery.
- **AC16–AC17:** truthful app-global disk/runtime recovery and concurrency-safe exact
  config mutation.
- **AC18:** focused interaction, layout, persistence, concurrency, hydration,
  endpoint, creation-path, and partial-failure coverage.
