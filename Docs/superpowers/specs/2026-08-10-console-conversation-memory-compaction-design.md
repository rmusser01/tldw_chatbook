# Console Conversation Memory and Auto-Compaction Design

Status: Reviewed design, ready for implementation planning

Date: 2026-08-10

Target branch: `dev`

Parent task: [TASK-14811](../../../backlog/tasks/task-14811%20-%20Console-conversation-memory-and-auto-compaction.md)

Normative decision: [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md)

## Document purpose

This design turns the Console's existing context estimate, manual rewind
summary, prompt customization, and token-aware safety trimming into one
understandable conversation-memory system. It defines the user language,
ownership, persistence, request math, runtime state machine, Console and
Settings surfaces, failure behavior, and evidence required to ship it.

## Product outcome

A user can see how much of the selected model's usable context the next
Console request will consume, set a smaller conversation budget if desired,
and choose whether Chatbook should ask, automatically summarize, or never
summarize older turns. They can inspect or reset the resulting memory. Global
defaults and the summary prompt remain configurable in canonical Settings.

The feature does not pretend that optional summarization is the safety
boundary. When a known provider request would overflow, deterministic
whole-unit windowing still prevents the overflow.

## Goals

- Explain model capacity, response reservation, request overhead, conversation
  budget, and compaction without using one overloaded token limit.
- Give the current conversation durable policy overrides.
- Make the exact next-request projection the source for UI and send-time
  decisions.
- Preserve all transcript content while replacing older request context with
  reviewable local memory.
- Make automatic work branch-safe, revision-safe, bounded, and cost-visible.
- Keep global defaults, model capabilities, current-conversation overrides,
  and prompt editing with their existing owners.
- Work at narrow terminal widths and entirely from visible keyboard-accessible
  controls.

## Non-goals

- A general-purpose memory/RAG framework.
- Deleting, rewriting, or syncing transcript content.
- Automatically creating a second visible conversation or session.
- Wall-clock- or turn-count-based compaction triggers.
- Per-model compaction profiles in the first release.
- User-defined provider roles, safety wrappers, or transcript delimiters.
- Summarizing transient tool schemas, source payloads, skills, or world-info
  blocks merely to make them fit.
- Claiming that a model-generated summary is lossless.

## Terminology and user language

| Term | Meaning | Editable here? |
| --- | --- | --- |
| Model window | Provider/model hard context capability. | Read-only in Console; repairable in Providers and Models. |
| Response max tokens | Maximum requested model output. | Current conversation/model setting. |
| Safety margin | Reserved uncertainty for tokenization/provider framing. | Application policy, not an ordinary user control. |
| Safe input ceiling | Model window minus response reservation and safety margin. | Calculated. |
| Request overhead | System contract, tools, sources, skills, and other non-conversation context. | Explained, not compacted by this feature. |
| Conversation budget | Maximum replaceable memory plus durable conversation units for this request. | Automatic or custom. |
| Next request estimate | Projected total provider input after current transforms. | Calculated. |
| Compaction | A separate model call that summarizes older durable units into memory. | Ask, Automatic, or Off. |
| Memory | Local derived summary carried into later requests. | Reviewable and resettable. |

User-facing copy says **conversation**, not **session**, except where the
provider or diagnostic concept truly is a runtime session. Existing `Max
tokens` copy becomes `Response max tokens` anywhere context controls appear.

## Default policy

| Setting | Default | Rationale |
| --- | --- | --- |
| Conversation budget | Automatic | Uses the selected model safely without requiring model knowledge. |
| Compaction | Ask | Does not incur an unexpected model call or latency. |
| Trigger | 80% | Leaves room before the hard boundary. |
| Target | 55% | Creates enough hysteresis to avoid repeated compaction. |
| Summary max | 1,024 tokens | Bounds cost and prevents a summary from consuming the reclaimed space. |
| Failure | Stop and ask | Avoids silently changing the context strategy. |
| Carry-forward | Memory with recent turns | Preserves immediate conversational texture. |

Validation requires `0 < target < trigger <= 0.95` and a minimum 15 percentage
point gap. Numeric token fields use bounded integer input, locale-independent
parsing, and explicit inline errors. The implementation derives practical
minimums and maximums from existing provider and configuration validation
constants instead of introducing unchecked magic values in widgets.

## Ownership and precedence

### Model capability owner

The model catalog/capability layer owns context-window values. Console reads an
effective capability snapshot. Providers and Models is the only Settings
surface that repairs an unknown or incorrect value.

### Global default owner

Canonical Settings > Console Behavior owns default budget mode, optional
custom budget, compaction mode, trigger, target, summary cap, failure policy,
and carry-forward mode. It uses the screen's existing staged-save behavior.

### Conversation owner

A persisted Console conversation owns only overrides from the global policy.
Saving in the Console modal applies to the current conversation. Reopening or
restarting restores those overrides. Removing an override returns that field
to the current global default rather than copying a default value into the
conversation.

Before the first message makes a Console conversation durable, overrides live
in the existing tab/session snapshot. Applying policy does not create an empty
conversation row. The staged overrides write through when the conversation is
first persisted. Closing an unsaved empty tab discards them; the UI states
that limit instead of promising restart persistence for an identity that does
not yet exist.

### Prompt owner

Internal Prompts owns `console.rewind_summarize`. Console Behavior offers
`Edit summary prompt...`, which navigates to that existing prompt. It does not
embed a second editor or save prompt text with Console Behavior.

### Resolution order

For every policy field:

1. valid current-conversation override;
2. valid global Console default;
3. application default from this design.

Model window, response reservation, and safety margin are inputs to policy
resolution, not lower-precedence policy values.

## Request accounting model

The estimator and dispatch path must consume the same immutable prepared
request. A provider-neutral `PreparedConsoleRequest` preserves named semantic
segments, including original system content and app-owned memory. The provider
gateway serializes it once into a sensitive `PreparedProviderRequest` carrying
the exact adapter kwargs/payload used for both token accounting and dispatch.
A UI-only approximation or a second payload builder cannot decide compaction.

For a known model window:

```text
effective_response_reservation =
    min(requested_response_max, advertised_model_output_cap when known)
safe_input_ceiling = min(
    advertised_model_input_cap when known,
    model_window - effective_response_reservation - safety_margin,
)
non_compactable_material =
    app/provider overhead + active request + mandatory pinned units
available_conversation_capacity =
    max(0, safe_input_ceiling - non_compactable_material)
configured_budget =
    available_conversation_capacity               when Automatic
    custom_budget_tokens                          when Custom
effective_budget = min(configured_budget, available_conversation_capacity)
conversation_load = memory + compactable_durable_prior_units
next_request_estimate = non_compactable_material + conversation_load
```

The projection reports requested and effective response reservation, plus
configured and effective conversation budget, separately. A lower effective
value caused by a provider cap or current overhead is a request fact, not a
mutation of the saved value. If mandatory material plus the requested response
reservation cannot fit, preflight asks the user to shorten the request, remove
overhead, reduce response max tokens, or change model. It does not silently
reuse the existing trimmer's historical half-window reservation clamp.

Token categories must be inspectable in tests and UI details:

- original user/character system contract;
- app-owned memory block;
- retained durable user/assistant/tool units;
- active draft/request and pinned prefill;
- tool definitions and tool-choice framing;
- staged sources/evidence, skills, world info, and other injected context;
- provider framing estimate;
- response reservation and safety margin.

The estimator uses the selected provider/model tokenizer when available and
the existing conservative fallback otherwise. It states when a value is an
estimate. Cache keys include every revision that can affect the projection;
any mismatch invalidates the cached result.

### Unknown model window

An unknown window shows `Model limit unknown` rather than a fabricated ratio.
Automatic budget is blocked. Automatic compaction is available only after the
user supplies a bounded custom conversation budget; the UI still labels that
threshold as user supplied and provider safety as unverified. Compaction may
reduce context, but it does not prove the request fits an unknown provider
limit. Existing provider error handling remains the final boundary and the UI
links to repair the model capability.

### Non-compactable overhead

If mandatory material leaves too little capacity, repeated summaries cannot
help. The send flow stops with a breakdown and actions relevant to the active
payload, such as shortening the active request, reducing sources, disabling
tools, selecting a larger-window model, or lowering response max tokens. It
never loops compaction.

## Durable conversation units

Compaction and safety trimming operate on atomic request units, not arbitrary
messages or strings.

- a user message and its assistant response form an exchange when complete;
- an assistant tool call, all corresponding tool results, and the completing
  assistant response remain together;
- the active user request is mandatory;
- system/app-context blocks are classified separately;
- pinned messages remain explicit mandatory units and may cause an actionable
  over-budget state;
- hidden/deleted/variant messages follow the active lineage contract used by
  provider dispatch.

The summary input includes the prior active memory, when present, and only the
complete durable units being replaced. Prior memory and original transcript
use distinct delimiters and provenance labels so iterative compaction can
retain older facts without pretending they are original messages. Transcript
serialization uses stable role/tool labels.

## Runtime state machine

### Preflight on Send

1. Resolve an immutable provider, model, policy, prompt, conversation lineage,
   and request revision snapshot.
2. Build the projected next request before ordinary safety trimming.
3. If projection is below the trigger, dispatch normally.
4. If mandatory material alone is the blocker, show recovery for the active
   request or request overhead.
5. If compactable material crosses the trigger:
   - **Off:** do not call the summarizer; apply deterministic safety windowing
     and disclose omitted earlier units when omission is necessary.
   - **Ask:** pause with `Compact and send`, `Send with older context omitted`,
     and `Cancel`.
   - **Automatic:** run the bounded compaction transaction.
6. Rebuild the projection from committed memory and current revisions.
7. Dispatch only if it fits; otherwise follow the configured failure path and
   never repeat automatically in the same send attempt.

`Compact now` uses the same transaction but is user-initiated and does not
automatically send the active draft.

### Compaction transaction

One transaction per conversation:

```text
idle -> admitted -> summarizing -> validating -> committing -> idle
                     |               |             |
                     +---- failed ---+---- stale --+
```

Admission captures conversation ID, active leaf and lineage digest, boundary,
request revision, policy revision, active-memory revision, provider/model,
prompt ID/digest, selected unit IDs and versions, summarized-prefix digest,
and target budget. A second request sees the busy state rather than starting
parallel work.

Validation requires:

- non-empty bounded output;
- no provider/tool envelope leakage;
- selected units and boundary still belong to the captured lineage and their
  prefix digest still matches;
- relevant revisions and provider/model still match;
- output plus required recent units makes meaningful progress toward target.

A stale or non-improving result is not committed. No automatic retry occurs.

## Persistence model

Implementation uses the next available `ChaChaNotes_DB` schema version at the
time TASK-14811.1 begins; the plan must not reserve a version that another
in-flight migration may consume.

### Conversation policy record

One record keyed by conversation ID stores optional overrides, a monotonic
revision, and timestamps. Enum and numeric values are strictly validated on
read and write. Corrupt values fail closed to an explicit recoverable state;
they are not silently rewritten as defaults.

### Memory records

Memory records store:

- conversation and derived-memory identity;
- boundary message ID and captured leaf/lineage digest;
- summary text and active/reset state;
- provider/model identity;
- prompt ID, prompt revision when available, and content digest;
- selected-unit IDs/versions and summarized-prefix digest, including selected
  variants and relevant attachments;
- input/output and before/after token counts;
- creation timestamp and revision.

The active request chooses the newest active memory whose boundary is present
in the current lineage and whose summarized-prefix digest still matches the
active prefix through that boundary. A descendant may reuse valid ancestral
memory. A sibling branch or in-place edit to summarized content cannot. Prefix
digest validation may be cached by message/payload revisions, but it occurs
before every injection. Reset deactivates derived memory; it never deletes
messages.

Legacy `context_summary` and `summary_boundary_message_id` data receives an
explicit compatibility migration. It may become a provenance-limited legacy
memory record only when its boundary can be validated. Otherwise it remains
reviewable but inactive until regenerated. Migration and rollback behavior
must be documented in the implementation task.

Memory is local-only private data, consistent with current summary storage. It
does not enter sync/export until a separate privacy and portability design
explicitly adds it.

## Summary generation boundary

The first release uses the exact active conversation provider/model. It does
not silently fall back to another model. A separately selectable summary model
is a future design because it adds independent credentials, capability,
pricing, and failure ownership.

Before admission, the service selects the largest oldest contiguous span of
complete compactable units that fits one auxiliary request:

```text
immutable wrapper + editable prompt + prior memory + selected units
    + effective summary output reserve
<= summarizer input ceiling
```

The effective summary cap is the minimum of the configured summary cap, the
model's advertised output cap, the auxiliary hard ceiling, and the output room
that can still make progress toward the post-compaction target. If no positive
allowance or useful contiguous span exists, the service fails before making a
provider call. One send attempt still makes at most one summary call; if one
bounded span cannot reach the target, the user receives the existing omission
or recovery choices.

The auxiliary call disables chat tools, RAG, source injection, skills,
streaming transcript writes, and ordinary conversation message persistence.
Cancellation is bounded and observable.

The stable prompt ID remains `console.rewind_summarize`. User customization
may specify preservation priorities and output organization. The application
always owns:

- transcript delimiters and role serialization;
- the instruction that transcript content is untrusted data;
- the output cap and admission snapshot;
- the provider message role used for memory injection;
- the memory safety wrapper.

The generated summary is not trusted merely because another model wrote it.
When carried forward, it remains a separate app-owned semantic segment with a
fixed label and provenance metadata outside model-visible free text. Providers
with one system/preamble field may serialize that tagged segment beside the
unchanged original system segment; storage and prepared-request ownership do
not collapse merely because the wire shape does.

### Auxiliary usage owner

The local conversation database owns a content-free auxiliary-call ledger.
Each admitted attempt records an operation ID, conversation ID, purpose,
provider/model, requested output cap, estimated input tokens, status, timing,
and pricing provenance. Provider-reported usage is added when the gateway
returns it. Failed, cancelled, and stale calls remain represented because they
may still have incurred cost. The gateway result contract therefore carries
normalized provider usage when available rather than only provider, model,
and text. No request, response, transcript, or summary body enters the ledger.

## Console experience

### Quick model popover

Keep the popover fast and compact:

```text
Request       ~63,400 / 96,000 safe input
Conversation ~42,000 / 70,000 budget
Compaction    [Ask                       v] at ~56,000

                   [Context & memory...] [Apply]
```

The quick popover changes compaction policy but does not grow a custom numeric
editor inside its fixed, non-scrollable geometry. `Context & memory...` opens
the full Console settings modal directly on that view, where the user can set
Automatic or custom budget tokens. Request and Conversation use different
denominators deliberately. `Apply` affects only the current conversation.

Status variants:

- `Request ~63,400 / 96,000 safe` and `Conversation ~42,000 / 70,000`;
- `Conversation ~58,200 / 70,000 - compaction will be offered`;
- `Model limit unknown - set a custom budget or repair model data`;
- `Request overhead exceeds available context`;
- `Compacting...` with controls disabled and a cancellable progress state.

### Full current-conversation modal

The existing scrollable Console Settings modal keeps one stable action bar and
uses two in-modal views rather than opening a nested modal:

- **Model & generation** contains the existing provider, model, identity,
  sampling, provider-specific, and streaming controls.
- **Context & memory** contains the capacity, policy, and memory workflow.

The Context & memory view uses progressive disclosure in four sections:

1. **Model capacity** — model window, response max tokens, safety margin, and
   safe input ceiling.
2. **Conversation budget** — Automatic/Custom, configured value, current
   effective value, total next-request estimate, and overhead breakdown.
3. **Compaction** — Ask/Automatic/Off, trigger, target, summary cap, failure
   behavior, carry-forward mode, and inherited/override badges.
4. **Current memory** — state, boundary, generated time, provider/model,
   prompt provenance, before/after tokens, and actions.

Memory review expands inline as a plain-text scroll region rather than opening
another modal. Actions are `Save`, `Reset overrides`, `Compact now`, and
`Reset current branch memory`. The current-branch reset deactivates only the
selected active memory and offers Undo. An advanced `Reset all conversation
memory...` action requires confirmation and deactivates every branch record;
neither action changes transcript messages.

Preserve the modal's existing global default workflow, but rename its button
and copy to `Save provider defaults`: it writes only provider, sampling, and
streaming defaults while also applying the current conversation draft.
Conversation-memory defaults remain exclusive to canonical Settings. Raw
prompt editing also remains in Internal Prompts.

### Ask-before-compacting dialog

Copy must expose the extra model call:

> This request has reached 80% of its conversation budget. Chatbook can make
> one additional model call to summarize older turns, keep your full
> transcript, and then send your message.

Actions:

- `Compact and send` (primary);
- `Send with older context omitted` (secondary, with omission explanation);
- `Cancel`.

An optional `Use this choice for this conversation` control may change Ask to
Automatic or Off only after explicit selection.

## Canonical Settings experience

### Console Behavior

Add a `Conversation memory` group:

- Default conversation budget: Automatic / Custom tokens.
- Default compaction: Ask / Automatic / Off.
- Trigger at: percentage and calculated example.
- Compact toward: percentage and calculated example.
- Summary max tokens.
- On summary failure: Stop and ask / Send with older context omitted.
- Carry forward: Memory with recent turns / Memory with latest exchange.
- `Edit summary prompt...` deep link.
- read-only preview explaining the resulting memory block and retained turns.

All values remain staged until the screen's existing Save action succeeds.
Changing defaults does not overwrite conversations with explicit overrides.

### Internal Prompts

Retitle the display name of `console.rewind_summarize` if helpful, but preserve
its stable ID. The deep link opens Internal Prompts filtered to that record.
The editor explains which instructions are customizable and which safety and
injection boundaries remain application-owned.

### Providers and Models

Show the detected context window near each model's generation limits. For
unknown or incorrect values, allow a validated local override with source and
reset-to-detected behavior. Do not add per-model compaction policy in this
tranche. Reuse the model-capability and config-override authority established
by TASK-320; this feature does not create a second context-window table or
configuration owner.

## Responsive, accessibility, and interaction requirements

- No new binding may use terminal-convention keys or shadow global bindings.
- Every core workflow is reachable through visible controls, not only the
  command palette.
- Labels do not rely on color alone; threshold states include text and symbols.
- Focus order follows visual order and returns to the invoking control when a
  modal closes.
- Busy work disables duplicate submission while leaving Cancel reachable.
- Errors appear adjacent to the invalid field and in a concise summary when
  Save/Apply is blocked.
- At narrow widths, label/value pairs stack and actions wrap without clipping,
  overlap, horizontal scrolling, or off-screen confirmation controls.
- Token values use grouped digits and `~` only for estimates. Exact and
  estimated values are not visually identical.
- Help copy is concise in the main flow; detailed math lives behind a visible
  `How this is calculated` disclosure.

## Failure and recovery matrix

| Condition | Required behavior |
| --- | --- |
| Summary provider error/timeout | Stop and ask by default; allow explicit send with omission; no retry loop. |
| User cancels | Discard partial output, leave draft/transcript/memory unchanged. |
| Result becomes stale | Discard it and explain which state changed. |
| Summary is empty/too large | Reject it as non-improving; use configured failure behavior. |
| Summary still above target | Do not automatically call again during this send; show breakdown and choices. |
| Mandatory material alone exceeds capacity | Do not summarize; show active-request/source/tool/model/response recovery actions. |
| Model window unknown | Disable Automatic budget; allow Automatic compaction only with a bounded custom threshold and a safety-unverified warning. |
| Custom budget exceeds current capacity | Preserve saved value, show lower effective value and cause. |
| Branch no longer contains memory boundary | Ignore that memory and select a valid ancestral record or none. |
| Policy record is invalid/corrupt | Fail closed to an explicit reset/repair path; do not silently persist defaults. |
| Current-branch memory reset | Deactivate the selected active record, rebuild the estimate, and offer Undo. |
| Reset all conversation memory | Confirm scope, deactivate every branch record, and leave the transcript intact. |

## Security and privacy requirements

- Treat transcript and generated summary as untrusted model-visible data.
- Never let summarized content alter the immutable wrapper or escape its
  delimiters.
- Never log transcript or summary bodies through diagnostics, usage, errors,
  or provider adapter debug output.
- Keep memory local-only under the same private database boundary as the
  conversation.
- Parameterize every migration/query and validate enum/numeric fields at the
  repository boundary.
- Sanitize memory preview Markdown using the same policy as chat content.
- Do not claim that local plaintext storage is encryption.
- The content-free auxiliary ledger exposes attempt status and provider/model/
  token/cost metadata without copying private content.

## Verification strategy

### Pure and repository tests

- policy defaults, overrides, validation, precedence, and model changes;
- prepared semantic segments, exact serialized accounting categories, and
  estimator/dispatch artifact identity;
- whole exchange and tool-call/result unit selection;
- trigger/target hysteresis and one-attempt guard;
- unknown model, overhead exhaustion, and custom/effective budget behavior;
- migration, close/resume/restart, reset, and corrupt-record handling;
- legacy summary compatibility and branch-valid memory selection;
- stale-result rejection for every captured revision;
- no transcript mutation or deletion.

### Provider-boundary tests

- summary call disables ordinary chat augmentations and bounds output;
- memory remains a distinct semantic segment, provider serialization is
  deterministic, and stored original system content remains byte-identical;
- transcript delimiters and immutable safety wrapper survive custom prompts;
- successful, failed, cancelled, and stale auxiliary attempts are recorded
  without a transcript message;
- content is absent from logs and error serialization.

### Mounted UI and geometry tests

- quick and full controls reflect inherited/overridden/effective values;
- all validation, busy, failure, stale, unknown-window, and overhead states;
- prompt deep link and model-window repair navigation;
- keyboard/focus and destructive confirmation behavior;
- 80x24 and narrower supported geometry with long model names and large token
  values.

### Live evidence

Use a real configured provider and capture redacted observations for:

- below-threshold send;
- Ask > Compact and send;
- Automatic compaction;
- Off > deterministic omission;
- summary failure/timeout;
- model switch to a smaller and larger window;
- branch switch and edit during a slow summary;
- close/resume and full application restart;
- overhead-exceeds-capacity recovery;
- narrow-terminal interaction.

Evidence must distinguish test harness behavior from the real provider wire
path and record the provider/model and observed token/usage metadata without
private prompt content.

## Delivery slices

1. [TASK-14811.1](../../../backlog/tasks/task-14811.1%20-%20Persist-and-resolve-Console-conversation-context-policy.md) — persistence and policy resolution.
2. [TASK-14811.2](../../../backlog/tasks/task-14811.2%20-%20Prepare-and-account-exact-Console-provider-requests.md) — prepared request, exact accounting, and deterministic safety.
3. [TASK-14811.2.1](../../../backlog/tasks/task-14811.2.1%20-%20Add-branch-safe-automatic-Console-conversation-compaction.md) — bounded compaction, valid memory, and auxiliary usage.
4. [TASK-14811.3](../../../backlog/tasks/task-14811.3%20-%20Expose-current-conversation-context-controls-in-Console.md) — current-conversation UX.
5. [TASK-14811.4](../../../backlog/tasks/task-14811.4%20-%20Add-global-conversation-memory-controls-and-prompt-routing-in-Settings.md) — global Settings and prompt routing.
6. [TASK-14811.5](../../../backlog/tasks/task-14811.5%20-%20Harden-and-live-verify-Console-conversation-memory.md) — edge-case hardening and live evidence.

TASK-14811.3 and TASK-14811.4 may proceed in parallel after their declared
foundations. The final hardening slice begins only after all user surfaces and
runtime behavior have landed.
