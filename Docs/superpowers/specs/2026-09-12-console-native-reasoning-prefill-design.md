# Console native reasoning prefill

Date: 2026-09-12
Status: User approved; implementation planning authorized
Task: [TASK-32521](../../../backlog/tasks/task-32521%20-%20Design-native-reasoning-prefill-for-Console.md)
ADR: [ADR-159](../../../backlog/decisions/159-console-native-reasoning-prefill.md)
Implementation plan: [Eight-task implementation plan](../plans/2026-09-12-console-native-reasoning-prefill.md)
Source feature: [SillyTavern PR 6001](https://github.com/SillyTavern/SillyTavern/pull/6001)

## Purpose and scope

Let a Console user supply the beginning of a model's reasoning and ask a qualified
backend to continue it natively. Provide both a next-send value and a value pinned
to the current conversation. Use one Console interface across providers, with
support determined by the actual execution target.

Native means the backend starts generation at the unfinished reasoning boundary.
Accepting a reasoning field, replaying historical reasoning, or influencing an
answer does not establish native prefill support. There is no conversion into
system/user instructions, silent omission, or automatic switch of provider, API
mode, endpoint, tools, or thinking settings to make a request succeed.

This is a Console feature, including ordinary replies and tool-enabled runs whose
exact combination is supported. It does not add arbitrary assistant prompt-block
placement, SillyTavern preset import, macro expansion, saved prompt-library
artifacts, or a general custom-wire editor. Text is literal in this version.
Existing response-prefill command syntax remains compatible.

## Approved behavior

Each chat has two independent values: **Next send** and **Pinned to this chat**.
Each value has text and an enabled state. Save with nonblank text enables the
value; Disable retains its text; Enable rearms it; Clear removes it. Saving an
unchanged enabled value also rearms it with a new revision.

An enabled next-send value overrides an enabled pinned value. Disabling or
clearing next-send exposes the enabled pin again, with an immediate summary of
that change. An unsupported active override blocks sending; it never falls
through to the pin. A disabled value does not participate in compatibility or
budget checks. With neither enabled, requests behave as before this feature.

| Operation | Reasoning seed selection |
| --- | --- |
| Manual send | Enabled next-send value, otherwise enabled pin |
| User-authored queued send | Same precedence, reserved for exactly that queued submission |
| Retry a failed turn | That turn's captured seed and execution settings |
| Regenerate an assistant answer | Current enabled pin; never an unrelated next-send value |
| Continue an existing answer | No new reasoning seed; preserve existing continuation rules |
| Tool round or internal recovery within a run | No new seed injection; use the owning run's continuation |
| Child agent, fork/new chat, background helper, automatic follow-up | No inheritance or consumption of the parent's configured seed |

Changing a pin affects future submissions and regenerations, not an accepted
turn or its retry. A retry that cannot reconstruct the original request must
offer explicit recovery instead of substituting the current pin.

## Console interface

Add a **Reasoning prefill** entry to the existing Context surface. It opens a
multiline editor with separate Next send and Pinned to this chat sections. Each
section exposes its enabled state, text, Save, and Clear. The modal uses the
existing dirty-draft dismissal contract: closing or switching chats cannot
silently apply or discard edits. Application targets the originating chat and
revision; a stale editor keeps its draft and reports the conflict.

The Context summary identifies the effective lifetime and reports **Supported**,
**Unsupported**, or **Unverified**, with one concise explanation and relevant
recovery action. It also shows when next-send is reserved by a queued/running
turn. A running turn keeps its captured value even if the editor is changed.
An older run's completion never changes a newer draft or armed revision.

`/reasoning-prefill` opens this same editor; help and command completion advertise
it. Do not overload `/prefill`, whose free-text argument could already begin with
words such as `reasoning`. No new terminal-convention keybinding is introduced.

The explicit **Next Send** preview displays the exact effective text under
**User-authored reasoning prefill**, its source lifetime, compatibility result,
and token estimate. Summary rows remain content-free apart from lengths/state;
they do not duplicate the seed into notices or diagnostics. A budget estimate is
labelled as an estimate when the target tokenizer is unavailable.

Unsupported or unverified active seeds block before provider contact and before
turn acceptance mutates the transcript. Recovery offers Edit, Disable, and Change
model. A conflict with tools, response prefill, or thinking-off settings names
that combination. Choosing another model never deletes the saved text.

Use the modal shell, stacked form sections, standard TextArea and Button states,
and semantic status classes from the [design language](../../../backlog/docs/design-language.md).
All token-covered visual values use existing `$ds-*` tokens. Edit stylesheet
sources and rebuild the generated bundle if CSS changes are needed.

## Ownership and lifecycle

Keep the state in three distinct owners:

1. **Configuration:** the conversation-owned pin and live-session next-send slot.
2. **Accepted submission:** an immutable seed snapshot, source revision, exact
   target, and unique submission identity. A queue entry owns this snapshot when
   enqueue succeeds. Failed enqueue leaves the slot untouched.
3. **Provider continuation:** completed call history constructed by the adapter,
   including the seed only when its continuation contract requires it.

The runtime reserves a next-send revision atomically for one submission. Later
queued submissions cannot reserve that same revision; they use the current pin
unless the user explicitly rearms next-send. Re-arming identical text creates a
new identity. A reserved revision cannot be claimed by autonomous work or by a
second submission while the first awaits tools, approvals, or recovery.

| Outcome for the owning submission | Reservation and next-send result |
| --- | --- |
| Validation/queue admission fails | Nothing reserved or consumed |
| Queued entry is removed before dispatch | Release the reservation; restore availability only if no newer slot revision exists |
| Model/run completes successfully | Consume only the captured revision |
| User stops after provider dispatch begins | Consume the captured revision; retain existing stopped-output semantics |
| User cancels before provider dispatch begins | Release without consumption, as for removing an undispatched entry |
| Provider or later tool/run failure | Keep the failed turn's seed and reservation for explicit retry or release |
| Delivery status is unknown | Keep recovery ownership; never automatically resend or make the reserved seed available to another submission |

A failed turn's retry uses its own snapshot even after the user rearms a new
next-send value. Resolving that retry cannot erase the new value. An explicit
release abandons the failed turn's reservation; the UI explains whether the
original slot becomes available or has already been superseded. Queue failure
and stop follow the existing queue-pause rules, preventing automatic reuse.

Consumption follows the terminal outcome of the owning Console run, not the
first model call's successful tool response. Changing pin/enable state during a
run affects future work only. A queued submission's settings are captured at
enqueue acceptance; target readiness is revalidated before execution without
silently replacing those settings. If revalidation fails, the queue pauses.

## Provider boundary and qualification

Use a small adapter-owned capability contract resolved against the frozen target:
provider, API mode, endpoint identity, model, relevant server/template identity
where available, thinking mode, tools, and simultaneous response prefill. It
returns status, a content-free reason, supported combinations, and the serializer
needed to build the first native request. It is not a generic arbitrary-kwargs
passthrough or a global boolean inferred from a provider name.

An unknown target is Unverified and blocked while a seed is active. Documented,
qualified combinations may be Supported without requiring every user to make a
probe request. Local support rules must declare their server/template constraints;
if those constraints cannot be established for a target, it remains Unverified.
An OpenAI-compatible label or support for reasoning replay does not satisfy this
contract. No automatic paid compatibility probes run at startup or send time.

The implementation must inventory all current Console adapters and record each
as qualified or unavailable, with evidence. The first qualification candidates
are DeepSeek Chat Completions and local vLLM/llama.cpp combinations, not promises
that they work in the checked-out application. Shipping an enabled feature
requires at least one qualified native target; tool support requires its own
successful multi-round qualification before being advertised.

Evidence must include the complete outbound request, the effective unfinished
reasoning boundary where the backend exposes template rendering, stream and
non-stream behavior, reasoning-to-answer transition, echo behavior, and tool
continuation when claimed. A normal HTTP response or an answer that follows the
seed's advice is not sufficient. For hosted APIs without a rendered-prompt
interface, combine their explicit native contract with wire and live generation
evidence. Record tested versions and constraints rather than implying universal
support from one model test.

Adapters construct at most one trailing assistant prefix message. Combining
response and reasoning prefill requires an explicit supported serialization;
never append two synthetic assistant messages or fall into the response-prefill
path that disables thinking or bypasses tools. Thinking-off plus an enabled seed
is a validation conflict, not permission to turn thinking on automatically.

## Tool replay, output, and budgeting

Inject the seed only into the first model call of the owning run. If tools are
called, the adapter constructs the completed reasoning record required by that
provider and replays it with its exact assistant/tool owner. Replaying this
completed record is not another prefill injection. Do not prepend the configured
pin at every tool round or copy it to child agents.

Displayable thinking contains newly generated reasoning only. For a backend that
returns only a suffix, keep that suffix separate from the seed. For a backend
that echoes the seed, remove it from the display projection only using the
qualified adapter's exact echo contract. A generic string-prefix heuristic could
erase genuinely generated text and is prohibited. If attribution is ambiguous,
the target is not qualified for this feature.

Required provider replay retains the correct complete reasoning inside the
existing protected continuation owner (ADR-063). Optional thinking replay follows
ADR-090: suffix-only display data is not treated as a complete replay record.
When replay requires a seed that cannot be reconstructed from the compatible
owner, Auto omits that optional block and Include reports incompatibility before
send. Never reconstruct it from today's pinned value. Required replay takes
precedence over optional Include/Exclude policy as it does today.

Budget from the same serialized projection sent to the backend. Count the seed
once as input, never as model-generated output. Mandatory continuation and its
assistant/tool owner are retained or evicted together. Reject oversize input;
never truncate a reasoning prefix or its required continuation to fit silently.
Provider-reported usage remains authoritative, including when a server echoes
input in its response. Generated thinking alone does not turn a failed or empty
answer into a successful final response.

## Persistence, privacy, and portability

Store the pin in a dedicated versioned `reasoning_prefill` object in
`conversations.metadata`, with `version: 1`, exact `text`, and boolean `enabled`.
Absence means no pin. The configured text must be nonblank and at most 4,000
Unicode code points, matching the current response-prefill size bound without
copying its trimming behavior. Reject invalid controls through the existing
boundary validator; never normalize or silently rewrite accepted whitespace.
Transient disabled empty editor drafts are not persisted as pins.

Pin updates merge only this key under the existing transaction and optimistic
revision rules. Persist both enabled and disabled pins. A failed save remains
visibly unsaved, and stale asynchronous completion cannot overwrite a newer edit.
Temporary chats keep the pin in memory; promotion persists the current value.
The pin is excluded from global/model defaults and from automatically inherited
fork/new-chat configuration, matching the existing fork prefill exclusion.

Conversation resume, lossless conversation export/import, and Sync v2 round-trip
the pin under message-content protection. Human-readable answer exports, FTS,
titles, summaries, speech, notifications, routine diagnostics, and error bodies
exclude it. Validate imports before mutation. Unsupported versions are preserved
on unrelated writes but cannot be executed; a present invalid/unknown pin causes
an explicit recovery state instead of silently behaving as though no pin exists.
Backends that cannot round-trip this object must reject persistent enabling/save
before provider contact. Do not describe it as local-only metadata.

The next-send slot and accepted submission's seed snapshot are process-memory
state, excluded from disk screen snapshots, sync/export, forks, and global
configuration. Retained sessions can navigate away and back without losing them.
After process loss, a pending turn whose exact seed is unavailable is not safely
reconstructable, even if its source was a pin: that pin may have changed. Recovery
must explain this and offer an explicit new generation or discard, following the
existing unknown-delivery guard. Do not rebuild a retry from a stored hash.

Memory-only configuration does not promise that input is absent from the provider
request or existing opt-in request capture. The seed can also remain inside
mandatory protected provider continuation when exact replay requires it. These
records follow their established retention, encryption, sync, and access rules;
they are never exposed as the model's newly generated thinking. Avoid a new
standalone historical seed column just to support the editor.

## Implementation boundaries

Use a focused pure reasoning-prefill module for validation, capability results,
and revision/reservation transitions. Integrate with the existing session store,
turn custody/preparation, queue, dispatch checkpoint, and terminal lifecycle.
`ChatScreen` owns entry points and view callbacks, not a second lifecycle engine.
Declare any new view hook in the shared slot contract and verify attach/detach.

Provider gateway and adapter integration use typed internal seed data stripped
before unrelated provider/trace serialization. Shared hosted wire contracts must
explicitly carry only the new fields their adapters require. A serializer alone
is insufficient if an intermediate normalization step drops the prefix flag.

Generation capture, continuation, context budgeting, persistence hydration,
import/export/sync validation, settings drafts, and fork exclusion all participate
in the same ownership contract. Reuse the existing versioned columns; increment
their envelope versions and advertised capabilities if their serialized contracts
change. Any actual SQL schema change follows the repository migration policy.

The existing response-prefill feature is a compatibility reference, not an
implementation shortcut. Its current durable-acceptance clearing differs from
its terminal-consumption helper, and its llama.cpp path forces thinking off.
Keep that existing behavior outside this feature's scope unless a separately
specified prerequisite is needed; write regressions proving it remains unchanged.

## Verification and delivery

Targeted checks must cover:

- Exact text, whitespace, blank/length/control validation; enable/disable/clear;
  precedence; unsupported overrides; thinking-off and response-prefill conflicts.
- Revision-safe reservation; two queued sends; identical-text rearming; queue
  removal; stale editor; predispatch cancel; stop; failed provider/tool round;
  retry after pin changes; unknown delivery; process-loss recovery.
- The real mounted Console and runtime hook path: editor to preview to adapter,
  navigation to another chat during execution, and completion of the original
  owner without mutating the new one.
- Stream and non-stream native serialization; precise attribution/echo behavior;
  required tool-history replay; no child inheritance or repeated injection;
  budget counting from the actual provider projection.
- Real SQLite save/resume/promotion; sibling metadata preservation; failed writes;
  lossless import/export/sync; unknown versions; unsupported persistent backends;
  ephemeral seed exclusion; existing fork/default exclusions.
- Existing response-prefill, thinking display/replay, agent continuation, queue,
  and recovery regressions, plus applicable lint, formatting, token-governance,
  and stylesheet bundle checks.

Live checks use an explicitly configured test target and report actual coverage.
No live provider calls or feature tests have run during this design work. Do not
run the full repository suite without the user's opt-in. Do not mark a provider
qualified from mocked tests alone.

Implementation planning follows approval of this written spec. Split runtime
foundation, provider qualification, and Console/persistence integration into
atomic tasks in dependency order; create each referenced task before linking it.
The shared controls remain disabled for unqualified targets throughout delivery.

## Review refinements incorporated

The design now specifies native continuation evidence rather than field
acceptance, one lifecycle independent of existing response-prefill shortcuts,
exact retry/queue ownership, seed-aware tool replay and display attribution,
Disable without deleting, whitespace preservation, explicit portability, and
honest recovery when an ephemeral seed has been lost.

Self-review completed for placeholders, contradictory lifetimes, unsupported
fallbacks, stale-owner mutations, tool replay, storage scope, and qualification
claims. The user approved this written version and authorized implementation planning.

## References

- [DeepSeek prefix API](https://api-docs.deepseek.com/api/create-chat-completion/):
  native prefix flag, reasoning input, and beta endpoint requirements.
- [vLLM chat protocol](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/):
  continuation and generation-prompt controls; these alone do not qualify every template.
- [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md),
  [ADR-064](../../../backlog/decisions/064-deepseek-dual-api-provider-boundary.md),
  [ADR-090](../../../backlog/decisions/090-console-thinking-block-ownership-and-replay.md),
  [ADR-092](../../../backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md),
  [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md).
- [Console wiring lessons](../../../backlog/docs/lessons-console-wiring.md) and
  [testing evidence lessons](../../../backlog/docs/lessons-testing-evidence.md).
