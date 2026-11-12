# Console prompt mix-ins

Date: 2026-11-01
Status: Accepted 2026-11-01; written spec reviewed by the user
Task: [TASK-38534](../../../backlog/tasks/task-38534%20-%20Design-prompt-mix-ins-for-Console-sends.md)
ADR: [ADR-286](../../../backlog/decisions/286-console-prompt-mix-ins.md)
Plan: [Prompt mix-in implementation](../plans/2026-11-01-prompt-mix-ins-implementation.md)
Implementation: Complete on focused dev branch, including all PR3346 review corrections; TASK-38544 Done. See durable QA for results and limitations.

## 1. Purpose and approved scope

A mix-in is a reusable instruction that modifies the response to the user's
message. Users manage it in the prompt library, select several for a chat, and
enable or disable them before sending. The composer and transcript retain the
original message. Mix-ins accompany only the current send, with a separate
record of the exact instructions used.

Examples include noir narration, concise answers, an assumptions section, or a
request to append a scene diagram in SVG/HTML. A mix-in may ask the model to use
an existing tool. It does not implement rendering, invoke a second generation,
enable a tool, or bypass tool permissions. Character chats retain their existing
tool restrictions. Requested output is subject to existing model and rendering
capabilities; raw HTML is not automatically executed or opened.

Approved v1 decisions:

- Mix-in is a dedicated prompt-library artifact type, separate from Prompt and
  Recipe, using the existing library infrastructure.
- Each mix-in has one literal instruction body and one Before/After placement.
  There are no variable forms, conditions, bundles, or automatic activation.
- Selected entries, their enabled states, and their order belong to one chat.
  Enabled entries stay enabled until changed. New chats start with none selected.
- Console access is part of the existing composer **Prompts** button and Prompt
  Workbench. There is no separate Mix-ins button or manager dialog.
- Library edits affect future sends. Running and queued sends retain their
  captured definitions. Saved chats restore their selections on reopening.
- Retry uses the original attempt's mix-ins; regeneration and edit-and-resend use
  current choices. Retry labels disclose the use of original mix-ins.
- Reliable off behavior takes precedence over maximum cache reuse. Preserve
  stable-prefix caching where supported without retaining old instructions in
  later model history.

This is a design-only document. The linked plan defines delivery work; no
migrations, runtime changes or provider qualification are delivered by this spec.

## 2. Existing owners and integration boundaries

The current library has typed Prompt/Recipe artifacts, local and server source
capabilities, retained versions, and portable record codecs. The Console has
per-session state, immutable turn configuration, a bounded queue, provider-neutral
prepared requests, and per-send transforms that leave stored user text intact.
Extend those seams rather than adding another prompt catalog or send engine.

| Concern | Existing owner to extend |
| --- | --- |
| Artifact model, validation and deterministic compilation | `tldw_chatbook/Prompt_Management/prompt_artifact_models.py`, `prompt_artifact_codec.py`, `prompt_block_compiler.py` |
| Library authority, version checks and source capabilities | `tldw_chatbook/Prompt_Management/prompt_scope_service.py`, `local_prompt_service.py`, `server_prompt_service.py`, `prompt_source_capabilities.py` |
| Library records and retained versions | `tldw_chatbook/DB/Prompts_DB.py`, existing file-backed migration mechanism |
| Portable records and server schemas | `tldw_chatbook/Prompt_Management/prompt_chatbook_record.py`, `prompt_markdown_export.py`, `tldw_chatbook/tldw_api/prompt_chatbook_schemas.py` |
| Library browse/editor | `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`, `library_prompt_work_pane.py`, existing Library prompt controllers |
| Console Workbench and entry point | `tldw_chatbook/Widgets/Console/console_prompts_modal.py`, `console_prompts_browse.py`, `console_prompts_state.py`, `tldw_chatbook/UI/Console_Modules/prompts.py`, existing composer action registry |
| Chat preference and durable attempt ownership | `tldw_chatbook/Chat/console_chat_store.py`, `chat_persistence_service.py`, `tldw_chatbook/DB/ChaChaNotes_DB.py` |
| Admission, queue and immutable send inputs | `tldw_chatbook/Chat/console_chat_controller.py`, `console_prompt_queue_coordinator.py`, `console_turn_context.py` |
| Provider payloads, capture descriptors and caching | `tldw_chatbook/Chat/console_prepared_request.py`, `console_provider_gateway.py`, `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` |

These are integration targets, not instructions to add mix-in logic to every
file. A focused, Textual-independent composition function should express the
ordering rule once. UI components emit intents; source services read/write
artifacts; the chat runtime owns selection and execution snapshots. Avoid a new
framework or expanding the existing large controller with a second orchestration
path.

## 3. Library artifact and compatibility

Reuse the existing name, description/details, author, tags, search, versioning,
deletion and collection facilities. Add `artifact_type = "mixin"` and a strict,
versioned definition identifying a literal text mix-in, its body and placement.
The definition is the single editable authority. The simple mix-in editor has
Name, Instructions, Placement, and the existing optional metadata; it does not
present System/User lanes or the block-prompt editor.

The implementation plan must allocate the schema version through the existing
explicit kind/version dispatch, keeping server-v1 and Console-v2 definitions
unchanged. A mix-in is not inferred from tags, titles, a body substring or a
compiled compatibility lane. Any compatibility text is a deterministic projection
and cannot make the artifact eligible for normal Prompt insertion or execution.
Whitespace-only instruction bodies and unknown placements are invalid. Validate
against source limits while preserving valid body text exactly, including braces,
newlines and leading/trailing whitespace. Do not apply Prompt variable parsing.

Library > Prompts distinguishes Prompts, Recipes and Mix-ins through a type
filter. Normal prompt pickers, `/prompt`, System application, recipe execution and
improvement paths must not accidentally accept mix-ins. Selecting a mix-in offers
**Enable in current chat** through the existing guarded, exact-target Console
handoff. It does not insert text into the composer or replace the System prompt.
If there is no valid Console destination, show the existing open-Console remedy
rather than creating an undisclosed chat. Re-enabling an already selected entry
does not create a duplicate or change its position.

Extend brief/detail/search/write/version/import/export contracts consistently.
Source capability negotiation must explicitly advertise mix-in support before
writes or selection. An older server is unavailable for these operations; never
downgrade the type to Prompt or switch to the local source silently. References
identify the source authority and stable record identity, not an unscoped row ID.
A profile, server, or runtime/source change cannot bind an old selection to an
unrelated record with the same numeric ID.

The portable Chatbook Prompt-record codec currently accepts only Prompt/Recipe
in record version 1. Introduce a new explicit record version for mix-ins while
retaining old readers for existing formats. Preserve type, body, placement and
ordinary portable library metadata. Older readers must reject unsupported records
clearly instead of importing them as normal prompts. Apply the same explicit-type
rule to Markdown export/import. Imported library mix-ins are never auto-enabled.
No cross-device chat-selection or retry-record sync protocol is introduced in v1.

## 4. Console interaction and ordering

Extend the existing composer **Prompts** button to open the Prompt Workbench with
a Mix-ins section. That same control reports the enabled count, with full wording
in its accessible description; enabled names may appear in its existing compact
summary when space permits. Narrow widths retain access through the same existing
Prompts action/menu route. Do not add a second manager, separate persistent
button, or another permanent toolbar row.

The Mix-ins view lists entries selected for the current chat, grouped into Before
and After. **Add mix-in** browses/searches the existing library with the Mix-in
filter and visible source authority. Each selected row provides:

- An explicit enabled/disabled checkbox and its name.
- Inspect/edit access to the instructions through the existing library owner.
- Move up / Move down within its placement group.
- Remove from chat, which does not delete the library artifact.

Adding an entry appends it to its placement group. Disabled entries remain in
that group's order; re-enabling restores their previous position. Removing an
entry removes that chat selection. Library sorting/search never changes injection
order. Placement is edited in the library definition, not overridden privately
per chat. Observing a placement edit moves the entry to the end of the new group,
preserving the relative order of other entries; the same rule applies when first
observed during send preparation after a reopen. Display the changed placement.

Toggles and ordering apply immediately to future sends in the exact originating
chat. Closing or escaping the selection view retains them. Library body edits
retain the editor's existing save/discard rules. A tab switch cannot retarget an
open Workbench transaction. Disabled/unavailable rows remain visible and cannot
make the enabled count misleading. If changed since last use, mark an entry as
updated; Next send shows the effective definition and placement.

Use the existing design tokens, forms/list/modal patterns and focus restoration.
Keyboard users can toggle, inspect, add, remove and reorder without dragging.
Do not claim terminal-convention keys or shadow global bindings. Footer hints
advertise only implemented actions. Verify narrow layouts rather than assigning
new literal dimensions.

## 5. Selection and attempt lifetime

Saved chats persist source-qualified selections, enabled states, group ordering
and last-used revision information as chat-owned local preferences. Empty is the
default; no global, workspace, persona or character default enables mix-ins.
Temporary chats retain selections and snapshots only within their existing
lifetime; promotion saves their current state with the conversation.

Reopening an existing chat restores its selections and resolves the current
library definitions for its next new send. New chat creation starts empty,
including a new chat created by copy/fork; inherited historical attempt records
remain historical data and never activate a selection. Branch changes within the
same chat retain that chat's selection. Unknown future selection formats are
preserved and shown as unavailable, not guessed or destructively overwritten.

| Operation | Definition and selection used |
| --- | --- |
| New manual send | Current selections and current library definitions, frozen during admission |
| Queue admission | Current selections and definitions captured when enqueued, not when drained |
| Already preparing/accepted execution | Its captured values; later toggles/edits cannot replace them |
| Retry a failed attempt | That exact attempt's recorded instructions, placement and order |
| Regenerate an existing reply | Current selections and current definitions, captured for a new attempt |
| Edit and resend | Current selections and current definitions, captured for a new attempt |
| Explicit Continue starting a new turn | Current selections and current definitions |
| Tool continuation or transport retry within one response | The same attempt snapshot, applied once |

Admission freezes selection identity/order for the owning chat before yielding
to asynchronous work. Resolve enabled definitions off the UI thread through the
source authority and capture their exact revisions before accepting the send or
queue entry. If a revision changes during the reads needed to form that snapshot,
reject stale preparation for retry; do not accept a mixed set. This is a source
snapshot plus an immutable execution input, not a claimed transaction across
the Prompt and conversation databases. Edits after successful capture affect the
next new capture. Source identity changes and stale session targets refuse
admission. Once accepted, a queued entry does not re-resolve its library bodies;
existing permission and execution-authority checks still apply at dispatch.

The existing queue remains process-memory-only and retains its existing capacity,
pause and quit behavior. This feature does not add crash replay of unsent entries.
Cancelled/refused preparation preserves the draft and changes no selection.

Retry is labeled **Retry with original mix-ins** when its snapshot is nonempty.
The confirmation/context surface identifies the original set even if current
toggles are off. An original empty set remains empty on Retry even when mix-ins
are now enabled. A missing/corrupt original snapshot cannot become an empty set
or a newer library version: disable that Retry and offer regeneration with current
choices. Legacy attempts predating this feature are explicitly treated as having
no mix-ins; a missing record for a new-format attempt is an error.

Automatic wakeups, goal iterations and machine follow-ups do not newly resolve
chat-selected mix-ins in v1. Existing instructions within an already accepted
response remain part of that attempt; a new automatic turn cannot inherit new
human prompt modifiers merely because they are selected in the chat.

## 6. Request composition and current-send-only semantics

Existing command recognition, skill substitution, dictionaries, world info and
retrieval preparation operate on their existing inputs. Literal mix-in bodies
are added afterward to the current user-message envelope, before final token
accounting, semantic capture binding and provider dispatch. They are not reparsed
as commands, template variables, world-info triggers or search queries.

The logical request order is:

```text
Existing system instructions and tool definitions
Existing conversation history without old mix-in instructions
--- end of reusable clean-history prefix ---
Enabled Before mix-ins, in visible order
Current prepared user message
Enabled After mix-ins, in visible order
```

For string content, join the nonempty Before bodies, prepared user content and
After bodies with two newlines between components. Do not trim or normalize body
contents. No mix-ins means the original payload is unchanged. For multipart
content, insert text blocks before/after the original ordered parts; preserve
image, attachment and other nontext parts exactly. Apply the same rule once per
prepared attempt across plain and agent execution. Tool-loop requests carry the
already composed current message without wrapping it a second time.

Keep the user's canonical composer text, stored message body and normal transcript
export unchanged. Usage records are inspection data, never synthetic conversation
messages. Exclude old mix-in bodies and usage receipts from later model history,
compaction summaries and retrieval queries. Toggling off stops explicit injection
for future new sends; prior assistant prose or generated artifacts may still
influence the model as ordinary history. Mix-in requests grant no new tool, file,
rendering or provider authority. A conflicting instruction is sent as authored;
there is no priority score or semantic conflict resolver.

Next send preview and existing token/cost projections account for enabled bodies
using the same frozen composition rules. Preview is read-only, performs no tool
or generation calls, and is provisional until admission validates the exact
source revisions. Turning off all entries leaves no hidden mix-in envelope.

## 7. Exact operational snapshots and privacy

Retain operational snapshots independently of optional diagnostic exchange
capture. A sanitized trace cannot serve as the exact retry source, and enabling
mix-ins must not silently enable exchange capture.

Use the existing conversation database and persistence transaction owner for:

- A chat-owned selection preference with source-qualified references and order.
- Immutable used-definition snapshots holding exact body, name at use, source
  reference, source version and placement.
- Attempt-owned ordered references to those snapshots, including an explicit
  recorded-empty state and an attempt identity independent of the eventual reply.

Persist a used definition once per distinct used revision within the owning
conversation and reference it from attempts; do not duplicate accumulated history
or complete provider requests. Bodies belong in dedicated content storage, not
generic message metadata, trajectory summaries, logs, notifications or exception
strings. Operational snapshots follow ordinary private conversation-storage and
backup boundaries. They are intentional retained user content, even with exchange
capture off; describe this in the inspection surface. Trace redaction/purge still
governs traces and must not be represented as deletion of these retry records.

Before dispatch for a durable attempt, record its mix-in snapshot references
through the existing attempt/admission persistence seam. Failure to retain them
blocks dispatch, with draft-preserving retry. A failure after provider execution
must use existing uncertain-outcome recovery and cannot invite an undisclosed
duplicate send. Temporary attempts retain the same structure in memory. Promotion
commits their still-owned snapshots with the conversation bundle.

Library edits/deletion never mutate used snapshots. Conversation copy/fork and
deletion preserve the existing transcript/attempt lineage boundary: history may
carry immutable receipts without activating selections; required retained bodies
cannot disappear while a surviving conversation owns a receipt. Hard purge of
the last owning conversation removes its operational content, while soft deletion
follows the existing recovery lifetime. Include new tables in existing backup and
restore inventory. Do not introduce another database, raw JSON sidecar or new
cross-device synchronization scheme.

Historical inspection shows the used names, source versions, placement and order,
and can explicitly expand the stored instructions. Inspection never re-fetches
current library content as though it were historical. Provider semantic traces
reference/project these sections through their existing credential/PII policy;
the operational snapshot cannot bypass trace export filtering.

## 8. Prompt caching

Before/After placement is relative to the current user message, not the start/end
of the entire request. Do not rewrite the system prompt or tool catalog when
mix-ins change. Preserve body bytes, selected order and deterministic separators.
Do not add timestamps, random labels or revision counters to model-facing text.

Exact prefix reuse is provider/model dependent. The next turn removes the previous
turn's injected instructions from its history, so it can differ from the previous
whole-request cache entry at that earlier insertion point even when the selection
stays enabled. A stable cache key alone cannot repair this mismatch.

Where supported, cache the clean conversation-history boundary immediately before
the current mix-in/user tail, preserving existing tool/system cache points. This
allows reuse of older history for which an eligible identical entry exists; the
newest exchange may still require processing again. An additional end-of-current-
attempt breakpoint can be useful for unchanged tool-loop continuations, but only
within the provider's supported breakpoint budget and existing configuration.
Do not silently enable caching, change TTLs, force cache writes or assume every
OpenAI-compatible endpoint accepts provider-specific fields.

The current Anthropic adapter marks tools, system and the final message. It needs
an explicit clean-history boundary for this feature; its prior end-of-user-message
write alone cannot guarantee reuse after removal of historical mix-ins. Preserve
existing kill switches, TTL/degrade behavior and provider limits. Keep OpenAI's
optional prefix-derived cache key stable when only mix-ins change; add explicit
breakpoints only for models/endpoints whose capabilities support them.

Cache misses never change correctness or reintroduce disabled instructions.
Report measured provider usage when available; do not promise a hit rate or savings
without live evidence. Official references consulted 2026-11-01:
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
and [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## 9. Failure behavior

| Condition | Required outcome |
| --- | --- |
| Deleted, unreadable or wrong-source enabled record | Block new admission; retain draft; identify entry and offer disable/remove or retry loading |
| Unavailable disabled record | Keep visible; do not read/inject its body or block an otherwise valid send |
| Unsupported artifact/source capability | Explain unavailable capability; never flatten, silently switch source or reinterpret as Prompt |
| Invalid placement, malformed definition or oversized body | Refuse save or admission at the boundary; never truncate or silently omit |
| Whole request exceeds context limits | Existing context-budget recovery; count mix-ins and never silently drop them |
| Stale session, source or selection during preparation | Refuse exact stale preparation without clearing a newer draft or mutating another chat |
| Durable selection save fails | Retain live choice, mark it unsaved, offer bounded exact-revision retry; never claim it will survive reopening |
| Durable attempt snapshot save fails | Do not dispatch; preserve draft and offer retry |
| Original retry record unavailable | Explain unavailable original instructions; offer explicit regeneration using current choices |
| Provider rejects optional cache controls | Existing supported degrade behavior; never change instruction content |
| Model ignores mix-in or emits unsupported output | Preserve ordinary response/error handling; no automatic repair or follow-up generation |

Selection-save retries target the originating conversation and current preference
revision so they cannot overwrite newer changes. Loading work runs in existing
workers with stale-result rejection. Existing storage admission, path validation,
parameterized SQL and backup/recovery rules remain mandatory.

## 10. Verification required during implementation

Use targeted tests touching the changed owners; a full suite requires separate
user authorization. No runtime test results are claimed by this spec.

1. Pure composition: zero/one/many Before and After entries, disabled entries,
   stable ordering, exact whitespace/braces, no recursive interpretation, string
   and multipart content, unchanged no-mix-in payload, no duplicate tool-loop wrap.
2. Artifact contracts: strict type/kind/version/placement/body validation; existing
   Prompt/Recipe compatibility; normal insertion and recipe paths reject Mix-ins;
   local/search/version/portable round trips and unsupported-server refusal.
3. Chat isolation and UI: existing composer Prompts entry point/count; add/toggle/
   remove/reorder; disabled position retention; library sort independence; placement
   edits; keyboard focus; exact-origin behavior across tab/workspace switches.
4. Lifetime: blank/copy/fork new chats start empty; reopen restores; temporary and
   promotion behavior; selection persistence failure/retry cannot overwrite newer
   choices; invalid future formats remain intact.
5. Send admission and queue: held asynchronous reads plus selection/library edits;
   missing enabled versus disabled definitions; captured queue content remains
   unchanged by later edits, deletion and toggles; cancellation preserves drafts.
6. Retry/regeneration: original nonempty and empty snapshots, library deletion,
   restart with diagnostic capture off, unavailable/corrupt new-format snapshots,
   legacy attempts, regenerate-current and edit-and-resend-current behavior.
7. Storage/privacy: transaction failure blocks pre-dispatch; snapshots reused across
   attempts; independent trace controls; no body-bearing metadata/logs; backup/
   restore, fork retention and final conversation purge preserve ownership.
8. Request semantics: plain, character and agent paths; original transcript and
   future history remain clean; summaries/retrieval omit old instructions; empty
   drafts and command-only operations cannot become sends through a mix-in.
9. Caching: stable clean prefixes under toggles/reorder/edits; correct eligible
   history boundary; Anthropic breakpoint budget, disable/degrade/TTL behavior;
   canonical versus custom OpenAI endpoint gating; no dependence on a cache hit.
10. Live UI: isolated profile, normal and narrow terminal sizes, actual composer
    Prompts action, preview, keyboard toggles/order, send/queue/retry/regenerate,
    reopen, and visible save/source failures. Exercise real provider usage for
    cache qualification when credentials are available; label unavailable evidence.

For a scene-diagram example, verify the instructions and any returned SVG/HTML as
text through existing handling. Only claim rendering or tool execution if that
existing surface/tool was actually exercised. Do not add a renderer to pass the
example. Run applicable scoped lint/format, schema, CSS-token and bundle checks
when their owners change. Update the living chat-pipeline, library, database and
provider architecture docs and user guide with the implemented behavior.

## 11. Alternatives and implementation handoff

Rejected alternatives are recorded in ADR-286: tags as behavior, a separate
catalog, editable per-chat copies, historical injection replay, System-lane
mutation, current-value retry, trace-only retry storage and cache-first semantics.

The user approved the written spec and ADR on 2026-11-01 and authorized the linked
implementation plan. That plan divides the work into atomic, testable PRs for
artifact/source compatibility, chat and attempt ownership, request/retry/cache
integration, and existing Workbench/Library UI with targeted/live qualification.
Every implementation task must link ADR-286 and record its own ADR check.
Allocate migration versions against the actual implementation baseline; do not
reserve stale numeric versions in this design.

Design provenance: the supplied
[pose visualizer example](https://www.reddit.com/r/SillyTavernAI/comments/1xib3jp/more_pics_from_my_pose_visualizer_prompt_from/)
motivates an optional output instruction, not an embedded renderer or an automatic
image-generation pipeline.
