# Local Citation Repair Transition Design

**Date:** 2026-07-26
**Backlog:** TASK-553.15
**Parent:** TASK-553
**Status:** Approved for written-spec review
**ADR required:** no
**ADR path:** `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

## Purpose

Keep a local RAG answer visibly provisional until its citation markers have
been structurally checked. When markers are missing or invalid, make exactly
one bounded repair request, preserve the answer's non-marker text, and resolve
the same assistant message to either the repaired body or the original body
with honest recovery copy.

This task implements the provisional-stream and visible-repair workstream from
the accepted citation-provenance design. It does not claim semantic support,
grounded status, or durable occurrence-backed provenance.

## Existing foundations

TASK-553.13 captures the exact local prompt evidence and assigns stable
`[S#]` ordinals. TASK-553.14 defers the first assistant write until terminal
selection and can atomically persist a message with a sealed local trace when
the builder can represent the selected body.

The current Console flow streams directly into one assistant placeholder and
then immediately marks it complete. Marker-bearing answers currently fail
closed at canonical finalization because occurrence mappings are not yet
available. That behavior must remain honest: this task may repair visible
markers, but it must not label or persist those answers as grounded.

Canonical citation writes are also controlled by a recovery switch that is
disabled by default. Structural citation checking is user-facing generation
behavior and must not depend on that persistence switch.

## Approved decisions

- The Console controller owns repair orchestration.
- A repair uses the same resolved provider, model, and sampling settings as
  the initial response.
- The repair request bypasses the agent loop and exposes no tools.
- Valid initial markers do not cause a second request.
- Missing or invalid markers cause at most one repair request.
- The initial body stays visible while repair runs.
- A repaired body replaces the same assistant message only when its
  non-marker text is unchanged and all eligible markers are valid.
- Cancellation, failure, invalid output, or a changed claim selects the
  original body with honest copy.
- The current-session original-attempt preview is transient and read-only.
- This task never adds a second assistant message, fetches more evidence,
  renumbers sources, reruns the RAG pipeline, or claims semantic support.

## Approaches considered

### Selected: controller-owned repair session

The controller already owns provider resolution, active-run cancellation, and
the direct-versus-agent dispatch boundary. A small request-scoped repair
session can span initial streaming, structural checking, one repair request,
and terminal selection without giving the store provider dependencies.

Pure validation and prompt construction live in a focused module rather than
expanding the already large controller.

### Rejected: store-owned repair

The store is the correct owner for message state and persistence deferral, but
not for provider prompts, async streams, cancellation, or model settings.
Putting repair in the store would couple data state to external generation.

### Rejected: general repair service

A separately injected orchestration service would be useful only after other
repair producers share the same contract. Server traces have different
authority and transport boundaries, so a general service now would be
speculative abstraction.

## Pure repair contracts

Create a focused `tldw_chatbook/Chat/citation_repair.py` module containing only
bounded, provider-independent contracts and functions.

### Hard limits

The module defines these non-configurable limits:

| Constant | Value | Purpose |
|---|---:|---|
| `REPAIR_EVIDENCE_CONTEXT_UTF8_BYTES_MAX` | `SNAPSHOT_TEXT_UTF8_BYTES_MAX` (`64 * 1024`) | Exact evidence context retained for repair |
| `REPAIR_ALLOWED_ORDINALS_MAX` | `EVIDENCE_ENTRIES_PER_PROMPT_MAX` (`64`) | Allowed ordinal count and maximum ordinal value |
| `REPAIR_MARKERS_MAX` | `CITATION_OCCURRENCES_MAX` (`512`) | Total eligible well-formed and malformed marker-like tokens scanned in either body |
| `REPAIR_MARKER_CHARACTERS_MAX` | `MARKER_CHARACTERS_MAX` (`32`) | Maximum ASCII characters in one eligible citation-like token |
| `REPAIR_ANSWER_BODY_UTF8_BYTES_MAX` | `ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX` (`1024 * 1024`) | Initial and repaired body buffer |
| `REPAIR_FIXED_OVERHEAD_UTF8_BYTES_MAX` | `8 * 1024` | Fixed instruction and literal delimiter content |
| `REPAIR_REQUEST_UTF8_BYTES_MAX` | `REPAIR_ANSWER_BODY_UTF8_BYTES_MAX + REPAIR_EVIDENCE_CONTEXT_UTF8_BYTES_MAX + REPAIR_FIXED_OVERHEAD_UTF8_BYTES_MAX` (`1,122,304`) | Sum of UTF-8 bytes in the canonical system and user message content before provider adaptation |

These limits are implementation constants, not settings. Exact-limit values
are accepted and limit-plus-one values fail closed. The fixed instruction and
literal delimiters must fit the fixed-overhead allocation; changing them
requires updating its boundary tests.

### `CitationRepairContract`

The local capture boundary returns an optional immutable contract containing:

- marker namespace, initially only `chatbook_s_v1`
- sorted unique allowed positive marker ordinals
- the exact prompt evidence context already sent to the initial provider
- a schema version

The contract requires one contiguous ordinal tuple exactly equal to
`(1, 2, ..., N)` for `1 <= N <= 64`, applies the hard limits above, and contains
no builder, database handle, source primary key, credential, provider object,
or mutable request state.

`LocalRagContextResult` carries this contract independently of its optional
canonical `CitationTraceBuilder`. A successfully formatted non-empty local
prompt can therefore be checked when canonical persistence is disabled or its
key material is unavailable.

Capture returns no repair contract when:

- no prompt evidence was submitted
- local evidence normalization or authorization failed
- the marker namespace is unsupported
- the context or ordinal set exceeds its limits

### Structural decision

A pure decision function examines the exact unrendered answer with the
Markdown-aware canonical marker scanner. Outside fenced code, inline code, and
escaped literals, a citation-like token is the exact bracket form matched by
`\[S[0-9,\t ]+\]`. A token is well formed only when its full text matches
`\[S[1-9][0-9]*\]`. Therefore `[S0]`, `[S01]`, comma-grouped forms such as
`[S1,S2]`, and TAB-or-U+0020-SPACE variants are malformed. A well-formed
positive ordinal absent from the contract is unknown. Both malformed and
unknown tokens are invalid.

The scanner applies the 32-character token bound before ordinal
classification. Overlong citation-like tokens are invalid. It never converts
untrusted decimal text to an integer: known ordinals are compared against the
precomputed ASCII strings `{"1", ..., str(N)}`. This keeps arbitrarily long
digit sequences bounded and avoids interpreter-dependent integer parsing.

Repair scanning and the existing canonical span scanner must share the same
Markdown-code and escape exclusion traversal. The implementation may promote
that traversal to one package-private helper consumed by both modules; it must
not duplicate fenced-code, inline-code, or preceding-backslash logic in
`citation_repair.py`.

It returns one of:

- `not_applicable`: no valid repair contract
- `valid`: at least one eligible marker exists and every eligible marker
  is well formed and belongs to the contract
- `repair_required_missing`: no eligible citation-like token exists
- `repair_required_invalid`: at least one eligible token is malformed or uses
  an unknown ordinal
- `unavailable`: the answer exceeds its UTF-8 bound or the combined count of
  eligible well-formed and malformed tokens exceeds `REPAIR_MARKERS_MAX`

Repeated, grouped, and reordered known markers are structurally valid. Markers
inside fenced code, inline code, or escaped literals are ignored. Structural
validity never means the cited snapshot supports the claim.

### Claim-preservation projection

Repair may alter citation marker syntax but no other answer text.

The projection uses the same Markdown-aware token ranges as the structural
decision, including malformed citation-like tokens in the initial body. It
walks token ranges from right to left. For each token it deletes the token's
exact Unicode-codepoint range and also deletes exactly one immediately
preceding U+0020 SPACE, if present. It never removes tabs, newlines, other
ASCII whitespace, punctuation, or more than one preceding space. Processing
right to left makes adjacent `[S1][S2]` and space-separated `[S1] [S2]`
deterministic without a separate “marker group” rule.

This allows conventional text such as:

```text
Original: Aurora began at 09:30 UTC.
Repaired: Aurora began at 09:30 UTC [S1].
```

The original and repaired projections must otherwise be byte-for-byte equal
as UTF-8 text. The projection does not normalize Unicode, punctuation,
newlines, general whitespace, Markdown, or case.

Unknown and malformed citation-like tokens are removed from the initial
projection, so `[S9]`, `[S0]`, or `[S1,S2]` may be replaced by `[S1]`. Code and
escaped literals are not eligible tokens and remain part of the compared text.

A repaired body is selectable only when:

- it is non-empty
- it fits the existing answer-attempt UTF-8 body limit
- its projection exactly equals the initial projection
- it has at least one eligible marker
- every eligible marker ordinal is allowed

Any failed condition selects the original body.

## Repair request

Repair is a direct provider request using the initial response's already
resolved provider/model/settings. It does not enter the agent loop and does not
advertise MCP, built-in tools, skills, or approval hooks.

The request contains only:

1. a fixed system instruction defining citation-marker-only repair
2. the exact bounded evidence context
3. the exact initial answer

Evidence and answer text are delimited as untrusted data. The instruction:

- permits insertion, deletion, replacement, grouping, or reordering of
  `[S#]` markers
- forbids changing any other text
- forbids new facts, explanations, prefaces, code fences, or metadata
- requires returning only the repaired answer

The controller must not replay conversation history. Before dispatch it
first applies the UTF-8 request limits above, then verifies the complete
two-message repair payload fits the resolved model window without trimming.
It counts the exact canonical system and user messages with
`count_console_messages_tokens`. The response reservation is the greater of:

- the positive resolved `max_tokens`, or `DEFAULT_RESPONSE_RESERVATION` (`1024`)
  when `max_tokens` is absent; and
- the same counter's token estimate for the exact initial answer as one
  assistant message.

The safety margin is `max(512, resolved_window // 50)`, matching the Console
history boundary. Dispatch is allowed only when `prompt_tokens +
response_reservation + safety_margin <= resolved_window`. The reservation is
not clamped and the payload is never trimmed. If the request byte cap, token
window lookup/count, or inequality fails, repair is unavailable and the
original is selected.

Repair chunks are collected off-screen into a bounded buffer. Empty output or
crossing the answer body byte limit aborts collection and selects the original.
No partial repaired body is ever shown.

## Controller lifecycle

The controller creates one request-scoped `ConsoleCitationRepairSession`
before appending a repair-eligible assistant placeholder. The session
contains:

- the immutable repair contract
- the resolved provider settings
- exact initial and repaired bodies only while needed
- one-attempt state
- terminal selection and safe presentation metadata

It is not stored in the message/session serialization model.

Repair eligibility is independent of canonical builder readiness. The store
therefore gains an explicit `defer_terminal_persistence` argument for an empty
assistant placeholder requested with `persist=True`. The argument:

- is valid only for an empty, attachment-free assistant placeholder
- arms terminal persistence whenever a persistence backend exists, even when
  no citation finalizer is installed
- does not consult canonical-write readiness
- prevents streaming materialization or UI polling from writing the initial
  body
- releases through the existing terminal completion, stop, failure, and
  cleanup paths

When a builder is ready, the same placeholder may carry both the citation
finalizer and the repair defer flag; the store keeps only one deferral entry.
When no builder is ready and persistence exists, terminal completion performs
one ordinary stable-ID write of the selected body. A no-op citation finalizer
must not be used to simulate repair deferral.

The store also gains one atomic
`replace_deferred_terminal_body(message_id, selected_body)` operation. It
accepts only a deferred, attachment-free assistant message still in
`pending`/`streaming` state and a non-empty bounded body. In one synchronous
mutation it replaces `message.content`, collapses the stream buffer to exactly
that body, updates the materialization count, and leaves status and persistence
unchanged. Only successful repaired selection calls it. The UI can therefore
never observe an empty or partially replaced body between reset and append.

The lifecycle is:

```text
initial_streaming
  -> citation_check
      -> initial_selected_valid
      -> repair_streaming
          -> repaired_selected
          -> original_selected_warning
          -> original_selected_canceled
```

The assistant message remains nonterminal throughout checking and repair.
Terminal persistence still occurs exactly once after a body is selected.

Both direct-provider and agent-generated initial answers enter the same
post-generation boundary. Agent repair is still a direct, tool-free provider
request; it does not resume or supersede the agent run.

### Shared post-generation selection seam

Only the initial send receives a `ConsoleCitationRepairSession`; retry,
regenerate, edit/resend, continue, and recovered-draft paths pass `None`.
Direct generation invokes one async selection coordinator after the provider
stream ends and before `mark_message_complete`. An agent run invokes the same
coordinator only after a genuine `RUN_DONE` outcome and before
`_complete_agent_message` or run-to-message anchoring.

The coordinator always materializes and reads the exact initial body from the
store-owned assistant placeholder. It does not use `outcome.final_text`, which
can diverge from the body after tool-turn resets or streamed normalization.
Provider failure, agent failure/cancellation, empty output, and synthesized
fallback copy keep their existing terminal behavior and never dispatch citation
repair.

Synthesized fallback identity is out-of-band, never inferred by comparing body
text. The outer response path creates one request-scoped
`ConsoleProviderStreamSignals` value containing only a thread-safe
`synthetic_fallback_emitted` event. The contract lives beside the gateway
contracts in `console_provider_gateway.py`. The optional signal is passed
unchanged through direct `stream_chat` calls and, for agent runs, through
`ConsoleAgentBridge` and its streaming model adapter to the same gateway call.
Existing callers may omit it.

The gateway sets the event only in the exact normalization branches where the
gateway itself emits fallback copy; it sets the event before yielding that
copy. It does not mark genuine provider content, even when genuine content is
byte-for-byte equal to a fallback string. The gateway's yielded chunk types and
text remain unchanged. If any synthesized fallback was emitted, the selection
coordinator bypasses citation repair conservatively and completes through the
existing path. The signal contains no answer text, evidence, exception,
provider credential, or mutable controller state; it is neither serialized nor
logged.

When repair is required, the coordinator first publishes
`CHECKING_CITATIONS` and the safe checking presentation, then yields to the
event loop once before provider dispatch. It rechecks message/session ownership
and the stop request after that yield. This makes the transition paintable and
gives Stop a real pre-dispatch boundary rather than a state that exists only
inside one uninterrupted callback.

The outer `_stream_assistant_response` call remains the sole owner of the
active asyncio task and assistant-message identity across direct dispatch,
agent dispatch, checking, repair, and terminal selection. Nested direct and
agent helpers must not clear active-run ownership in their own `finally`
blocks. Agent thread cancellation retains its existing per-run
`threading.Event`; the citation repair phase uses the outer asyncio task and
the controller's stop flag.

### Active-run ownership

The same active asyncio task and controller stop request remain authoritative
from initial dispatch through terminal selection. During the agent phase, its
existing per-run thread event mirrors that stop request for the worker thread;
it does not replace the outer task's ownership. Existing direct and agent
`finally` blocks must not clear active-run state before repair resolves.

Stop behavior is:

- stop during initial generation: preserve existing stopped-response behavior
- stop after the initial body but before repair dispatch: signal and cancel the
  repair phase without synchronously terminalizing the message
- stop during repair: cancel collection without calling
  `mark_message_stopped`; the coordinator selects the original, calls
  `mark_message_complete`, sets run status `STOPPED`, and shows
  `Citation repair canceled`
- a late repair chunk after cancellation is discarded

The linearization point is the coordinator's synchronous selection commit. A
stop observed before that commit wins and selects the original. A stop arriving
after the commit sees no active run and is a no-op. Because initial generation
already completed, a canceled repair leaves the assistant message itself
`complete`, not `stopped`, so normal actions and future provider history remain
honest. `stop_active_run` does not append its durable system row while the
assistant is still deferred. After `mark_message_complete` persists the
selected original, the coordinator appends the existing durable user-stop
record with phase-specific copy `Citation repair canceled by user.` rather
than the inaccurate `Response stopped by user.`. This ordering keeps the
persisted parent chain and active leaf correct.

Closing the owning session cancels the active request, discards repair output,
clears its preview entry, and never recreates the session or message.

## Console presentation

Add `CHECKING_CITATIONS` to `ConsoleRunStatus`. Send remains disabled and Stop
remains enabled in that state.

The message stays in its existing nonterminal status until selection. It may
carry only safe transient presentation metadata:

- phase
- notice code
- whether the original attempt is currently available

No original body, repaired body, evidence text, provider prompt, or exception
is placed in presentation metadata.

Visible copy is:

- while checking or repairing: `Checking citations…`
- successful selection: `Citations repaired · View original attempt`
- unavailable, failed, or invalid repair:
  `Citation repair unavailable · Original response kept`
- canceled repair: `Citation repair canceled`

The repaired notice is structural. Every notice test asserts that it does not
show a grounded badge or imply that support was checked.

### Original-attempt preview

The controller keeps a bounded current-session map from message ID to original
body only for successfully repaired messages. Each body already satisfies the
answer body limit. The map retains at most eight entries; inserting a ninth
evicts the least recently used entry and removes that message's
`original_available` presentation flag.

The transcript passes `original_attempt_available` as an explicit optional
context argument to `ConsoleMessageActionService.available_actions`; the
service exposes `View original attempt` only when that argument is true.
Existing callers, `plain_action_row`, and exports omit the argument and remain
byte-identical. Availability may also ride on the safe transient message
presentation metadata for row signatures, but the original body never does.

Activation retrieves the body through the controller and stores it in
screen-local ephemeral preview state. The transcript renders a clearly
labeled:

```text
Original attempt (not selected)
```

block under the selected message. Activating the action again hides the block.
The preview never changes `ConsoleChatMessage.content`, variants, persistence,
copy action output, TTS input, export data, or provider history.

Session close, controller shutdown, message deletion, cache eviction, or
starting a replacement lifecycle for that message clears both the controller
entry and any screen-local preview. Restart does not restore it in this task.

## Persistence and provenance honesty

The selected visible body is the only body passed to ordinary message
persistence and future provider history.

This task does not create canonical citation occurrences. Marker-bearing
answers therefore continue to fail closed at the existing citation finalizer
and persist through the ordinary message path without a grounded association.
The repair notice remains a structural-generation notice, not provenance
status.

No incomplete builder, transient repair session, or original preview is
serialized. No schema migration is required.

## Failure handling

All repair failures retain the initial body:

| Failure | Result |
|---|---|
| Missing repair contract | Existing completion behavior |
| Exact repair request does not fit | Original + `Citation repair unavailable · Original response kept` |
| Provider unavailable or raises | Original + unavailable notice |
| Empty repair output | Original + unavailable notice |
| Output exceeds byte limit | Original + unavailable notice |
| Output changes non-marker text | Original + unavailable notice |
| Output still has missing/invalid markers | Original + unavailable notice |
| User cancels repair | Complete original + `Citation repair canceled`; run status `STOPPED` |
| Session closes | Discard result; no resurrection |

Unexpected errors log only operation, reason code, provider family when safe,
bounded sizes/counts, and lifecycle state. They never log answer text,
evidence, source identity, locator, prompt, exception text, or traceback.

## Compatibility and rollout

- Non-RAG sends are byte-for-byte behaviorally unchanged.
- RAG sends without a repair contract complete as they do today.
- Valid markers add validation work but no second provider request.
- Canonical-write readiness does not control repair eligibility.
- Existing retry, regenerate, edit/resend, continue, and recovered-draft paths
  do not inherit the initial repair session or original preview.
- The existing canonical-writes recovery switch remains unchanged.
- No new configuration is introduced.

## Testing

### Pure contract tests

- contract schema, contiguous `1..N` ordinals, count, marker-length, and byte
  limits
- valid, missing, unknown, zero, leading-zero, comma-grouped, repeated,
  space-separated, adjacent, and reordered markers
- exact 32-character marker acceptance, overlong-marker invalidity, and a
  body-sized decimal marker that never enters integer parsing
- escaped, inline-code, and fenced-code literals
- projection equality for insert, replace, remove, adjacent, space-separated,
  malformed, and unknown markers, including the exact one-U+0020 deletion rule
- rejection of punctuation, case, Unicode, newline, and general-whitespace
  changes
- empty, oversized, and marker-flood outputs
- exact-limit and limit-plus-one checks for evidence bytes, ordinals, marker
  count, answer bytes, fixed overhead, and total request bytes
- exact bounded repair prompt, untrusted-data delimiters, response reservation,
  and safety-margin window inequality

### Controller tests

- valid direct answer completes without repair
- missing/invalid direct answer repairs once with the same resolution
- agent answer uses direct tool-free repair rather than re-entering the agent
- direct and agent paths both read the initial body from the store and cross the
  same async selection seam before any completion or agent-run anchoring
- empty provider output, agent failure/cancellation, and synthesized fallback
  copy never dispatch repair
- direct and agent gateway calls propagate the same request-scoped fallback
  signal; synthesized copy marks it before yield while genuine provider text
  equal to the fallback string leaves it unset
- omitting the optional signal preserves existing gateway output types and
  behavior, and the signal never carries governed text
- repair remains active and defers persistence when no builder is available
- one repair maximum even when repaired output remains invalid
- checking state is published before an event-loop yield, remains stoppable and
  send-blocking, and a stop at that yield prevents provider dispatch
- stop before dispatch and during repair select the original as a complete
  message, then set run state `STOPPED` and append
  `Citation repair canceled by user.` as the durable stop record after the
  assistant write
- stop after the synchronous selection commit is a no-op; a late repair chunk
  cannot overwrite the original
- session close and late-chunk races
- repair failure selects the original
- successful repair passes only the repaired body to terminal persistence
- retry, regenerate, edit/resend, and continue do not inherit repair state

### Store and presentation tests

- the same assistant row remains visible throughout the transition
- no terminal write occurs before selection, including builder-unavailable
  repair sessions
- atomic deferred-body replacement exposes neither an empty intermediate body
  nor a partial repaired body and performs no write
- valid, repaired, failed-repair, and canceled-repair completion each perform
  one stable-ID terminal write when persistence exists
- presentation metadata contains no governed text
- notice and action-row signatures update without remounting unrelated rows
- all checking, success, failure, and canceled notices remain structural and
  never claim semantic support, verification, grounding, or canonical
  association
- original preview toggles with mouse and keyboard
- the original-attempt action is absent from default action-service calls,
  plain rows, and exports unless the transcript passes explicit availability
- preview does not alter message content, copy, TTS, export, or provider history
- eight-entry LRU eviction and all cleanup paths

### Privacy tests

Use unique sentinels for the initial body, repaired body, evidence, source
identity, prompt, and provider exception. Assert that none appears in stdlib or
Loguru output on every failure path.

### Scoped verification

Run only the pure citation-repair, local-capture, Console controller/store,
transcript action, and native Console integration files touched by this task,
plus Ruff lint/format checks and `git diff --check`. Repository-wide baseline
repair remains tracked separately.

## Acceptance mapping

1. The lifecycle and presentation contracts keep one message provisional until
   selection.
2. The structural decision and repair request enforce valid/no-repair and
   missing-or-invalid/one-repair behavior.
3. The projection and output checks prevent claim changes and define honest
   fallback/cancellation.
4. Safe presentation metadata and the ephemeral controller lookup provide the
   same-message transition and current-session original preview.
5. The repair contract is independent of builder readiness; all inputs,
   buffers, previews, and diagnostics are bounded and privacy-safe.
6. The shared post-generation boundary and scoped tests cover direct and agent
   paths without changing unrelated Console behavior.

## Explicit exclusions

- canonical occurrence parsing or legacy numeric marker mapping
- sealing marker-bearing repaired traces
- semantic support evaluation
- grounded badges or Sources footer
- shared evidence inspector or source resolvers
- server-owned trace production or Chatbook server-trace mapping
- new retrieval, evidence renumbering, or pipeline reruns
- artifact, export, import, or Sync v2 provenance transport
- restart restoration of the transient original-attempt preview
