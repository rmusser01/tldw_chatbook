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

### `CitationRepairContract`

The local capture boundary returns an optional immutable contract containing:

- marker namespace, initially only `chatbook_s_v1`
- sorted unique allowed positive marker ordinals
- the exact prompt evidence context already sent to the initial provider
- a schema version

The contract validates hard size and count limits. It contains no builder,
database handle, source primary key, credential, provider object, or mutable
request state.

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
Markdown-aware canonical marker scanner.

It returns one of:

- `not_applicable`: no valid repair contract
- `valid`: at least one eligible marker exists and every eligible marker
  ordinal belongs to the contract
- `repair_required_missing`: no eligible marker exists
- `repair_required_invalid`: at least one eligible marker uses an unknown
  ordinal
- `unavailable`: the answer or marker count exceeds a bound

Repeated, grouped, and reordered known markers are structurally valid. Markers
inside fenced code, inline code, or escaped literals are ignored. Structural
validity never means the cited snapshot supports the claim.

### Claim-preservation projection

Repair may alter citation marker syntax but no other answer text.

The comparison projection removes each eligible canonical marker group and,
when present, the single ASCII separator immediately before that group. This
allows conventional text such as:

```text
Original: Aurora began at 09:30 UTC.
Repaired: Aurora began at 09:30 UTC [S1].
```

The original and repaired projections must otherwise be byte-for-byte equal
as UTF-8 text. The projection does not normalize Unicode, punctuation,
newlines, general whitespace, Markdown, or case.

Unknown positive-ordinal markers are removed from the original projection so
`[S9]` may be replaced by `[S1]`. Code and escaped literals are not eligible
markers and remain part of the compared text.

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
verifies that the complete fixed instruction, evidence block, and answer block
fit the resolved model window without trimming. If the exact request cannot
fit, repair is unavailable and the original is selected.

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
therefore gains an explicit terminal-defer flag for an empty persisted
assistant placeholder. The flag:

- is valid only for an empty, attachment-free assistant placeholder
- arms terminal persistence even when no citation finalizer is installed
- does not consult canonical-write readiness
- prevents streaming materialization or UI polling from writing the initial
  body
- releases through the existing terminal completion, stop, failure, and
  cleanup paths

When a builder is ready, the same placeholder may carry both the citation
finalizer and the repair defer flag; the store keeps only one deferral entry.
When no builder is ready, terminal completion performs one ordinary stable-ID
write of the selected body. A no-op citation finalizer must not be used to
simulate repair deferral.

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

### Active-run ownership

The same active asyncio task and cancellation signal remain authoritative from
initial dispatch through terminal selection. Existing direct and agent
`finally` blocks must not clear active-run state before repair resolves.

Stop behavior is:

- stop during initial generation: preserve existing stopped-response behavior
- stop after the initial body but before repair dispatch: select the original
  with `Citation repair canceled`
- stop during repair: cancel collection, select the original, and show the
  same cancellation copy
- a late repair chunk after cancellation is discarded

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
- successful selection: `Citation markers repaired · View original attempt`
- failed or invalid repair: a concise citation-warning notice
- canceled repair: `Citation repair canceled`

The repaired notice is structural. It does not show a grounded badge or imply
support was checked.

### Original-attempt preview

The controller keeps a bounded current-session map from message ID to original
body only for successfully repaired messages. Each body already satisfies the
answer body limit. The map retains at most eight entries; inserting a ninth
evicts the least recently used entry and removes that message's
`original_available` presentation flag.

The transcript action service exposes `View original attempt` only while the
controller reports that the body is available. Activation retrieves the body
through the controller and stores it in screen-local ephemeral preview state.
The transcript renders a clearly labeled:

```text
Original attempt (not selected)
```

block under the selected message. Activating the action again hides the block.
The preview never changes `ConsoleChatMessage.content`, variants, persistence,
copy action output, TTS input, export data, or provider history.

Session close, controller shutdown, message deletion, cache eviction, or
starting a replacement lifecycle for that message clears the preview. Restart
does not restore it in this task.

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
| Exact repair request does not fit | Original + warning |
| Provider unavailable or raises | Original + warning |
| Empty repair output | Original + warning |
| Output exceeds byte limit | Original + warning |
| Output changes non-marker text | Original + warning |
| Output still has missing/invalid markers | Original + warning |
| User cancels repair | Original + canceled notice |
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

- contract schema, ordinal uniqueness, count, and byte limits
- valid, missing, unknown, repeated, grouped, and reordered markers
- escaped, inline-code, and fenced-code literals
- projection equality for insert, replace, remove, and grouped markers
- rejection of punctuation, case, Unicode, newline, and general-whitespace
  changes
- empty, oversized, and marker-flood outputs
- exact bounded repair prompt and untrusted-data delimiters

### Controller tests

- valid direct answer completes without repair
- missing/invalid direct answer repairs once with the same resolution
- agent answer uses direct tool-free repair rather than re-entering the agent
- repair remains active and defers persistence when no builder is available
- one repair maximum even when repaired output remains invalid
- checking state remains stoppable and send-blocking
- stop before dispatch, during repair, and after the last chunk
- session close and late-chunk races
- repair failure selects the original
- successful repair passes only the repaired body to terminal persistence
- retry, regenerate, edit/resend, and continue do not inherit repair state

### Store and presentation tests

- the same assistant row remains visible throughout the transition
- no terminal write occurs before selection, including builder-unavailable
  repair sessions
- presentation metadata contains no governed text
- notice and action-row signatures update without remounting unrelated rows
- original preview toggles with mouse and keyboard
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
