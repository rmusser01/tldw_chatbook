# Console `/rewind` Summarize-from-here Design

Status: Approved design

Date: 2026-08-28

Target branch: `dev`

Task: [TASK-575](../../../backlog/tasks/task-575%20-%20Console-rewind-add-a-Summarize-from-here-complement-to-Summarize-up-to-here.md)

Normative decision: [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md), including its 2026-08-28 amendment

## Product outcome

The Console `/rewind` menu gains **Summarize from here**. A user selects one
of their prompts, Chatbook summarizes that prompt through the complete current
conversation tip, and later provider requests retain the earlier framing while
replacing the selected recent tangent with branch-valid generated memory. The
stored transcript, active tree, variants, and selected branch remain
unchanged.

This complements **Summarize up to here**, which summarizes the older prefix
and retains the selected prompt and recent tail. The two directions do not
stack: a successful manual summary replaces the effective memory on the
current branch.

## Approved semantics

### Inclusive selected range

The range begins with the selected user prompt and ends with the current
provider-visible durable leaf, inclusive. For:

```text
U1 -> A1 -> U2 -> A2 -> U3 -> A3
```

selecting `U2` summarizes `U2, A2, U3, A3`. The effective context for the
next user request is:

```text
app-owned range memory + U1 + A1 + next user request
```

After later descendants exist, they remain verbatim after the summarized
range:

```text
app-owned range memory + U1 + A1 + U4 + A4 + next user request
```

The memory is serialized in the provider-safe app-context position rather
than as a fake chronological message. Its immutable wrapper explains that the
summary describes a removed middle range occurring after retained earlier
turns and before retained later turns.

### Complete durable end

The captured end must close a provider-visible durable conversation unit. An
unanswered user prompt, partial tool sequence, failed placeholder, ephemeral
row, or otherwise incomplete tip is not silently rounded backward or omitted.
The action is unavailable with a plain recovery explanation until the current
exchange is complete or repaired.

This rule prevents the retained request from containing an assistant reply or
tool result whose owning user/tool-call row was removed by the range.

### Replacement, not layering

A successful Summarize-from-here replaces the current branch's effective
manual up-to, manual range, or automatic prefix memory. Records valid only on
other branches are not deactivated. A failed, cancelled, stale, unsafe, or
non-improving attempt leaves the prior memory active.

Manual summary content is always generated from authoritative raw transcript
rows in the selected span. The previous memory participates only in the
admission fence; it is not recursively included in the manual summarizer
input.

### Transcript neutrality

No message is created, removed, reparented, hidden, or rewritten. The summary
body and its marker are never transcript/tree nodes. Full source history stays
visible and exportable.

## Architecture

### Existing owners reused

- `console_context_compaction.py` owns durable-unit planning, the bounded
  auxiliary completion, provenance, validation, and stale-result transaction.
- `console_context_repository.py` owns local policy, branch memory, auxiliary
  attempt persistence, and atomic memory replacement.
- `console_prepared_request.py` and the provider gateway own semantic memory,
  provider role mapping, exact token accounting, and safety windowing.
- `console_chat_controller.py` owns admission from the active Console session
  and applies one validated memory projection before dispatch.
- `console_rewind_modal.py`, `chat_screen.py`, and `console_transcript.py` own
  choice, activity feedback, and derived presentation respectively.
- Internal Prompts remains the only editor of
  `console.rewind_summarize`. No second prompt is introduced.

### Additive scope persistence

The implementation adds a local-only one-to-one
`console_conversation_memory_scopes` table rather than rebuilding
`console_conversation_memories`:

```text
memory_id                    TEXT primary/composite foreign key
conversation_id              TEXT composite foreign key
coverage_kind                "prefix" | "range"
origin_kind                  "automatic" | "manual_rewind"
selection_anchor_message_id  nullable composite message foreign key
```

The memory/conversation pair references the same pair on
`console_conversation_memories`. The selection anchor references a message in
the same conversation. Deleting the owning memory cascades to its scope row.
The selection-anchor relationship is restrictive rather than cascading: a
scope row must never disappear while leaving its base memory to be
reinterpreted. Ordinary messages are soft-deleted; any exceptional hard
deletion removes/deactivates the owning memory first. Check constraints require
automatic scope to be prefix with no selection anchor, and manual scope to
have a selection anchor.

The migration deterministically backfills every existing
`source_kind="generated"` record as `coverage_kind="prefix"`,
`origin_kind="automatic"`. Existing inactive legacy records remain outside
the selector. Every new generated-memory insert writes its scope in the same
transaction. A generated record with a missing, corrupt, or contradictory
scope is ineligible and fails open to raw history; readers never guess its
meaning.

The base record retains its existing boundary meaning: the last summarized
durable message.

- Automatic prefix: unchanged.
- Manual up-to: boundary is the durable predecessor of the selected prompt;
  selection anchor is the selected prompt.
- Manual from-here: boundary is the captured complete leaf; selection anchor
  is the inclusive range start.

The existing prefix digest remains conservative for range memory: it covers
the raw active lineage through the end boundary. Therefore a mutation in the
retained early framing may invalidate a range summary even when the selected
rows themselves did not change. This is an accepted fail-open trade-off
because the summary may rely on that earlier framing.

### Legacy manual summary compatibility

The old conversation-field pair remains read-only compatibility state:

```text
context_summary
summary_boundary_message_id
```

A valid pair is treated as the effective manual up-to memory and suppresses
automatic-memory selection, preventing the current two mechanisms from
stacking. A dangling, cross-conversation, or off-lineage legacy boundary is
ignored and full history is used. No provenance is invented to convert an
unverifiable legacy row into a generated record.

The next successful manual swap clears the legacy pair in the same database
transaction that inserts the new memory/scope record and deactivates the
expected current-branch memory. New manual summaries never write the legacy
fields.

### Atomic branch-current replacement

The repository exposes one guarded transaction that receives:

- the new memory and scope;
- the expected current memory ID/revision, or no-memory expectation;
- the durable conversation and captured branch facts; and
- whether a valid legacy pair is expected.

The transaction rechecks the expected record, deactivates only that effective
branch-current record, inserts the replacement, clears the expected legacy
pair, and commits. A mismatch returns stale without partial mutation. Memory
records that are not valid candidates on the captured branch are untouched.

The provider call happens before this transaction. The old memory remains
effective throughout summarization.

## Manual range planning

### Admission

Selecting Summarize from here captures:

- durable conversation ID and runtime session ID;
- selected prompt native and persisted IDs;
- current complete end native and persisted IDs;
- active lineage and leaf;
- payload, identity, policy, prompt, and active-memory revisions;
- resolved provider/model and prompt digest; and
- summary output cap and exact provider capacity.

The action is refused before a provider call if the selected message is not a
user prompt on the active path, either persisted anchor is unavailable, the
end does not close a complete unit, the start follows the end, or the run is
not idle.

### Exact one-call bound

The planner serializes the exact inclusive raw range with the existing stable
role/tool delimiters and selected attachment/visual representation policy.
Transient tool schemas, skills, sources, world-info injections, failed rows,
and other request-only material are excluded, matching ADR-052.

The exact prompt, immutable input wrappers, selected range, and requested
output allowance must fit one auxiliary request for the active provider/model.
If they do not fit, Chatbook makes zero provider calls and asks the user to
choose a later start. TASK-575 does not add hierarchical, partial, or retry
summarization.

The manual planner ignores prior memory content. It uses the previous memory
ID/revision only to reject a stale replacement.

### Completion validation

The generated body must be non-empty, within the requested cap, free of
provider/tool envelopes and immutable wrapper tags, and smaller than the
provider context it replaces after wrapper cost. A result that does not reduce
the exact prepared request is reported as non-improving and is not committed.

After completion, all admission facts are revalidated. Session/branch edits,
new sends, selected variants, provider/model changes, policy or prompt
changes, memory reset/replacement, deletion, or a changed tip make the result
stale. The auxiliary ledger records the content-free terminal outcome.

## Request projection and leak rule

One pure projection step consumes the annotated provider rows and a validated
effective memory. For range memory it:

1. resolves the selection/start and boundary/end persisted IDs to the active
   runtime lineage;
2. finds both identities in the annotated outgoing payload;
3. verifies `start_index <= end_index` and that the end boundary is present;
4. retains the leading system contract and durable rows before the start;
5. removes the inclusive start-to-end rows;
6. retains durable rows after the end, including the active request; and
7. attaches one immutable app-owned range-memory segment.

If either identity is absent, reversed, dangling, cross-conversation, or
removed by an unexpected transform, the projection returns the original raw
history and no memory. It never guesses by content, turn number, or nearest
row.

The end identity is the activation boundary. A regenerate/retry/continue/edit
payload for a point before or inside the summarized range cannot contain that
end row, so it receives full raw history and no future summary. Descendant
requests after the end contain both anchors and may use the summary.

Preview/token validation and real dispatch consume the same immutable
projection. The memory remains separately owned and non-compactable in
`PreparedConsoleRequest`; distinct-role and single-preamble provider
serializers preserve the existing mapping and token ownership.

Private identity annotations and thinking/continuation/attachment/tool
sidecars owned only by removed rows are filtered before provider dispatch.
No provider sees Chatbook's private anchor keys.

### Later automatic compaction

Range memory does not turn off Ask or Automatic policy. If a later automatic
compaction is admitted, its effective input consists of retained early raw
units, the validated range memory, and eligible later complete units. A
successful automatic transaction may replace the range with a normal prefix
memory only after summarizing that complete effective context. It may not
drop the retained early framing while summarizing only the range memory and
later tail.

## User experience

### Rewind menu

The second-level action view contains:

- Restore to here
- Summarize up to here
- Summarize from here
- Never mind

Summary actions display `Uses the active model once`. When effective memory
exists, they also display `Replaces current conversation memory`.

The `/rewind` modal remains openable before general Send readiness. It may
disable summarization from synchronously known facts such as an active run or
an incomplete tip, but it does not duplicate asynchronous provider readiness.
The controller authoritatively repeats all checks after dismissal.

The new worker activity copy is `Summarizing selected range...`. Successful
completion reports `Conversation memory updated.` Failure copy is bounded and
content-free. No additional confirmation modal is added because the explicit
second-level action plus cost/replacement copy is the confirmation.

### Derived banner and memory review

Manual range memory renders above its selection anchor:

```text
Context uses a summary of turns #3-#5 - full transcript remains visible.
```

The bounds are user-turn ordinals derived from the valid active lineage, not
database IDs. Generated summary content never enters the banner. Manual
prefix memory retains the existing earlier-turns wording.

The banner renders only for effective validated memory whose required anchors
are on the active path. Restoring before the range end or switching branches
hides it and makes the memory inert. Returning to its valid lineage restores
it. Context & memory identifies the record as manual range memory and remains
the review/reset surface for its summary body.

The modal and banner retain existing keyboard navigation, focus restoration,
Escape behavior, narrow-terminal fit, and forbidden-binding rules.

## Error and recovery behavior

| Condition | Behavior |
| --- | --- |
| Run active or tip incomplete | Disable/refuse with recovery guidance; zero auxiliary calls. |
| Selected prompt or persisted anchor missing | Refuse; transcript and memory unchanged. |
| Exact range does not fit the auxiliary request | Ask for a later start; zero calls. |
| Provider/model not ready | Existing bounded provider-readiness copy; zero summary commit. |
| Empty, unsafe, over-cap, or non-improving output | Record failed attempt; retain previous memory. |
| Admission facts changed during the call | Record stale attempt; retain previous memory. |
| Cancellation | Record cancelled attempt; retain previous memory. |
| Range invalid at later dispatch | Fail open to full raw history; no memory injection. |
| Valid legacy manual pair exists | Use it instead of stacking automatic memory until successful replacement. |

Diagnostics and ledger rows may contain sizes, IDs, revisions, provider/model
identity, usage, pricing provenance, reason codes, and elapsed time. They must
not contain transcript text or generated summary text.

## Verification

### Persistence and repository

- Migration creates the additive scope table at the implementation-time next
  schema version, with composite same-conversation foreign keys, checks,
  indexes, idempotence, owning-memory cascade, and restrictive selection-anchor
  behavior.
- Existing generated records are backfilled as automatic prefix scope;
  generated records missing valid scope afterward are ineligible.
- Prefix/range and automatic/manual scope round-trip without sync-log writes.
- Guarded swap preserves the old record on mismatch, deactivates only the
  expected branch-current record on success, preserves other-branch records,
  and clears only the expected legacy pair.
- Valid legacy memory suppresses automatic stacking; dangling or off-lineage
  legacy state fails open.

### Planner and transaction

- Inclusive range selection uses raw durable snapshots and ignores prior
  memory content.
- Start validation, complete-unit end validation, tool-group integrity,
  attachment/visual representation, exact capacity, output cap, wrapper
  rejection, progress, cancellation, and every stale fence are covered.
- Valid manual actions make exactly one auxiliary call. Invalid, incomplete,
  oversized, or stale-before-admission requests make zero calls. No retry or
  hierarchical call occurs.
- Success/failure/cancelled/stale ledger rows remain content-free.

### Request and leak safety

- Direct-chat and agent requests retain rows before and after the range and
  remove only the inclusive covered rows.
- Distinct-role, single-preamble, original-system, and no-system providers
  preserve memory ownership, wrapper integrity, exact order, and accounting.
- Regenerate/retry/continue/edit-resend before or inside the end boundary
  receives raw history without range memory.
- Descendants after the end receive range memory plus retained early and later
  rows.
- Restoring before the end and switching branches makes memory inert; returning
  to the valid lineage restores it.
- Thinking, continuation, attachment, and tool sidecars owned by removed rows
  do not leak; private message IDs are stripped before provider dispatch.
- Later automatic replacement includes retained early context before creating
  a prefix memory.

### UI and lifecycle

- Modal action, cost/replacement copy, disabled guidance, authoritative
  revalidation, focus order, Escape dismissal, and narrow geometry are tested
  in mounted Textual tests.
- Range and prefix banners use distinct copy, show no generated body, create no
  transcript row, and preserve tree/message counts.
- Close/resume/restart restores the scope, effective payload, banner anchor,
  and user-turn bounds.
- Reset and branch navigation hide/reactivate the banner according to the
  effective-memory selector.

### Live verification

Use an isolated scratch `TLDW_CONFIG_PATH` whose `[paths].data_dir` is also a
scratch directory. Verify the exact database path before launch. Drive the
mounted flow through modal selection, one auxiliary completion, next-send
payload inspection, close/resume, and restored banner/payload. Never launch a
schema-bumping branch against the shared development database.

Focused DB, service, controller, request, mounted-UI, lint, and static checks
are required. A repository-wide test sweep remains opt-in under project rules.

## Baseline evidence

On current `origin/dev` before TASK-575 edits, the focused rewind/memory run
used Python 3.12 and collected 105 tests. It passed 104 and exposed one existing
failure:

```text
Tests/Chat/test_console_context_compaction.py::
test_context_repository_init_failure_is_observable_without_error_content
```

That test's minimal `SimpleNamespace` store lacks the `sessions()` method now
required later in `ConsoleChatController.__init__`; the expected repository
warning is emitted before the unrelated fixture failure. TASK-575 must keep
this baseline distinct from regressions introduced by its own changes and
recheck it after rebasing.

## Out of scope

- Layering multiple manual summary ranges.
- Hierarchical or partial range summarization.
- A dedicated summary provider/model selector.
- A fake transcript summary node or active-branch rewind.
- New prompt ownership or editable memory safety wrappers.
- A new global keybinding or other `/rewind` discoverability; that remains
  TASK-576.
