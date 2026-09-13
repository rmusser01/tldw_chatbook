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
other branches remain selectable there. Replacement does not globally
deactivate a memory row: it appends a branch-anchored selection event whose
activation message is the captured current leaf. The newest valid selection
event decides the current branch's memory. A failed, cancelled, stale, unsafe,
or non-improving attempt appends no event and leaves the prior selection
effective.

Current-branch reset appends a branch-anchored reset tombstone rather than
deactivating the selected memory globally. The tombstone selects no memory and
prevents an older selection from silently resurfacing on that branch. Undo
optimistically deactivates that exact tombstone revision. Branches that do not
contain the selection/reset activation message retain their own prior state.

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
  attempt persistence, and atomic branch-selection events.
- `console_prepared_request.py` and the provider gateway own semantic memory,
  provider role mapping, exact token accounting, and safety windowing.
- `console_chat_controller.py` owns admission from the active Console session
  and applies one validated memory projection before dispatch.
- `console_rewind_modal.py`, `chat_screen.py`, and `console_transcript.py` own
  choice, activity feedback, and derived presentation respectively.
- Internal Prompts remains the only editor of
  `console.rewind_summarize`. No second prompt is introduced.

### Additive scope and branch-selection persistence

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

Before adding the composite memory/scope foreign key, the migration adds an
explicit unique parent key on
`console_conversation_memories(id, conversation_id)`. Migration verification
has the migration runner fetch `PRAGMA foreign_key_check` after the schema and
backfill are installed and raise/roll back before the version bump if any row
is returned.

The memory/conversation pair references the same pair on
`console_conversation_memories`. The selection anchor references a message in
the same conversation. Deleting the owning memory cascades to its scope row.
The selection-anchor relationship is restrictive rather than cascading: a
scope row must never disappear while leaving its base memory to be
reinterpreted. Ordinary messages are soft-deleted; individual hard deletion of
a referenced message or memory is rejected rather than deleting events and
exposing older state. Whole-conversation deletion cascades every derived row.
Check constraints require automatic scope to be prefix with no selection
anchor, and manual scope to have a selection anchor.

Branch choice is represented separately by an append-mostly local-only
`console_conversation_memory_selections` table:

```text
sequence                     INTEGER primary key autoincrement
selection_id                 TEXT unique stable identifier
conversation_id              TEXT conversation foreign key
activation_message_id        TEXT same-conversation message foreign key
selected_memory_id           nullable same-conversation memory foreign key
event_kind                   "select" | "reset"
suppresses_legacy            boolean
created_at                   timestamp
revision                     positive integer
active                       boolean
```

`select` requires a selected memory; `reset` requires none. Every reset and
manual selection suppresses legacy. Automatic selections inherit the current
applicable event's suppression bit, defaulting false when there is no event;
backfilled automatic selections use false. The activation message is the
durable active leaf captured when the event is created. The database-owned
autoincrement sequence is the only authoritative recency order; `created_at`
is display metadata. Message and memory relationships are restrictive so a
selection event cannot vanish and expose older state accidentally.
Conversation deletion still cascades the entire local derived graph. An
individual hard delete of a referenced message/memory is rejected; ordinary
message removal is soft deletion, and whole-conversation deletion is the only
supported hard-delete path. The existing memory-row `active` flag remains a
coarse record-availability/reset-all compatibility flag; it no longer means
"selected on every branch."

The effective selector walks active selection events by descending sequence.
An event is branch-valid only when its activation message is on the active
durable lineage in the same conversation. The first valid event is the branch
head. If a valid legacy pair exists and the head is absent or does not suppress
legacy, legacy is effective. Otherwise the head is terminal: `reset` returns
no memory; `select` returns its referenced memory only if that memory's scope,
anchors, base `active` flag, boundary digest, and coverage rules validate. A
corrupt or invalid selected memory fails open to raw history rather than
falling through to an older memory, because replacement must not silently
reveal superseded context. Events on sibling lineages are skipped, so their
selections remain independent. An invalid/off-lineage legacy pair never
blocks a valid generated branch head.

The migration deterministically backfills every existing
`source_kind="generated"` record as `coverage_kind="prefix"`,
`origin_kind="automatic"`. Each active generated record with a usable captured
leaf receives a deterministic non-suppressing `select` event anchored at that
leaf and inserted in original memory-row insertion order; records without a
valid activation anchor remain inert. Existing inactive legacy records remain
outside the selector. Every new generated-memory insert writes its scope and
selection event in the same transaction. A generated record with a missing,
corrupt, or contradictory scope is ineligible and fails open to raw history;
readers never guess its meaning.

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

The old conversation-field pair remains baseline compatibility state and is
never written with new summary content:

```text
context_summary
summary_boundary_message_id
```

A valid pair is the compatibility baseline. It remains effective when no
branch-valid selection head explicitly suppresses it, preventing old
non-suppressing automatic selections from stacking with it. A dangling,
cross-conversation, or off-lineage legacy boundary is ignored; selection then
falls through to the branch-valid generated event stream. No provenance is
invented to convert an unverifiable legacy row into a generated record.

A successful manual selection leaves the conversation-global legacy pair
unchanged and inserts a branch event with `suppresses_legacy=1`. That override
applies only on descendants containing the event's activation message, so a
sibling where legacy remains valid keeps it. New manual summaries never write
the legacy fields. Only separately confirmed Reset all may clear the legacy
pair conversation-wide.

While a valid legacy pair is effective and the applicable branch head does not
suppress it, Ask/Automatic compaction and Compact now are ineligible and make
zero auxiliary calls. They must not commit an immediately hidden generated
selection or retrigger on later sends. Requests continue using the legacy
projection plus ordinary deterministic safety windowing. If that effective
request is known to overflow, the configured failure behavior still governs;
the recovery copy directs the user to make an explicit manual rewind summary
or reset current memory first.

Context & memory exposes a valid legacy pair as **Legacy manual prefix
memory**, shows its summary body, and labels unavailable provenance honestly.
Current-branch reset leaves the legacy pair unchanged and appends a suppressing
reset tombstone, so neither legacy nor an older generated selection surfaces
on that branch. Undo needs no copy of summary content: it deactivates the exact
tombstone only while that tombstone remains the current applicable selection
head at the expected revision, revealing the prior branch baseline. Generated-
memory Undo uses the same head/revision fence. Reset all clears the legacy
pair, deactivates all generated selection events/memory records, and increments
their revisions so outstanding current-reset Undo tokens expire; as today,
reset all has no Undo.

### Atomic branch-current selection

The repository exposes one guarded transaction that receives:

- the new memory and scope;
- a new branch selection event anchored at the captured leaf;
- the exact expected effective state: legacy boundary plus summary digest, or
  generated selection sequence/ID/revision plus memory ID/revision, or an
  explicit no-effective-memory sentinel;
- the expected active branch-selection head for the captured lineage,
  including a verified no-selection state;
- the persisted active cursor pair; and
- the captured ordered durable lineage with message IDs, parents, versions,
  deletion state, and selected-variant/attachment digests needed by the base
  memory fence.

Inside one SQLite write transaction, the repository reconstructs the captured
lineage from the persisted cursor, compares every durable row/version and the
applicable selection head, and compares the exact legacy pair/digest when it
participated in admission, then inserts the memory, scope, and a suppressing
manual selection event. It neither clears legacy nor deactivates the previously
selected memory. A mismatch returns stale without partial mutation. A
simultaneous job admitted with no memory cannot also win: after the first event
commits, the second no-selection comparison fails. An unrelated sibling event
does not invalidate the transaction because it is not applicable to the
captured lineage.

The provider call happens before this transaction. The old memory remains
effective throughout summarization. Immediately before entering the
non-awaiting transaction, the controller also rechecks its runtime payload,
identity, policy, provider/model, prompt, lineage, and selection admission;
the database checks close the remaining persistence race.

## Manual range planning

### Normative complete-unit predicate

One pure helper owns complete durable conversation-unit grouping for both
manual directions, modal eligibility, controller admission, automatic
planning, and tests. A complete unit:

- begins with one persisted, non-deleted user row on the captured active path;
- contains only provider-visible durable rows with positive persisted
  versions;
- keeps every assistant tool call with all of its durable tool results and
  terminal assistant outcome; and
- ends in a persisted assistant row whose status is `complete`.

An unanswered user row, generating/stopped/failed assistant placeholder,
orphan tool result, partial tool-call group, system/ephemeral presentation row,
or unavailable persisted version makes the candidate span ineligible. The
planner never silently skips or rounds around such a row. Leading system
contract and seeded greeting material are mandatory context outside the unit
set and are never summarized by a manual rewind action.

### Admission

Selecting Summarize from here captures:

- durable conversation ID and runtime session ID;
- selected prompt native and persisted IDs;
- current complete end native and persisted IDs;
- active lineage and leaf;
- payload, identity, policy, prompt, effective-memory, and applicable
  selection-event revisions;
- the exact legacy boundary/summary digest or explicit no-legacy sentinel;
- the persisted active cursor and durable message/parent/version facts;
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
selection and legacy state only to reject a stale replacement.

### Manual prefix parity

Summarize up to here moves onto the same planner and transaction instead of
retaining the legacy rolling/truncating implementation. Its authoritative raw
span is every complete durable unit strictly before the selected user prompt.
The leading system contract and seeded greeting remain verbatim. The selected
prompt is the scope's render anchor; the base memory boundary is the final row
of the last covered unit.

The action makes zero provider calls when there is no complete prior unit, the
predecessor is incomplete, an anchor/version is unavailable, or the exact raw
span plus output allowance cannot fit one auxiliary request. It never folds a
prior summary, silently drops the oldest rows to a fixed budget, retries, or
partially summarizes. Its output validation, idle progress projection,
admission fences, suppressing branch selection event, legacy-baseline
preservation, and one-call rule are identical to Summarize from here.
Successful output creates
`coverage_kind="prefix"`, `origin_kind="manual_rewind"`.

### Canonical idle progress projection

Manual rewind actions run while idle, when no next user request exists, but
`PreparedConsoleRequest` requires a non-empty active-request segment. Progress
therefore uses a canonical comparison artifact that is never dispatched. Both
the before and after artifacts contain the exact current system/identity
contract plus the same fixed app-owned empty-request sentinel. Both omit
future-only tools, sources, skills, world-info injections, continuations, and
other material that does not yet exist.

The before artifact uses authoritative raw history with no prior generated or
legacy memory. The after artifact applies only the candidate new prefix/range
memory and its exact retained raw rows. Both go through the active provider's
normal `prepare_chat_request(..., apply_safety_window=False)` mapping. A result
makes progress only when the exact after total is smaller than the exact before
total, the covered raw material saves more tokens than the memory wrapper and
body cost, and the after artifact fits the safe provider input capacity.
`before_tokens` and `after_tokens` record these two candidate projections.

The currently effective request is deliberately not the manual comparison
baseline: replacing an older prefix summary can restore early raw framing and
therefore be larger than that older compressed request while still correctly
shrinking the newly selected raw span.

### Completion validation

The generated body must be non-empty, within the requested cap, free of
provider/tool envelopes and immutable wrapper tags, and smaller than the
canonical raw candidate it replaces after wrapper cost. A result that does not
reduce the canonical idle projection is reported as non-improving and is not
committed.

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

Those are the two supported wire contracts. App memory is never downgraded to
an ordinary user-authored row. If a future adapter cannot preserve either
contract, request preparation must refuse that memory projection and fail open
to raw history rather than inventing a third unsafe mapping.

Private identity annotations and thinking/continuation/attachment/tool
sidecars owned only by removed rows are filtered before provider dispatch.
No provider sees Chatbook's private anchor keys.

### Later automatic compaction

Range memory does not turn off Ask or Automatic policy. If a later automatic
compaction is admitted, a distinct range-to-prefix planner constructs an
ordered effective chronology:

1. every complete retained raw unit before the range start, all mandatory;
2. one sealed prior-memory unit containing the range memory plus its ID,
   revision, start/end anchors, and content-free provenance; and
3. the largest eligible consecutive prefix of complete later units allowed by
   the carry-forward policy, target, and one-call auxiliary capacity.

The auxiliary envelope preserves that chronological order even though normal
provider dispatch serializes app memory in the provider-safe preamble
position. The mandatory early units and sealed range memory are indivisible;
if they cannot fit one auxiliary call, automatic compaction does not run and
the configured failure behavior applies. It may not summarize the range
memory and later tail while dropping retained early framing.

The resulting generated memory is a normal automatic prefix. Its boundary is
the range end when no later unit is included, otherwise the last row of the
last included later unit. Its prefix digest covers authoritative raw lineage
through that boundary. `selected_units_json` records the early raw units, one
prior-memory provenance marker, and any later units without duplicating the
summary body. Automatic progress compares the exact current effective request
with the exact candidate prefix request and still enforces the existing target
and safe capacity. Success inserts the new prefix memory/scope and a selection
event anchored at the current leaf, carrying forward the applicable head's
legacy-suppression bit; it leaves prior records available to sibling branches.

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
hides it and makes that selection event inert; the effective selector may then
show the older state belonging to that historical/sibling lineage. Returning
to the selection's valid lineage restores it unless a later valid reset event
supersedes it. Context & memory identifies the record as manual range memory
and remains the review/reset surface for its summary body. A reset tombstone
shows no banner and does not create a transcript row.

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
| Valid legacy manual pair exists | Use it as the baseline until a suppressing manual-selection/reset event applies on this branch. |
| Automatic/Ask/Compact now while unsuppressed legacy is effective | Make zero auxiliary calls; keep legacy and use normal safety/failure behavior. |
| Legacy pair is invalid on this lineage | Ignore it and continue generated-event selection. |
| Exact manual prefix does not fit | Ask for an earlier selected prompt; zero calls. |

Diagnostics and ledger rows may contain sizes, IDs, revisions, provider/model
identity, usage, pricing provenance, reason codes, and elapsed time. They must
not contain transcript text or generated summary text.

## Verification

### Persistence and repository

- Migration creates the additive scope and selection-event tables at the
  implementation-time next schema version, with the required composite unique
  parent key, same-conversation foreign keys, checks, indexes, idempotence,
  restrictive anchors, deterministic backfill, and a clean foreign-key audit.
  The migration runner must fetch every `PRAGMA foreign_key_check` row and
  raise before the schema-version commit when any row exists; merely executing
  the PRAGMA is not a passing check.
- Existing generated records are backfilled as automatic prefix scope and
  usable active records receive deterministic branch selection events;
  generated records missing valid scope/selection afterward are ineligible.
- Prefix/range scope and select/reset events round-trip without sync-log
  writes.
- Guarded selection preserves every old record on mismatch and success,
  inserts exactly one branch event on success, preserves sibling-branch
  selection, and leaves the exact admitted legacy baseline unchanged.
- Simultaneous no-memory jobs, concurrent legacy replacement, cursor movement,
  and durable message-version changes make the losing transaction stale.
- Valid legacy memory suppresses automatic stacking; dangling or off-lineage
  legacy state falls through to generated selection.
- Effective unsuppressed legacy makes Automatic, Ask, and Compact now zero-call
  ineligible; no hidden generated selection is committed or repeatedly
  retriggered, and overflow follows the existing failure policy.
- Manual select/reset overrides legacy only through a suppressing branch event;
  Undo reveals the prior baseline, while reset all alone clears legacy
  conversation-wide.
- Selection recency uses the database autoincrement sequence, not timestamps;
  individual referenced-row hard deletes are rejected and whole-conversation
  cascade leaves no foreign-key violations.

### Planner and transaction

- Inclusive range and exclusive-prefix selection use raw durable snapshots and
  ignore prior memory content.
- One normative complete-unit helper governs eligibility, tool-group closure,
  persisted versions, and terminal assistant state across UI/controller/tests.
- Start validation, complete-unit end validation, tool-group integrity,
  attachment/visual representation, exact capacity, output cap, wrapper
  rejection, canonical idle progress, cancellation, and every stale fence are
  covered.
- Valid manual actions make exactly one auxiliary call. Invalid, incomplete,
  oversized, or stale-before-admission requests make zero calls. No retry or
  hierarchical call occurs.
- Success/failure/cancelled/stale ledger rows remain content-free.

### Request and leak safety

- Direct-chat and agent requests retain rows before and after the range and
  remove only the inclusive covered rows.
- Distinct-role and single-preamble providers preserve original system
  content, memory ownership, wrapper integrity, exact order, and accounting;
  no user-role fallback exists.
- Regenerate/retry/continue/edit-resend before or inside the end boundary
  receives raw history without range memory.
- Descendants after the end receive range memory plus retained early and later
  rows.
- Restoring before the end and switching branches makes memory inert; returning
  to the valid lineage restores it.
- Thinking, continuation, attachment, and tool sidecars owned by removed rows
  do not leak; private message IDs are stripped before provider dispatch.
- Later automatic replacement treats retained early units plus the sealed
  range-memory unit as mandatory, records the exact new boundary/provenance,
  and creates a prefix selection event without mutating sibling state.

### UI and lifecycle

- Modal action, cost/replacement copy, disabled guidance, authoritative
  revalidation, focus order, Escape dismissal, and narrow geometry are tested
  in mounted Textual tests.
- Range and prefix banners use distinct copy, show no generated body, create no
  transcript row, and preserve tree/message counts.
- Close/resume/restart restores the scope, effective payload, banner anchor,
  selection event, and user-turn bounds.
- Generated and legacy current-branch reset, reset-all, and Undo follow the
  exact selection/tombstone lifecycle; branch navigation shows the newest
  event applicable to that lineage.

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
