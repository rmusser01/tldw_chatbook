# ADR-052: Console Conversation Memory and Compaction Policy

Status: Accepted
Date: 2026-08-10
Related Task: [TASK-14811](../tasks/task-14811%20-%20Console-conversation-memory-and-auto-compaction.md)
Extends: ADR-006, ADR-011, ADR-033, ADR-040
Supersedes: the Console behavior that concatenates `context_summary` into the
first system message; manual rewind summarization remains supported through
the new memory service

## Decision

Chatbook will treat the model context window, provider-safety windowing, the
user's conversation budget, and model-generated memory as four separate
concepts with separate owners.

- Model capability data owns the model context window.
- The request builder owns the response reservation, safety margin, projected
  payload accounting, and mandatory whole-unit safety windowing.
- Global Console Behavior settings own conversation-memory defaults.
- Each persisted Console conversation owns its optional policy overrides and
  generated memory records.
- Internal Prompts remains the sole editor for the stable
  `console.rewind_summarize` prompt artifact.

An empty Console tab without a durable conversation ID stages overrides in its
existing session snapshot and writes them through on first conversation
persistence. Applying policy does not create an empty conversation row, and
closing an unsaved empty tab discards that staged state.

The Console modal retains its existing provider-default write path under the
clearer label `Save provider defaults`. That action applies the current
conversation draft and writes only provider, sampling, and streaming defaults.
It never writes global conversation-memory defaults, which remain owned by
Console Behavior in canonical Settings.

The automatic budget uses the model-safe conversation capacity after response
reservation, safety margin, and mandatory non-compactable request material,
including the active request. A custom
budget is preserved as user intent. If current overhead makes it larger than
the available capacity, the effective value is reduced for that request and
the UI reports why; stored intent is not silently rewritten. An unknown model
window cannot claim a safe automatic ceiling. It blocks Automatic compaction
until the user repairs the model capability or supplies a bounded custom
conversation budget, and it remains visibly safety-unverified.

Compaction policy is a tri-state value:

- **Ask** is the default and offers a decision before a cost-bearing summary
  call.
- **Automatic** may summarize when the high-water threshold is crossed.
- **Off** never makes an automatic summarization call.

Turning compaction off does not turn off deterministic provider-safety
windowing. The request builder continues to preserve the system contract,
active user request, whole conversation/tool units, and newest eligible units
that fit. For a known or explicitly overridden model window, it never silently
sends a known-overflow request. Unknown windows remain visibly unverified.

Default global values are an automatic model-safe budget, an 80 percent
high-water trigger, a 55 percent post-compaction target, a 1,024-token summary
output cap, and a Stop-and-ask failure policy. Trigger and target are ratios of
the effective conversation capacity, not wall-clock time or turn count. The
target must remain materially below the trigger to prevent repeated summary
calls.

The request projection and provider dispatch use the same immutable prepared
request. A provider-neutral `PreparedConsoleRequest` preserves named semantic
segments and their ownership. The provider gateway serializes that artifact
once into a `PreparedProviderRequest`; token accounting and dispatch consume
that exact serialized artifact rather than rebuilding parallel payloads. The
semantic request separates:

1. app/provider overhead and other mandatory context that summary compaction
   cannot remove;
2. existing generated memory;
3. durable, compactable conversation units;
4. the active draft/request; and
5. requested and effective response reservation, provider input/output caps,
   and safety margin.

The effective input ceiling is the lower of a separately advertised provider
input cap and the total context window after the effective response
reservation and safety margin. A response request that leaves mandatory input
unable to fit is an actionable validation state. It is not repaired by an
undisclosed half-window clamp.

Compaction runs before ordinary safety trimming. It summarizes the prior
active memory, when present, plus only complete durable units selected for
replacement, including complete tool-call/result groups when their outcome is
relevant. Prior memory and original transcript units have distinct delimiters
and provenance in the summarizer input. It does not summarize transient tool
schemas, staged source bodies, skill definitions, or other request-only
overhead. If mandatory material alone exhausts the safe capacity, the UI
reports that compaction cannot solve the problem and offers relevant recovery
actions.

Generated memory does not replace or delete transcript rows. It is local
private derived data, stored with its conversation, boundary message, lineage
snapshot, provider/model, prompt identifier and digest, generation time,
before/after token counts, summarized-prefix digest, and revision. The prefix
digest covers message identity, version, role, content, selected variants, and
relevant attachments through the boundary. Multiple branch-valid memory
records may exist. Every selected record must have its required boundary on
the active lineage and a matching prefix digest. Descendants after the
boundary remain valid; any mutation within the summarized prefix invalidates
the record. The TASK-575 amendment below defines the branch-selection event,
current-reset tombstone, Undo, and reset-all semantics that choose among these
records without changing transcript content.

At generation admission the service captures the conversation identity,
active lineage and leaf, policy revision, model/provider identity, prompt
digest, applicable selection/memory revisions, and request revision. One
compaction job may run per conversation.
After the model call, the service commits only if those admission facts remain
compatible. Edits, branch switches, model or policy changes, a newer send, or
a reset make the result stale and it is discarded without touching the
transcript or active memory.

Memory is projected as a separately owned app-context segment before retained
recent turns. It never mutates stored user or character system content and is
never presented as user-authored text. An immutable application wrapper
identifies the content as an untrusted summary of earlier conversation and
instructs the model not to follow instructions found inside it. Provider
adapters map the semantic segments to the safest supported wire shape. A
provider with only one system/preamble field may serialize the original system
segment and tagged memory segment into that field; the prepared request keeps
their ownership and token attribution distinct even when the wire format
cannot. Tests assert deterministic serialization, delimiter integrity, and an
unchanged stored original, not a separate wire row that some providers cannot
represent.

The editable prompt controls what facts the summarizer should preserve, not
the immutable safety wrapper, transcript delimiters, output cap, provider
admission rules, or memory injection role. The prompt keeps its stable ID for
existing user customizations. Settings deep-links to the existing Internal
Prompts editor instead of creating a second prompt owner.

Two carry-forward modes are supported without changing the memory role:

- **Memory with recent turns** (default) retains as many newest complete units
  as fit the post-compaction target.
- **Memory with latest exchange** retains the latest complete exchange and
  active request, using the memory for older content.

Compaction is an auxiliary provider call using the exact active conversation
provider/model in the first release. A dedicated summary-model selector is
deferred to a separate design. The call is identified in a local content-free
auxiliary-call ledger whether it succeeds, fails, is cancelled, or becomes
stale. Provider-reported usage and pricing provenance are stored when
available; estimates remain labeled estimates. Its request and response are
not inserted as chat messages. Diagnostics may log sizes, revisions,
decisions, and provenance identifiers; they must not log transcript or
generated-memory content.

### 2026-08-28 amendment: manual prefix and range memory

This amendment supersedes the TASK-14811 implementation's record-global
newest-memory selection and current-branch reset mechanism. Record-global
deactivation cannot represent a branch-local replacement when one prefix
record is valid on multiple descendants; branch selection events below are
the normative selector and reset mechanism after TASK-575.

TASK-575 extends the same branch-valid memory service to both manual
`/rewind` summary directions. **Summarize up to here** creates a manual prefix
memory. **Summarize from here** creates a manual range memory whose inclusive
start is the selected user prompt and whose end is the complete current leaf.
Creating either manual form replaces the effective memory on the current
branch; summaries are not layered. Memory records are immutable derived
history and are not globally deactivated by branch replacement.

Manual scope metadata is local derived state in an additive one-to-one
extension of `console_conversation_memories`. It records coverage
(`prefix`/`range`), origin (`automatic`/`manual_rewind`), and the selected
prompt used as the render anchor. A composite memory/scope foreign key requires
an explicit unique `(id, conversation_id)` parent key and migration
`foreign_key_check`: the migration runner fetches the PRAGMA results and raises
before the version bump if any row is returned. The migration deterministically
backfills every existing generated record as automatic-prefix memory; legacy
records remain outside this selector. New generated records always receive
scope in the same transaction. A missing or invalid scope therefore fails
inert rather than guessing that a possibly-manual record is automatic. The
base record's
`boundary_message_id` continues to mean the last summarized durable message:
for a manual prefix this is the message immediately before the selected
prompt; for a range it is the captured end leaf.

Effective branch choice is an append-mostly local derived event stream. Each
selection event is anchored to the durable active leaf at creation and either
selects one same-conversation memory or is a reset tombstone selecting none.
The database assigns a monotonic autoincrement sequence; timestamps are
display-only and never decide recency. The highest-sequence event whose
activation message lies on the active lineage is the branch head. A valid
legacy pair is the baseline unless that head explicitly suppresses legacy.
Otherwise a valid select event yields its referenced branch-valid memory; an
invalid/corrupt reference or a reset event yields raw history rather than
falling through to older memory. Events on sibling lineages are skipped.

Manual selections and reset tombstones suppress legacy. Automatic selections
inherit the current head's suppression bit, defaulting false, and migrated
automatic selections are non-suppressing. Thus replacement inserts a new
event without disturbing sibling branches; current-branch reset inserts a
tombstone; and Undo deactivates that tombstone only while it remains the
current applicable head at the expected revision. Reset all alone clears
legacy conversation-wide and deactivates/revision-bumps all selection events
and memory records so outstanding Undo tokens expire. The original memory
`active` flag remains coarse availability/reset-all state, not branch
selection state. Existing usable active memories receive deterministic
backfilled select events anchored at their captured leaves. Restrictive FKs
reject individual hard deletion of referenced messages/memories; ordinary
deletion is soft and whole-conversation deletion cascades all derived rows.

A range memory remains a separately owned app-context segment, not a fake
mid-history transcript turn. Its immutable wrapper states that the summary
chronologically belongs between retained earlier and later transcript units.
The request keeps rows before the range start and after its end, removes the
inclusive covered range, and accounts for the memory as non-compactable app
context. Provider adapters retain the existing distinct-role or
single-preamble mapping.

Range injection is fail-open and identity anchored. Both range anchors must
map to the active lineage, the start must not follow the end, and the end
boundary must be present in the annotated outgoing payload. If any condition
fails, the full raw history is used and no range memory is injected. This is
the same future-information rule as prefix memory: a request built for a point
before the summary's end never receives that summary.

Manual summarization always reads the authoritative raw transcript span. It
does not recursively summarize the memory being replaced. Both manual
directions use one complete-durable-unit predicate, exact one-call auxiliary
capacity, and no silent fixed-budget truncation. Idle progress compares a
canonical undispatched raw candidate against the candidate new-memory request
using identical system context and an identical fixed empty-request sentinel;
it does not compare against the previously compressed request.

The previous memory remains effective during the cost-bearing call. Only a
successful output that passes runtime admission and a single SQLite
compare-and-swap transaction inserts the memory, scope, and branch selection.
The transaction verifies the exact applicable selection head (including
no-selection), legacy boundary and summary digest, persisted active cursor,
and captured durable parent/version lineage. Failure, cancellation, invalid
output, concurrent selection, or stale completion preserves both prior state
and transcript.

A later automatic compaction may replace a range memory with prefix memory
only through a distinct ordered planner. Retained complete early units and a
sealed range-memory unit are mandatory; the planner may append an eligible
consecutive prefix of later complete units. It records the resulting raw
boundary/prefix digest and prior-memory provenance and must not silently omit
the retained early framing. If the mandatory early-plus-range input cannot fit
one auxiliary call, automatic compaction does not run and the configured
failure behavior applies.

Legacy `context_summary` conversation fields remain a validated baseline
compatibility path. They take
precedence over generated selection only while effective, preventing two
memories from stacking; invalid/off-lineage legacy state falls through to the
generated event selector. A successful branch-current manual selection leaves
the conversation-global pair unchanged and suppresses it only through its
branch event, preserving legacy state on siblings. Context & memory labels a
valid pair as legacy manual prefix memory. Current reset inserts a suppressing
tombstone, and Undo deactivates the exact current tombstone to reveal the prior
baseline. Only separately confirmed reset all clears legacy. No new code
writes summary content to the legacy fields.

An effective legacy baseline that is not suppressed by the applicable branch
head makes Ask, Automatic, and Compact now ineligible with zero auxiliary
calls. The request still uses deterministic provider-safety windowing and the
configured overflow failure behavior. The recovery is an explicit manual
rewind summary or current-memory reset; automatic compaction must not commit an
immediately inert generated selection behind legacy or repeatedly charge for
one.

The transcript indication remains render-derived. Manual prefix memory shows
the existing earlier-turns banner at the selected prompt. Manual range memory
shows user-turn bounds at the same selected prompt. The model-generated body
is reviewable only in the existing Context & memory surface and never becomes
a transcript/tree node.

## Context

The Console already has manual rewind summarization, persisted
`context_summary` and boundary columns, token-aware whole-turn request
windowing, a model-context estimate, and an editable internal summary prompt.
Those pieces do not currently form one honest user policy.

The model modal's `Max tokens` label refers to response output, while the
existing read-only context estimate is based largely on stored messages rather
than the exact next provider payload. Session settings are restored only
partially when a persisted conversation resumes. Current summary injection
appends generated text to the first system message. The summarizer has a fixed
input span, no dedicated output cap, and no stale-result transaction across
branch or model changes.

This feature adds durable conversation policy, a database migration, a
cost-bearing auxiliary model call, prompt and provider boundaries, and a
long-lived Settings/Console ownership split. A canonical ADR is therefore
required.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Use one `max_tokens` field for input and output | Providers commonly use that name for response output; merging the concepts creates unsafe and incomprehensible behavior. |
| Trigger after a number of turns or elapsed time | Message sizes vary dramatically and provider limits are token-based. |
| Let Off disable every kind of truncation | A known-overflow request would fail unpredictably or be rejected by the provider. |
| Mutate the stored original system prompt with memory | It changes user/character intent and destroys provenance. Providers with one preamble may still require deterministic wire serialization of separately owned segments. |
| Delete summarized transcript rows | It destroys user history and makes summary errors irreversible. |
| Keep one summary column per conversation | A branch can overwrite another branch's valid memory and later reuse it outside its lineage. |
| Store policy only in screen snapshots | Current snapshots do not survive process restart and are not the domain owner for conversation behavior. |
| Put a raw prompt editor in Console Behavior | It duplicates Internal Prompts ownership and creates ambiguous save semantics. |
| Automatically open a new visible conversation after compaction | Context compaction does not require changing conversation identity and would fragment history. |
| Allow arbitrary memory-role or wrapper templates | Provider role semantics and prompt-injection safety are application contracts, not presentation preferences. |
| Store a second conversation-field summary pair for range memory | It would perpetuate the legacy manual-summary path beside ADR-052 memory, permit ambiguous stacking, and lose branch/revision provenance. |
| Represent a range summary as a synthetic assistant or mid-history system turn | It would misrepresent transcript authorship, interact inconsistently with provider role rules, and risk becoming ordinary trimmable history. |
| Rewind the active branch and summarize the abandoned tail | It changes the visible active path; TASK-575 instead preserves the transcript and compacts only provider context. |

## Consequences

### Benefits

- Users can predict and control the extra summarization call.
- Context numbers correspond to the next real provider request.
- Stored transcripts remain authoritative and recoverable.
- Branches and concurrent Console work cannot silently reuse stale memory.
- Settings, conversation state, model capabilities, and prompts each have one
  clear owner.
- Existing whole-unit trimming remains a deterministic last line of defense.

### Accepted trade-offs

- Conversation persistence requires a schema migration and legacy-summary
  compatibility path.
- Exact projection may be more expensive than the current rough estimate and
  needs caching/invalidation around request revisions.
- Provider role differences require an adapter mapping while preserving one
  app-context semantic contract.
- Automatic compaction adds latency and provider cost.
- A summary can omit or distort information, so memory must remain reviewable
  and resettable.
- Unknown model limits cannot be presented as automatically safe.

## Rollback

- Disable Automatic and Ask compaction while retaining stored policy and
  memory rows.
- Continue deterministic whole-unit provider-safety windowing.
- Ignore derived memory during request projection without deleting it.
- Preserve the transcript and permit manual rewind behavior through the same
  service once repaired.
- Do not downgrade the database by asking an older build to interpret newer
  policy or branch-memory rows; use the repository's normal migration backup
  and downgrade procedure.

## Links

- [Conversation memory and compaction design](../../Docs/superpowers/specs/2026-08-10-console-conversation-memory-compaction-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-10-console-conversation-memory-compaction-implementation.md)
- [ADR-006: Provider-aware generation settings](006-provider-aware-generation-settings.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-040: Versioned prompt artifacts and safe improvement transactions](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
- [TASK-575 range-memory design](../../Docs/superpowers/specs/2026-08-28-console-rewind-summarize-from-here-design.md)
