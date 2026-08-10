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
records may exist. The active request selects the newest record whose boundary
is in the active lineage and whose prefix digest still matches. Descendants
after the boundary remain valid; any mutation within the summarized prefix
invalidates the record. Current-branch reset deactivates the selected active
record and supports Undo. A separately confirmed reset-all operation
deactivates every branch record. Neither changes transcript content.

At generation admission the service captures the conversation identity,
active lineage and leaf, policy revision, model/provider identity, prompt
digest, active-memory revision, and request revision. One compaction job may
run per conversation.
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
