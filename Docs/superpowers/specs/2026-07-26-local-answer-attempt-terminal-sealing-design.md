# Local Answer-Attempt and Terminal Citation Sealing Design

**Status:** Approved design for TASK-553.14
**Date:** 2026-07-26
**Parent:** TASK-553
**Depends on:** TASK-553.13

## Purpose

Complete one eligible marker-free local RAG generation by binding the exact
final assistant body to the request-scoped citation builder, sealing a
canonical `CitationTrace`, and persisting the assistant message and provenance
in one idempotent transaction.

TASK-553.13 already captures ordered local retrieval runs and exact prompt
evidence sets. This task adds the next lifecycle boundary:

```text
EvidenceRun
  -> PromptEvidenceSet
  -> initial AnswerAttempt
  -> terminal CitationTrace seal
  -> atomic assistant-message plus provenance persistence
```

The trace remains internal. It must not yet be presented as having cited
sources because canonical occurrence parsing and visible citation trust are
separate workstreams.

## ADR check

ADR required: yes
ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`
Reason: This task directly implements ADR-024's accepted request-scoped
builder, terminal seal, governed answer body, message ownership, and atomic
persistence contracts. It does not introduce or change an architectural
decision, so no new ADR is required.

## Scope

### In scope

- record a bounded governed initial answer-attempt payload
- bind the attempt to the exact final materialized assistant body with the
  existing secret-scoped message-body fingerprint codec
- seal only answers with no eligible marker syntax under the selected prompt
  namespace; marker-bearing answers remain ordinary ungrounded messages until
  canonical occurrence parsing is implemented
- seal the request-scoped builder exactly once
- derive selected-attempt completeness using the canonical reducer
- carry repository-owned policy version and capabilities into the trace
- defer citation-bearing assistant persistence until terminal completion
- atomically persist the final message, trace, governed payloads, references,
  and owner association
- use stable message and trace identities for one same-identity retry after an
  ambiguous transaction failure
- preserve ordinary ungrounded answers when canonical writes are
  deterministically unavailable
- cover both direct-provider and agent success paths

### Out of scope

- citation occurrence parsing or legacy numeric marker mapping
- semantic support verification
- citation repair or provisional repair presentation
- retrieval/generation reruns, retry, regenerate, edit/resend, or cache reuse
- Sources footer, inspector, source opening, or resolver capabilities
- saved-artifact ownership, export, import, server traces, or Sync v2
- changing the canonical payload schema or database schema

## Approaches considered

### Selected: builder-owned sealing plus a terminal persistence callback

`CitationTraceBuilder` records the answer attempt and seals its own immutable
graph. The Console store receives a narrow callback that accepts the exact
materialized final body and returns a `SealedCitationWrite`. The store invokes
the callback after stream-buffer materialization but before the first durable
assistant write.

This preserves three existing boundaries:

- the builder owns construction and secret fingerprint use
- the store owns exact visible-body materialization and terminal state
- `ChatPersistenceService` and `CitationTraceRepository` own the atomic
  transaction

### Rejected: controller-side chunk reconstruction

Rebuilding the final body from provider chunks in the controller can diverge
from the visible store body because of prefill text, empty-answer fallback,
agent output normalization, resets, and stream-buffer materialization. A trace
must bind to the body actually persisted, not an approximation.

### Rejected: repository seals mutable builders

Passing the mutable builder into the repository mixes construction and
persistence authority, expands the repository's secret-bearing surface, and
makes it harder to prove that the exact visible body was selected before the
transaction.

### Rejected: a separate finalization coordinator

A coordinator could wrap builder, store, and repository calls, but it would
duplicate lifecycle ownership already present in those components. A typed
terminal callback is sufficient and has a smaller compatibility surface.

## Builder contract

### Repository-owned seal policy

The repository factory supplies each local builder with a frozen, versioned
local seal policy. The policy contains:

- a bounded policy version identifier
- `VIEW_SNAPSHOT`
- `VIEW_SOURCE_IDENTITY`

It does not advertise `RESOLVE_CURRENT_SOURCE`; current native locators and
resolver navigation are not implemented yet. The controller never chooses or
widens capabilities.

### Request-scoped state

The builder adds:

- one opaque trace ID allocated once and retained through persistence retry
- bounded `AnswerAttempt` metadata
- bounded governed `AnswerAttemptPayload` values
- the repository-owned seal policy
- a sealed-state guard and the resulting immutable `SealedCitationWrite`

All mutation methods reject calls after sealing.

### Answer-attempt recording

The initial attempt records:

- a new opaque attempt ID and ordinal
- `AnswerAttemptKind.INITIAL`
- the explicit prompt-evidence-set ID returned by prompt capture
- a governed answer payload reference
- the exact final assistant body
- a `MESSAGE_BODY` domain fingerprint in `body_integrity_hmac`
- no occurrence mappings yet
- structural and semantic summaries derived by the existing model contracts
- an aware completion timestamp that is not earlier than the selected prompt
  evidence set's `created_at`

The exact body and its integrity fingerprint remain governed payload fields and
must not appear in immutable trace JSON, logs, exceptions, or diagnostics.

Before constructing the attempt, the builder uses the existing bounded,
Markdown-aware marker-span helper only as an eligibility guard. If the exact
body contains any eligible marker under the selected prompt set's namespace,
recording fails atomically with a fixed reason code. TASK-553.14 does not
construct occurrence records, infer evidence mappings, or weaken the model
invariant that a retained body's eligible markers must be represented exactly.
Workstream 13 removes this temporary restriction by adding canonical occurrence
parsing.

The answer payload participates in both its per-payload byte cap and the
aggregate governed-payload cap. Validation constructs all prospective objects
before mutating builder collections, so a rejected body leaves no partial
attempt.

### One-shot seal

`seal()` requires:

- at least one evidence run
- the selected prompt set
- the selected answer attempt
- a terminal timestamp not preceding request, retrieval, prompt, or attempt
  boundaries

It constructs a local, sealed `CitationTrace`, computes
`completeness_at_seal` with `reduce_selected_attempt_completeness()`, and then
constructs a full `SealedCitationWrite`. Validation completes before the
builder changes to sealed state.

A second seal call is rejected. Persistence retry reuses the same returned
sealed write and stable IDs; it never calls `seal()` again.

## Prompt-capture handoff

`LocalRagContextResult` gains an optional prompt-evidence-set ID. Capture paths
set it only when the builder successfully records a non-empty prompt evidence
set. The Console request keeps the tuple of:

- canonical context
- request-local builder
- explicit prompt-evidence-set ID

If any element required for terminal provenance is missing, generation remains
compatible but no terminal citation callback is installed.

## Console completion contract

### Deferred first durable write

Today an empty assistant placeholder is queued for persistence, and any UI
poll that materializes the first stream chunk can persist it immediately.
That behavior would make a later message-plus-trace transaction impossible.

Provenance-eligible initial assistant placeholders therefore opt into
terminal-deferred persistence:

- stream chunks and UI polling may materialize visible content in memory
- pending/streaming states cannot flush the assistant row
- successful completion releases the deferral only through the citation-aware
  terminal path
- stopped, canceled, failed, or empty terminal paths release the deferral and
  retain existing ordinary persistence behavior without a trace
- each terminal path consumes or clears its callback exactly once
- closing or removing a session clears both callback and deferral bookkeeping

Non-citation messages keep the current first-content persistence behavior.
Callbacks live only in a bounded transient store map keyed by native message ID;
they are not serialized into message or session models and cannot transfer to a
replacement, retry, regenerated variant, or runtime-written assistant row.

### Exact-body callback

Initial send installs a request-local callback with the shape:

```text
exact materialized assistant body -> SealedCitationWrite or no write
```

On successful direct-provider or agent completion, the store:

1. materializes the final stream buffer
2. applies existing successful-response prefill behavior
3. gives the resulting exact body to the callback once
4. marks the message complete
5. persists the message with the returned sealed write, or ordinarily without
   provenance when the callback rejects marker-bearing content

Retry, regenerate, edit/resend, continuation, stopped, canceled, failed, and
empty-generation paths do not receive this callback in TASK-553.14. An agent
success outcome with no generated final text may retain its current visible
fallback copy, but that fallback is not an answer attempt and is not sealed.

## Atomic persistence and identity

When a sealed write exists, `ConsoleChatStore` passes it through the optional
`citation_write` parameter already supported by the real
`ChatPersistenceService.create_message()` implementation.

The store also supplies the native Console message UUID as the explicit
database message ID. This is required even for ordinary text assistants when a
citation write is attached:

- message retry targets the same row
- the trace retains the same stable identity
- an uncertain commit cannot create a duplicate assistant message
- `ChatPersistenceService` can verify an existing row and idempotently finish
  or return the committed aggregate

The existing persistence service preflights the sealed write, then commits the
message, trace, governed payloads, references, tombstone checks, and owner
association in its existing single SQLite transaction. No partial builder or
retrieval rows are written before terminal completion.

## Failure handling

### Builder or seal unavailable

If answer-attempt validation or sealing fails:

- log only a fixed reason code and structural counts
- do not include answer, query, title, snapshot, source identity, fingerprint,
  locator, or exception text
- complete and persist the assistant normally without a citation write
- do not retain a partial attempt or sealed trace

Eligible marker syntax is one such deterministic validation denial for this
task. It is recorded only as a fixed `occurrence_mapping_unavailable` reason;
the marker text and answer body are not logged. The answer is preserved as an
ordinary ungrounded message until workstream 13 supplies exact occurrence
mappings.

### Deterministic canonical persistence denial

If persistence raises `CitationPersistenceUnavailable`, the transaction has
not produced a usable canonical trace. Attempt to persist the same assistant
answer normally, without provenance, under its stable message ID. If that
stable-ID insert also fails, retain the completed answer in memory and do not
overwrite, update, or allocate a different durable row. The answer remains
visible and is not labeled grounded.

### Ambiguous persistence failure

For a non-policy exception whose commit outcome may be uncertain:

1. retry exactly once with the same explicit message ID and the same
   `SealedCitationWrite`
2. rely on existing message/trace idempotency verification if the first
   transaction committed
3. if the retry also fails, keep the completed answer visible in memory,
   record a fixed content-free diagnostic, and do not attempt a potentially
   conflicting ordinary insert

The failed persistence state must not cause the controller to reclassify a
successful generation as a provider or agent failure.

## Trust semantics

This task establishes complete or reduced prompt-provenance storage, not a
claim that the answer cited or was supported by the evidence.

- `occurrences` remains empty
- only marker-free answers can produce a trace in this task
- semantic status remains not checked
- no Sources footer or interactive marker is enabled
- no source-resolution capability is advertised
- later occurrence parsing and UI workstreams decide cited-vs-submitted
  presentation

The stored trace can therefore survive restart and be inspected internally
without overstating structural or semantic trust.

## Security and privacy

- no raw answer, query, title, snapshot, identity, locator, or fingerprint in
  immutable trace JSON or logs
- no logging of `str(exception)` on the citation finalization path
- no builder, callback, or sealed write stored on the app, module globals, or
  serialized/long-lived Console session models; the store holds only bounded
  request-local transient state cleared on every terminal or removal path
- policy metadata comes only from the repository-owned factory
- selected-answer body integrity uses the existing keyed fingerprint codec
- disabled canonical writes cannot create partial canonical rows

## Testing strategy

### Builder tests

- answer attempt exact-body payload and immutable-trace privacy
- marker-bearing body rejection with no partial mutation
- answer-body and aggregate byte bounds
- explicit prompt-set linkage and timestamp ordering
- atomic rejection with no partial mutation
- repository-owned policy metadata and bounded capabilities
- deterministic completeness
- one-shot seal and post-seal mutation rejection
- stable sealed write reuse for persistence retry

### Store and persistence tests

- UI materialization does not persist a terminal-deferred assistant early
- exact materialized body reaches the callback once
- stable native message ID reaches `create_message`
- message and trace commit atomically
- deterministic denial falls back to an ordinary ungrounded message
- ambiguous failure retries once with the same IDs and sealed write
- second ambiguous failure leaves the answer visible without a conflicting
  ordinary insert
- non-citation messages preserve current persistence timing

### Console integration tests

- direct-provider success seals and persists exact output
- agent success seals and persists exact output
- successful prefill generation binds the exact visible body
- empty provider and empty agent generation do not seal
- marker-bearing answers persist ordinarily without provenance
- capture/seal diagnostics contain no sentinel content
- stopped, canceled, failed, empty, retry, and regenerate paths do not seal

Verification remains scoped to touched builder, Console controller/store,
persistence, and citation tests. Repository-wide baseline repair remains
separate.

## Rollout and rollback

Canonical writes remain behind the existing runtime recovery switch. Disabling
the switch stops new terminal traces while preserving ordinary messages and
readability of already stored provenance. No down-migration or schema rollback
is required.
