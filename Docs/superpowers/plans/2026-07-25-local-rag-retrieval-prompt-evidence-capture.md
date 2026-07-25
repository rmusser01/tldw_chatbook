# Local RAG Retrieval and Prompt-Evidence Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture each local RAG retrieval execution and the exact marked evidence text submitted to the provider in a request-scoped canonical citation builder.

**Architecture:** Add a pure mutable `CitationTraceBuilder` beside the frozen citation contracts, but stop before answer-attempt capture or sealing. The existing citation repository remains the trusted composition boundary for local identity and keyed fingerprints. Local RAG keeps its legacy string behavior when canonical writes are unavailable; when enabled, the final ranked results are normalized, scope-checked again, formatted once with stable `[S#]` markers, recorded as governed snapshots, and passed through the existing post-chat-dictionary RAG payload seam so those snapshot bytes reach the provider unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2 citation contracts, Textual chat event handlers, SQLite-backed citation repository composition, pytest, Ruff.

---

## Scope and file structure

**Create**

- `tldw_chatbook/Chat/citation_trace_builder.py` — request-scoped local capture state and mutations only; no persistence or sealing.
- `tldw_chatbook/RAG_Search/local_citation_capture.py` — normalize final local `SearchResult` dictionaries, re-check scope, and format exact canonical prompt blocks.
- `Tests/Chat/test_citation_trace_builder.py` — pure builder RED/GREEN coverage.
- `Tests/RAG/test_local_citation_capture.py` — prompt formatting, scope, truncation, and chat-boundary integration coverage.

**Modify**

- `tldw_chatbook/Chat/citation_trace_repository.py` — safe factory/fingerprint seam using the repository’s existing identity and secret.
- `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py` — opt-in capture-aware result API while preserving `get_rag_context_for_chat()`.
- `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py` — use the capture-aware API and carry the builder into the request worker.
- `tldw_chatbook/Event_Handlers/worker_events.py` — keep the builder request-scoped and remove it before provider dispatch; answer capture is explicitly deferred.
- `tldw_chatbook/Chat/Chat_Functions.py` — sanitize provider-bound logging; canonical evidence uses its existing post-dictionary `media_content` seam.
- `tldw_chatbook/RAG_Search/pipeline_functions_simple.py` — correct semantic result source identity normalization needed by canonical capture.
- `Tests/Chat/test_citation_trace_repository.py` — repository factory readiness and fail-closed tests.
- `Tests/Chat/test_chat_function_local_citation_boundary.py` — real provider-payload and sensitive-log regression coverage.
- Existing RAG/chat event tests only where the production call seam changes.
- `Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md` — link the concrete child task/plan without changing ADR semantics.
- `backlog/tasks/task-553.13 - Capture-local-RAG-retrieval-runs-and-exact-prompt-evidence-sets.md` — final AC checks and concise implementation notes.

## Non-goals

- No `AnswerAttempt`, citation occurrence parsing, visible repair, trace seal, or message persistence.
- No current-source resolution, native/external source opening, export, import, artifacts, or Sync v2.
- No server `grounding_trace/v1` production or adaptation.
- No dual-write to the legacy JSON sidecar.
- No global or app-owned mutable builder registry.

## ADR check

ADR required: yes  
ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`  
Reason: This plan implements ADR-024’s local retrieval-run and exact prompt-boundary evidence capture contracts. It does not introduce a new architectural choice.

---

### Task 1: Build the request-scoped local citation accumulator

**Files:**

- Create: `Tests/Chat/test_citation_trace_builder.py`
- Create: `tldw_chatbook/Chat/citation_trace_builder.py`

- [ ] **Step 1: Write failing constructor and privacy tests**

Add tests that construct the builder with a fixed `LocalCitationIdentityContext`,
`CitationFingerprintCodec`, `request_id`, and `generation_id`. Assert:

```python
builder = CitationTraceBuilder.local(
    request_id="request-1",
    generation_id="generation-1",
    identity_context=identity,
    fingerprint_codec=codec,
    created_at=NOW,
)

assert builder.request_id == "request-1"
assert builder.generation_id == "generation-1"
assert builder.evidence_runs == ()
assert builder.prompt_evidence_sets == ()
assert builder.is_sealed is False
assert "secret query" not in repr(builder)
```

Also assert frozen identity inputs are validated and the builder cannot be
constructed without a local authority/profile binding.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_citation_trace_builder.py -q
```

Expected: collection/import failure because `citation_trace_builder.py` does
not exist.

- [ ] **Step 3: Implement the minimal builder constructor**

Create a mutable class whose public collection properties return tuples. Keep
secret material private and redact it from `repr`:

```python
class CitationTraceBuilder:
    @classmethod
    def local(
        cls,
        *,
        request_id: str,
        generation_id: str,
        identity_context: LocalCitationIdentityContext,
        fingerprint_codec: CitationFingerprintCodec,
        created_at: datetime | None = None,
    ) -> "CitationTraceBuilder":
        ...

    @property
    def evidence_runs(self) -> tuple[EvidenceRun, ...]:
        return tuple(self._evidence_runs)
```

The builder must not expose a generic `seal()` in this task.

- [ ] **Step 4: Write failing retrieval-run tests**

Define narrow typed inputs such as `LocalRetrievalCandidateCapture` and
`LocalRetrievalRunMetadata`; do not import `SearchResult` into the Chat layer
and do not accept a free-form metadata dictionary. Test ordered candidates,
typed score semantics, bounded allowlisted metadata, HMAC query identity, and
absence of candidate content:

```python
run_id = builder.record_retrieval_run(
    stage="hybrid",
    raw_query="secret query",
    candidates=(candidate_one, candidate_two),
    retrieval_metadata=LocalRetrievalRunMetadata(
        search_mode="hybrid",
        requested_top_k=5,
        max_context_characters=10_000,
        rerank_enabled=True,
        source_kinds=(CanonicalSourceKind.MEDIA_DB,),
        scope_state="unscoped",
    ),
    started_at=NOW,
    ended_at=NOW,
)

assert builder.evidence_runs[0].run_id == run_id
assert builder.evidence_run_payloads[0].raw_query is None
assert builder.evidence_run_payloads[0].query_fingerprint.startswith(
    "hmac-sha256-v1:"
)
assert [item.rank for item in builder.evidence_run_payloads[0].candidates] == [1, 2]
assert "content" not in builder.evidence_run_payloads[0].model_dump()
```

Reject duplicate/unknown run references, non-finite scores, excess candidates,
unknown metadata fields, path/URL-shaped metadata values, and payloads beyond
existing model caps.

- [ ] **Step 5: Run and verify RED, then implement retrieval capture**

Run the focused test, confirm failures are caused by the missing method, then
implement only enough to create `EvidenceRun` and `EvidenceRunPayload` using:

- `CitationFingerprintDomain.RAW_QUERY`
- `new_opaque_id("evidence-run")`
- `new_opaque_id("run-payload")`
- typed `RetrievalScoreKind` and `RetrievalScoreScale`

Serialize only the typed metadata model into
`EvidenceRunPayload.retrieval_metadata`. Never log query, title, source
identity, locator, lineage, or fingerprints.

- [ ] **Step 6: Write failing prompt-set tests**

Test exact bytes, stable ordinals, transformations, and run linkage:

```python
prompt_set_id = builder.record_prompt_evidence_set(
    run_id=run_id,
    evidence=(
        LocalPromptEvidenceCapture(
            candidate_rank=1,
            snapshot_text="[S1] MEDIA — Alpha\nExact transformed text",
            transformations=("heading_injected",),
        ),
    ),
    created_at=NOW,
)

entry = builder.prompt_evidence_sets[0].entries[0]
snapshot = builder.evidence_snapshot_payloads[0]
assert entry.marker_ordinal == 1
assert entry.evidence_ordinal == 1
assert entry.run_id == run_id
assert snapshot.snapshot_text == "[S1] MEDIA — Alpha\nExact transformed text"
assert snapshot.storage_mode is EvidenceStorageMode.EMBEDDED
```

Assert no prompt set is created for zero evidence, ordinals are unique,
snapshots use keyed exact/comparison fingerprints, and no partial state is
appended when validation fails.

- [ ] **Step 7: Run and verify RED, implement prompt capture, then run GREEN**

Build all Pydantic objects before mutating builder lists so a failure is atomic.
Use `MarkerNamespace.CHATBOOK_S_V1`. The builder remains unsealed and
non-persistable.

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_citation_trace_builder.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add Tests/Chat/test_citation_trace_builder.py \
  tldw_chatbook/Chat/citation_trace_builder.py
git commit -m "feat(rag): add request-scoped citation trace builder"
```

---

### Task 2: Normalize and format exact local prompt evidence

**Files:**

- Create: `Tests/RAG/test_local_citation_capture.py`
- Modify: `Tests/RAG/test_scope_pipeline_enforcement.py`
- Create: `tldw_chatbook/RAG_Search/local_citation_capture.py`
- Modify: `tldw_chatbook/RAG_Search/pipeline_functions_simple.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`

- [ ] **Step 1: Write failing source-normalization tests**

Cover the three allowed local families and common aliases:

```python
assert normalize_local_result(media_result).source_kind is CanonicalSourceKind.MEDIA_DB
assert normalize_local_result(note_result).source_kind is CanonicalSourceKind.NOTES
assert normalize_local_result(conversation_result).source_kind is CanonicalSourceKind.CHAT_HISTORY
```

Normalize semantic media results from their governed metadata when their
top-level source is currently `"unknown"`. Reject empty IDs, unknown source
kinds, non-string title/content, control-bearing titles, unsafe metadata, and
non-finite scores with a non-content reason code.

Use an explicit final-score mapping driven by producer metadata, never by the
bare float alone:

- `_final_score_kind=reranker` written by a successful `rerank_results()` call
  -> `RERANKER` / `UNBOUNDED`
- RRF fusion metadata, only when no later final-score marker exists -> `RRF` /
  `NON_NEGATIVE`
- semantic similarity metadata, only when no later final-score marker exists
  -> `VECTOR_SIMILARITY` / `UNBOUNDED`
- all custom, FTS placeholder, or otherwise opaque scores -> `LEGACY` /
  `UNBOUNDED`

Update `rerank_results()` test-first so successful FlashRank output writes the
explicit final-score marker when it overwrites `SearchResult.score`. Import,
initialization, and execution fallback must not write that marker; the retained
prior score is then classified from reliable prior metadata or as `LEGACY`.
Never infer BM25 or normalized similarity semantics from a bare float.

- [ ] **Step 2: Run RED, then implement a strict allowlisted normalizer**

The normalizer may retain only:

- canonical source kind and opaque item ID
- bounded title
- exact content for the live formatter
- typed score kind/scale/value
- allowlisted non-executable lineage keys

Do not copy arbitrary metadata, URLs, paths, or citation dictionaries into
source identity or locator fields. Resolver-family tasks will add typed native
locators later.

- [ ] **Step 3: Write failing canonical formatter tests**

Test that one formatting function produces both the provider context and the
exact per-entry blocks:

```python
formatted = format_local_evidence_context(
    normalized_results,
    max_length=90,
)

assert formatted.context == "\n---\n".join(
    entry.snapshot_text for entry in formatted.entries
)
assert formatted.entries[0].snapshot_text.startswith("[S1] ")
assert len(formatted.context) <= 90
assert formatted.omitted_candidate_ranks == (3,)
```

Include:

- exact-fit and one-character-short budgets
- UTF-8/Unicode input while the configured budget remains the existing
  character budget
- ellipsis included inside the captured snapshot
- stable contiguous marker ordinals after invalid/unauthorized candidates are
  excluded
- no snapshots for omitted candidates

- [ ] **Step 4: Run RED, implement the formatter, then run GREEN**

Build each block as:

```text
[S1] MEDIA — Title
exact submitted content
```

Use a single calculation for returned context and snapshot text. Never parse
the aggregate context afterward to reconstruct entries.

- [ ] **Step 5: Write failing prompt-boundary authority tests**

Test a fresh, uncached post-retrieval check:

- a media/note/conversation result is retrieved, then its backing row is
  deleted or soft-deleted before prompt assembly; it is excluded
- a conversation or workspace scope is narrowed after retrieval without
  reusing the old `ScopeCache` entry; newly unauthorized evidence is excluded
- conversations remain excluded under an active local RAG scope
- an `empty` or failed authority/existence read rejects every entry

- [ ] **Step 6: Implement the fresh authority/existence seam and run GREEN**

Add an explicit uncached mode to the existing scope-resolution seam and a
batched current-existence query for all three local source families. Prompt
assembly must re-read the conversation/workspace scope and current backing rows
after retrieval, off the event loop where appropriate. Do not consult the
pre-retrieval `EffectiveScope` or `ScopeCache` for this decision. Query/read
errors fail closed. The check consumes canonical item IDs, not display titles
or raw metadata, and runs before marker ordinals are assigned.

- [ ] **Step 7: Correct semantic source propagation test-first**

Add focused regressions proving:

- `search_semantic()` uses the result metadata’s allowlisted source when no
  top-level source attribute exists
- successful reranking marks the overwritten score as
  `RERANKER`/`UNBOUNDED`
- reranker import/execution fallback retains honest prior RRF/semantic metadata
  or degrades an opaque score to `LEGACY`

Preserve the existing return shape and citation metadata behavior.

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/RAG/test_semantic_honest_states.py \
  Tests/RAG/test_fusion.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  tldw_chatbook/RAG_Search/local_citation_capture.py \
  tldw_chatbook/RAG_Search/pipeline_functions_simple.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
git commit -m "feat(rag): format exact canonical prompt evidence"
```

---

### Task 3: Compose builders through the existing citation repository

**Files:**

- Modify: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`

- [ ] **Step 1: Write failing readiness/factory tests**

Test:

```python
assert disabled_repository.create_local_trace_builder(
    request_id="request-1",
    generation_id="generation-1",
) is None

builder = enabled_repository.create_local_trace_builder(
    request_id="request-1",
    generation_id="generation-1",
)
assert builder is not None
```

Also cover missing identity, missing fingerprint key, and persisted identity
mismatch. None of these states may break ordinary RAG generation.

- [ ] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_citation_trace_repository.py \
  -k "local_trace_builder or canonical_capture" -q
```

Expected: failures because the factory does not exist.

- [ ] **Step 3: Implement the minimal safe factory**

The repository owns access to `_fingerprint_codec`; do not add a public secret
or codec property. Re-read the persisted identity before issuing a builder, as
the persistence path already does:

```python
def create_local_trace_builder(
    self,
    *,
    request_id: str,
    generation_id: str,
) -> CitationTraceBuilder | None:
    if not self.canonical_writes_enabled:
        return None
    ...
```

Return `None` for unavailable prerequisites and use only non-content diagnostic
codes if logging is necessary.

- [ ] **Step 4: Run GREEN and broader repository tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_citation_trace_identity.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Tests/Chat/test_citation_trace_repository.py \
  tldw_chatbook/Chat/citation_trace_repository.py
git commit -m "feat(rag): compose local citation capture securely"
```

---

### Task 4: Wire capture through every local Chat RAG mode

**Files:**

- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `Tests/RAG/test_rag_ui_integration.py`
- Modify: `Tests/RAG/test_scope_pipeline_enforcement.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`

- [ ] **Step 1: Write failing capture-aware API tests**

Add a frozen result wrapper:

```python
@dataclass(frozen=True)
class LocalRagContextResult:
    context: str | None
    citation_builder: CitationTraceBuilder | None
```

Test a new `get_rag_context_capture_for_chat()` while retaining the old API:

```python
captured = await get_rag_context_capture_for_chat(app, "query")
assert captured.context.startswith("[S1] ")
assert len(captured.citation_builder.evidence_runs) == 1
assert len(captured.citation_builder.prompt_evidence_sets) == 1

legacy = await get_rag_context_for_chat(app, "query")
assert isinstance(legacy, str)
```

- [ ] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py -q
```

- [ ] **Step 3: Implement opt-in capture without changing pipeline tuples**

At request start:

1. Allocate opaque request/generation IDs.
2. Ask `app.citation_trace_repository` for a builder.
3. Capture the active conversation/workspace/session identity and retrieval
   start time.
4. Execute the existing selected pipeline unchanged.
5. When no builder exists, return the pipeline’s existing context byte for
   byte.
6. When a builder exists, normalize the final ordered results, re-read the
   original request’s conversation/workspace scope without `ScopeCache`
   (never whichever session happens to be active after retrieval), batch-check
   current backing-row existence for media, notes, and conversations, exclude
   anything no longer authorized/available, record the full retrieval run,
   format the exact marked prompt evidence, record the prompt set, and return
   that exact context.
7. Exclude individual malformed or unauthorized results before assigning
   markers. If the canonical capture operation itself cannot complete
   atomically, return no RAG context and no builder so ordinary chat remains
   available without submitting uncaptured evidence or retaining partial
   provenance.

Sanitize the modified RAG path’s existing query/error logs as part of the same
RED/GREEN change. Log only mode, counts, timing, and fixed non-content reason
codes; never log the user query, formatted context, title, identity, locator,
lineage, snapshot, fingerprint, or `str(ValidationError)`.

- [ ] **Step 4: Test all pipeline selection branches**

Parameterize `plain`, `semantic`, `hybrid`, and a custom pipeline. The tests
may monkeypatch their `perform_*` functions to return the same ranked results;
assert each branch produces one equivalent run/prompt capture.

Cover:

- empty results/context
- pipeline exception
- malformed result exclusion
- scope exclusion
- backing-row deletion between retrieval and prompt assembly
- scope narrowing between retrieval and prompt assembly despite a populated
  pre-retrieval cache
- canonical writes disabled
- repository absent
- key/identity unavailable

Add log-capture assertions using unique sentinel query/title/content strings.
Neither successful capture, malformed-result rejection, nor validation failure
may place the sentinels into log messages.

- [ ] **Step 5: Run focused GREEN and compatibility suites**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/RAG/test_semantic_honest_states.py \
  Tests/RAG/test_fusion.py -q
```

Expected: all tests pass and the old string-return assertions remain valid.

- [ ] **Step 6: Commit**

```bash
git add Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
git commit -m "feat(rag): capture local retrieval and prompt evidence"
```

---

### Task 5: Keep the builder alive across the current request worker

**Files:**

- Modify: `Tests/Event_Handlers/Chat_Events/test_chat_events.py`
- Create: `Tests/Event_Handlers/test_worker_local_citation_capture.py`
- Create: `Tests/Chat/test_chat_function_local_citation_boundary.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Modify: `tldw_chatbook/Event_Handlers/worker_events.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`

- [ ] **Step 1: Write a failing chat-send seam test**

Assert the send handler uses `get_rag_context_capture_for_chat()` and forwards
the same builder object as an internal-only worker keyword. Canonical context
must use the existing `media_content`/`selected_parts` seam instead of being
concatenated into `message`, because `Chat_Functions.chat()` applies chat
dictionary transformations to `message` before provider dispatch:

```python
assert captured_chat_wrapper_kwargs["citation_trace_builder"] is builder
assert captured_chat_wrapper_kwargs["message"] == original_user_message
assert captured_chat_wrapper_kwargs["media_content"] == {
    "evidence": captured.context,
}
assert captured_chat_wrapper_kwargs["selected_parts"] == ["evidence"]
```

- [ ] **Step 2: Write failing worker and real provider-boundary tests**

Assert `chat_wrapper_function()` removes `citation_trace_builder` before
calling `core_chat_function`, so it cannot reach provider adapters or generic
payload parameter routing. It must leave canonical `media_content` and
`selected_parts` intact. The wrapper keeps the object in a local variable for
the lifetime of generation, then drops it because answer-attempt capture is the
next task.

In `Tests/Chat/test_chat_function_local_citation_boundary.py`, call the real
`Chat_Functions.chat()` with:

- canonical evidence containing unique title/body sentinels
- an evidence block with intentional leading/trailing whitespace
- a chat-dictionary entry that would mutate those sentinels if it saw them
- `media_content={"evidence": canonical_context}`
- `selected_parts=["evidence"]`
- a patched `chat_api_call` that captures `messages_payload`

Assert every `EvidenceSnapshotPayload.snapshot_text` remains an exact substring
of the provider-bound user text after `process_user_input()` runs, while the
ordinary user prompt still receives its configured dictionary transformation.
The whitespace-bearing snapshot must also match byte for byte despite
`Chat_Functions.chat()` assembling the overall message with `.strip()`. This is
the terminal prompt-boundary proof for this task.

- [ ] **Step 3: Run RED, implement the minimal threading seam, then run GREEN**

When `citation_builder is None`, keep the current legacy behavior byte for byte:
prepend the old RAG context to `message` and pass empty `media_content` /
`selected_parts`. When a builder exists, keep `message` free of canonical
evidence and pass the canonical context through the dedicated RAG seam above.
Do not store the builder on `app`, a module global, a sidecar, or SQLite. Do not
log or serialize it.

Sanitize existing logs on this modified path:

- `chat_rag_events.py`: log mode/count/timing only, never raw query or
  validation exception text
- `chat_events.py`: log message length/history count, never message previews
- `Chat_Functions.py`: log content types and lengths only, never input text,
  prompt previews, provider-payload text previews, custom-prompt text, answer
  previews, or post-generation replacement previews

Tests must attach capture handlers with unique query, prompt, title, snapshot,
and answer sentinels and assert none appear in emitted records. Validation
failures log a fixed reason code, not `str(ValidationError)`. Exercise both the
real local pipeline functions and streaming/non-streaming worker paths so
downstream query and answer logging is covered.

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Event_Handlers/Chat_Events/test_chat_events.py \
  Tests/Event_Handlers/test_worker_local_citation_capture.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/Chat/test_citation_trace_builder.py -q
```

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py \
  tldw_chatbook/Event_Handlers/worker_events.py \
  tldw_chatbook/Chat/Chat_Functions.py \
  Tests/Event_Handlers/Chat_Events/test_chat_events.py \
  Tests/Event_Handlers/test_worker_local_citation_capture.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py
git commit -m "feat(rag): carry citation builder through generation request"
```

---

### Task 6: Documentation, self-review, and verification

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md`
- Modify: `backlog/tasks/task-553.13 - Capture-local-RAG-retrieval-runs-and-exact-prompt-evidence-sets.md`

- [ ] **Step 1: Update implementation links and boundary notes**

Record that workstreams 10 is implemented while workstreams 11–30 remain
follow-ons. Do not mark the epic complete.

- [ ] **Step 2: Run formatting/static checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/citation_trace_builder.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/RAG_Search/local_citation_capture.py \
  tldw_chatbook/RAG_Search/pipeline_functions_simple.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py \
  tldw_chatbook/Event_Handlers/worker_events.py \
  tldw_chatbook/Chat/Chat_Functions.py \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/Event_Handlers/test_worker_local_citation_capture.py
```

Expected: no findings in changed files.

- [ ] **Step 3: Run the complete focused regression set**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_models.py \
  Tests/Chat/test_citation_trace_identity.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  Tests/Event_Handlers/Chat_Events/test_chat_events.py \
  Tests/Event_Handlers/test_worker_local_citation_capture.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/RAG/test_semantic_honest_states.py \
  Tests/RAG/test_fusion.py -q
```

Expected: zero failures.

- [ ] **Step 4: Run repository-wide verification without concurrent pytest**

First verify no other pytest process is active, then run:

```bash
../../.venv/bin/python -m pytest
```

Expected: zero failures. If unrelated baseline failures exist, record exact
node IDs and compare them against a fresh `origin/dev` worktree before making
any completion claim.

- [ ] **Step 5: Review the diff and task acceptance criteria**

```bash
git diff --check
git status --short
git diff --stat origin/dev...
git diff origin/dev... -- \
  tldw_chatbook Tests Docs backlog/tasks
```

Confirm:

- no sensitive content logging
- no arbitrary metadata copied into locator/source fields
- no partial builder persistence
- disabled mode preserves existing prompt bytes
- exact captured snapshot blocks compose the provider context
- no answer-attempt, marker-occurrence, repair, UI, resolver, server, or sync
  scope leaked into this task

- [ ] **Step 6: Complete Backlog hygiene**

Check all six acceptance criteria, add concise implementation notes with test
evidence and the ADR-024 link, and set TASK-553.13 to Done only after every DoD
gate passes.

- [ ] **Step 7: Commit documentation/task closeout**

```bash
git add \
  Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md \
  Docs/superpowers/plans/2026-07-25-local-rag-retrieval-prompt-evidence-capture.md \
  'backlog/tasks/task-553.13 - Capture-local-RAG-retrieval-runs-and-exact-prompt-evidence-sets.md'
git commit -m "docs(rag): record local citation capture delivery"
```
