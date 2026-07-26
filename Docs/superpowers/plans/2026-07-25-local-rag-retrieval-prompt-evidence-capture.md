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

- [x] **Step 1: Write failing constructor and privacy tests**

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

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_citation_trace_builder.py -q
```

Expected: collection/import failure because `citation_trace_builder.py` does
not exist.

- [x] **Step 3: Implement the minimal builder constructor**

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

- [x] **Step 4: Write failing retrieval-run tests**

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

- [x] **Step 5: Run and verify RED, then implement retrieval capture**

Run the focused test, confirm failures are caused by the missing method, then
implement only enough to create `EvidenceRun` and `EvidenceRunPayload` using:

- `CitationFingerprintDomain.RAW_QUERY`
- `new_opaque_id("evidence-run")`
- `new_opaque_id("run-payload")`
- typed `RetrievalScoreKind` and `RetrievalScoreScale`

Serialize only the typed metadata model into
`EvidenceRunPayload.retrieval_metadata`. Never log query, title, source
identity, locator, lineage, or fingerprints.

- [x] **Step 6: Write failing prompt-set tests**

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

- [x] **Step 7: Run and verify RED, implement prompt capture, then run GREEN**

Build all Pydantic objects before mutating builder lists so a failure is atomic.
Use `MarkerNamespace.CHATBOOK_S_V1`. The builder remains unsealed and
non-persistable.

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_citation_trace_builder.py -q
```

Expected: all tests pass.

- [x] **Step 8: Commit**

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

- [x] **Step 1: Write failing source-normalization tests**

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

- [x] **Step 2: Run RED, then implement a strict allowlisted normalizer**

The normalizer may retain only:

- canonical source kind and opaque item ID
- bounded title
- exact content for the live formatter
- typed score kind/scale/value
- allowlisted non-executable lineage keys

Do not copy arbitrary metadata, URLs, paths, or citation dictionaries into
source identity or locator fields. Resolver-family tasks will add typed native
locators later.

- [x] **Step 3: Write failing canonical formatter tests**

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

- [x] **Step 4: Run RED, implement the formatter, then run GREEN**

Build each block as:

```text
[S1] MEDIA — Title
exact submitted content
```

Use a single calculation for returned context and snapshot text. Never parse
the aggregate context afterward to reconstruct entries.

- [x] **Step 5: Write failing prompt-boundary authority tests**

Test a fresh, uncached post-retrieval check:

- a media/note/conversation result is retrieved, then its backing row is
  deleted or soft-deleted before prompt assembly; it is excluded
- a conversation or workspace scope is narrowed after retrieval without
  reusing the old `ScopeCache` entry; newly unauthorized evidence is excluded
- conversations remain excluded under an active local RAG scope
- an `empty` or failed authority/existence read rejects every entry

- [x] **Step 6: Implement the fresh authority/existence seam and run GREEN**

Add an explicit uncached mode to the existing scope-resolution seam and a
batched current-existence query for all three local source families. Prompt
assembly must re-read the conversation/workspace scope and current backing rows
after retrieval, off the event loop where appropriate. Do not consult the
pre-retrieval `EffectiveScope` or `ScopeCache` for this decision. Query/read
errors fail closed. The check consumes canonical item IDs, not display titles
or raw metadata, and runs before marker ordinals are assigned.

- [x] **Step 7: Correct semantic source propagation test-first**

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

- [x] **Step 8: Commit**

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

- [x] **Step 1: Write failing readiness/factory tests**

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

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_citation_trace_repository.py \
  -k "local_trace_builder or canonical_capture" -q
```

Expected: failures because the factory does not exist.

- [x] **Step 3: Implement the minimal safe factory**

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

- [x] **Step 4: Run GREEN and broader repository tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_citation_trace_identity.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit**

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

- [x] **Step 1: Write failing capture-aware API tests**

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

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py -q
```

- [x] **Step 3: Implement opt-in capture without changing pipeline tuples**

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

- [x] **Step 4: Test all pipeline selection branches**

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

- [x] **Step 5: Run focused GREEN and compatibility suites**

```bash
../../.venv/bin/python -m pytest \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/RAG/test_semantic_honest_states.py \
  Tests/RAG/test_fusion.py -q
```

Expected: all tests pass and the old string-return assertions remain valid.

- [x] **Step 6: Commit**

```bash
git add Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_rag_ui_integration.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
git commit -m "feat(rag): capture local retrieval and prompt evidence"
```

---

### Task 5: Keep the builder alive across the current Console generation request

**Files:**

- Create: `Tests/Chat/test_console_local_citation_boundary.py`
- Create: `Tests/UI/test_console_local_citation_capture.py`
- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [x] **Step 1: Rebase the boundary plan onto the native Console architecture**

Latest `dev` retired the legacy `chat_events.py` send path planned originally.
The live boundary is now `ChatScreen` → `ConsoleChatController.submit_draft()`
→ direct provider or agent dispatch. Keep the retired entrypoints absent and
port the same request-local ownership and exact prompt-boundary guarantees to
the native Console path.

- [x] **Step 2: Write failing Console prompt-boundary tests**

Cover:

- canonical evidence containing unique title/body sentinels
- an evidence block with intentional leading/trailing whitespace
- a chat-dictionary entry that would mutate those sentinels if it saw them
- stored/visible user text remains the original draft
- the ordinary prompt still receives dictionary/world-info transformations
- canonical evidence is added only after those transforms
- exact snapshots reach both direct-provider and agent payloads byte for byte
- multimodal image parts remain unchanged
- the request-local builder remains alive through generation and is then
  released
- capture failures log fixed structural diagnostics without sentinels

- [x] **Step 3: Stage and reauthorize the complete Console evidence bundle**

Stage every retrieved Library-RAG result, not only the first display result.
At send time, validate the serialized `EvidenceBundle`, reject non-local or
unavailable references, re-read source existence and active scope without the
cache, format the canonical markers once, and record the run plus exact prompt
evidence set. If no repository/builder is available, preserve the compatible
context-only behavior.

- [x] **Step 4: Thread request-local capture through native generation**

Inject a request-local capture provider into `ConsoleChatController`. Keep the
builder in a local variable across the awaited direct-provider or agent
generation; never store, log, serialize, or persist it. Canonical evidence is
prefixed after ordinary prompt transforms, while the no-builder compatibility
path preserves its prior ordering.

- [x] **Step 5: Verify and commit the native Console port**

The final touched-code gate is recorded in Task 6. Commit the Console port with
the task closeout after the rebased diff is reviewed.

---

### Task 6: Documentation, self-review, and verification

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md`
- Modify: `backlog/tasks/task-553.13 - Capture-local-RAG-retrieval-runs-and-exact-prompt-evidence-sets.md`

- [x] **Step 1: Update implementation links and boundary notes**

Record that workstreams 10 is implemented while workstreams 11–30 remain
follow-ons. Do not mark the epic complete.

- [x] **Step 2: Run formatting/static checks**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/UI/test_console_local_citation_capture.py \
  Tests/RAG/test_local_citation_capture.py
```

Expected: no findings in changed files.

Result: `ruff check` passed for the six citation/Console-RAG code and test
files. `ruff format --check` passed for the three touched test files. The large
native Console modules retain the current `dev` formatting baseline to avoid
unrelated mechanical churn.

- [x] **Step 3: Run the complete focused regression set**

Run the three direct citation files plus only the existing Console seams changed
by the native port: provider dictionary/world-info ordering, Library-RAG
staging/authority display, query validation, and retired-entrypoint guards.

Expected: zero failures.

Result: `110 passed, 1 dependency-version warning in 25.74s`.

- [x] **Step 4: Record the repository-wide verification deviation**

The repository-wide run was stopped at the user's direction after several
hours, and all lingering broad `tldw_chatbook` UI/RAG pytest processes were
terminated. Verification is intentionally limited to touched code for this
task.

Two failures encountered while narrowing the gate reproduce on pristine
`origin/dev` (`af2aee6cd`) and are tracked separately:

- TASK-761:
  `Tests/UI/test_console_dictionary_send_integration.py::test_native_send_applies_conversation_dictionary_agent_branch`
- TASK-762:
  `Tests/UI/test_console_internals_decomposition.py::test_console_rag_action_without_service_stages_recoverable_blocker`

- [x] **Step 5: Review the diff and task acceptance criteria**

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
- no answer-attempt, marker-occurrence, repair, citation-presentation/source-open
  UI, resolver, server, or sync scope leaked into this task

- [x] **Step 6: Complete Backlog hygiene**

Check all six acceptance criteria, add concise implementation notes with test
evidence and the ADR-024 link, and set TASK-553.13 to Done only after every DoD
gate passes.

- [x] **Step 7: Commit documentation/task closeout**

```bash
git add \
  Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md \
  Docs/superpowers/plans/2026-07-25-local-rag-retrieval-prompt-evidence-capture.md \
  'backlog/tasks/task-553.13 - Capture-local-RAG-retrieval-runs-and-exact-prompt-evidence-sets.md'
git commit -m "docs(rag): record local citation capture delivery"
```
