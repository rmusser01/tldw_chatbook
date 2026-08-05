# Local Answer-Attempt and Terminal Citation Sealing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist one eligible marker-free local RAG answer and its complete prompt provenance atomically from the exact terminal Console body.

**Architecture:** Extend the request-scoped citation builder with governed answer attempts and one-shot sealing, then carry an explicit prompt-set ID and a transient finalizer into the native Console assistant placeholder. The store owns exact-body materialization, terminal deferral, stable message identity, and bounded retry/fallback behavior; the existing persistence service and citation repository continue to own the single SQLite transaction. Production composition passes the same repository instance to both capture and persistence, while unsupported adapters remain on the ordinary message path.

**Tech Stack:** Python 3.11+, Pydantic v2, Textual Console store/controller, SQLite/FTS-backed `CharactersRAGDB`, pytest, Loguru.

---

## Planning constraints

- Required implementation discipline: `@superpowers:test-driven-development`.
- Completion verification: `@superpowers:verification-before-completion`.
- ADR required: yes.
- ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`.
- Reason: this task directly implements ADR-024's existing request-scoped builder, terminal seal, governed answer-body, message ownership, and atomic persistence decisions. It does not make a new architectural decision.
- Base dependency: TASK-553.13 / commit `22e9e6b9b`.
- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-answer-attempt-sealing`.
- Do not add occurrence mappings, citation repair, Sources UI, source opening, retry/regenerate provenance, server traces, or Sync v2 trace transport.
- Do not weaken `SealedCitationWrite`'s exact occurrence/body validation.
- Run only the focused tests named in this plan. Repository-wide baseline repair remains separate.
- Never log answer bodies, queries, titles, snapshots, source identities, locators, fingerprints, or exception text on the finalization path.

## File responsibility map

| File | Responsibility in this task |
| --- | --- |
| `tldw_chatbook/Chat/citation_trace_builder.py` | Validate and retain the initial governed answer attempt, enforce marker-free eligibility and chronology, and seal one immutable write. |
| `tldw_chatbook/Chat/citation_trace_repository.py` | Own the fixed local seal policy and supply it through the builder factory. |
| `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py` | Return the exact prompt-evidence-set ID alongside context and builder. |
| `tldw_chatbook/Chat/chat_persistence_service.py` | Advertise fail-closed canonical-write readiness from the real repository/database binding. |
| `tldw_chatbook/Chat/console_chat_store.py` | Atomically register transient finalization state, defer early persistence, invoke the exact-body finalizer, and implement stable retry/fallback behavior. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Create the request-local finalizer, install it only for initial sends, and clear it on every exit. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Pass the app's exact citation repository into the Console persistence service. |
| `Tests/Chat/test_citation_trace_builder.py` | Builder mutation, privacy, chronology, completeness, and one-shot seal tests. |
| `Tests/Chat/test_citation_trace_repository.py` | Repository-owned seal-policy factory tests. |
| `Tests/RAG/test_local_citation_capture.py` | Prompt-evidence-set ID handoff tests for local capture paths. |
| `Tests/RAG/test_scope_pipeline_enforcement.py` | Existing direct-builder fixture compatibility after policy injection becomes required. |
| `Tests/Chat/test_chat_function_local_citation_boundary.py` | Existing direct-builder boundary fixture plus prompt-set ID handoff coverage. |
| `Tests/Chat/test_console_chat_store.py` | Existing non-citation persistence timing regressions. |
| `Tests/Chat/test_console_terminal_citation_persistence.py` | New focused store retry/fallback, cleanup, exact-body, and real atomic persistence tests. |
| `Tests/Chat/test_console_local_citation_boundary.py` | Direct-provider and agent request lifecycle integration tests. |
| `Tests/UI/test_console_native_chat_flow.py` | Production repository-composition test. |
| `backlog/tasks/task-553.14 - Capture-answer-attempts-and-seal-terminal-local-citation-traces.md` | Implementation plan, checked acceptance criteria, verification evidence, and implementation notes. |

### Task 1: Add repository-owned local policy and one-shot builder sealing

**Files:**
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py:488-517`
- Modify: `tldw_chatbook/Chat/citation_trace_builder.py:21-605`
- Modify: `Tests/Chat/test_citation_trace_builder.py`
- Modify: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `Tests/RAG/test_scope_pipeline_enforcement.py`
- Modify: `Tests/Chat/test_chat_function_local_citation_boundary.py`

- [ ] **Step 1: Add failing builder tests for the initial answer attempt**

Add tests that construct one run and prompt set, then call the proposed terminal APIs:

```python
def _record_prompt(builder: CitationTraceBuilder) -> str:
    run_id = _record_run(builder)
    return builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text="[S1] MEDIA — Alpha\nexact evidence",
            ),
        ),
        created_at=NOW + timedelta(seconds=1),
    )


def test_local_builder_records_exact_governed_initial_answer() -> None:
    builder = _builder()
    prompt_id = _record_prompt(builder)

    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_id,
        answer_body="marker-free exact answer",
        completed_at=NOW + timedelta(seconds=2),
    )

    assert builder.answer_attempts[0].attempt_id == attempt_id
    assert builder.answer_attempts[0].kind is AnswerAttemptKind.INITIAL
    assert builder.answer_attempt_payloads[0].answer_body == "marker-free exact answer"
    assert builder.answer_attempt_payloads[0].body_integrity_hmac
    assert "marker-free exact answer" not in builder.answer_attempts[0].model_dump_json()
```

Also add tests for:

- single and multiple eligible `[S#]` markers both raising the same safe `occurrence_mapping_unavailable` reason
- markers inside Markdown code or escaped literals remaining eligible for marker-free sealing
- answer-body byte cap and aggregate governed-payload cap
- unknown prompt-set ID
- attempt completion before prompt creation
- rejected recording leaving both attempt collections unchanged

- [ ] **Step 2: Run the new answer-attempt tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_citation_trace_builder.py \
  -k 'initial_answer or marker_free or answer_body or attempt_completion'
```

Expected: failures because `record_initial_answer_attempt`, `answer_attempts`, and `answer_attempt_payloads` do not exist.

- [ ] **Step 3: Add failing seal and policy tests**

Cover:

```python
def test_local_builder_seals_once_with_repository_policy() -> None:
    builder = _builder()
    prompt_id = _record_prompt(builder)
    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_id,
        answer_body="final answer",
        completed_at=NOW + timedelta(seconds=2),
    )

    write = builder.seal(
        selected_attempt_id=attempt_id,
        sealed_at=NOW + timedelta(seconds=3),
    )

    assert write.trace.origin is TraceOrigin.LOCAL
    assert write.trace.lifecycle is TraceLifecycle.SEALED
    assert write.trace.policy_version == "local-prompt-provenance-v1"
    assert write.trace.policy_capabilities == (
        PolicyCapability.VIEW_SNAPSHOT,
        PolicyCapability.VIEW_SOURCE_IDENTITY,
    )
    assert write.trace.completeness_at_seal is CitationCompleteness.COMPLETE
    assert write.trace.answer_attempts[0].occurrences == ()
    assert builder.is_sealed is True
    with pytest.raises(ValueError, match="sealed"):
        builder.seal(selected_attempt_id=attempt_id, sealed_at=NOW + timedelta(seconds=4))
```

Add the remaining privacy assertions:

```python
assert write.trace.answer_attempts[0].occurrences == ()
assert "final answer" not in write.trace.model_dump_json()
assert write.answer_attempt_payloads[0].answer_body == "final answer"
```

Also test:

- every run requires non-null `ended_at`
- each prompt entry's `run.ended_at <= prompt.created_at`
- each attempt's `prompt.created_at <= attempt.created_at`
- every run, prompt, and attempt boundary is `<= sealed_at`
- failed sealing does not set `is_sealed`
- every mutation method rejects after a successful seal
- a second initial-attempt recording is rejected without partial mutation
- the returned `SealedCitationWrite` object can be reused unchanged
- repository factory builders receive the fixed version and only the two approved capabilities
- disabled or identity/key-unready repositories still return no builder

- [ ] **Step 4: Run the seal/policy tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_repository.py \
  -k 'seal or policy or closed_retrieval or lifecycle_order'
```

Expected: failures because the builder cannot seal and the repository factory does not supply policy metadata.

- [ ] **Step 5: Implement the minimal repository-owned policy**

In `citation_trace_repository.py`, keep the values private and non-configurable:

```python
_LOCAL_TRACE_POLICY_VERSION = "local-prompt-provenance-v1"
_LOCAL_TRACE_POLICY_CAPABILITIES = (
    PolicyCapability.VIEW_SNAPSHOT,
    PolicyCapability.VIEW_SOURCE_IDENTITY,
)
```

Pass both values from `create_local_trace_builder()` into `CitationTraceBuilder.local()`. Do not add them to `CitationProvenanceRuntimePolicy`.

Make `policy_version` and `policy_capabilities` required builder-construction
arguments, so a seal-capable builder cannot silently invent or default policy
metadata. Update every existing direct test builder listed in this task with
explicit test policy values; production continues to construct builders only
through the repository factory.

- [ ] **Step 6: Implement bounded answer-attempt state**

In `citation_trace_builder.py`:

- import the existing attempt, trace, lifecycle, policy, reducer, and marker helper models
- add answer-attempt and answer-payload collections, stable trace ID, policy metadata, and sealed-write state to `__slots__`
- add deep-copy tuple properties for governed payloads
- extend `_ensure_governed_payload_capacity()` and `_canonical_payload_bytes()` to include `AnswerAttemptPayload`
- add `_ensure_unsealed()` and invoke it before every mutation
- make `is_sealed` reflect the sealed-write state
- reject a second initial attempt in TASK-553.14; do not add repair or rerun mutation APIs
- compute `body_integrity_hmac` with the existing `MESSAGE_BODY` fingerprint domain

Use a safe typed denial:

```python
class CitationTraceBuildUnavailable(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)
```

Export this exception from the builder module for the Console boundary.

Implement the marker guard without changing the shared helper:

```python
try:
    spans = eligible_citation_marker_spans(
        answer_body,
        prompt_set.marker_namespace,
        max_count=1,
    )
except ValueError:
    raise CitationTraceBuildUnavailable(
        "occurrence_mapping_unavailable"
    ) from None
if spans:
    raise CitationTraceBuildUnavailable(
        "occurrence_mapping_unavailable"
    )
```

Construct all prospective `AnswerAttemptPayload` and `AnswerAttempt` objects, validate aggregate capacity, and only then append them.

- [ ] **Step 7: Implement one-shot sealing**

Implement:

```python
def seal(
    self,
    *,
    selected_attempt_id: str,
    sealed_at: datetime | None = None,
) -> SealedCitationWrite:
    ...
```

The method must:

1. reject an already sealed builder
2. require runs, prompt sets, attempts, and a known selected attempt
3. require every local run to have `ended_at`
4. validate the complete temporal chain
5. build a provisional `CitationTrace` only to call `reduce_selected_attempt_completeness()`
6. rebuild the validated trace with the reduced completeness
7. build a complete `SealedCitationWrite`
8. assign `_sealed_write` only after all validation succeeds
9. return the same immutable write on the persistence path without exposing the codec

Do not parse or create occurrences.

- [ ] **Step 8: Run focused builder and factory tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_repository.py \
  -k 'local_builder or create_local_trace_builder'
```

Expected: all selected tests pass.

- [ ] **Step 9: Run direct-builder fixture compatibility tests**

Run the existing tests whose fakes construct builders directly:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  -k 'canonical_evidence_at_provider_boundary or capture_api_records or console_staged_local_evidence or empty_retrieval or pipeline_exception or malformed_result or off_selection_source or validation_failure or fresh_scoped_existence_failure or backing_row_deleted or original_session_fresh_scope or original_unpersisted_holder'
```

Expected: all selected tests pass without builder-construction errors.

- [ ] **Step 10: Commit the builder boundary**

```bash
git add \
  tldw_chatbook/Chat/citation_trace_builder.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py
git commit -m "feat(rag): seal local citation answer attempts"
```

### Task 2: Carry the exact prompt-set identity through local capture

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py:67-73,1415-1515,1570-1705`
- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `Tests/Chat/test_chat_function_local_citation_boundary.py`

- [ ] **Step 1: Write failing prompt-ID handoff tests**

Extend the successful pipeline and Console evidence capture tests:

```python
result = await prepare_local_rag_context(...)

assert result.citation_builder is builder
assert result.prompt_evidence_set_id == builder.prompt_evidence_sets[-1].prompt_set_id
```

Assert `prompt_evidence_set_id is None` for:

- no builder
- empty formatted evidence
- authority/capture failure
- a builder run that cannot record a prompt set

- [ ] **Step 2: Run the handoff tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  -k 'prompt_evidence_set_id or canonical_capture or local_citation'
```

Expected: failure because `LocalRagContextResult` has no prompt-set ID.

- [ ] **Step 3: Implement the explicit handoff**

Add a backward-compatible default:

```python
@dataclass(frozen=True)
class LocalRagContextResult:
    context: str | None
    citation_builder: CitationTraceBuilder | None
    prompt_evidence_set_id: str | None = None
```

Capture the exact returned ID rather than rediscovering the last prompt set:

```python
prompt_evidence_set_id = None
if formatted.entries:
    prompt_evidence_set_id = builder.record_prompt_evidence_set(...)
return LocalRagContextResult(
    context,
    builder,
    prompt_evidence_set_id,
)
```

Apply the same rule to every local capture path. A builder without a recorded non-empty prompt set must not carry an ID.

- [ ] **Step 4: Run focused capture tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  -k 'prompt or canonical or builder'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the capture handoff**

```bash
git add \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py
git commit -m "feat(rag): return captured prompt evidence identity"
```

### Task 3: Advertise persistence readiness and wire the production repository

**Files:**
- Modify: `tldw_chatbook/Chat/citation_trace_repository.py:488-578`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:18-28`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:2959-2977`
- Modify: `Tests/Chat/test_citation_trace_repository.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing readiness tests**

Add focused service tests:

```python
def test_canonical_citation_writes_ready_requires_matching_enabled_repository(
    db_instance,
) -> None:
    repository = citation_repository(db_instance)
    service = ChatPersistenceService(
        db_instance,
        citation_repository=repository,
    )
    assert service.canonical_citation_writes_ready is True


def test_canonical_citation_writes_ready_is_false_without_repository(
    db_instance,
) -> None:
    assert ChatPersistenceService(
        db_instance
    ).canonical_citation_writes_ready is False
```

Cover disabled, identity/key-unready, and database-mismatched repositories as false.

Add repository tests for a dedicated public
`local_citation_writes_ready` property. It must match the builder factory's
actual prerequisites, including equality with the persisted singleton
identity, rather than reusing the migration-named readiness property.

- [ ] **Step 2: Write the failing ChatScreen composition test**

Build a bare `ChatScreen` using the existing native Console harness pattern. Put a sentinel repository on `app_instance`, call `_ensure_console_chat_store()`, and assert:

```python
assert store.persistence is not None
assert store.persistence.citation_repository is repository
assert store.persistence.db is repository.db
```

Also assert a mismatched repository is not passed into the service.

- [ ] **Step 3: Run the readiness/composition tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py::TestChatPersistenceService::test_canonical_citation_writes_ready_requires_matching_enabled_repository \
  Tests/Chat/test_chat_persistence_service.py::TestChatPersistenceService::test_canonical_citation_writes_ready_is_false_without_repository \
  Tests/UI/test_console_native_chat_flow.py::test_console_store_uses_app_citation_repository \
  -k 'local_citation_writes_ready or canonical_citation_writes_ready or console_store_uses_app_citation_repository'
```

Expected: failures because readiness is not exposed and `ChatScreen` omits the repository.

- [ ] **Step 4: Implement fail-closed service readiness**

Add a read-only property:

```python
@property
def canonical_citation_writes_ready(self) -> bool:
    repository = self.citation_repository
    return bool(
        repository is not None
        and repository.db is self.db
        and repository.local_citation_writes_ready
    )
```

Add `CitationTraceRepository.local_citation_writes_ready` as a content-free,
read-only check for the same canonical-write switch, identity, codec, and
persisted-identity equality required by `create_local_trace_builder()`. Reuse
that property inside the factory to prevent readiness drift. It may read the
persisted singleton but must not load keys, mutate configuration, or log
governed values. A failed persisted-identity read returns false.

- [ ] **Step 5: Wire the exact app repository**

In `ChatScreen._ensure_console_chat_store()`:

```python
repository = getattr(
    self.app_instance,
    "citation_trace_repository",
    None,
)
if repository is not None and getattr(repository, "db", None) is not db:
    repository = None
persistence = ChatPersistenceService(
    db,
    workspace_registry=...,
    citation_repository=repository,
)
```

Keep `persistence=None` when the app has no ChaChaNotes DB.

- [ ] **Step 6: Run the focused readiness/composition tests**

Run the three node IDs from Step 3 plus the disabled/mismatch parameterized node IDs.

Expected: all pass.

- [ ] **Step 7: Commit production composition**

```bash
git add \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/UI/test_console_native_chat_flow.py
git commit -m "fix(console): wire canonical citation persistence"
```

### Task 4: Add transient terminal finalization to `ConsoleChatStore`

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1-105,228-320,543-578,810-858,1232-1268,1478-1555,1980-2190,2793-2823`
- Create: `Tests/Chat/test_console_terminal_citation_persistence.py`
- Modify: `Tests/Chat/test_console_chat_store.py`

- [ ] **Step 1: Write failing capability and early-persistence tests**

Create focused fakes:

```python
class _CitationPersistenceFake:
    db = None
    canonical_citation_writes_ready = True

    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []

    def create_message(self, **kwargs) -> str:
        self.create_calls.append(kwargs)
        return str(kwargs["message_id"])
```

Test that:

- a ready adapter accepting `citation_write` atomically arms the placeholder
- an adapter without `citation_write`, readiness, or persistence does not arm it
- `get_message()` and `messages_for_session()` materialize streamed text without calling persistence while terminal deferral is active
- ordinary assistant messages retain current first-content persistence

- [ ] **Step 2: Run the new store tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'capability or deferred or ordinary'
```

Expected: failures because `append_message()` cannot accept a finalizer and the store has no terminal deferral.

- [ ] **Step 3: Add the optional protocol and transient state**

Update `ConsoleChatPersistence.create_message()` with optional `citation_write`, and document that narrow fakes may omit it.

Add:

```python
TerminalCitationFinalizer = Callable[
    [str],
    SealedCitationWrite | None,
]
```

Store only transient state:

```python
self._terminal_citation_finalizers: dict[
    str, TerminalCitationFinalizer
] = {}
self._terminal_persistence_deferred_ids: set[str] = set()
```

Add `_citation_persistence_ready()` requiring:

- persistence exists
- `create_message` accepts `citation_write`
- `canonical_citation_writes_ready is True`

Any readiness property/signature inspection failure returns false without
logging exception text. Existing fakes that omit readiness remain fail-closed
and ordinary.

- [ ] **Step 4: Install finalizer and deferral atomically**

Extend `append_message()` with:

```python
terminal_citation_finalizer: TerminalCitationFinalizer | None = None
```

Before mutating the tree, reject callback placement on a non-assistant or
non-empty/attachment-bearing message, evaluating attachments after the
existing scalar-image-to-tuple normalization. A valid callback arms only when:

- role is assistant
- content and attachments are empty/pending
- `persist is True`
- persistence readiness passes

An unavailable persistence capability is fail-closed ordinary behavior, not a
caller error. After registering the new assistant tree node and before any
persistence call, install both transient entries together. If the subsequent
session-persistence/defer setup raises before `append_message()` can return,
clear both transient entries before re-raising. Never leave only one entry
installed.

- [ ] **Step 5: Block UI polling from flushing deferred messages**

Add the terminal-deferral membership check to `_persist_pending_message_if_ready()`. Do not change `_materialize_stream_buffer()`; it must still update visible in-memory content.

Add one idempotent cleanup method:

```python
def clear_terminal_citation_state(self, message_id: str) -> None:
    self._terminal_citation_finalizers.pop(message_id, None)
    self._terminal_persistence_deferred_ids.discard(message_id)
```

Call the same cleanup from session close, subtree deletion, store clear/shutdown, and every non-success terminal transition.

- [ ] **Step 6: Run capability/deferral tests**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 7: Write failing exact-body and stable-ID tests**

Test that successful completion:

- materializes the exact buffered body first
- calls the finalizer exactly once with that body
- supplies the same native message UUID as `message_id`
- passes the returned `SealedCitationWrite` by object identity
- clears finalizer and deferral after completion
- does not call the finalizer again on later reads
- an exception during append-time session persistence leaves neither transient entry behind

Use a marker-free sealed-write fixture whose selected answer body exactly matches the test message.

- [ ] **Step 8: Write failing fallback and ambiguous-retry tests**

Use behavior-controlled fakes to cover:

1. finalizer returns `None` → one ordinary stable-ID create
2. first citation call raises `CitationPersistenceUnavailable` → one ordinary stable-ID create
3. first citation call raises an ambiguous exception → one retry with the same ID and same sealed-write object
4. both ambiguous calls fail → no ordinary insert, message remains complete in memory
5. the ambiguous retry raises `CitationPersistenceUnavailable` → still no ordinary insert
6. deterministic ordinary fallback also fails → no new ID and no later polling-triggered create

After terminal abandonment, call both `get_message()` and `messages_for_session()` and assert the fake's call count does not increase. The store must remove the ID from ordinary pending persistence when it intentionally abandons a conflicting durable write.

- [ ] **Step 9: Run exact-body/retry tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'exact_body or stable_id or fallback or ambiguous or abandoned'
```

Expected: failures because terminal citation persistence is not implemented.

- [ ] **Step 10: Implement the citation-aware create call**

Refactor `_persist_new_message()` just enough to build `create_kwargs` once and accept:

```python
citation_write: SealedCitationWrite | None = None
force_stable_message_id: bool = False
```

Use `message.id` when either `generation_metadata` or `force_stable_message_id` is true. Keep attachment, metadata, parent, feedback, active-leaf, and Sync v2 behavior unchanged.

Isolate retry around only:

```python
self.persistence.create_message(**create_kwargs)
```

Apply this fixed first-failure disposition:

```text
CitationPersistenceUnavailable first
  -> remove citation_write
  -> one ordinary same-ID attempt

any other exception first
  -> one same-ID, same-write retry
  -> never ordinary fallback after that branch
```

Catch persistence failures inside the provenance-aware terminal path so a completed provider/agent run remains complete. Log only fixed reason codes.

If the final durable attempt fails, discard the message ID from ordinary pending persistence so later UI polling cannot insert a conflicting row.

Once `create_message()` succeeds, assign/discard the durable identity before
active-leaf and Sync v2 bookkeeping. A later bookkeeping failure is logged with
a fixed diagnostic and never replays the atomic create call.

- [ ] **Step 11: Implement exact-body terminal completion**

On `mark_message_complete()`:

1. materialize the stream buffer
2. atomically pop the finalizer while retaining a local stable-ID flag
3. invoke it once with `message.content`
4. mark the message complete
5. persist through the citation-aware create helper
6. clear terminal state in `finally`

Defensively translate a finalizer exception to a fixed `terminal_finalizer_unavailable` diagnostic and ordinary stable-ID persistence. Never log `str(exception)`.

On stopped, failed, canceled, and empty paths, clear terminal state before using existing ordinary persistence behavior.

- [ ] **Step 12: Run focused store tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_chat_store.py \
  -k 'citation or persist or stream or complete or stopped or failed'
```

Expected: all selected tests pass.

- [ ] **Step 13: Commit the store lifecycle**

```bash
git add \
  tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): finalize citation traces at terminal persistence"
```

### Task 5: Integrate initial direct and agent Console generations

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:667-855,3351-3370,3485-3710,3962-4170`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`
- Modify: `Tests/Chat/test_console_terminal_citation_persistence.py`

- [ ] **Step 1: Write failing direct-provider integration tests**

Use a real `CitationTraceBuilder` and a ready citation persistence fake. Have the capture provider return:

```python
SimpleNamespace(
    context=context,
    citation_builder=builder,
    prompt_evidence_set_id=prompt_id,
)
```

Cover:

- marker-free direct success seals the exact visible output
- successful prefill plus streamed output seals the concatenated visible body
- marker-bearing output persists ordinarily with no citation write
- an empty provider stream does not seal
- provider failure and user stop do not seal
- no builder/finalizer remains reachable from store transient state after any exit

- [ ] **Step 2: Write failing agent integration tests**

The fake agent bridge must stream into the actual placeholder before returning `RUN_DONE`:

```python
def run_reply(**kwargs):
    store.append_stream_chunk(
        kwargs["assistant_message_id"],
        "agent answer",
    )
    return "run-test", RunOutcome(
        status=RUN_DONE,
        steps=[],
        final_text="agent answer",
    )
```

Cover:

- agent success seals its exact materialized body
- `RUN_DONE` with empty `final_text` clears terminal citation state before adding `"No response was generated."`
- canceled, stopped, and failed agent outcomes do not seal
- a missing/replaced placeholder never transfers the finalizer to another assistant row
- after an initial RAG send exits, `retry_message()` and
  `regenerate_message()` run without a terminal callback, do not add citation
  persistence calls, and do not mutate or inherit the original builder

- [ ] **Step 3: Run direct/agent tests and confirm they fail**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  -k 'terminal or seal or marker_bearing or empty or stopped or agent or retry or regenerate'
```

Expected: failures because the controller drops the builder without installing a finalizer.

- [ ] **Step 4: Return the full capture triple**

Change `_capture_rag_context()` to return:

```python
tuple[str | None, CitationTraceBuilder | None, str | None]
```

Validate all three values independently. Context-only compatibility remains available, but no terminal finalizer is built unless context, builder, and explicit prompt ID are all present.

- [ ] **Step 5: Build a content-safe request-local finalizer**

Add a narrow helper returning `TerminalCitationFinalizer | None`. Its closure:

```python
def finalize(exact_body: str) -> SealedCitationWrite | None:
    try:
        terminal_at = datetime.now(UTC)
        attempt_id = builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_evidence_set_id,
            answer_body=exact_body,
            completed_at=terminal_at,
        )
        return builder.seal(
            selected_attempt_id=attempt_id,
            sealed_at=terminal_at,
        )
    except CitationTraceBuildUnavailable as exc:
        logger.warning(
            "Console citation finalization unavailable; reason={}",
            exc.reason_code,
        )
    except Exception:
        logger.warning(
            "Console citation finalization unavailable; "
            "reason=attempt_or_seal_failure"
        )
    return None
```

Do not include the exception object, body, builder representation, or governed values in any log call.

- [ ] **Step 6: Install and clear the finalizer across the whole send**

In `submit_draft()`:

1. receive context, builder, and prompt ID
2. create the finalizer before appending the assistant
3. pass it into the same `append_message()` call that creates the empty placeholder
4. keep the existing user-message persistence ordering
5. in the outer `finally`, call `store.clear_terminal_citation_state(assistant.id)` and delete the local builder reference

Do not pass a finalizer from retry, regenerate, edit/resend, or continuation entry points.

- [ ] **Step 7: Disarm the empty-agent fallback**

Before `_complete_agent_message()` appends `"No response was generated."` for an empty final result, call the idempotent terminal-state cleanup. Successful non-empty agent output keeps the finalizer until `mark_message_complete()`.

- [ ] **Step 8: Add content-free diagnostic tests**

Use sentinels in:

- answer body
- marker-bearing body
- a forged finalizer exception
- query/title/snapshot fields

Capture Loguru output and assert none of those sentinels occur. Assert only fixed reason codes are present.

- [ ] **Step 9: Run focused controller integration tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'direct or agent or terminal or marker or empty or stop or retry or regenerate or diagnostic'
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit the Console integration**

```bash
git add \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_terminal_citation_persistence.py
git commit -m "feat(console): seal initial local RAG answers"
```

### Task 6: Prove real atomic persistence and close TASK-553.14

**Files:**
- Modify: `Tests/Chat/test_console_terminal_citation_persistence.py`
- Modify: `backlog/tasks/task-553.14 - Capture-answer-attempts-and-seal-terminal-local-citation-traces.md`

- [ ] **Step 1: Add the real database integration fixture**

Use `CharactersRAGDB`, the persisted local identity, the test fingerprint codec, enabled `CitationProvenanceRuntimePolicy`, `CitationTraceRepository`, and `ChatPersistenceService` to build a real `ConsoleChatStore`.

The repository instance passed to `ChatPersistenceService` must be the same instance used to create the builder.

- [ ] **Step 2: Add the failing end-to-end atomic persistence test**

Drive a marker-free direct-provider answer through `ConsoleChatController.submit_draft()`, then assert:

```python
assistant = next(
    message
    for message in store.messages_for_session(session.id)
    if message.role is ConsoleMessageRole.ASSISTANT
)
assert assistant.persisted_message_id == assistant.id
assert db.get_message_by_id(assistant.id)["content"] == "exact final answer"
```

Query the citation tables or use the existing repository hydration/owner APIs to assert:

- one trace exists
- its selected attempt body equals the persisted message body
- run, prompt snapshot, attempt payload, reference, and owner rows exist
- the immutable trace JSON contains neither answer body nor integrity HMAC
- `occurrences == ()`
- restart/reload can find the active owner

- [ ] **Step 3: Add the failing rollback/fallback integration test**

Use a test-only repository subclass that overrides the existing row-family
failure seam to raise
`CitationPersistenceUnavailable("forced_deterministic_unavailable")` after a
chosen row family. Do not use the stock `failure_after_row_family` behavior
here: it raises `RuntimeError`, which is intentionally ambiguous and must not
fall back to an ordinary insert. Assert:

- no partial trace/run/snapshot/attempt/reference/owner rows remain
- the ordinary assistant message is still persisted under its native ID when the failure is deterministic
- it has no active trace owner
- the answer remains complete in memory

Do not duplicate repository transaction matrix tests already covered in `test_citation_trace_repository.py`; this test proves only the Console-to-service integration seam.

- [ ] **Step 4: Run the new real integration tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'real_atomic or real_rollback'
```

Expected: both pass.

- [ ] **Step 5: Run the final scoped verification**

Run only touched-code coverage:

```bash
../../.venv/bin/pytest -q \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/RAG/test_scope_pipeline_enforcement.py \
  Tests/Chat/test_chat_function_local_citation_boundary.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/UI/test_console_native_chat_flow.py \
  -k 'initial_answer or marker_free or answer_body or seal or local_builder or create_local_trace_builder or prompt_evidence_set_id or local_citation or canonical_capture or canonical_evidence_at_provider_boundary or capture_api_records or console_staged_local_evidence or empty_retrieval or pipeline_exception or malformed_result or off_selection_source or validation_failure or fresh_scoped_existence_failure or backing_row_deleted or original_session_fresh_scope or original_unpersisted_holder or canonical_citation or citation_repository or terminal_citation or console_citation or marker_bearing or exact_body or deferred or fallback or ambiguous or abandoned or retry or regenerate or diagnostic or real_atomic or real_rollback'
```

Expected: all selected tests pass with no collection errors.

Then run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 6: Self-review the exact diff**

Check:

- no occurrence parsing or UI presentation entered the diff
- no raw governed value or exception text appears in logs
- every transient map/set is swept on completion, failure, cancellation, deletion, close, and clear
- the first ambiguous failure alone determines retry disposition
- an abandoned write cannot later fall through ordinary pending persistence
- production `ChatPersistenceService` owns the same repository instance as capture
- only initial sends receive finalizers

- [ ] **Step 7: Update Backlog implementation notes and acceptance criteria**

Use Backlog CLI to:

1. check all six acceptance criteria only after their tests pass
2. add concise implementation notes listing the builder, capture handoff, store lifecycle, controller integration, production wiring, and scoped verification
3. retain the ADR check:

```text
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Direct implementation of the accepted terminal seal and atomic ownership contract; no new decision.
```

4. set TASK-553.14 to Done only after every Definition-of-Done item is satisfied

- [ ] **Step 8: Commit closeout metadata**

```bash
git add \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  'backlog/tasks/task-553.14 - Capture-answer-attempts-and-seal-terminal-local-citation-traces.md'
git commit -m "test(rag): verify terminal citation persistence"
```

- [ ] **Step 9: Verify TASK-553.14 closeout**

Run:

```bash
backlog task 553.14 --plain
```

Expected: status `Done`, all acceptance criteria checked, implementation plan and implementation notes present, and ADR linkage retained.
