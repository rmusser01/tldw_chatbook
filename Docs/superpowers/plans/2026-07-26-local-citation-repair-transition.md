# Local Citation Repair Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep one local RAG assistant message provisional through structural citation checking, make at most one bounded claim-preserving repair request, and expose an honest transient original-attempt preview when repair succeeds.

**Architecture:** Add a provider-independent citation-repair contract beside the canonical trace models, carry it independently of builder readiness from exact local evidence capture, and let the Console controller own one request-scoped selection session across direct and agent generation. The store defers the one terminal write and provides atomic repaired-body replacement; the provider gateway reports synthesized fallback out of band; transient presentation and preview state never enter message content, persistence, exports, TTS, or provider history.

**Tech Stack:** Python 3.11+, dataclasses and enums, Textual Console controller/store/widgets, asyncio plus thread-safe events, Pydantic-backed citation limits, pytest/pytest-asyncio, Loguru, Ruff.

---

## Planning constraints

- Required implementation discipline: `@superpowers:test-driven-development`.
- Completion verification: `@superpowers:verification-before-completion`.
- ADR required: no.
- ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`.
- Reason: TASK-553.15 directly implements ADR-024's accepted provisional-stream and visible-repair behavior. It introduces no new storage, ownership, sync, security, dependency, or service-boundary decision.
- Approved spec: `Docs/superpowers/specs/2026-07-26-local-citation-repair-transition-design.md`.
- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-citation-repair-transition`.
- Rebase onto the then-current `origin/dev` before implementation begins; resolve documentation-only conflicts first, then re-check every path and line reference in this plan before production edits.
- Run only the focused tests named in this plan. Do not launch the repository-wide suite. Repository-wide baseline repair remains separate.
- Do not add canonical occurrence mappings, semantic support checks, grounded badges, a Sources footer, source resolvers, server trace mapping, pipeline reruns, new retrieval, evidence renumbering, new settings, schema migrations, or restart restoration.
- Only an initial local RAG send receives repair state. Retry, regenerate, edit/resend, continue, and recovered drafts pass no repair session.
- Never log or serialize answer bodies, repaired bodies, evidence, source identity, locators, prompts, credentials, raw provider exceptions, or tracebacks from the repair path.
- Preserve existing message content as the only copy/TTS/export/provider-history input. Presentation metadata contains codes and booleans only.
- Preserve existing non-RAG and no-contract call shapes by passing new optional gateway/action arguments only on repair-eligible paths.

## File responsibility map

| File | Responsibility in this task |
| --- | --- |
| `tldw_chatbook/Chat/citation_repair.py` | New pure bounded contract, Markdown-aware structural decision, claim-preservation projection, repaired-output selection, exact prompt construction, and model-window gate. |
| `tldw_chatbook/Chat/citation_trace_models.py` | Existing package-private Markdown/code/escape traversal reused by citation repair; no duplicate traversal is introduced. |
| `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py` | Carry an optional repair contract independently of the canonical builder and create it only from exact authorized formatted evidence. |
| `tldw_chatbook/Chat/console_provider_gateway.py` | Add the optional content-free synthesized-fallback signal and mark only gateway-authored fallback branches before yield. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Pass one signal through every primary, intermediate tool-turn, and subagent gateway call for the whole agent run. |
| `tldw_chatbook/Chat/console_chat_models.py` | Add `CHECKING_CITATIONS` and bounded transient citation presentation codes/flags. |
| `tldw_chatbook/Chat/console_chat_store.py` | Add explicit terminal-persistence deferral, atomic repaired-body replacement, safe presentation mutation, and cleanup. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Own the request-scoped repair session, shared direct/agent post-generation selection seam, cancellation linearization, preview LRU, and final completion ordering. |
| `tldw_chatbook/Chat/console_message_actions.py` | Offer `View original attempt` only from an explicit optional availability argument; default/plain/export behavior remains unchanged. |
| `tldw_chatbook/Widgets/Console/console_transcript.py` | Render structural notices, pass explicit action availability, and render a screen-provided read-only original-attempt block without mutating the message. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Own ephemeral visible preview state, route the preview action, keep checking runs polling/stoppable, and clear previews on lifecycle changes. |
| `Tests/Chat/test_citation_repair.py` | New pure contract, scanner, projection, limits, prompt, selection, and window-budget tests. |
| `Tests/RAG/test_local_citation_capture.py` | Builder-ready, builder-unavailable, capture-failure, limit, and exact-context repair-contract handoff tests. |
| `Tests/UI/test_console_local_citation_capture.py` | Console-staged evidence composition and controller handoff coverage. |
| `Tests/Chat/test_console_provider_gateway.py` | Synthesized fallback provenance, genuine-equal-text, optional-signal, and no-governed-content tests. |
| `Tests/Chat/test_console_agent_bridge.py` | Whole-run signal reuse across primary/intermediate/subagent gateway calls. |
| `Tests/Chat/test_console_terminal_citation_persistence.py` | No-builder deferral, atomic replacement, one-write outcomes, cleanup, and stable-ID persistence tests. |
| `Tests/Chat/test_console_local_citation_boundary.py` | Direct/agent selection, one-repair, fallback bypass, stop races, session close, and provider-history tests. |
| `Tests/Chat/test_console_message_actions.py` | Explicit availability and byte-identical default/plain action contracts. |
| `Tests/UI/test_console_native_transcript.py` | Notice rendering, row signatures, keyboard action, preview rendering, and non-mutation tests. |
| `Tests/UI/test_console_native_chat_flow.py` | Native screen action routing and cleanup integration tests. |
| `backlog/tasks/task-553.15 - Add-provisional-citation-checking-and-one-visible-repair-transition.md` | Approved-plan link, checked acceptance criteria, scoped verification evidence, ADR check, and implementation notes. |

### Task 1: Build the pure bounded citation-repair contract

**Files:**
- Create: `tldw_chatbook/Chat/citation_repair.py`
- Create: `Tests/Chat/test_citation_repair.py`
- Read/reuse: `tldw_chatbook/Chat/citation_trace_models.py:28-43, 889-1075`
- Read/reuse: `tldw_chatbook/Chat/console_history_budget.py:14-52`

- [ ] **Step 1: Write failing contract and structural-decision tests**

Create `Tests/Chat/test_citation_repair.py` with helpers that build only contiguous contracts:

```python
def _contract(
    *,
    context: str = "[S1] MEDIA — Alpha\nexact evidence",
    ordinals: tuple[int, ...] = (1,),
) -> CitationRepairContract:
    return CitationRepairContract(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=ordinals,
        evidence_context=context,
    )
```

Cover:

- exact contiguous `(1, ..., N)` acceptance for `N=1` and `N=64`
- empty, gapped, reordered, duplicated, boolean, non-integer, zero, negative, and `N=65` ordinal rejection
- non-string/empty/over-64-KiB context rejection
- unsupported namespace and schema version rejection
- `not_applicable` for `None`
- `valid` for repeated, reordered, adjacent, and space-separated known markers
- `repair_required_missing` when no eligible token exists
- `repair_required_invalid` for `[S0]`, `[S01]`, `[S1,S2]`, TAB/SPACE grouped forms, unknown ordinals, and an over-32-character token
- `unavailable` for an answer over 1 MiB and the 513th eligible token
- fenced-code, inline-code, and odd-backslash-escaped literals ignored
- even-backslash markers treated as eligible
- a body-sized digit sequence returns invalid without invoking integer conversion

Use a digit sequence longer than Python's guarded decimal-conversion limit.
If the implementation calls `int()` on it, the test will raise instead of
returning the required bounded invalid decision.

- [ ] **Step 2: Run the contract tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_repair.py \
  -k 'contract or structural or marker or markdown or integer'
```

Expected: collection fails because `tldw_chatbook.Chat.citation_repair` does not exist.

- [ ] **Step 3: Implement immutable contracts, limits, and the shared scanner**

In `citation_repair.py`, import the existing canonical limits rather than restating numeric literals:

```python
REPAIR_EVIDENCE_CONTEXT_UTF8_BYTES_MAX = SNAPSHOT_TEXT_UTF8_BYTES_MAX
REPAIR_ALLOWED_ORDINALS_MAX = EVIDENCE_ENTRIES_PER_PROMPT_MAX
REPAIR_MARKERS_MAX = CITATION_OCCURRENCES_MAX
REPAIR_MARKER_CHARACTERS_MAX = MARKER_CHARACTERS_MAX
REPAIR_ANSWER_BODY_UTF8_BYTES_MAX = ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX
REPAIR_FIXED_OVERHEAD_UTF8_BYTES_MAX = 8 * 1024
REPAIR_REQUEST_UTF8_BYTES_MAX = (
    REPAIR_ANSWER_BODY_UTF8_BYTES_MAX
    + REPAIR_EVIDENCE_CONTEXT_UTF8_BYTES_MAX
    + REPAIR_FIXED_OVERHEAD_UTF8_BYTES_MAX
)
```

Define:

```python
class CitationRepairDecision(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    VALID = "valid"
    REPAIR_REQUIRED_MISSING = "repair_required_missing"
    REPAIR_REQUIRED_INVALID = "repair_required_invalid"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CitationRepairContract:
    schema_version: int
    marker_namespace: MarkerNamespace
    allowed_ordinals: tuple[int, ...]
    evidence_context: str
```

Its `__post_init__` validates exact types, schema `1`, namespace
`CHATBOOK_S_V1`, the tuple `tuple(range(1, len(ordinals) + 1))`, the 64-entry
cap, and the exact UTF-8 context boundary.

Compile only these repair-specific expressions:

```python
_CITATION_LIKE_TOKEN = re.compile(r"\[S[0-9,\t ]+\]")
_WELL_FORMED_TOKEN = re.compile(r"\[S([1-9][0-9]*)\]\Z")
```

Import and use `citation_trace_models._eligible_marker_matches` for fenced
code, inline code, and escape exclusion. Do not copy `_markdown_code_intervals`,
`_fenced_code_intervals`, `_inline_code_intervals`, or backslash counting.
Apply the 32-character bound before matching the captured digits against
`frozenset(str(value) for value in contract.allowed_ordinals)`.

- [ ] **Step 4: Run the structural tests and confirm they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing projection and repaired-selection tests**

Add table-driven tests for:

```python
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Alpha [S1].", "Alpha."),
        ("Alpha  [S1].", "Alpha ."),
        ("Alpha\t[S1].", "Alpha\t."),
        ("Alpha\n[S1].", "Alpha\n."),
        ("Alpha [S1][S2].", "Alpha."),
        ("Alpha [S1] [S2].", "Alpha."),
        ("Alpha [S0].", "Alpha."),
        ("Alpha `[S1]`.", "Alpha `[S1]`."),
        (r"Alpha \[S1].", r"Alpha \[S1]."),
    ],
)
def test_claim_projection_deletes_only_tokens_and_one_ascii_space(...):
    ...
```

Cover selectable insert/replace/remove/reorder cases whose projections are
identical, and reject:

- empty repaired output
- oversized repaired output
- missing, malformed, unknown, or flooded repaired markers
- punctuation, case, Unicode normalization, newline, tab, or general-space changes
- a provider fallback string that is not structurally valid repair output

The selection result must carry only a selected-body choice and a safe reason
code; it must not contain evidence, prompts, or exception objects.

- [ ] **Step 6: Run projection tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_repair.py \
  -k 'projection or selection or repaired'
```

Expected: failures because projection and selection functions are absent.

- [ ] **Step 7: Implement right-to-left projection and strict selection**

Add:

```python
@dataclass(frozen=True, slots=True)
class CitationRepairSelection:
    selected_body: str
    repaired: bool
    reason_code: str


def claim_preservation_projection(answer_body: str) -> str:
    ...


def select_repaired_body(
    initial_body: str,
    repaired_body: str,
    contract: CitationRepairContract,
) -> CitationRepairSelection:
    ...
```

Build ranges from the same repair token scan, walk them from right to left,
delete the exact token range, and additionally delete only the immediately
preceding U+0020 space. Compare projected strings by exact Python string
equality after validating both UTF-8 bounds; do not normalize or case-fold.

- [ ] **Step 8: Run projection tests and confirm they pass**

Run the Step 6 command.

Expected: all selected tests pass.

- [ ] **Step 9: Write failing prompt, byte-limit, and model-window tests**

Cover:

- fixed system/user message shape with evidence and initial answer treated as untrusted data
- exact evidence, answer, fixed-overhead, and total-request limits accepted
- each limit plus one rejected without trimming
- fixed literal instruction/delimiters fitting the 8-KiB overhead allocation
- response reservation `max(positive max_tokens, initial-answer token estimate)`
- absent, zero, negative, boolean, or malformed `max_tokens` falling back to 1024
- safety margin `max(512, resolved_window // 50)`
- exact equality to the model window accepted; one token over rejected
- token counter/window lookup exceptions or invalid windows failing closed

- [ ] **Step 10: Run prompt/window tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_repair.py \
  -k 'prompt or request or overhead or window or reservation'
```

Expected: failures because request construction and window checking are absent.

- [ ] **Step 11: Implement exact bounded request construction and window checking**

Add a fixed two-message request builder that returns `None` on any bound
failure. Its total counts only UTF-8 bytes from the canonical system and user
message `content` fields before provider adaptation.

Add a primitive-only window function:

```python
def repair_request_fits_model_window(
    messages: list[dict[str, str]],
    *,
    initial_answer: str,
    model: str,
    provider: str,
    max_tokens: int | None,
    count_fn: Callable[..., int] = count_console_messages_tokens,
    window_fn: Callable[[str, str], int] = get_model_token_limit,
) -> bool:
    ...
```

Never clamp the reservation and never trim the request.

- [ ] **Step 12: Run the complete pure repair file**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_citation_repair.py
```

Expected: all tests pass.

- [ ] **Step 13: Commit the pure repair contract**

```bash
git add \
  tldw_chatbook/Chat/citation_repair.py \
  Tests/Chat/test_citation_repair.py
git commit -m "feat(rag): define bounded citation repair contracts"
```

### Task 2: Carry repair eligibility independently through local capture

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py:65-79, 1439-1529, 1540-1707, 1712-1951`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:780-838, 3375-3411`
- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `Tests/UI/test_console_local_citation_capture.py`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`

- [ ] **Step 1: Add failing local-capture contract tests**

Extend `LocalRagContextResult` expectations to include
`citation_repair_contract`, defaulting to `None` so existing two- and
three-positional-argument fixtures remain valid.

Test:

- builder-ready local pipeline capture returns exact context, builder,
  prompt-set ID, and contract
- builder-unavailable capture still normalizes, authorizes, formats, and
  returns the exact context plus contract
- Console-staged evidence does the same when repository/key readiness yields no builder
- contract ordinals exactly equal `1..len(formatted.entries)`
- empty formatted evidence, authorization failure, unsupported source,
  unsupported namespace, or limits produce no contract
- a canonical builder recording failure degrades to exact context plus
  contract and no builder/prompt ID after successful normalization and
  authorization
- a legacy raw pipeline fallback may retain its existing context but never
  receives a repair contract
- contract-backed context is prepended only after chat-dictionary and
  world-info transforms, so those transforms cannot rewrite the exact evidence
  recorded in the contract
- the existing early-prepend ordering remains only for legacy/raw context with
  neither a repair contract nor a builder

- [ ] **Step 2: Run focused capture tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_local_citation_capture.py \
  -k 'repair_contract or builder_unavailable or canonical_capture_failure'
```

Expected: failures because `LocalRagContextResult` has no repair contract and
builder-unavailable paths do not construct one.

- [ ] **Step 3: Add the contract field and one safe formatted-evidence helper**

Update the result contract:

```python
@dataclass(frozen=True)
class LocalRagContextResult:
    context: str | None
    citation_builder: CitationTraceBuilder | None
    prompt_evidence_set_id: str | None = None
    citation_repair_contract: CitationRepairContract | None = None
```

Add one private helper that accepts only `LocalEvidenceContext`, returns `None`
for empty context/entries, and otherwise creates:

```python
CitationRepairContract(
    schema_version=1,
    marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
    allowed_ordinals=tuple(range(1, len(formatted.entries) + 1)),
    evidence_context=formatted.context,
)
```

Catch validation failure with one safe reason-code log; do not log the context.

- [ ] **Step 4: Refactor local and staged capture without weakening authorization**

For local pipeline capture, run normalization, current-authority checks, and
`format_local_evidence_context` whether or not a builder exists. Record
canonical run/prompt objects only when the builder exists. If safe formatting
succeeds but builder construction/recording is unavailable, return exact
formatted context plus the repair contract and clear builder identity.

For Console-staged capture, build the repair contract immediately after exact
authorized formatting, before the optional builder branch. Preserve existing
fail-closed behavior for normalization and authority failures.

Do not derive ordinals by reparsing aggregate context.

- [ ] **Step 5: Run focused capture tests and confirm they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 6: Add failing controller handoff tests**

In `Tests/UI/test_console_local_citation_capture.py`, assert the controller's
capture boundary returns the same validated contract object alongside context,
builder, and prompt-set ID. Malformed duck-typed objects must yield `None`
instead of entering repair.

- [ ] **Step 7: Run the handoff tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_local_citation_capture.py \
  -k 'controller and repair_contract'
```

Expected: failure because `_capture_rag_context` returns only three values.

- [ ] **Step 8: Extend the controller capture boundary**

Return a four-tuple from `_capture_rag_context` and accept only an actual
`CitationRepairContract`. Keep context, builder, prompt ID, and repair contract
independent; never synthesize a contract in the controller.

At submit, use this ordering rule:

```python
has_exact_citation_context = (
    citation_trace_builder is not None
    or citation_repair_contract is not None
)
```

- early-prepend only legacy/raw context when
  `has_exact_citation_context is False`
- apply chat dictionaries and world info
- late-prepend context when `has_exact_citation_context is True`

Add a controller test whose dictionary/world-info doubles rewrite existing
user content. Assert the final provider payload contains the repair contract's
evidence context byte-for-byte and that only the non-evidence user content was
transformed.

- [ ] **Step 9: Run all touched capture and ordering tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_local_citation_capture.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  -k 'LocalRagContextResult or local_citation or repair_contract or canonical_capture or exact_context_order'
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit local repair-contract capture**

```bash
git add \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_local_citation_capture.py \
  Tests/Chat/test_console_local_citation_boundary.py
git commit -m "feat(rag): carry citation repair eligibility"
```

### Task 3: Report synthesized provider fallback out of band

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:20-75, 918-1117`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:428-557, 966-1025, 1319-1327`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`

- [ ] **Step 1: Write failing gateway-signal tests**

Test an optional `ConsoleProviderStreamSignals` with:

- one private `threading.Event`
- read-only `synthetic_fallback_emitted`
- no content, evidence, provider, exception, or credential fields
- genuine provider content byte-for-byte equal to
  `NO_PROVIDER_CONTENT_COPY` or `UNSUPPORTED_PROVIDER_RESPONSE_COPY` leaving
  the event unset
- each gateway-authored fallback normalization branch setting the event before
  the first fallback chunk is observed
- tools-mode fallback suppression leaving it unset
- omitted signal preserving exact yielded item types/text
- signal repr and public state containing no governed text

- [ ] **Step 2: Run gateway-signal tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  -k 'stream_signal or synthetic_fallback or real_answer_equal'
```

Expected: failures because the signal contract and optional argument are absent.

- [ ] **Step 3: Implement the content-free signal and mark-at-source behavior**

Add:

```python
@dataclass(slots=True)
class ConsoleProviderStreamSignals:
    _synthetic_fallback: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )

    @property
    def synthetic_fallback_emitted(self) -> bool:
        return self._synthetic_fallback.is_set()

    def mark_synthetic_fallback(self) -> None:
        self._synthetic_fallback.set()
```

Thread optional `signals=None` through `stream_chat`,
`_stream_generic_chat`, and `normalize_provider_response`. Set the event in the
same branch immediately before yielding gateway-authored fallback. Do not
compare yielded text to fallback constants.

- [ ] **Step 4: Run gateway-signal tests and confirm they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing whole-agent-run propagation tests**

In `Tests/Chat/test_console_agent_bridge.py`, use a scripted multi-turn gateway
and assert:

- `run_reply(..., provider_stream_signals=signals)` passes the exact same
  object on every adapter gateway call
- the signal survives a primary tool-call turn and final-answer turn
- the signal survives subagent turns
- no adapter call resets an already-set event
- omitting the signal preserves existing bridge fake signatures/call behavior

- [ ] **Step 6: Run agent propagation tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_agent_bridge.py \
  -k 'provider_stream_signal or synthesized_fallback'
```

Expected: failures because `run_reply` and `_StreamingModelAdapter` do not
accept or forward the signal.

- [ ] **Step 7: Thread one signal through every adapter call**

Add optional `provider_stream_signals` to `ConsoleAgentBridge.run_reply` and
`_StreamingModelAdapter.__init__`. Store the same object on the one adapter
created for the run. In `_consume`, add the keyword only when non-`None`,
alongside the existing conditional `tools` keyword, so legacy bridge fakes
remain compatible when no signal is requested.

Never recreate or clear the signal between calls.

- [ ] **Step 8: Run gateway and bridge signal tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_agent_bridge.py \
  -k 'stream_signal or synthetic_fallback or provider_stream_signal or real_answer_equal'
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit fallback provenance**

```bash
git add \
  tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_agent_bridge.py
git commit -m "feat(console): report synthesized provider fallback"
```

### Task 4: Add independent provisional selection, terminal deferral, and atomic body selection

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:20-29, 276-306`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:280-318, 826-904, 1446-1619, 2053-2095, 2986-3009`
- Modify: `Tests/Chat/test_console_terminal_citation_persistence.py`

- [ ] **Step 1: Write failing store deferral tests**

Add tests for `append_message(..., defer_terminal_persistence=True)`:

- only an empty, attachment-free assistant placeholder accepts it
- the flag records logical provisional-selection eligibility when
  `persist=False` or no persistence backend exists
- a real persistence backend arms deferral even when canonical writes are
  disabled, the citation kwarg is absent, or no finalizer is supplied
- no persistence backend makes only persistence deferral a no-op; it does not
  remove provisional-selection eligibility
- a ready finalizer plus explicit deferral uses one deferral entry
- stream append, `get_message`, `messages_for_session`, and repeated
  materialization perform zero writes before terminal selection
- completion, failure, stop, close, delete, restore, and explicit cleanup
  release both provisional-selection and persistence-deferral state

- [ ] **Step 2: Run deferral tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'repair_deferral or builder_unavailable_deferral'
```

Expected: failures because `defer_terminal_persistence` is absent.

- [ ] **Step 3: Implement explicit deferral independently of finalizer readiness**

Add the keyword with default `False`. Validate its call shape separately from
the finalizer. Track `_provisional_terminal_selection_ids` independently of
`_terminal_persistence_deferred_ids`, and compute:

```python
arm_finalizer = (
    terminal_citation_finalizer is not None
    and persist
    and self._citation_persistence_ready()
)
arm_provisional_selection = defer_terminal_persistence
arm_terminal_deferral = (
    persist
    and self.persistence is not None
    and (defer_terminal_persistence or arm_finalizer)
)
```

Register the finalizer only when ready, but register one terminal deferral
whenever `arm_terminal_deferral` is true. Register provisional selection
whenever `arm_provisional_selection` is true, including in-memory mode. Keep
existing failure/cleanup behavior, release both sets on every terminal or
cleanup path, and do not install a no-op finalizer.

- [ ] **Step 4: Run deferral tests and confirm they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing atomic replacement and presentation tests**

Cover `replace_deferred_terminal_body(message_id, selected_body)`:

- accepts only a provisionally selection-eligible, attachment-free assistant
  in pending/streaming state, whether or not persistence is configured
- rejects empty, non-string, over-1-MiB, unknown, non-eligible, attached,
  non-assistant, and terminal messages
- synchronously sets `message.content`, stream buffer `[selected_body]`, and
  materialized count `1`
- does not persist or change status
- a poll immediately before/after sees initial or repaired content, never empty
  or partial replacement
- a no-persistence Console successfully replaces the same in-memory row and
  completes it without attempting a write

Add a transient presentation contract containing only:

```python
phase
notice_code
original_attempt_available
```

Test that store snapshots preserve it in memory, persistence calls never
receive it, and restore defaults it to `None`.

- [ ] **Step 6: Run replacement/presentation tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'replace_deferred or citation_presentation or atomic_repair'
```

Expected: failures because the operation and presentation model are absent.

- [ ] **Step 7: Implement atomic replacement and safe presentation mutation**

In `console_chat_models.py`, add string enums or Literals for the approved
phase/notice codes and a frozen `ConsoleCitationPresentation`; add an optional
field to `ConsoleChatMessage`.

In the store:

```python
def replace_deferred_terminal_body(
    self,
    message_id: str,
    selected_body: str,
) -> ConsoleChatMessage:
    ...

def set_citation_presentation(
    self,
    message_id: str,
    presentation: ConsoleCitationPresentation | None,
) -> ConsoleChatMessage:
    ...
```

Perform body replacement in one synchronous mutation and do not call
`_materialize_stream_buffer` afterward.

- [ ] **Step 8: Add one-write terminal outcome tests**

For valid-initial, repaired, unavailable, and canceled-repair simulations,
assert:

- no pre-selection write
- exactly one stable-ID assistant create on `mark_message_complete` when a
  persistence backend exists
- only the selected body reaches persistence
- the same successful repaired-body selection works with no persistence
  backend, updates the same row, and performs zero writes
- ordinary persistence still works when canonical finalization is unavailable
- a ready canonical finalizer remains fail-closed for marker-bearing selected
  bodies and falls back to the ordinary message write without a grounded trace

- [ ] **Step 9: Run the scoped store tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  -k 'repair_deferral or builder_unavailable_deferral or replace_deferred or citation_presentation or atomic_repair or one_terminal_write'
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit store selection primitives**

```bash
git add \
  tldw_chatbook/Chat/console_chat_models.py \
  tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_terminal_citation_persistence.py
git commit -m "feat(console): defer and atomically select repaired replies"
```

### Task 5: Orchestrate one repair on the direct-provider path

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:533-635, 780-884, 3572-3799, 4580-4687`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`

- [ ] **Step 1: Write failing request-session and direct-selection tests**

Add focused helpers for one repair-eligible initial send and scripted initial
and repair streams. Cover:

- one `ConsoleCitationRepairSession` is created only for an initial send with a
  valid contract
- the assistant placeholder receives explicit terminal deferral independent of
  builder/finalizer readiness
- valid initial markers complete with zero repair calls
- missing or invalid initial markers make exactly one second gateway call
- repair call reuses the exact `ConsoleProviderResolution` and therefore the
  same provider/model/sampling settings
- repair messages contain only fixed instructions, exact evidence, and exact
  initial body; no history, tools, skills, MCP, approvals, or agent bridge
- successful repair atomically replaces the same message
- an unavailable initial decision (oversized body or marker flood) makes zero
  repair calls and keeps the original with unavailable notice
- failed request fit, provider error, empty/oversized repair output, invalid
  markers, changed claims, or a second invalid repair keep the original with
  unavailable notice
- valid-initial and every failure outcome perform one terminal write
- retry/regenerate/edit-resend/continue paths never receive a repair session

- [ ] **Step 2: Run direct orchestration tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  -k 'citation_repair and direct'
```

Expected: failures because the controller has no repair session or selection
coordinator.

- [ ] **Step 3: Add the bounded controller-owned session**

Define a request-local mutable dataclass in `console_chat_controller.py`:

```python
@dataclass(slots=True)
class ConsoleCitationRepairSession:
    contract: CitationRepairContract
    resolution: ConsoleProviderResolution
    attempt_started: bool = False
    selection_committed: bool = False
    phase: str = "initial_streaming"
    cancel_reason: Literal["user", "session_close", "shutdown"] | None = None


@dataclass(frozen=True, slots=True)
class ConsoleCitationSelectionOutcome:
    selected_body: str
    state: Literal["bypassed", "valid", "repaired", "unavailable", "canceled"]
```

Do not store it on a message or session serialization model. Keep answer
bodies in local variables whenever possible; clear any session buffers at
terminal selection.

At submit:

1. receive the validated repair contract from `_capture_rag_context`
2. construct the session from that contract and the already-resolved provider
3. append the assistant placeholder with
   `defer_terminal_persistence=session is not None`
4. pass the session only into this initial `_stream_assistant_response`
5. create one `ConsoleProviderStreamSignals` for the repair-eligible request
   and reuse it for the initial direct stream and any repair stream

The append flag must be passed even when `persist=False`; the store uses it to
authorize logical same-row provisional selection independently of whether it
also has a terminal write to defer.

- [ ] **Step 4: Make the outer response coroutine own active state**

Set `_active_assistant_message_id`, `_active_stream_task`, and
`_stop_requested` once in `_stream_assistant_response` before the
direct/agent branch. Add
`self._active_citation_repair_session: ConsoleCitationRepairSession | None`,
bind the request's session beside those fields, and clear it only in that
outer method's `finally` when it is still the same object. This transient
controller reference is the sole inspection/mutation seam used by
`stop_active_run`, `close_session`, and `shutdown`; it is never serialized or
copied onto the message.

Remove active-run clearing from the direct nested block and `_run_agent_reply`.
Keep the agent's per-run `threading.Event` lifecycle in `_run_agent_reply`, but
do not let it replace or clear outer asyncio ownership. The nested agent
`finally` may detach its own `_active_cancel_event` when appropriate, but it
must not reset `_active_assistant_message_id`, `_active_stream_task`,
or `_stop_requested`.

On the direct initial stream, pass `signals=stream_signals` only when a repair
session exists. Plain, retry, regenerate, edit/resend, and continue paths omit
the optional keyword and retain their existing gateway-double compatibility.

Run the existing active-state tests named below after the refactor:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_chat_controller.py \
  -k 'active_stream or stop_active or session_closed'
```

Expected: existing selected tests pass before repair logic continues.

- [ ] **Step 5: Implement one shared async post-generation coordinator**

Add an async method used by both branches:

```python
async def _select_post_generation_body(
    self,
    *,
    assistant_message_id: str,
    repair_session: ConsoleCitationRepairSession,
    stream_signals: ConsoleProviderStreamSignals,
) -> CitationRepairSelectionOutcome:
    ...
```

It must:

1. call `store.get_message` and use that exact materialized body
2. bypass repair for an empty body or a marked synthesized-fallback signal
3. return immediately on structurally valid markers
4. on an `unavailable` structural decision, synchronously select the original
   with unavailable notice and make no provider call
5. on missing/invalid, transition the active session from
   `initial_streaming` to `checking`, set only safe checking presentation,
   and publish `ConsoleRunStatus.CHECKING_CITATIONS`
6. `await asyncio.sleep(0)` before repair dispatch
7. recheck message ownership, session existence, stop state, and request fit
8. set `attempt_started=True` and transition `checking` to `repair_streaming`
   immediately before one direct, tool-free gateway call
9. collect string chunks off-screen with incremental UTF-8 size accounting
10. discard non-string/tool-call output and late chunks after cancellation
11. call `select_repaired_body`
12. invoke `replace_deferred_terminal_body` only on successful repaired selection
13. synchronously set `phase="selected"` and `selection_committed=True` at
    the final body/notice commit

Check `stream_signals.synthetic_fallback_emitted` both before dispatch and
after repair collection so gateway-authored fallback during repair cannot be
selected.

- [ ] **Step 6: Complete only after selection**

On the direct success path, invoke the coordinator before
`mark_message_complete`. Then:

- complete the selected body exactly once
- set `COMPLETED` for valid/repaired/unavailable outcomes
- treat a canceled outcome as already completed/persisted by the coordinator
  and do not call `mark_message_complete` a second time
- keep existing failure behavior for initial provider failure/empty output
- preserve one-shot prefill consumption ordering
- return `completed.content`, never a repair buffer

- [ ] **Step 7: Run direct orchestration and ownership tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_chat_controller.py \
  -k 'citation_repair and direct or active_stream or stop_active'
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit direct repair orchestration**

```bash
git add \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): repair provisional direct RAG replies"
```

### Task 6: Unify agent selection and repair cancellation

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:1618-1691, 3572-4261, 4580-4687`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`
- Modify: `Tests/Chat/test_console_agent_swap.py`

- [ ] **Step 1: Write failing agent-selection tests**

Cover:

- only a genuine `RUN_DONE` reaches the shared async selection coordinator
- agent failure, runtime cancellation, missing placeholder, empty final text,
  and synthesized fallback retain existing terminal behavior and never repair
- a successful agent repair uses the direct gateway with no tools and never
  re-enters `ConsoleAgentBridge`
- the exact store body, not `outcome.final_text`, drives structural checking
- repair completes before `_complete_agent_message` and before
  `_record_run_assistant_message`
- the persisted run anchor points to the one selected assistant row
- fallback from any earlier primary tool turn, intermediate turn, or subagent
  call marks the shared signal and conservatively bypasses final-answer repair
- genuine provider text equal to fallback copy does not bypass repair

- [ ] **Step 2: Run agent-selection tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  -k 'citation_repair and agent'
```

Expected: failures because agent finalization is synchronous and terminalizes
before the shared coordinator.

- [ ] **Step 3: Move agent success across the async selection seam**

Pass the request signal to `ConsoleAgentBridge.run_reply`. Make the controller's
agent finalization path async only where required:

- classify cancellation/failure exactly as today
- for `RUN_DONE`, verify the existing placeholder and non-empty genuine output
- await `_select_post_generation_body`
- only then call `_complete_agent_message`, except for a canceled repair whose
  original was already completed/persisted by the coordinator
- only after persistence call `_record_run_assistant_message`

Do not use `outcome.final_text` as the selected body.

- [ ] **Step 4: Run agent-selection tests and confirm they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing checking/repair stop-race tests**

Use events to pause:

- immediately after `CHECKING_CITATIONS` is published but before dispatch
- during repair collection before any chunk
- after one repair chunk but before selection
- immediately before and immediately after synchronous selection commit

Assert:

- Send remains blocked and Stop remains enabled while checking
- the checking state is observable after one event-loop yield
- stop before dispatch makes zero repair calls
- stop during repair cancels collection and never calls
  `mark_message_stopped`
- cancellation selects the original, calls `mark_message_complete`, then sets
  Console run state `STOPPED`
- only after assistant persistence, one durable system row is appended with
  `Citation repair canceled by user.`
- shutdown (`record_user_stop=False`) and session close append no user-stop row
  and never recreate a closed session/message
- the system row parents after the persisted assistant
- late repair chunks cannot overwrite the original
- stop after `selection_committed` is a no-op
- initial-generation stop retains existing stopped-message behavior
- session close discards output, clears preview state, and never recreates a
  message/session

- [ ] **Step 6: Run stop-race tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_agent_swap.py \
  -k 'citation_repair and (stop or cancel or close or late)'
```

Expected: failures because `stop_active_run` currently terminalizes every
active message synchronously.

- [ ] **Step 7: Implement phase-aware cancellation and the selection linearization point**

Add `CHECKING_CITATIONS` to `ConsoleRunStatus`; make
`ConsoleRunState.is_stop_allowed` true for both streaming and checking.

While an uncommitted repair session is active, `stop_active_run` must:

1. first inspect `_active_citation_repair_session.phase`
2. use the special deferred-selection behavior only for phase `checking` or
   `repair_streaming`; phase `initial_streaming` retains the existing
   `_mark_stream_stopped` initial-generation behavior
3. signal `_stop_requested` and the agent cancel event
4. record `cancel_reason="user"` only when `record_user_stop=True`; shutdown
   records `"shutdown"` and `close_session` records `"session_close"`
5. avoid `_mark_stream_stopped`
6. avoid appending the ordinary stop row
7. cancel the outer task when needed

The coordinator catches repair-phase `CancelledError` only when its own stop
flag is set, commits the original synchronously, completes/persists it, appends
the phase-specific durable system row only for `cancel_reason="user"`, and
returns a stopped result. It rechecks message/session ownership first, so
session-close teardown cannot resurrect either object. Unrelated cancellation
still propagates.

Append the user-stop row with `persist=self.store.persistence is not None`
after `mark_message_complete` returns, so “durable” is literal and its
persisted parent is the selected assistant. Do not reuse the current
non-persisting default call shape.

At the synchronous selection commit, set `selection_committed=True`.
`stop_active_run` returns false/no-ops for that committed session even if the
outer coroutine is finishing bookkeeping.

- [ ] **Step 8: Run direct, agent, and cancellation lifecycle tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_chat_controller.py \
  -k 'citation_repair or active_stream or stop_active or session_closed'
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit shared agent and cancellation lifecycle**

```bash
git add \
  tldw_chatbook/Chat/console_chat_models.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): unify agent citation repair lifecycle"
```

### Task 7: Render honest notices and the transient original-attempt preview

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py:54-220, 250-374`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:95-305, 506-620, 1050-1580`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:435-447, 2030-2125, 10820-11055, 13139-13370, 13919-13955, 15330-15380`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:533-635, 1007-1030`
- Modify: `Tests/Chat/test_console_message_actions.py`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing action-service tests**

Test:

```python
service.available_actions(
    repaired_message,
    original_attempt_available=True,
)
```

offers one `view-original-attempt` action. Default calls, explicit `False`,
failed/non-assistant messages, `plain_action_labels`, and `plain_action_row`
must omit it. `dispatch` returns only the target message ID and safe copy, never
the original body.

- [ ] **Step 2: Run action tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_message_actions.py \
  -k 'original_attempt'
```

Expected: failure because the optional argument and action do not exist.

- [ ] **Step 3: Implement explicit action availability**

Add `original_attempt_available: bool = False` to `available_actions` and
insert `View original attempt` before regenerate only when true. Do not pass
the flag from plain/export helpers. Add a pure dispatch result for the action.

- [ ] **Step 4: Write failing notice and preview-LRU tests**

Controller tests must prove:

- successful repair stores only the original body in an `OrderedDict`-style
  LRU keyed by message ID
- the maximum is eight
- access refreshes recency
- the ninth insertion evicts the oldest and clears its availability flag
- unavailable/canceled/valid-initial messages store no preview
- close session, shutdown, delete, edit/resend, regenerate/replacement, and
  explicit clear remove entries
- restart/restore creates no entries

Transcript/screen tests must prove exact copy:

- checking/repairing: `Checking citations…`
- success with available original:
  `Citations repaired · View original attempt`
- success after eviction: `Citations repaired`
- failure: `Citation repair unavailable · Original response kept`
- cancellation: `Citation repair canceled`

Every case must assert no `grounded`, `verified`, `supported`, or canonical
association badge/copy appears.

- [ ] **Step 5: Run notice/LRU tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/UI/test_console_native_transcript.py \
  -k 'original_attempt or citation_notice or repaired_notice'
```

Expected: failures because the LRU and rendering are absent.

- [ ] **Step 6: Implement the controller LRU and safe notice rendering**

Controller methods:

```python
def original_attempt_for_message(self, message_id: str) -> str | None:
    ...

def clear_original_attempt(self, message_id: str) -> None:
    ...

def clear_original_attempts_for_session(self, session_id: str) -> None:
    ...
```

Only successful repair inserts. On eviction, update the message's safe
presentation flag if it still exists.

In transcript rendering, derive notice text exclusively from presentation
codes/booleans and include that presentation in message/action signature
tokens so the row updates without remounting unrelated rows.

- [ ] **Step 7: Write failing interactive preview tests**

Test mouse and Enter activation through the real button route:

- parser recognizes `view-original-attempt` before less-specific prefixes
- activation asks the controller for the body
- screen toggles an ephemeral per-message preview map
- transcript renders a distinct row/block labeled
  `Original attempt (not selected)`
- activating again hides it
- stale/evicted entries disappear during UI sync
- body is never assigned to `ConsoleChatMessage.content` or presentation

Assert selected message content, copy result, TTS event text, plain export,
save payload, and next provider history remain the repaired body before,
during, and after preview.

- [ ] **Step 8: Run interactive preview tests and confirm they fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  -k 'original_attempt or view_original'
```

Expected: failures because screen/transcript preview state and routing are absent.

- [ ] **Step 9: Implement screen-local preview rendering and cleanup**

Add a screen-side `dict[str, str]` for currently visible previews. Before each
transcript sync, remove keys no longer available from the controller and pass a
copy to the transcript through a dedicated setter.

Render preview as its own `_TranscriptRow` immediately after the owning message,
using literal `Content`/`Static` text with the fixed label. It must not modify
`_message_body`, message variants, or action-service dispatch content.

Call controller/screen cleanup from:

- edit/edit-resend callback before mutation
- regenerate/replacement before dispatch
- confirmed delete
- session close
- controller shutdown

Add `CHECKING_CITATIONS` to `CONSOLE_ACTIVE_RUN_STATUSES` and the transcript
jump-pill active copy so the 0.2-second sync continues while repair runs.

- [ ] **Step 10: Run action, transcript, and native-flow tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_message_actions.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  -k 'original_attempt or view_original or citation_notice or checking_citations'
```

Expected: all selected tests pass.

- [ ] **Step 11: Commit transient repair presentation**

```bash
git add \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): show citation repair transition"
```

### Task 8: Prove privacy, scoped compatibility, and TASK-553.15 closeout

**Files:**
- Modify: `Tests/Chat/test_citation_repair.py`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`
- Modify: `Tests/Chat/test_console_terminal_citation_persistence.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Docs/superpowers/specs/2026-07-26-local-citation-repair-transition-design.md`
- Modify: `backlog/tasks/task-553.15 - Add-provisional-citation-checking-and-one-visible-repair-transition.md`

- [ ] **Step 1: Add failing privacy-sentinel tests**

Use distinct sentinels for:

- initial body
- repaired body
- evidence
- source identity
- locator
- complete repair prompt
- provider exception

For request-fit failure, provider raise, empty output, oversized output, invalid
markers, changed claims, user cancellation, late chunk, session close, and
fallback bypass, capture stdlib and Loguru output and assert none of the
sentinels appears.

Also inspect:

- `ConsoleProviderStreamSignals`
- `ConsoleCitationPresentation`
- `ConsoleCitationRepairSession` after cleanup
- persisted message payload
- controller run-state copy

Only the selected message body may appear in its governed message persistence
payload; no diagnostic/presentation object may contain governed text.

- [ ] **Step 2: Run privacy tests and confirm any gaps fail**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_repair.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_provider_gateway.py \
  -k 'privacy or sentinel or governed_text'
```

Expected: new tests pass only after all unsafe exception/log/state paths are
removed. Fix production code, never weaken sentinel assertions.

- [ ] **Step 3: Run the complete new pure test file**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_citation_repair.py
```

Expected: all tests pass.

- [ ] **Step 4: Run only focused touched-code regression selections**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_local_citation_capture.py \
  -k 'repair_contract or canonical_capture or builder_unavailable'
```

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_agent_bridge.py \
  -k 'stream_signal or synthetic_fallback or provider_stream_signal or real_answer_equal'
```

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_chat_controller.py \
  -k 'citation_repair or repair_deferral or replace_deferred or citation_presentation or active_stream or stop_active'
```

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_message_actions.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py \
  -k 'original_attempt or view_original or citation_notice or checking_citations'
```

Expected: every selected test passes. Do not broaden these commands to parent
directories or the full repository. If an unrelated baseline failure is
encountered despite the filters, record its exact node ID in a separate
Backlog task and keep TASK-553.15 scoped.

- [ ] **Step 5: Run touched-file static checks**

Run Ruff only over touched Python files:

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Chat/citation_repair.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Chat/console_chat_models.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_citation_repair.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_local_citation_capture.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_local_citation_boundary.py \
  Tests/Chat/test_console_agent_swap.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_native_chat_flow.py
```

```bash
../../.venv/bin/ruff format --check \
  tldw_chatbook/Chat/citation_repair.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Chat/console_chat_models.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_citation_repair.py
```

Expected: zero Ruff errors and no formatting drift in the named files.

- [ ] **Step 6: Perform focused self-review**

Inspect:

```bash
git diff --check
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Chat/citation_repair.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/UI/Screens/chat_screen.py
```

Review specifically for:

- a second repair attempt
- non-marker text normalization
- integer parsing of marker digits
- duplicated Markdown traversal
- early persistence/materialization
- nested active-run cleanup
- repair re-entering the agent or advertising tools
- string-equality fallback detection
- a stop path calling `mark_message_stopped` during repair
- original/evidence/prompt text in presentation/logs
- preview affecting content/history/export/TTS
- grounded/support claims
- unrelated refactors

- [ ] **Step 7: Update approved spec status and task closeout**

Change the spec status to `Approved and implemented`.

In TASK-553.15:

1. link this implementation plan
2. preserve the ADR check (`ADR required: no`, ADR-024 path, reason)
3. check all six acceptance criteria only after their mapped tests pass
4. add concise `## Implementation Notes` covering approach, trade-offs,
   modified files, privacy/cancellation behavior, and exact scoped verification
5. set Done only after every Definition-of-Done requirement is satisfied

Use Backlog CLI for status:

```bash
backlog task edit 553.15 -s Done
```

- [ ] **Step 8: Verify TASK-553.15 closeout**

Run:

```bash
backlog task 553.15 --plain
git diff --check
git status --short
```

Expected:

- status `Done`
- all acceptance criteria checked
- Implementation Plan and Implementation Notes present
- ADR-024 linked with no new ADR
- scoped verification recorded
- no unintended uncommitted changes

- [ ] **Step 9: Commit implementation notes and closeout**

```bash
git add \
  Docs/superpowers/specs/2026-07-26-local-citation-repair-transition-design.md \
  "backlog/tasks/task-553.15 - Add-provisional-citation-checking-and-one-visible-repair-transition.md"
git commit -m "docs(rag): record citation repair transition delivery"
```

## Acceptance-criteria traceability

| Acceptance criterion | Planned proof |
| --- | --- |
| AC1: one provisional message through selection | Tasks 4-6 store deferral, shared coordinator, same-row atomic replacement, and direct/agent lifecycle tests |
| AC2: valid skips; missing/invalid repairs at most once with same provider/model | Tasks 1, 5, and 6 decision/request tests and exact resolution identity assertions |
| AC3: unchanged claims and honest original fallback/cancellation | Tasks 1, 5, and 6 projection, repaired-selection, failure, and stop-race tests |
| AC4: visible same-message repair and keyboard-accessible current-session original preview | Task 7 action, transcript, screen, LRU, cleanup, and non-mutation tests |
| AC5: independent readiness plus bounded/private state | Tasks 1-4 contract limits, builder-unavailable capture, independent deferral, signal privacy, and Task 8 sentinels |
| AC6: direct/agent stop and session-close compatibility | Tasks 5-6 outer ownership, agent seam, cancellation linearization, late-chunk, close, and scoped regression tests |
