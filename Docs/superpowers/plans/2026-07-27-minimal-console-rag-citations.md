# Minimal Console RAG Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the `[S#]` mappings for the selected local RAG answer, show a `Sources (N)` footer beneath persisted cited Console answers, display the exact cited chunks in one modal, and open supported originals through Library's existing exact-ID path.

**Architecture:** Extend the existing citation trace builder to retain eligible marker occurrences, then read active traces through one repository-owned helper that keeps the fingerprint codec private. Console caches only deduplicated source counts until the user opens a modal; the modal performs authorized all-or-nothing hydration and revalidation. It passes a bounded source type/ID pair to Library, which delegates to its existing `_open_library_item_by_id()` method.

**Tech Stack:** Python 3.11+, Textual, Pydantic citation models, SQLite, pytest, Ruff

---

## Scope and references

- Backlog task: `TASK-553.16`
- Approved design: `Docs/superpowers/specs/2026-07-27-citation-evidence-inspector-design.md`
- Existing architecture: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`
- Delivery: one task, one PR, scoped verification only

ADR required: no

ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

Reason: this work reuses the existing citation trace, authorization, hydration, and Library navigation contracts without changing storage, ownership, security, or service boundaries.

## Guardrails

- Do not add tables, migrations, trace formats, policy versions, or a second citation model.
- Do not add fact/claim/semantic-support storage, resolver registries, event buses, or `tldw_server` work.
- Do not hydrate chunk bodies during transcript composition or footer discovery.
- Do not preflight Library items or add a return journey.
- Do not repair unrelated baseline failures. Record them separately if a scoped command exposes one.

Run every Python command below from this worktree with the repository's Python
3.12 environment at `../../.venv/bin/python`. Do not use the system Python 3.9.

Before the first pytest command in a clean environment, create these two
temporary guard files with `apply_patch` (not in the repository):

```text
/tmp/tldw_chatbook_task55316_no_mlx/parakeet_mlx.py
/tmp/tldw_chatbook_task55316_no_mlx/lightning_whisper_mlx.py
```

Each file contains exactly:

```python
raise ImportError("optional MLX backend disabled for scoped headless tests")
```

Prefix every pytest command with
`PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx`. This is the established
headless collection guard for the separately tracked TASK-839 optional-MLX
baseline; it prevents unrelated MLX initialization without modifying
application code or this task's test scope. The guard was validated with a
representative citation/repository/Console `--collect-only` run.

### Task 1: Persist selected-answer marker occurrences

**Files:**

- Modify: `tldw_chatbook/Chat/citation_trace_builder.py`
- Modify: `Tests/Chat/test_citation_trace_builder.py`
- Modify: `Tests/Chat/test_console_terminal_citation_persistence.py`

- [ ] **Step 1: Replace the marker-rejection test with occurrence-mapping tests**

In `Tests/Chat/test_citation_trace_builder.py`, replace the current
`test_initial_answer_with_eligible_markers_is_unavailable_and_atomic` coverage
with `test_initial_answer_records_known_repeated_and_unknown_occurrences` and
`test_initial_answer_occurrence_overflow_is_atomic`, proving:

- `[S1]` and repeated `[S1]` occurrences are both retained with exact offsets;
- a known marker maps to its prompt entry's `evidence_ordinal` and is `VALID`;
- an unknown marker such as `[S99]` has no evidence ordinal and is
  `UNKNOWN_MARKER`;
- escaped markers and markers inside Markdown code remain excluded;
- more than `CITATION_OCCURRENCES_MAX` eligible markers fails atomically.

Representative assertions:

```python
attempt = builder.answer_attempts[0]
assert [item.raw_marker for item in attempt.occurrences] == [
    "[S1]",
    "[S1]",
    "[S99]",
]
assert [item.evidence_ordinal for item in attempt.occurrences] == [1, 1, None]
assert [item.structural_state for item in attempt.occurrences] == [
    StructuralValidationState.VALID,
    StructuralValidationState.VALID,
    StructuralValidationState.UNKNOWN_MARKER,
]
```

- [ ] **Step 2: Add a terminal-persistence restart regression**

In `Tests/Chat/test_console_terminal_citation_persistence.py`, add
`test_terminal_selected_answer_citations_survive_restart`. Use the real
SQLite/repository stack to finalize and persist an answer containing known,
repeated, and unknown markers. Close/reopen the DB and assert the active trace's
selected attempt still contains every eligible occurrence and only the known
ones resolve to evidence.

The test must use the final body passed to the terminal citation finalizer, not
an earlier streamed draft.

- [ ] **Step 3: Run the focused tests and confirm the expected failure**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_trace_builder.py::test_initial_answer_records_known_repeated_and_unknown_occurrences \
  Tests/Chat/test_citation_trace_builder.py::test_initial_answer_occurrence_overflow_is_atomic \
  Tests/Chat/test_citation_trace_builder.py::test_marker_free_eligibility_ignores_markdown_code_and_escaped_literals \
  Tests/Chat/test_console_terminal_citation_persistence.py::test_terminal_selected_answer_citations_survive_restart
```

Expected: the new marker tests fail because
`record_initial_answer_attempt()` still rejects eligible markers.

- [ ] **Step 4: Map all eligible spans in the existing builder method**

In `CitationTraceBuilder.record_initial_answer_attempt()`:

1. Parse with `eligible_citation_marker_spans(..., max_count=CITATION_OCCURRENCES_MAX)`.
2. Build one `marker_ordinal -> evidence_ordinal` mapping from the referenced
   final `PromptEvidenceSet`.
3. Create one `CitationOccurrence` per span in source order.
4. Use `VALID` for known ordinals and `UNKNOWN_MARKER` for unknown ordinals.
5. Keep payload/attempt mutation after all validation and governed-payload
   capacity checks so failures remain atomic.

Core shape:

```python
evidence_by_marker = {
    entry.marker_ordinal: entry.evidence_ordinal
    for entry in prompt_set.entries
}
occurrences = tuple(
    CitationOccurrence(
        occurrence_id=new_opaque_id("citation-occurrence"),
        occurrence_ordinal=index,
        raw_marker=span.raw_marker,
        marker_namespace=prompt_set.marker_namespace,
        evidence_ordinal=evidence_by_marker.get(span.marker_ordinal),
        marker_start=span.marker_start,
        marker_end=span.marker_end,
        structural_state=(
            StructuralValidationState.VALID
            if span.marker_ordinal in evidence_by_marker
            else StructuralValidationState.UNKNOWN_MARKER
        ),
    )
    for index, span in enumerate(marker_spans, start=1)
)
```

Do not add another attempt type or semantic-support calculation. The existing
terminal finalizer already supplies the body that will be persisted.

- [ ] **Step 5: Run the focused tests**

Run the same four-node pytest command.

Expected: PASS.

- [ ] **Step 6: Commit the pipeline change**

```bash
git add \
  tldw_chatbook/Chat/citation_trace_builder.py \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_console_terminal_citation_persistence.py
git commit -m "feat: persist local RAG citation markers"
```

### Task 2: Add a repository-owned active-trace lookup

**Files:**

- Modify: `tldw_chatbook/Chat/citation_trace_repository.py`
- Modify: `Tests/Chat/test_citation_trace_repository.py`

- [ ] **Step 1: Write helper contract tests**

Add
`test_active_trace_for_current_message_uses_persisted_revision` and
`test_active_trace_for_current_message_rejects_missing_or_mismatched_body` for
a public narrow helper named
`get_active_trace_for_current_message(message_id, current_body)`:

- it reads the non-deleted message's current `version` internally;
- it returns the same issued active result as
  `get_active_trace_for_message()` for a matching body;
- it returns `NOT_FOUND` for a missing/deleted message;
- it returns `UNVERIFIABLE` when the persisted body differs;
- callers do not supply or receive the repository's fingerprint codec;
- the returned active result still passes `verify_active_trace_result()`.

- [ ] **Step 2: Run the repository tests and confirm the expected failure**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_uses_persisted_revision \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_rejects_missing_or_mismatched_body \
  Tests/Chat/test_citation_trace_repository.py::test_active_lookup_verifies_body_and_preserves_historical_summary_on_mismatch \
  Tests/Chat/test_citation_trace_repository.py::test_active_result_requires_repository_issuance_and_exact_object_verification
```

Expected: FAIL because the new helper does not exist.

- [ ] **Step 3: Implement the narrow wrapper**

Query only `messages.version` and `messages.content` for the exact non-deleted
message ID. Return bounded inactive states for missing or mismatched rows, then
delegate to the existing method with `self._fingerprint_codec`:

```python
return self.get_active_trace_for_message(
    message_id,
    int(row["version"]),
    current_body,
    self._fingerprint_codec,
)
```

Do not expose a codec property and do not duplicate the existing owner,
fingerprint, visibility, or repository-issuance checks.

- [ ] **Step 4: Run the repository tests**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_uses_persisted_revision \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_rejects_missing_or_mismatched_body \
  Tests/Chat/test_citation_trace_repository.py::test_active_lookup_verifies_body_and_preserves_historical_summary_on_mismatch \
  Tests/Chat/test_citation_trace_repository.py::test_active_result_requires_repository_issuance_and_exact_object_verification
```

Expected: PASS.

- [ ] **Step 5: Commit the repository helper**

```bash
git add \
  tldw_chatbook/Chat/citation_trace_repository.py \
  Tests/Chat/test_citation_trace_repository.py
git commit -m "feat: look up active citation trace by message body"
```

### Task 3: Discover footer counts and render `Sources (N)`

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Create: `tldw_chatbook/Widgets/Console/console_citation_sources_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_citation_sources.py`
- Modify: `Tests/UI/test_console_transcript_selection_contract.py`

- [ ] **Step 1: Add pure count and transcript-row tests**

Create `Tests/UI/test_console_citation_sources.py` with a small trace fixture
proving that selected-attempt evidence ordinals:

- include only `StructuralValidationState.VALID`;
- deduplicate repeated evidence ordinals;
- retain first-citation order;
- ignore unknown markers and legacy selected attempts with no occurrences.

In `Tests/UI/test_console_transcript_selection_contract.py`, assert:

- `set_citation_counts({"assistant-native-id": 2})` adds one focusable
  `Sources (2)` row directly after that assistant message;
- zero/absent counts render no row;
- the footer belongs to `PROTECTED_CLICK_CLASSES` so activating it does not
  change or clear message selection;
- updating only the count reconciles that row without rebuilding unrelated
  transcript message rows.

- [ ] **Step 2: Add ChatScreen discovery tests**

Use a fake repository to prove:

- only complete assistant messages with a persisted ID are queried;
- streaming, pending, user, uncited, stale, and body-mismatched messages have
  no count;
- footer discovery receives only message ID/body and never a codec;
- repeating the 0.2-second sync with the same message signature does not
  dispatch duplicate discovery work;
- a late worker result is discarded after message ID, body, or request
  generation changes;
- the transcript cache contains only integer counts, never chunk text.

- [ ] **Step 3: Run the focused UI tests and confirm the expected failures**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_console_transcript_selection_contract.py
```

Expected: FAIL because citation count discovery and footer rows do not exist.

- [ ] **Step 4: Add the minimal selected-attempt ordinal reducer**

Create `console_citation_sources_modal.py` with only the pure reducer needed by
this slice. It accepts a `CitationTrace`, locates `selected_attempt_id`, and
returns distinct valid `evidence_ordinal` values in occurrence order. Task 4
will add the display-row and `ModalScreen` code to this same module. The reducer
must not infer markers from answer text or inspect semantic support.

- [ ] **Step 5: Add screen-owned footer discovery**

In `ChatScreen`, add:

- a `dict[native_message_id, int]` count cache;
- an input signature made from native ID, persisted ID, current body, and
  status for eligible messages;
- a monotonically increasing request generation;
- one exclusive Textual worker group for changed signatures.

The worker calls
`repository.get_active_trace_for_current_message(persisted_id, current_body)`,
requires `ACTIVE` plus a verified active result, computes the deduplicated
count from non-governed trace metadata, and applies results only if the
captured signature and generation still match.

Call `transcript.set_citation_counts(...)` from the existing native transcript
sync path. Fold the sorted count mapping into `refresh_key` so an async count
arrival refreshes rows even when messages are otherwise unchanged.

- [ ] **Step 6: Add one transcript row kind**

In `ConsoleTranscript`:

- store counts in `_citation_counts`;
- add `set_citation_counts()`;
- add a `"citations"` `_TranscriptRow` directly after each qualifying
  `"message"` row;
- build one small `Button` with label `Sources (N)`, a stable ID/class, and the
  native message ID attached as an attribute;
- include the protected footer class in `PROTECTED_CLICK_CLASSES`.

Do not call the repository from `compose()`, `_transcript_rows()`, or
`_build_row_widget()`.

- [ ] **Step 7: Run the focused UI tests**

Run the same two-file pytest command.

Expected: PASS.

- [ ] **Step 8: Commit the footer slice**

```bash
git add \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/Widgets/Console/console_citation_sources_modal.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_console_transcript_selection_contract.py
git commit -m "feat: show cited source counts in Console"
```

### Task 4: Hydrate exact chunks in one Sources modal

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_citation_sources_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_console_citation_sources.py`

- [ ] **Step 1: Add modal transformation and security tests**

Extend the focused test file to prove:

- only the selected attempt's valid, referenced evidence entries appear;
- repeated evidence appears once in first-citation order;
- title, exact `snapshot_text`, `source_kind`, and `source_id` come from the
  hydrated snapshot payload referenced by the final prompt set;
- unsupported source kinds retain visible chunk text but have no Open action;
- non-string, empty, or over-limit `source_kind`/`source_id` values fail
  closed and never produce an Open action;
- literal chunk text such as `[link](https://example.invalid)`, Rich tags, and
  ANSI-looking content is rendered as text rather than markup or links.

- [ ] **Step 2: Add modal loading-contract tests**

With a fake repository and a Textual pilot, assert:

- opening the footer creates the same `ModalScreen` at narrow and wide sizes;
- no `hydrate_trace()` call occurs before the footer is activated;
- authorization is constructed from `repository.identity_context` with only
  `view_snapshot` and `view_source_identity` true;
- `verify_active_trace_result()` is called on the same active result after
  hydration and before rows are applied;
- denied/unavailable hydration shows one `Sources unavailable` state;
- a dismissed modal or changed message body discards late hydration results.

- [ ] **Step 3: Run the focused test file and confirm the expected failures**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q Tests/UI/test_console_citation_sources.py
```

Expected: FAIL because the modal and lazy hydration path do not exist.

- [ ] **Step 4: Implement the modal and row builder in one module**

Extend the module created in Task 3 with:

- an immutable display-row dataclass;
- the valid-evidence-ordinal reducer shared with footer counts;
- a pure hydrated-result-to-display-rows function;
- one `ConsoleCitationSourcesModal`.

The row builder joins:

```text
selected AnswerAttempt
  -> its PromptEvidenceSet
  -> PromptEvidenceEntry.snapshot_payload_ref
  -> GovernedCitationPayloads.evidence_snapshot_payloads
```

Reject incomplete joins as one unavailable result. Use the static open mapping
only:

```python
OPEN_SOURCE_TYPES = {
    "media_db": "media",
    "notes": "notes",
    "chat_history": "conversations",
}
```

Treat hydrated `source_identity` as untrusted JSON: an Open action requires
`source_kind` and `source_id` to be bounded, non-empty strings before the kind
is mapped. Invalid identity disables Open without interpreting or coercing the
stored values.

Render exact chunks with `Static(Text(snapshot_text), markup=False)` (or an
equivalent literal Rich `Text` path). Do not use Markdown widgets and do not
auto-link.

- [ ] **Step 5: Implement lazy, revalidated hydration**

When the footer is activated, pass only the native/persisted message identity,
current body, and repository to the modal. Its worker:

1. calls `get_active_trace_for_current_message()`;
2. requires an issued active result;
3. builds `CitationReadAuthorization` from `repository.identity_context` with
   `AuthorityScope.LOCAL_PROFILE`, matching profile/governance scope, the one
   local authority, and only the two read flags;
4. calls `hydrate_trace(active.summary.namespace, authorization=...)`;
5. calls `verify_active_trace_result(active)` again;
6. applies rows only if the modal is still mounted and its request generation
   is current.

No governed payload is retained by `ChatScreen` or `ConsoleTranscript`.

- [ ] **Step 6: Add styles and regenerate the committed CSS bundle**

Add only the modal/list/detail/footer rules needed for a usable scrollable
layout to `css/components/_agentic_terminal.tcss`, then run:

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

Expected: both commands succeed and the generated modular bundle matches its
sources.

- [ ] **Step 7: Run the modal tests**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q Tests/UI/test_console_citation_sources.py
```

Expected: PASS.

- [ ] **Step 8: Commit the modal slice**

```bash
git add \
  tldw_chatbook/Widgets/Console/console_citation_sources_modal.py \
  tldw_chatbook/Widgets/Console/__init__.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_citation_sources.py
git commit -m "feat: inspect exact cited chunks in Console"
```

### Task 5: Open supported citation sources through Library

**Files:**

- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/Widgets/Console/console_citation_sources_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_console_citation_sources.py`
- Modify: `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Add bounded navigation-context tests**

In `Tests/UI/test_library_shell.py`, cover two new context keys:

```python
LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE = "open_source_type"
LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID = "open_source_id"
```

Assert:

- only `media`, `notes`, and `conversations` are accepted;
- type and ID must both be bounded non-empty strings;
- valid context calls the existing `_open_library_item_by_id()` exactly once;
- pre-mount context is deferred and opened once on mount;
- mounted dirty-note handling still flushes/vetoes through the existing
  `apply_navigation_context()` path;
- invalid, unsupported, or incomplete context is a no-op.

- [ ] **Step 2: Add Console action tests**

In `Tests/UI/test_console_citation_sources.py`, activate Open for one media,
note, and conversation row and assert the modal dismisses/navigates with only:

```python
{
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: expected_type,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: source_id,
}
```

Assert unsupported rows render no Open button. Do not add a Library existence
preflight.

- [ ] **Step 3: Run the focused navigation tests and confirm failure**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_library_shell.py \
  -k "citation or open_source"
```

Expected: FAIL because the navigation context is not implemented.

- [ ] **Step 4: Wire the modal action to existing screen navigation**

The modal returns or posts the mapped source type/ID. `ChatScreen` sends a
normal `NavigateToScreen("library", context)` using only the two constants.
Do not add source resolution or item lookup to Console.

- [ ] **Step 5: Validate and defer the exact-ID open in Library**

In `LibraryScreen._apply_navigation_context_state()`:

- reject non-string values, then validate both strings through the existing
  bounded `_safe_text()` helper;
- accept only `media`, `notes`, or `conversations`;
- retain at most one pending `(source_type, source_id)` pair when pre-mount;
- when mounted, schedule a small async method that clears the pending pair and
  awaits `_open_library_item_by_id(source_type, source_id)`;
- in `on_mount()`, schedule the same method once after normal setup.

Library's existing opener owns missing-item warnings. Do not duplicate its
media/note/conversation state transitions.

- [ ] **Step 6: Run focused navigation tests**

Run the same two-file pytest command.

Expected: PASS.

- [ ] **Step 7: Run the complete touched-scope verification**

Run:

```bash
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_citation_trace_builder.py::test_initial_answer_records_known_repeated_and_unknown_occurrences \
  Tests/Chat/test_citation_trace_builder.py::test_initial_answer_occurrence_overflow_is_atomic \
  Tests/Chat/test_citation_trace_builder.py::test_marker_free_eligibility_ignores_markdown_code_and_escaped_literals \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_uses_persisted_revision \
  Tests/Chat/test_citation_trace_repository.py::test_active_trace_for_current_message_rejects_missing_or_mismatched_body \
  Tests/Chat/test_citation_trace_repository.py::test_active_lookup_verifies_body_and_preserves_historical_summary_on_mismatch \
  Tests/Chat/test_citation_trace_repository.py::test_active_result_requires_repository_issuance_and_exact_object_verification \
  Tests/Chat/test_console_terminal_citation_persistence.py::test_terminal_selected_answer_citations_survive_restart \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_console_transcript_selection_contract.py
PYTHONPATH=/tmp/tldw_chatbook_task55316_no_mlx ../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  -k "citation_source or open_source"
```

Then run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/citation_trace_builder.py \
  tldw_chatbook/Chat/citation_trace_repository.py \
  tldw_chatbook/Widgets/Console/console_transcript.py \
  tldw_chatbook/Widgets/Console/console_citation_sources_modal.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Constants.py \
  Tests/Chat/test_citation_trace_builder.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_console_transcript_selection_contract.py \
  Tests/UI/test_library_shell.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

Expected: all scoped commands pass. If an unrelated baseline failure appears,
record it separately and do not broaden this task.

- [ ] **Step 8: Update task completion records**

After implementation and verification:

- mark all `TASK-553.16` acceptance criteria complete;
- add concise Implementation Notes listing the pipeline, footer, modal, and
  Library changes plus the scoped verification results;
- retain the ADR-required-no decision and ADR-024 reference;
- set the task to Done only after the Definition of Done is satisfied.

- [ ] **Step 9: Commit the navigation and task completion**

```bash
git add \
  tldw_chatbook/Constants.py \
  tldw_chatbook/Widgets/Console/console_citation_sources_modal.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_console_citation_sources.py \
  Tests/UI/test_library_shell.py \
  "backlog/tasks/task-553.16 - Add-minimal-RAG-citation-Sources-modal-to-Console.md"
git commit -m "feat: open cited sources in Library"
```

## Completion checkpoint

Before creating the PR, review the diff against the approved design and delete
any code that introduces a second provenance model, speculative source
abstraction, responsive rehosting, server dependency, or unrelated repair.
The finished behavior should remain:

```text
persist selected [S#] mappings
  -> Sources (N)
  -> one lazy hydrated modal
  -> existing Library exact-ID open
```
