# Console Context-Cost Evidence Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Console context and cost estimates use the same canonical local-evidence normalization, prompt framing, and 64-entry cap as the send path while remaining zero-I/O.

**Architecture:** Add one pure Console-reference adapter and one pure Console prompt-formatting wrapper to the existing local citation boundary. Reuse both from the send adapter and from a private Console display-state formatter, then derive estimated text and count from that one formatted result. Preserve the send path's asynchronous authority check and document that estimate-time results are pre-authority.

**Tech Stack:** Python 3.11+, Pydantic evidence models, pytest, Textual Console display state

**Design:** `Docs/superpowers/specs/2026-08-27-console-context-cost-evidence-parity-design.md`

**ADR required:** no  
**ADR path:** N/A  
**Reason:** Routine parity bug fix reusing existing normalization, authority, and formatting boundaries.

---

### Task 1: Pin canonical estimate behavior with failing tests

**Files:**
- Modify: `Tests/UI/test_console_staged_evidence_strip.py:44-115,183-220`
- Modify: `Tests/RAG/test_local_citation_capture.py:1220-1485`

- [ ] **Step 1: Extend the local evidence test builder only where needed**

Allow `_reference()` to accept optional `source_type`, `snippet`, `score`, and `metadata` values so the tests can express empty content, rejected source kinds, score fallback, and chunk lineage without constructing ad-hoc payload dictionaries.

- [ ] **Step 2: Write failing public-estimator tests**

Add focused tests equivalent to:

```python
def test_prompted_evidence_uses_canonical_headers_and_separators() -> None:
    launch = _mixed_launch()
    assert console_prompted_evidence_text(launch) == (
        "[S1] MEDIA — Source 1\nBody 1\n---\n"
        "[S2] MEDIA — Source 3\nBody 3"
    )
    assert console_prompted_source_count(launch) == 2


def test_prompted_evidence_applies_the_send_cap() -> None:
    launch = _launch(65)
    text = console_prompted_evidence_text(launch)
    assert console_prompted_source_count(launch) == 64
    assert "[S64] MEDIA — Source 64" in text
    assert "Source 65" not in text
    assert text.count("\n---\n") == 63


def test_prompted_evidence_keeps_empty_local_content_as_header_only() -> None:
    launch = _launch_from_references(
        _reference(1, snippet=""),
    )
    assert console_prompted_evidence_text(launch) == "[S1] MEDIA — Source 1\n"
    assert console_prompted_source_count(launch) == 1
```

- [ ] **Step 3: Write a failing adapter-contract test**

Import the proposed `normalize_console_evidence_references()` helper and verify that a valid chunked reference, an invalid source-kind reference, and a second valid reference produce two normalized candidates with ranks `(1, 2)`, preserve `source_id`, select the non-empty `chunk_id` as result lineage, and map `score=None` to `0.0`.

- [ ] **Step 4: Run the RED tests**

Run:

```bash
../../.venv/bin/pytest -q \
  Tests/UI/test_console_staged_evidence_strip.py \
  -k "prompted_evidence or prompted_source_count"
../../.venv/bin/pytest -q \
  Tests/RAG/test_local_citation_capture.py \
  -k "console_evidence_reference_adapter"
```

Expected: failures show raw snippet joining, a count of 65, empty-content omission, and the missing adapter helper.

---

### Task 2: Extract the shared normalization and formatting boundary

**Files:**
- Modify: `tldw_chatbook/RAG_Search/local_citation_capture.py:1-40,294-480`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py:40-55,1690-1760`
- Test: `Tests/RAG/test_local_citation_capture.py`

- [ ] **Step 1: Add the minimal pure Console-reference adapter**

Implement the existing send mapping once:

```python
def normalize_console_evidence_references(
    references: Sequence[EvidenceReference],
) -> tuple[NormalizedLocalResult, ...]:
    normalized: list[NormalizedLocalResult] = []
    for reference in references:
        if reference.source_owner.strip().lower() != "local":
            continue
        metadata: dict[str, Any] = {
            "source_type": reference.source_type,
            "source_id": reference.source_id,
        }
        chunk_id = reference.metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            metadata["chunk_id"] = chunk_id
        try:
            normalized.append(
                normalize_local_result(
                    {
                        "source": reference.source_type,
                        "id": chunk_id if isinstance(chunk_id, str) and chunk_id else reference.source_id,
                        "title": reference.title,
                        "content": reference.snippet,
                        "score": reference.score if reference.score is not None else 0.0,
                        "metadata": metadata,
                    },
                    candidate_rank=len(normalized) + 1,
                )
            )
        except LocalResultNormalizationError:
            continue
    return tuple(normalized)
```

- [ ] **Step 2: Add the shared Console formatting wrapper**

```python
def format_console_evidence_context(
    normalized_results: Sequence[NormalizedLocalResult],
) -> LocalEvidenceContext:
    return format_local_evidence_context(
        normalized_results,
        max_length=sum(
            len(candidate.title) + len(candidate.content) + 32
            for candidate in normalized_results
        ),
    )
```

Export both helpers through `__all__`. Do not change the generic formatter's 90-character default or other RAG callers.

- [ ] **Step 3: Route the send adapter through the shared helpers**

Replace only the duplicated Console reference loop and allowance expression. Preserve logging with:

```python
references = bundle.available_references()
normalized = normalize_console_evidence_references(references)
rejected_count = len(references) - len(normalized)
```

After the existing authority check, call `format_console_evidence_context(authorization.candidates)`. Leave all authority, builder, repair-contract, and error behavior unchanged.

- [ ] **Step 4: Run the adapter tests GREEN**

Run:

```bash
../../.venv/bin/pytest -q Tests/RAG/test_local_citation_capture.py \
  -k "console_evidence_reference_adapter or console_staged_local_evidence"
```

Expected: PASS.

- [ ] **Step 5: Commit the shared boundary**

```bash
git add tldw_chatbook/RAG_Search/local_citation_capture.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  Tests/RAG/test_local_citation_capture.py
git commit -m "refactor(console): share staged evidence formatting"
```

---

### Task 3: Derive estimate text and count from one formatted result

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py:1-25,797-862`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4320-4345,5270-5305,6590-6610`
- Modify: `Tests/UI/test_console_staged_evidence_strip.py`

- [ ] **Step 1: Add one private formatted-context seam**

```python
def _console_prompted_evidence_context(
    launch: ConsoleLiveWorkLaunch | None,
) -> LocalEvidenceContext | None:
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return None
    return format_console_evidence_context(
        normalize_console_evidence_references(bundle.available_references())
    )
```

- [ ] **Step 2: Make both public estimate helpers projections**

`console_prompted_source_count()` returns `len(formatted.entries)` and `console_prompted_evidence_text()` returns `formatted.context`, with zero/empty fallbacks when no bundle exists. Remove both duplicated owner predicates and raw snippet joining.

- [ ] **Step 3: Correct semantic documentation**

Update both helper docstrings and the relevant context/cost comments to say “formatted pre-authority estimate.” Update the sent-notice fallback documentation to remain authoritative when repair-contract ordinals exist and explicitly best-effort only when it must fall back to the launch estimate.

- [ ] **Step 4: Run the estimator tests GREEN**

Run:

```bash
../../.venv/bin/pytest -q Tests/UI/test_console_staged_evidence_strip.py \
  -k "prompted_evidence or prompted_source_count or staged_source_count"
../../.venv/bin/pytest -q Tests/Chat/test_console_session_settings.py \
  -k "context_estimate"
```

Expected: PASS.

- [ ] **Step 5: Commit estimator parity**

```bash
git add tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_staged_evidence_strip.py
git commit -m "fix(console): align evidence cost estimate with prompt"
```

---

### Task 4: Pin authority shrinkage and verify the finished task

**Files:**
- Modify: `Tests/RAG/test_local_citation_capture.py`
- Modify: `backlog/tasks/task-2525 - Console-context-cost-estimate-has-three-small-modelling-gaps.md`

- [ ] **Step 1: Add a focused authority-shrink regression**

Build a two-reference Console launch whose estimate includes both canonical prompt blocks, then send through an app authority double that exposes only the second backing row. Assert that estimate-time code performs no authority I/O, the send context contains only the surviving source re-numbered as `[S1]`, and the repair contract contains one ordinal.

- [ ] **Step 2: Run one-time dependency and static checks**

```bash
../../.venv/bin/python -c \
  "import tldw_chatbook.Chat.console_display_state; import tldw_chatbook.RAG_Search.local_citation_capture"
../../.venv/bin/ruff check \
  tldw_chatbook/RAG_Search/local_citation_capture.py \
  tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/RAG/test_local_citation_capture.py \
  Tests/UI/test_console_staged_evidence_strip.py
git diff --check origin/dev...HEAD
```

Expected: all commands exit 0. Do not add a permanent subprocess import test.

- [ ] **Step 3: Run focused behavior verification**

```bash
../../.venv/bin/pytest -q Tests/RAG/test_local_citation_capture.py \
  -k "console_staged or console_evidence or authority"
../../.venv/bin/pytest -q Tests/UI/test_console_staged_evidence_strip.py \
  -k "prompted_evidence or prompted_source_count or staged_source_count"
../../.venv/bin/pytest -q Tests/Chat/test_console_session_settings.py \
  -k "context_estimate"
```

Expected: PASS. The unrelated full-file mounted-send baseline remains tracked separately: its second durable turn is refused before capture because the thread-local `:memory:` fixture opens without schema. Do not mask that failure or alter production behavior in this task.

- [ ] **Step 4: Complete backlog and implementation notes**

Use the Backlog CLI to check every acceptance criterion, add concise Implementation Notes with the ADR decision and verification evidence, and set TASK-2525 to Done only after every task Definition-of-Done requirement is satisfied.

- [ ] **Step 5: Commit task closeout**

```bash
git add backlog/tasks/task-2525\ -\ Console-context-cost-estimate-has-three-small-modelling-gaps.md \
  Tests/RAG/test_local_citation_capture.py
git commit -m "test(console): verify evidence estimate parity"
```

