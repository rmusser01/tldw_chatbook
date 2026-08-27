---
id: TASK-2525
title: Console context/cost estimate has three small modelling gaps
status: In Progress
assignee: []
created_date: '2026-08-06 02:21'
updated_date: '2026-08-27 14:06'
labels:
  - console
  - rag
  - cost
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T2 Task 6 made staged Library evidence count toward Console's context estimate and cost chip instead of
reporting zero, by tokenizing `EvidenceReference.snippet` (via the new `console_prompted_evidence_text()`,
`Chat/console_display_state.py`). Task 6's own review (whole-branch review, dual-blind against the actual send
path) found the chosen source is honest and correctly bounded, but flagged three small, deliberately-deferred
modelling gaps between what the estimate counts and what the send actually assembles/limits:

1. **Per-source framing overhead isn't counted.** The send path's `format_local_evidence_context()`
   (`RAG_Search/local_citation_capture.py:359-420`) prepends a header `f"[S{ordinal}] {label} — {result.title}\n"`
   (`:386-388`) and a separator between blocks to every staged source — the estimate tokenizes only the raw
   snippet text, not these headers/separators. Real but bounded under-count, roughly 20-60 characters per staged
   source.

2. **The 64-entries-per-prompt cap isn't modelled.** `EVIDENCE_ENTRIES_PER_PROMPT_MAX = 64`
   (`Chat/citation_trace_models.py:32`), enforced in the same `format_local_evidence_context()` loop
   (`RAG_Search/local_citation_capture.py:383-385`: entries past the cap are appended to `omitted` rather than
   included). The estimate has no equivalent truncation, so with more than 64 staged sources it would over-count
   relative to what's actually sent — the safe direction, but still a real divergence from "what will be sent".

3. **Duplicated filter predicate.** `console_prompted_source_count()` and `console_prompted_evidence_text()`
   (`Chat/console_display_state.py:607-635` and `:636-672`) both independently write
   `reference.source_owner.strip().lower() == "local"` (`:632` and `:671`) to decide which references are
   prompt-eligible. Identical today, but two copies of "which references reach the model" can silently drift if
   either is edited without the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The context/cost estimate accounts for the per-source header + separator overhead `format_local_evidence_
      context` adds (either by including a fixed/estimated per-source overhead, or by computing the estimate
      through the same formatting function used at send time), OR the estimate's docstring/UX copy is updated to
      state the known under-count explicitly if not fixed
- [ ] #2 The estimate does not silently over-count past `EVIDENCE_ENTRIES_PER_PROMPT_MAX` staged sources — either
      it caps its own count at the same limit, or this is explicitly documented as a known, safe-direction
      divergence
- [ ] #3 `console_prompted_source_count` and `console_prompted_evidence_text`'s shared `source_owner == "local"`
      eligibility predicate is extracted into one helper both functions call, so the two can no longer drift
- [ ] #4 Existing tests for both functions (`Tests/UI/test_console_staged_evidence_strip.py`,
      `Tests/Chat/test_console_session_settings.py`) stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red tests for canonical framing, normalization, empty content, and the 64-entry cap.\n2. Extract shared Console evidence normalization and full-candidate formatting helpers, then route the send adapter through them.\n3. Derive estimate text and count from one formatted pre-authority result and correct semantic documentation.\n4. Add authority-shrink coverage, run focused verification and static checks, then complete task notes and acceptance criteria.\n\nADR required: no\nADR path: N/A\nReason: Routine parity bug fix reusing existing normalization, authority, and formatting boundaries.
<!-- SECTION:PLAN:END -->
