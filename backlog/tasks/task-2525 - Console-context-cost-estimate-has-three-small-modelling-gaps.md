---
id: TASK-2525
title: Console context/cost estimate has three small modelling gaps
status: Done
assignee: []
created_date: '2026-08-06 02:21'
updated_date: '2026-08-27 15:17'
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
- [x] #1 The context/cost estimate accounts for the per-source header + separator overhead `format_local_evidence_
      context` adds (either by including a fixed/estimated per-source overhead, or by computing the estimate
      through the same formatting function used at send time), OR the estimate's docstring/UX copy is updated to
      state the known under-count explicitly if not fixed
- [x] #2 The estimate does not silently over-count past `EVIDENCE_ENTRIES_PER_PROMPT_MAX` staged sources — either
      it caps its own count at the same limit, or this is explicitly documented as a known, safe-direction
      divergence
- [x] #3 `console_prompted_source_count` and `console_prompted_evidence_text`'s shared `source_owner == "local"`
      eligibility predicate is extracted into one helper both functions call, so the two can no longer drift
- [x] #4 Existing tests for both functions (`Tests/UI/test_console_staged_evidence_strip.py`,
      `Tests/Chat/test_console_session_settings.py`) stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red estimator tests plus a green send-adapter characterization.
2. Extract shared Console evidence normalization and full-candidate formatting helpers, then route the send adapter through them.
3. Add authority-shrink RED coverage, derive estimate text and count from one formatted pre-authority result, and correct semantic documentation.
4. Run focused isolated verification and static checks, then complete task notes and acceptance criteria.

ADR required: no
ADR path: N/A
Reason: Routine parity bug fix reusing existing normalization, authority, and formatting boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned Console context/cost estimates with the send path by sharing canonical EvidenceReference normalization and prompt formatting. The zero-I/O estimator now includes source headers and separators, excludes noncanonical references, and applies the 64-entry cap; send-time authority remains authoritative and may shrink the pre-authority estimate.

Files: tldw_chatbook/RAG_Search/local_citation_capture.py, tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py, tldw_chatbook/Chat/console_display_state.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/RAG/test_local_citation_capture.py, and Tests/UI/test_console_staged_evidence_strip.py.

Verification: Ruff passed on all changed Python files; git diff --check passed; 6 focused RAG capture/authority tests passed; 7 prompted/staged estimator tests passed; the exact pre-send context-estimate test passed; and 8 existing Chat context-estimate tests passed. Full suite was not run per repository guidance.

Tradeoff: estimates intentionally avoid authority I/O and remain formatted pre-authority previews; the send path rechecks authority and fails closed. ADR required: no. ADR path: N/A. Reason: routine parity fix reusing existing boundaries. No new generalizable lesson was produced.

Final review corrected `build_console_context_estimate()` documentation to describe `staged_text` as canonical formatted pre-authority evidence that authoritative send capture may shrink; runtime behavior was unchanged.

PR review follow-up named and documented the per-candidate Console framing allowance, and replaced the authority-shrink test's database double with a real in-memory SQLite `Media` table. The focused authority/send tests passed (2), the prompted/staged estimator selection passed (7), Ruff passed, and `git diff --check` passed; runtime behavior remained unchanged.
<!-- SECTION:NOTES:END -->
