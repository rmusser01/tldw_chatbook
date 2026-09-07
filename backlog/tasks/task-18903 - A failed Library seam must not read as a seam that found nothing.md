---
id: TASK-18903
title: A failed Library seam must not read as a seam that found nothing
status: Done
assignee: []
created_date: '2026-08-18'
labels: [library, rag, correctness]
dependencies: []
priority: high
---

## Description (the why)

Every Library keyword seam ended `except Exception: return True, []` — `True`
meaning **available**. A seam whose backend threw reported itself healthy and
empty, and the merge site (which gates on "any seam available") could not tell
the difference.

**Traced end to end, the consequence was not merely cosmetic:**

1. all four seams throw → each returns `(True, [])`
2. `any(available)` passes the gate
3. zero rows, unscoped → a plain `{"results": [], …}` dict
4. `_outcome_from_service_result`: `if not rows: status="empty"`
5. `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES = frozenset({"ready", "empty"})`
6. so the RAG answer path **ran** — generating an answer from **no retrieved
   context** and presenting it as Library-grounded

A total backend outage produced a confident, ungrounded answer.

This is the same collapse that produced TASK-17855's wrong defect filing and
TASK-18255 — a value meaning "could not measure" rendering identically to
"measured, found nothing" — but reaching the user rather than the instrument.

## Acceptance Criteria (the what)

- [x] A seam whose backend throws reports `FAILED`, not available
- [x] **Total failure is NON-answerable**: status `failed`, excluded from
      `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES`, pinned by a test
- [x] All seams merely unconfigured still yields `blocked` — unchanged
- [x] Partial failure returns surviving results AND records the failed seams
      in diagnostics, appended to a list
- [x] The user is told which seams failed, through the existing rendered
      route-notes channel
- [x] `_empty_scoped_seam` maps to `AVAILABLE`
- [x] `seam_effect.py`'s `if not available:` guard updated — under an enum it
      would go silently inert
- [x] `Tests/Library` green (2031 passed); RAG-eval gate `PASSED: No
      regression. 105 metric(s)` with every cell at +0.000

## Implementation Notes

**`failed` is REUSED, not newly minted** (owner ruling). Design review found
`run_library_rag_search` already returns `status="failed"` with a recovery
state for a raised search; it already means "retrieval did not happen" and is
already absent from every answerable allowlist. A second failure status would
have been two vocabularies for one condition.

**Two channels, mirroring existing precedent rather than inventing one.**
Structured state goes to `KEYWORD_SEAM_DIAGNOSTICS_KEY` as a LIST of
`{"status", "seam", "message"}` — appended, never assigned, for the same
reason the scope slot is (task-9 review finding: two failures in one call must
not overwrite each other). The human sentence goes through the existing
`LIBRARY_RAG_ROUTE_NOTES_KEY`, which the panel already renders, and whose own
docstring makes this task's argument: *"zero rows is exactly when it matters
most"*.

**Born-red on BEHAVIOUR, not ImportError.** Per
`lessons-testing-evidence.md`, an ImportError red is not born-red evidence, so
the enum and constants landed first with today's semantics preserved; the
tests then failed 5/8 on behaviour, and the fix turned them green.

**Mutation-verified twice.** Restoring `return SeamState.AVAILABLE, []` on the
exception path reds 4 tests including both headline pins. Reverting the gate
to `if not any(state ...)` — the enum-truthiness trap, since every `Enum`
member is truthy — reds `test_no_configured_seam_still_blocks`, which exists
precisely to catch that.

**One existing test pinned the defect** and was rewritten, not deleted:
`test_erroring_only_seam_returns_empty_results_not_blocked` asserted a plain
dict with zero results, reasoning that an erroring seam "is still available
(attribute present)". Its real intent — don't show *setup* advice for a
configured seam — is preserved by returning `failed` rather than `blocked`;
only the part encoding the bug changed. Four assertions in
`test_library_keyword_and_then_prefix.py` moved from `available is True` to
`state is SeamState.AVAILABLE` (my design review wrongly claimed they
discarded the flag — they assert on it).

**Files:** `library_local_rag_search_service.py` (enum, four seams, gate,
recovery state, route note), `Tests/Library/test_library_seam_availability.py`
(new, 10 pins), two existing test files, `seam_effect.py`, the design spec.
