---
id: TASK-16588
title: >-
  Carry expand_document's identity to the semantic/hybrid routes: chunk_start
  and the provenance id fallbacks
status: In Progress
assignee: []
created_date: '2026-08-15 23:55'
updated_date: '2026-08-16 14:17'
labels:
  - rag
  - agents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16174 shipped `expand_document` and wired a per-row policy plus the
identity the tool requires into `Agents/library_rag_tool_provider.
_project_row`. Its Phase E oracle run measured the tool end to end on the
**`plain`** route (the Library's four-seam keyword path) and found
tool-OFF 0/8 vs tool-ON 7/8
(`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md`).

That run **cannot exercise two known gaps**, because the `plain` route
structurally avoids both: it emits no chunked rows and its `source_id` is
always a real database id. Task 3b pre-registered them, and the run
recorded them as **unrefuted rather than refuted** — every probe reported
`chunked_rows = []`, every expansion returned `status: "ok"`, and no
identity fallback was ever needed. Both remain live for the
`semantic`/`hybrid` routes, which are what a user on a vector or fused
profile actually gets.

**(a) `chunk_start` is not in the projected payload.** The tool accepts it
as the window anchor and `_semantic_row` already copies it into
`provenance`, but Task 3b's scope stopped at `source_type`/`source_id`
(+`chunk_id`). So a chunked row expands from `offset = 0` — the document
HEAD — not around the matched chunk. On a long document that returns a
budget's worth of the wrong text while reporting `status: "ok"`, which is
the failure mode hardest to notice: it looks like a successful expansion.
`chunk_id` cannot substitute — it is an INDEX (`f"{doc_id}_chunk_{i}"`),
not a character offset, and no index→offset path exists by design.

**(b) a semantic `source_id` can be a vector-store point id.**
`_semantic_row` resolves `source_id` as `metadata["source_id"] ||
metadata["document_id"] || the chroma point id`, so for some rows the
point id is what surfaces and the real document identity lives only in the
provenance extras (`note_id` / `doc_id`). The tool accepts those as
identity fallbacks, but they are **not** in the payload — so such a row is
declared `expandable` by the hint and can still come back `not_found`. A
verdict the agent cannot act on is exactly what Task 3b existed to
prevent, one route over.

The fix for both is the same shape as Task 3b's: a payload ADDITION under
the hint's own precondition, keeping `{expandable, reason}` and every
existing key unchanged. The cost is bytes in a sealed payload
(Task 3b measured +40.0 B/row for the identity keys; +97.5 B/row
cumulative for the arc), so the byte budget must be re-measured, not
assumed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A projected row that has a `chunk_start` in its provenance carries it in the payload, under exactly the same precondition as the existing identity keys, so a chunked hit expands around the match rather than from the document head
- [ ] #2 A row whose `source_id` is a vector-store point id carries the provenance identity fallback(s) the tool accepts (`note_id` / `doc_id`), so a row the hint declares expandable can actually be fetched
- [ ] #3 Both are measured on a route that can exercise them (`semantic` and/or `hybrid`), not only asserted in unit tests: a run records, per row, whether the returned window contains the matched chunk and whether any declared-expandable row returned `not_found`
- [ ] #4 The sealed-payload byte cost is re-measured on a normal ten-row payload and stated (Task 3b's method: strip-and-reserialize), and the sealing loop's progress guarantee still holds with the added keys
- [ ] #5 `expand_hint`'s pinned `{expandable, reason}` interface, the tool's contract, and row order/count are unchanged; `citations` and the `provenance` mapping still never leave the adapter
- [ ] #6 The gated retrieval suite still reads "PASSED: No regression. 105 metric(s)" with every cell at (+0.000) — this is a payload addition, not a retrieval change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-16-rag-semantic-identity-design.md (corrected at 279619d86). Plan: Docs/superpowers/plans/2026-08-16-rag-semantic-identity.md.
1. Task 1 - payload addition: RED-first unit pins in Tests/Agents/test_library_tool_provider.py (fallbacks projected verbatim; absent when provenance lacks them; ride the hint precondition; empty dropped; media_id never projected; sealed ten-row payload survives with the strip-and-reserialize byte cost stated), then a small helper in Agents/library_rag_tool_provider.py mirroring _chunk_start's shape (string-coerce, drop empty) called in the same 'if hint is not None' block. Gate: RAG_EVAL=1 pytest Tests/RAG_Eval/ reads 105 metric(s) verbatim.
2. Task 2 - dual-index route probe (canonical + non-canonical seeds, semantic + hybrid routes, direct expand per declared-expandable row, pre/post-fix arms), report.md, then closure of all six ACs on evidence.
<!-- SECTION:PLAN:END -->

## Notes

- **AC#1 shipped early, at the unit level only (TASK-16174 fix wave,
  2026-08-15).** The final whole-branch review called the dead `chunk_id`
  parameter a blocker, and the wire-or-retire fix for it was to retire
  `chunk_id` from the tool schema AND emit `chunk_start` from `_project_row`
  whenever a row's provenance carries a usable anchor (`> 0`). So the payload
  addition exists and is pinned by tests; what is still open on AC#1 is the
  route MEASUREMENT in AC#3 — no `semantic`/`hybrid` run has yet shown a
  window that actually contains the matched chunk.
- **In-scope evidence for AC#3 (final review finding 6):** rows whose raw
  provenance `source_type` is a canonicalization VARIANT — `media_chunk`,
  `chat`, or a plural spelling, all of which `_SEMANTIC_SOURCE_TYPE_MAP`
  treats as live — get no hint and therefore no identity at all under the
  policy's singular-only `EXPANDABLE_SOURCE_TYPES`, so the route measurement
  should count them rather than only the point-id case (carried as its own
  finding in TASK-16688).
