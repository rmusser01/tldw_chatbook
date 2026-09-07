---
id: TASK-16588
title: >-
  Carry expand_document's identity to the semantic/hybrid routes: chunk_start
  and the provenance id fallbacks
status: Done
assignee: []
created_date: '2026-08-15 23:55'
updated_date: '2026-08-16 07:45'
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
- [x] #1 A projected row that has a `chunk_start` in its provenance carries it in the payload, under exactly the same precondition as the existing identity keys, so a chunked hit expands around the match rather than from the document head
- [x] #2 A row whose `source_id` is a vector-store point id carries the provenance identity fallback(s) the tool accepts (`note_id` / `doc_id`), so a row the hint declares expandable can actually be fetched
- [x] #3 Both are measured on a route that can exercise them (`semantic` and/or `hybrid`), not only asserted in unit tests: a run records, per row, whether the returned window contains the matched chunk and whether any declared-expandable row returned `not_found`
- [x] #4 The sealed-payload byte cost is re-measured on a normal ten-row payload and stated (Task 3b's method: strip-and-reserialize), and the sealing loop's progress guarantee still holds with the added keys
- [x] #5 `expand_hint`'s pinned `{expandable, reason}` interface, the tool's contract, and row order/count are unchanged; `citations` and the `provenance` mapping still never leave the adapter
- [x] #6 The gated retrieval suite still reads "PASSED: No regression. 105 metric(s)" with every cell at (+0.000) — this is a payload addition, not a retrieval change
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

## Implementation Notes

Two tasks. **Task 1** added the payload's identity FALLBACKS; **Task 2**
measured `expand_document` on the two routes TASK-16174's oracle run
structurally could not reach, before and after, against two index kinds.

**The change (one file).** `Agents/library_rag_tool_provider.py` gained
`_IDENTITY_FALLBACK_KEYS = ("note_id", "doc_id")` and `_identity_fallbacks()`,
mirroring `_chunk_start`'s shape (string-coerce, drop `None`/empty/whitespace),
called inside the SAME `if hint is not None:` block every identity key already
rides — so verdict and identity still cannot drift apart. `media_id` is
deliberately never projected: the tool accepts it but no indexing builder
writes it (spec verification item 2). The tool, the policy helper, the hint
interface and the sealing loop's shrink order are untouched. Six RED-first
pins in `Tests/Agents/test_library_tool_provider.py` (47 → 55 items); the one
that could not be RED before the code existed — an absence-only guard on a
variant row — is disclosed as a regression pin rather than evidence, and every
other test carries a same-payload positive control so none can pass vacuously.

**The measurement (AC#1's residue, AC#3).** A committed, mechanical probe —
no LLM, no network, no spend, no TUI boot —
`Docs/superpowers/qa/2026-08-16-rag-semantic-identity/` (`route_probe.py`,
`report.md`, `probe-artifacts.json`). It seeds TWO indexes in TWO isolated
scratch profiles: a CANONICAL one (20 items through `note_document` /
`media_document` / `conversation_document` + `index_entries`) and a
NON-CANONICAL one (the same 12 notes through a hand-built `IndexEntry` with
`{"type","note_id","title"}` metadata — TASK-15810's committed seeder shape).
Both routes are driven through the production surface
(`LibraryRagToolProvider.invoke`, `mode="rag"`), with the direct engine as the
metadata control, and every identity-bearing row is expanded by a DIRECT
`ExpandDocumentTool().execute(...)` call in three arms: `pre` (the two
fallback keys stripped by the probe — the before/after without a checkout
dance), `post` (as shipped), and `head` (`post` minus `chunk_start`).

| index × route | rows | hinted | expandable | not_found PRE (hinted/expandable) | not_found POST | chunk_start carried | variant rows w/o hint |
|---|---|---|---|---|---|---|---|
| canonical × semantic | 100 | 100 | 64 | 0 / 0 | 0 / 0 | 69 | 0 |
| canonical × hybrid | 100 | 100 | 61 | 0 / 0 | 0 / 0 | 56 | 0 |
| non-canonical × semantic | 70 | 70 | 49 | 70 / 49 | 0 / 0 | 45 | 0 |
| non-canonical × hybrid | 70 | 70 | 45 | 66 / 45 | 0 / 0 | 43 | 0 |

All three pre-registered expectations HELD, so the plan's STOP-and-report
condition never fired and Task 2 touched no production code.

**AC#1 (the window).** 22 of 22 marker rows — across note, media and
conversation seams, on both routes, every one at rank 1 — returned a
`chunk_start`-anchored window that CONTAINS the matched chunk's planted
marker (e.g. marker at char 9732 of a 12,380-char note, window 4380–12380).
0 of 22 head windows (0–8000) contain it, which is the control proving the
corpus can actually fail this check. Long-doc windows missing their marker: 0.
Instrument limit, disclosed: all 22 anchored windows are the document TAIL
(`[total − 8000, total]`), so this proves the window is off the HEAD and on
the match's side of the document — not that it is centred on the match. See
the QA report's Disclosed limits for what a future extender must vary.

**AC#2 (the identity).** On the non-canonical index `_semantic_row` resolves
`source_id` to the vector store's POINT id (`note_<uuid>_chunk_4`), which
names nothing fetchable: every row the hint declared expandable returned
`not_found` before the fallbacks shipped (49/49 semantic, 45/45 hybrid) and
`ok` after. On the canonical index the reading was 0 both before and after —
`source_id` already resolves there — so the fix is defensive-plus-legacy, and
the report says so rather than overclaiming. The four non-canonical hybrid
rows that resolved pre-fix are the engine's FTS keyword-leg rows (identity
from the notes DB, not from vector metadata) and are all `expandable: false`.

**AC#4 (bytes).** Strip-and-reserialize, ten-row payload, five carriers:
**+15.0 B per carrying row, +75 B total, payload 4,085 B of 32,768 B,
headroom 28,683 B.** Provenance: that figure is computed inside
`test_sealed_payload_survives_fallbacks` and rendered into its **assertion
message**, which pytest emits only when the assert FAILS — it is not printed
output of the passing suite; the final review reproduced it independently in
a standalone script and got the byte-identical string. The fixture's ids are
deliberately short, and both the fixture and the assertion now carry that
caveat in-file. Re-measured by the same method on the 34 REAL route payloads:
**45.94 B/row canonical (a redundant `doc_id`) and 102.0 B/row non-canonical
(`note_id` + `doc_id`)** — 3–7× the synthetic figure because real ids are
UUIDs. Largest observed payload 17,350 B, 53 % of the ceiling, 15,418 B of
headroom; `returned == 10` on all 34, so the sealing loop dropped nothing and
its progress guarantee (two hostile-metadata termination pins) still holds.

**AC#5.** `{expandable, reason}`, row order and count unchanged;
`citations` and the `provenance` mapping still never leave the adapter;
`Tools/document_expansion_tool.py` untouched (it already accepted these
kwargs).

**AC#6.** `[rag-eval baselines] PASSED: No regression. 105 metric(s) within
0.05 of baseline.` — 105 of 105 gated cells read `(+0.000)`, 0 moved, read
cell-by-cell in both tasks.

**Also verified.** Batteries `Tests/Agents/` 1445 passed, `Tests/Library/`
1986 passed / 2 skipped, `Tests/Tools/` 572 passed / 15 skipped,
provider file 55 passed. Full-suite collection at HEAD: 48,280 items, delta
+8 vs merge-base `bab84f7d9`, fully accounted for by this branch's one
changed test file (47 → 55); the only two collection errors
(`Tests/Web_Scraping/Confluence/*`) are a missing `playwright` optional dep
in this venv, on files this branch never touches. `ruff check` clean on both
touched files (`ruff format` drift on them predates HEAD and was left alone).

**Deliberately NOT done.** The canonicalization-VARIANT count
(`media_chunk`/`chat`/plurals receiving no hint) was **0** on every
(index × route), with a positive control proving the detector fires on every
variant spelling and no singular one — no writer this corpus can exercise
emits a variant into chunk metadata. Per the plan's rule, nothing was
broadened and nothing was appended to TASK-16688; the finding stays there.
Also unmeasured here by design: `label_only` rows never appear on
`semantic`/`hybrid` (they are a `plain`-route product of the Library's own
four-seam keyword path), so TASK-16174's oracle run and this probe are
complements, not overlaps.

**Files.** Modified: `tldw_chatbook/Agents/library_rag_tool_provider.py`,
`Tests/Agents/test_library_tool_provider.py`, `Tests/RAG_Eval/README.md`.
Added: `Docs/superpowers/specs/2026-08-16-rag-semantic-identity-design.md`,
`Docs/superpowers/plans/2026-08-16-rag-semantic-identity.md`,
`Docs/superpowers/qa/2026-08-16-rag-semantic-identity/{route_probe.py,report.md,probe-artifacts.json}`.
