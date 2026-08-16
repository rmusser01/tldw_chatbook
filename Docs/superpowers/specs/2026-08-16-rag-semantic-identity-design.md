# Semantic-route identity: carry the fallbacks, measure the window (TASK-16588)

Date: 2026-08-16
Status: draft-pending-user-review
Programme: RAG server-port (eleven merged; last: 16174 agentic expansion
#1712 → dev `dec2c467f`)
Worktree: `.worktrees/rag-16588-route`, branch
`feat/rag-16588-semantic-identity`, off dev `bab84f7d9`.

## What this closes

TASK-16174's oracle run measured `expand_document` on the `plain` route
only — which structurally cannot produce a chunked row or a vector-store
point id. Its two pre-registered suspects are therefore UNREFUTED, and
both live on the routes a vector/fused profile actually uses:

- **(a)** `chunk_start` now ships in the payload (16174's fix wave,
  unit-pinned) — but no semantic/hybrid run has ever shown a returned
  window that actually contains the matched chunk. AC#1's residue is
  the MEASUREMENT, not the code.
- **(b)** a semantic row's `source_id` can be a chroma point id while the
  real identity sits in provenance extras (`note_id`/`doc_id`) that the
  tool accepts but the payload WITHHOLDS — so a row the hint declares
  expandable can return `not_found`. This is the code gap (AC#2).

**Verified at this HEAD, and it sharpens (b):** the app's own indexing
path is two-vocabulary. `ingestion_indexing.py` builds documents whose
metadata carries `source_id`/`source_type` — yet the 15810 live check's
scratch profile, seeded through the app's own `index_entries`, produced
chunk metadata of `doc_id`/`note_id`/`doc_title`/`type` with **no
`source_id` key** (recorded vector-store stats, 2026-08-15). Somewhere in
the chunk-indexing layer the keys are transformed, so on real app-built
indexes `_semantic_row`'s `source_id` falls through to the point id and
the fallback path is not an edge case — it may be the COMMON case.
Plan-phase verification item 1 pins down exactly where the
transformation happens and which vocabulary reaches `provenance`.

## The change (AC#2, #5): one payload addition, Task 3b's shape exactly

`_project_row` additionally emits the provenance identity fallbacks the
tool accepts — `note_id` / `doc_id` (and `media_id` if verification finds
it emitted) — under **the hint's own precondition**, same as every
identity key before them. Pinned invariants unchanged: `{expandable,
reason}`, row order/count, `citations`/`provenance` never leave, and the
tool's contract untouched (it already accepts these kwargs; nothing on
the tool side changes at all).

## The measurement (AC#1's residue, #3): mechanical, no LLM, no spend

A committed probe (`Docs/superpowers/qa/.../route_probe.py`), not a
pytest (network irrelevant but the live-run conventions apply: scratch
profile, config-hash before/after, teardown; seeded via the app's own
APIs with a real embedding index so the semantic route is genuinely
exercised). For `semantic` AND `hybrid` routes, over queries chosen so
chunked rows and long documents actually appear:

- per returned row: the hint verdict, the identity keys present, and —
  for every row declared expandable — a DIRECT `expand_document` call
  recording `status`, whether `chunk_start` was carried, and whether the
  returned window CONTAINS the matched chunk's text (substring check
  against the row's own snippet);
- counted: `not_found` on declared-expandable rows (the (b) defect —
  target 0 after the fix; each occurrence pre-fix is evidence, not
  failure); windows that miss their chunk (the (a) defect on long docs);
  and canonicalization-VARIANT rows (`media_chunk`/`chat`/plurals) that
  get no hint at all — counted per the task's note as in-scope evidence,
  fixed only if the count is nonzero AND the fix is a one-line allowlist
  broadening (else the count feeds TASK-16688 where the finding lives).

Run BEFORE the payload change (expect (b) occurrences on point-id rows)
and AFTER (expect 0), so the fix has a measured before/after, not a
unit-test-only story.

## Invariants (AC#4, #6)

- Byte cost re-measured by Task 3b's strip-and-reserialize method on a
  normal ten-row payload, stated in the report; sealing-loop progress
  re-proven with the added keys.
- The gated suite reads verbatim **PASSED 105/105 with every cell
  (+0.000)** — payload addition, never a retrieval change.

## Out of scope (declared)

- The policy allowlist broadening beyond the one-line case above
  (TASK-16688 finding 6), reranking (3502), the `allowed_tools`
  runtime-schemas bypass (16788), any tool-contract change.

## Plan-phase verification (before tasks are cut)

1. WHERE the metadata vocabulary transforms: trace `index_entries` →
   chunk metadata; name the function that drops/renames
   `source_id`/`source_type` into `doc_id`/`type`, and record which keys
   reach `_semantic_row`'s `provenance` on an app-built index.
2. Whether `media_id` ever appears in provenance (the tool accepts it;
   emit only what actually occurs).
3. How the probe drives the `semantic`/`hybrid` routes headless
   (15810's `probe_headless.py` pattern + a real embedding index — the
   eval harness ingest already builds one; reuse, don't reinvent).
4. The seeding corpus: long documents (> the tool's 8000-char default
   budget) with distinctive mid-document chunks, so the window-contains-
   chunk check can actually fail if (a) is real.
