# Semantic-route identity: carry the fallbacks, measure the window (TASK-16588)

Date: 2026-08-16
Status: accepted
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

**Verified at this HEAD (correcting this spec's own first draft):** the
transformation hypothesis was traced end to end and REFUTED for the
canonical path. `store_documents_batch`
(`RAG_Search/simplified/indexing_helpers.py:181`, reached from
`index_entries` → `index_batch_optimized`) SPREADS the entry's metadata
into every chunk and adds `doc_id` (the PREFIXED entry id, `note_5`) +
`doc_title` + `chunk_start`/`chunk_end`/`chunk_index`. The app's
canonical builders (`note_document`/`media_document`/
`conversation_document`) all write `source_id`/`source_type`, so on a
canonically-built index `_semantic_row` resolves `source_id` correctly
and the point-id fallback never fires. The 15810 scratch's
`doc_id`/`note_id`/`type` vocabulary came from that QA script's OWN
hand-built `IndexEntry` metadata (`seed_profile.py:72`), not the app's.
What this means for the arc, pre-registered:
- (b) is real for any non-canonical `IndexEntry` producer — one such
  producer is committed in this repo (TASK-15810's QA seeder), which is
  proof such producers EXIST, not that one has ever run against a real
  profile: that script asserts its own data_dir is scratch-isolated
  (`seed_profile.py:51`). Whether a shipped app version ever wrote an
  index without `source_id` was NOT established — no such vintage of the
  indexing path was found. It is NOT the common case on a fresh canonical
  index.
- The route probe therefore seeds BOTH: a canonical index (expected
  pre-fix `not_found` on declared-expandable rows: **0** — and a nonzero
  reading there would be a NEW finding) and a non-canonical one built the
  15810 script's way (expected pre-fix: nonzero — the fallback's
  evidence; post-fix: 0).
- On canonical indexes the payload's `doc_id` fallback is ALWAYS present
  (prefixed; the tool already strips prefixes), so the fix's value is
  defensive-plus-legacy, and the spec says so rather than overclaiming.
- `media_id` never appears in any builder's metadata (verification item
  2, answered): the payload addition emits `note_id`/`doc_id` only.

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
