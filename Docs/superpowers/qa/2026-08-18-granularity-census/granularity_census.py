"""TASK-18155 census: could a granularity router change any golden outcome?

Run:  RAG_EVAL=1 python Docs/superpowers/qa/2026-08-18-granularity-census/granularity_census.py

Answers two populations, because "chunk vs document granularity" has TWO
mechanisms and the cheap count only sees the first:

1. DIRECT -- the query's own relevant document is multi-chunk, so retrieving
   it whole rather than in pieces changes what is scored for that query.
2. DISPLACEMENT -- some OTHER document occupies several of the top-k row
   slots (`canonicalize.py`: "one document can occupy several of the top-k
   slots"), because the top-k cut happens at ROW level and rows collapse to
   documents only afterwards. Collapsing at retrieval would free those slots
   for documents currently pushed out. This one needs a live run.

Only a query whose relevant document is CURRENTLY MISSED can be rescued, so
the qualifying population is (slots freed > 0) AND (relevant doc absent).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

if os.environ.get("RAG_EVAL") != "1":
    raise SystemExit("refusing to run without RAG_EVAL=1 (this builds a real index)")

import tldw_chatbook

# Provenance: a census that silently measured another checkout would be
# worthless, and this arc already shipped one instrument that could not see
# what it was asked about (TASK-18255).
print(f"PROVENANCE tldw_chatbook <- {tldw_chatbook.__file__}")
assert str(REPO) in tldw_chatbook.__file__, f"WRONG TREE: expected under {REPO}"

from Tests.RAG_Eval.harness.canonicalize import canonical_source_type, slug_lookup_from
from Tests.RAG_Eval.harness.goldenset import load_fixtures
from Tests.RAG_Eval.harness.ingest import build_eval_runtime
from Tests.RAG_Eval.harness.runner import SOURCE_TYPES, _extract_rows, build_query_scope

CHUNK_SIZE = 384          # hybrid_basic profile
CHUNK_OVERLAP = 64
K = 10                    # run_eval's default
BAR = 5                   # inherited verbatim from the PRF + clarification-gate arcs
MODES = ("semantic", "hybrid")   # `plain` returns whole items: no chunks, so no granularity


def row_slug(row, lookup):
    """Map ONE row to its fixture slug, mirroring rows_to_doc_ids without collapsing."""
    prov = row.get("provenance") or {}
    st = canonical_source_type(prov.get("source_type"))
    sid = str(row.get("source_id", ""))
    for key in ((st, sid), (st, sid.split("_", 1)[-1] if "_" in sid else sid)):
        if key in lookup:
            return lookup[key]
    return f"unknown:{st}:{sid}"


def main() -> int:
    import tempfile

    corpus, golden = load_fixtures()
    print(f"PROBE PROOF: corpus={len(corpus)} docs, golden={len(golden)} queries")

    # --- population 1: DIRECT, from the corpus alone (no run needed) ---
    from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

    svc = ChunkingService()
    chunks_per_doc = {
        d.slug: len(svc.chunk_text(d.content, chunk_size=CHUNK_SIZE,
                                   chunk_overlap=CHUNK_OVERLAP, method="words"))
        for d in corpus
    }
    multi = {s: n for s, n in chunks_per_doc.items() if n > 1}
    print(f"PROBE PROOF: docs chunked={len(chunks_per_doc)}, "
          f"total chunks={sum(chunks_per_doc.values())}, multi-chunk docs={len(multi)}")
    for s, n in sorted(multi.items(), key=lambda kv: -kv[1]):
        print(f"    multi-chunk: {s:38s} {n} chunks")

    direct = [q for q in golden if any(s in multi for s in q.relevant_slugs)]
    print(f"\n=== POPULATION 1 (DIRECT): {len(direct)} of {len(golden)} queries ===")
    for q in direct:
        print(f"    {q.id:30s} [{q.category}] -> "
              f"{[s for s in q.relevant_slugs if s in multi]}")

    # --- population 2: DISPLACEMENT, needs a live index ---
    with tempfile.TemporaryDirectory() as tmp:
        runtime = build_eval_runtime(corpus, tmp)
        from tldw_chatbook.Library.library_local_rag_search_service import (
            LibraryLocalRagSearchService,
        )

        seam = LibraryLocalRagSearchService(runtime.app)
        lookup = slug_lookup_from(runtime.slug_to_source)
        cfg = runtime.service.config.search
        original = getattr(cfg, "default_search_mode", None)

        qualifying: dict[str, list[str]] = {m: [] for m in MODES}
        any_dup: dict[str, list[str]] = {m: [] for m in MODES}
        excluded: dict[str, list[str]] = {m: [] for m in MODES}
        direct_state: dict[str, list[str]] = {m: [] for m in MODES}
        try:
            for mode in MODES:
                cfg.default_search_mode = mode
                for q in golden:
                    scope = build_query_scope(runtime.slug_to_source, q)
                    result = runtime.run(
                        seam.search(q.query, SOURCE_TYPES, "rag", top_k=K, scope=scope)
                    )
                    rows, _backend, err = _extract_rows(result)
                    if err:
                        print(f"    !! {mode}/{q.id}: {err}")
                        continue
                    slugs = [row_slug(r, lookup) for r in rows]
                    counts = Counter(slugs)
                    freed = len(slugs) - len(counts)          # slots a router would free
                    hit = any(s in counts for s in q.relevant_slugs)
                    if q in direct:
                        direct_state[mode].append(
                            f"{q.id}({'HIT' if hit else 'MISS'},+{freed})"
                        )
                    if freed > 0:
                        any_dup[mode].append(
                            f"{q.id}[{q.category}](+{freed},{'HIT' if hit else 'MISS'})"
                        )
                        if not hit:
                            # A query can only be RESCUED if it has a target to
                            # find AND that target is reachable in this mode.
                            # Two exclusions, both structural:
                            #  - `negative` queries have NO relevant slug, so
                            #    `hit` is False by construction and a miss is
                            #    the CORRECT outcome, not a rescuable failure.
                            #  - `prompt` targets have no vector index at all
                            #    (B2 gave prompts an FTS sub-leg deliberately
                            #    without one), so in a vector mode no freed slot
                            #    can admit them -- they are not in the index.
                            if not q.relevant_slugs:
                                excluded[mode].append(f"{q.id}(negative: no target)")
                            elif q.category == "prompt" and mode == "semantic":
                                excluded[mode].append(f"{q.id}(prompt: not vector-indexed)")
                            else:
                                qualifying[mode].append(q.id)
        finally:
            cfg.default_search_mode = original

    print(f"\n=== POPULATION 1 current state (can a HIT be rescued? no) ===")
    for mode in MODES:
        print(f"  {mode}: {direct_state[mode]}")

    print(f"\n=== POPULATION 2 (DISPLACEMENT), K={K} ===")
    for mode in MODES:
        print(f"  {mode}: queries with duplicate-doc slots = {len(any_dup[mode])}")
        for entry in any_dup[mode]:
            print(f"      {entry}")
        if excluded[mode]:
            print(f"  {mode}: excluded as structurally unrescuable = {excluded[mode]}")
        print(f"  {mode}: QUALIFYING (slots freed AND a reachable target missed) = "
              f"{len(qualifying[mode])} {qualifying[mode]}")

    # A query already HITTING cannot be rescued -- a router could only
    # preserve or harm it. The bar's precedent (PRF clause 1) counts
    # RESCUES, so the qualifying population is misses only.
    print(f"\n=== VERDICT vs pre-registered bar of {BAR} (inherited verbatim) ===")
    rescuable_by_mode = {}
    for mode in MODES:
        direct_missed = [e for e in direct_state[mode] if "(MISS," in e]
        total = len(direct_missed) + len(qualifying[mode])
        rescuable_by_mode[mode] = total
        print(f"  {mode:9s}: direct-missed={len(direct_missed)} "
              f"{[e.split('(')[0] for e in direct_missed]} + "
              f"displacement-qualifying={len(qualifying[mode])} -> rescuable={total}")
    best = max(rescuable_by_mode.values())
    print(f"  exposure (currently-HIT queries a reorder could only move DOWN): "
          f"{ {m: sum(1 for e in any_dup[m] if ',HIT)' in e) for m in MODES} }")
    print(f"  BEST-MODE RESCUABLE: {best}  vs BAR {BAR}")
    print(f"  RESULT: {'CLEARS' if best >= BAR else 'BELOW'} bar -> "
          f"{'probe' if best >= BAR else 'NULL, arc ends -- no probe, no production code'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
