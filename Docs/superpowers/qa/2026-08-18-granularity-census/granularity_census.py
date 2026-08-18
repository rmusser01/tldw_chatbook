"""TASK-18155 census: could a granularity router change any golden outcome?

Run::

    RAG_EVAL=1 PYTHONPATH=$(pwd) <venv>/bin/python \
        Docs/superpowers/qa/2026-08-18-granularity-census/granularity_census.py

"Chunk vs document granularity" has TWO mechanisms, and a corpus count only
sees the first:

1. DIRECT -- the query's own relevant document is multi-chunk, so retrieving
   it whole rather than in pieces changes what is scored for that query.
2. DISPLACEMENT -- some OTHER document occupies several of the top-k row
   slots (`canonicalize.py`: "one document can occupy several of the top-k
   slots"), because the top-k cut happens at ROW level and rows collapse to
   documents only afterwards. Collapsing at retrieval would free those slots
   for documents currently pushed out. This one needs a live index.

Only a query whose relevant document is CURRENTLY MISSED can be rescued, and
only if that document is reachable in the mode at all -- see `classify`.

The pure helpers (`parity_text`, `classify`) are importable without the
`RAG_EVAL` gate so `Tests/RAG_Eval/test_granularity_census.py` can pin them;
the gate guards `main`, which builds a real index.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Tests.RAG_Eval.harness.canonicalize import canonical_source_type  # noqa: E402

CHUNK_SIZE = 384          # hybrid_basic profile, the harness's own
CHUNK_OVERLAP = 64
K = 10                    # run_eval's default
BAR = 5                   # inherited verbatim from PRF clause 1 + the clarification gate
MODES = ("semantic", "hybrid")   # `plain` returns whole items: no chunks, no granularity

#: Reasons a missed query still cannot be rescued by freeing a top-k slot.
EXCLUDED_NEGATIVE = "negative: no target to find"
EXCLUDED_UNINDEXED = "prompt: not vector-indexed"


def parity_text(source_type: str, content: str) -> str:
    """Return the text production actually chunks for a document.

    Conversations are indexed as a transcript with each message's sender
    prepended (`ingestion_indexing.conversation_document` builds
    ``f"{sender}: {content}"``), so chunking the raw fixture content would
    measure a slightly shorter document than the one that exists in the
    index. Media and notes index their content unchanged.

    Args:
        source_type: The fixture document's source type.
        content: The document's raw fixture content.

    Returns:
        The text the indexer would chunk for this document.
    """
    if source_type == "conversation":
        return f"user: {content}"
    return content


def classify(
    relevant_slugs: tuple[str, ...],
    category: str,
    mode: str,
    hit: bool,
) -> tuple[str, str]:
    """Decide whether a query could be RESCUED by a granularity router.

    A query already retrieving its target cannot be rescued -- a router could
    only preserve or harm it -- so the bar's population (which counts
    rescues, following PRF clause 1) is misses only. Two kinds of miss are
    nonetheless structurally unrescuable:

    * a `negative` query has no relevant document, so ``hit`` is False **by
      construction** and the miss is the CORRECT outcome; and
    * a `prompt` target has no vector index at all (B2 gave prompts an FTS
      sub-leg deliberately without one), so in a vector mode no freed slot
      can admit a document that is not in the index.

    Args:
        relevant_slugs: The query's ground-truth fixture slugs.
        category: The query's golden-set category.
        mode: Retrieval mode being measured (``semantic`` or ``hybrid``).
        hit: Whether any relevant slug appeared in the retrieved rows.

    Returns:
        ``(verdict, reason)`` where verdict is one of ``"hit"``,
        ``"qualifying"`` or ``"excluded"``; ``reason`` is empty unless
        excluded.
    """
    if hit:
        return "hit", ""
    if not relevant_slugs:
        return "excluded", EXCLUDED_NEGATIVE
    if category == "prompt" and mode == "semantic":
        return "excluded", EXCLUDED_UNINDEXED
    return "qualifying", ""


def slot_summary(
    slugs: list[str], relevant_slugs: tuple[str, ...]
) -> tuple[int, bool, int]:
    """Summarize one query's retrieved rows for the displacement question.

    Args:
        slugs: One fixture slug per retrieved ROW, in rank order, uncollapsed.
        relevant_slugs: The query's ground-truth slugs.

    Returns:
        ``(freed, hit, unmapped)`` where ``freed`` is how many top-k slots a
        granularity router would recover (rows minus distinct documents),
        ``hit`` is whether any relevant document was retrieved, and
        ``unmapped`` counts rows no fixture claimed.

        ``unmapped`` is reported, not discarded: every unmapped row is a
        DISTINCT ``unknown:*`` key, so a mapping failure would drive ``freed``
        to 0 and the census would report "no displacement" when it had in
        fact measured nothing.
    """
    counts = Counter(slugs)
    freed = len(slugs) - len(counts)
    hit = any(s in counts for s in relevant_slugs)
    unmapped = sum(1 for s in slugs if s.startswith("unknown:"))
    return freed, hit, unmapped


def row_slug(row: Mapping[str, Any], lookup: Mapping[tuple[str, str], str]) -> str:
    """Map ONE retrieved row to its fixture slug, without collapsing.

    Mirrors `canonicalize.rows_to_doc_ids`, but per row: this census needs to
    know how many SLOTS a document occupies, which the collapsing version
    deliberately discards.

    Args:
        row: One row as the Library search seam emits it.
        lookup: ``(canonical source_type, source_id) -> slug``, from
            `canonicalize.slug_lookup_from`.

    Returns:
        The fixture slug, or a synthesized ``"unknown:<type>:<id>"`` when no
        fixture claims the row (kept, never dropped -- an unrecognized row
        still occupies a top-k slot).
    """
    prov = row.get("provenance") or {}
    st = canonical_source_type(prov.get("source_type"))
    sid = str(row.get("source_id", ""))
    stripped = sid.split("_", 1)[-1] if "_" in sid else sid
    for key in ((st, sid), (st, stripped)):
        if key in lookup:
            return lookup[key]
    return f"unknown:{st}:{sid}"


def main() -> int:
    """Run the census and print the verdict against the pre-registered bar.

    Returns:
        Process exit code: 0 when the census completed over the FULL query
        population, 1 when any query errored (an incomplete population cannot
        support a NULL, so no verdict is claimed).
    """
    if os.environ.get("RAG_EVAL") != "1":
        raise SystemExit("refusing to run without RAG_EVAL=1 (this builds a real index)")

    # One contiguous local-import group. These are deferred to call time
    # rather than module scope so the pure helpers above stay importable
    # without the gate (and without building a retrieval stack) for
    # `Tests/RAG_Eval/test_granularity_census.py`.
    import tempfile

    import tldw_chatbook
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )
    from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

    from Tests.RAG_Eval.harness.canonicalize import slug_lookup_from
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import (
        SOURCE_TYPES,
        _extract_rows,
        build_query_scope,
    )

    # A census that silently measured another checkout would be worthless,
    # and this programme already shipped one instrument that could not see
    # what it was asked about (TASK-18255).
    print(f"PROVENANCE tldw_chatbook <- {tldw_chatbook.__file__}")
    assert str(REPO) in tldw_chatbook.__file__, f"WRONG TREE: expected under {REPO}"

    corpus, golden = load_fixtures()
    print(f"PROBE PROOF: corpus={len(corpus)} docs, golden={len(golden)} queries")

    # --- population 1: DIRECT, from the corpus alone ---
    svc = ChunkingService()
    chunks_per_doc = {
        d.slug: len(
            svc.chunk_text(
                parity_text(d.source_type, d.content),
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                method="words",
            )
        )
        for d in corpus
    }
    multi = {s: n for s, n in chunks_per_doc.items() if n > 1}
    print(
        f"PROBE PROOF: docs chunked={len(chunks_per_doc)}, "
        f"total chunks={sum(chunks_per_doc.values())}, multi-chunk docs={len(multi)}"
    )
    for s, n in sorted(multi.items(), key=lambda kv: -kv[1]):
        print(f"    multi-chunk: {s:38s} {n} chunks")

    direct = [q for q in golden if any(s in multi for s in q.relevant_slugs)]
    print(f"\n=== POPULATION 1 (DIRECT): {len(direct)} of {len(golden)} queries ===")
    for q in direct:
        print(
            f"    {q.id:30s} [{q.category}] -> "
            f"{[s for s in q.relevant_slugs if s in multi]}"
        )

    # --- population 2: DISPLACEMENT, needs a live index ---
    errors: list[str] = []
    total_rows = 0
    total_unmapped = 0
    qualifying: dict[str, list[str]] = {m: [] for m in MODES}
    any_dup: dict[str, list[str]] = {m: [] for m in MODES}
    excluded: dict[str, list[str]] = {m: [] for m in MODES}
    direct_state: dict[str, list[str]] = {m: [] for m in MODES}

    with tempfile.TemporaryDirectory() as tmp:
        runtime = build_eval_runtime(corpus, tmp)
        seam = LibraryLocalRagSearchService(runtime.app)
        lookup = slug_lookup_from(runtime.slug_to_source)
        cfg = runtime.service.config.search
        original = getattr(cfg, "default_search_mode", None)
        try:
            for mode in MODES:
                cfg.default_search_mode = mode
                for q in golden:
                    scope = build_query_scope(runtime.slug_to_source, q)
                    try:
                        result = runtime.run(
                            seam.search(q.query, SOURCE_TYPES, "rag", top_k=K, scope=scope)
                        )
                        rows, _backend, err = _extract_rows(result)
                    except Exception as exc:                      # noqa: BLE001
                        rows, err = [], f"{type(exc).__name__}: {exc}"
                    if err:
                        # A skipped query SHRINKS the population, and a
                        # shrunken population cannot support a NULL: a
                        # qualifying query could be hiding behind the error.
                        errors.append(f"{mode}/{q.id}: {err}")
                        continue
                    slugs = [row_slug(r, lookup) for r in rows]
                    freed, hit, unmapped = slot_summary(slugs, q.relevant_slugs)
                    total_rows += len(slugs)
                    total_unmapped += unmapped
                    verdict, reason = classify(q.relevant_slugs, q.category, mode, hit)
                    if q in direct:
                        direct_state[mode].append(
                            f"{q.id}({'HIT' if hit else 'MISS'},+{freed})"
                        )
                    if freed > 0:
                        any_dup[mode].append(
                            f"{q.id}[{q.category}](+{freed},{'HIT' if hit else 'MISS'})"
                        )
                        if verdict == "qualifying":
                            qualifying[mode].append(q.id)
                        elif verdict == "excluded":
                            excluded[mode].append(f"{q.id}({reason})")
        finally:
            cfg.default_search_mode = original

    print("\n=== POPULATION 1 current state (a HIT cannot be rescued) ===")
    for mode in MODES:
        print(f"  {mode}: {direct_state[mode]}")

    print(f"\n=== POPULATION 2 (DISPLACEMENT), K={K} ===")
    for mode in MODES:
        print(f"  {mode}: queries with duplicate-doc slots = {len(any_dup[mode])}")
        for entry in any_dup[mode]:
            print(f"      {entry}")
        if excluded[mode]:
            print(f"  {mode}: excluded as structurally unrescuable = {excluded[mode]}")
        print(
            f"  {mode}: QUALIFYING (slots freed AND a reachable target missed) = "
            f"{len(qualifying[mode])} {qualifying[mode]}"
        )

    print(f"\n=== VERDICT vs pre-registered bar of {BAR} (inherited verbatim) ===")
    print(
        f"  PROBE PROOF: rows canonicalized={total_rows}, unmapped={total_unmapped} "
        f"({100.0 * total_unmapped / total_rows if total_rows else 0.0:.1f}%) -- "
        "an unmapped row is a DISTINCT unknown key, so a mapping failure would "
        "drive freed-slot counts to 0 and fake a 'no displacement' result"
    )
    if total_rows == 0:
        print("  NO VERDICT CLAIMED: zero rows canonicalized -- nothing was measured.")
        return 1
    if errors:
        print(f"  !! {len(errors)} query/queries ERRORED -- population INCOMPLETE:")
        for e in errors:
            print(f"       {e}")
        print("  NO VERDICT CLAIMED. A shrunken population cannot support a NULL;")
        print("  a qualifying query could be hiding behind any one of these.")
        return 1

    rescuable_by_mode = {}
    for mode in MODES:
        direct_missed = [e for e in direct_state[mode] if "(MISS," in e]
        total = len(direct_missed) + len(qualifying[mode])
        rescuable_by_mode[mode] = total
        print(
            f"  {mode:9s}: direct-missed={len(direct_missed)} "
            f"{[e.split('(')[0] for e in direct_missed]} + "
            f"displacement-qualifying={len(qualifying[mode])} -> rescuable={total}"
        )
    best = max(rescuable_by_mode.values())
    exposure = {m: sum(1 for e in any_dup[m] if ",HIT)" in e) for m in MODES}
    print(f"  exposure (currently-HIT queries a reorder could only move DOWN): {exposure}")
    print(f"  errors: 0 -- population COMPLETE ({len(golden)} queries x {len(MODES)} modes)")
    print(f"  BEST-MODE RESCUABLE: {best}  vs BAR {BAR}")
    print(
        f"  RESULT: {'CLEARS' if best >= BAR else 'BELOW'} bar -> "
        f"{'probe' if best >= BAR else 'NULL, arc ends -- no probe, no production code'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
