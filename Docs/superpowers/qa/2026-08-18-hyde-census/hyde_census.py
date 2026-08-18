"""TASK-18514 census: how many golden queries can HyDE even act on?

Run::

    RAG_EVAL=1 PYTHONPATH=$(pwd) <venv>/bin/python \
        Docs/superpowers/qa/2026-08-18-hyde-census/hyde_census.py

HyDE replaces the query embedding with a hypothetical answer's embedding, so
it acts ONLY on the semantic leg. Its reachable population is therefore:

    queries that currently MISS, whose target is vector-indexed at all.

Two exclusions are structural and were registered in the task BEFORE this ran:
`negative` queries have no target (a miss is correct), and `prompt` targets
have no vector index (B2 gave prompts an FTS sub-leg deliberately without
one), so no change to the query embedding can reach them.

The census reports, per mode and per query, whether the target is retrieved
at k=10 and -- for misses -- whether the target is reachable AT ALL by
deepening k, which separates "the query embedding points elsewhere" (HyDE's
actual theory of the case) from "this document is not findable by vector
search on this corpus".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

K = 10          # the gate's k
K_DEEP = 200    # "reachable at all?" depth, following PRF's own deep probe
BAR = 5         # registered in the task before this file existed
MODES = ("semantic", "hybrid")


def classify(
    has_target: bool,
    category: str,
    hit_at_k: bool,
    hit_at_deep: bool,
) -> str:
    """Bucket one query for the HyDE-reachable population.

    HyDE rewrites the query embedding, so it can only act on a query that
    misses today AND whose target is in the vector index at all. Two
    exclusions are structural and were registered in TASK-18514 before this
    ran:

    * `negative` queries have no target, so a miss is the CORRECT outcome —
      it is not a rescuable failure; and
    * `prompt` targets have no vector index (B2 gave prompts an FTS sub-leg
      deliberately without one), so no change to the query vector reaches
      them.

    Args:
        has_target: Whether the query has any ground-truth slug.
        category: The golden-set category.
        hit_at_k: Target retrieved at the gate's k.
        hit_at_deep: Target retrieved at the deep probe depth.

    Returns:
        One of ``"hitting"``, ``"excluded_negative"``, ``"excluded_prompt"``,
        ``"reachable"`` (misses now, present deeper — HyDE's case), or
        ``"unfindable"`` (absent even at depth).
    """
    if hit_at_k:
        return "hitting"
    if not has_target:
        return "excluded_negative"
    if category == "prompt":
        return "excluded_prompt"
    return "reachable" if hit_at_deep else "unfindable"


def main() -> int:
    """Run the census and print the reachable population against the bar.

    Returns:
        0 when the census covered every query; 1 when any query errored, in
        which case no verdict is claimed.
    """
    import tempfile

    if os.environ.get("RAG_EVAL") != "1":
        raise SystemExit("refusing to run without RAG_EVAL=1 (this builds a real index)")

    import tldw_chatbook
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids, slug_lookup_from
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import (
        SOURCE_TYPES,
        _extract_rows,
        build_query_scope,
    )

    print(f"PROVENANCE tldw_chatbook <- {tldw_chatbook.__file__}")
    assert str(REPO) in tldw_chatbook.__file__, f"WRONG TREE: expected under {REPO}"

    corpus, golden = load_fixtures()
    print(f"PROBE PROOF: corpus={len(corpus)} docs, golden={len(golden)} queries")

    errors: list[str] = []
    rows_seen = 0
    # per mode: query id -> (hit@K, hit@K_DEEP)
    state: dict[str, dict[str, tuple[bool, bool]]] = {m: {} for m in MODES}

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
                    hits: list[bool] = []
                    for depth in (K, K_DEEP):
                        try:
                            result = runtime.run(
                                seam.search(
                                    q.query, SOURCE_TYPES, "rag", top_k=depth, scope=scope
                                )
                            )
                            rows, _backend, err = _extract_rows(result)
                        except Exception as exc:                    # noqa: BLE001
                            rows, err = [], f"{type(exc).__name__}: {exc}"
                        if err:
                            errors.append(f"{mode}/{q.id}@{depth}: {err}")
                            hits.append(False)
                            continue
                        rows_seen += len(rows)
                        docs = set(rows_to_doc_ids(rows, lookup))
                        hits.append(any(s in docs for s in q.relevant_slugs))
                    state[mode][q.id] = (hits[0], hits[1])
        finally:
            cfg.default_search_mode = original

    by_id = {q.id: q for q in golden}
    print(f"\nPROBE PROOF: rows retrieved across the census = {rows_seen}")
    print(f"PROBE PROOF: errors = {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"    !! {e}")
        print("NO VERDICT CLAIMED -- population incomplete.")
        return 1

    print(f"\n=== reachable population for HyDE (acts on the SEMANTIC leg only) ===")
    reachable: dict[str, list[str]] = {}
    for mode in MODES:
        excl_neg, excl_prompt, deep_only, unfindable, hitting = [], [], [], [], []
        bucket = {
            "hitting": hitting,
            "excluded_negative": excl_neg,
            "excluded_prompt": excl_prompt,
            "reachable": deep_only,
            "unfindable": unfindable,
        }
        for qid, (h10, h200) in state[mode].items():
            q = by_id[qid]
            bucket[classify(bool(q.relevant_slugs), q.category, h10, h200)].append(qid)
        reachable[mode] = deep_only
        print(f"\n  --- {mode} ---")
        print(f"    hitting at k={K}                      : {len(hitting)}")
        print(f"    excluded, negative (no target)        : {len(excl_neg)}")
        print(f"    excluded, prompt (no vector index)    : {len(excl_prompt)} {excl_prompt}")
        print(f"    MISS but found by k={K_DEEP} (HyDE's case): {len(deep_only)} {deep_only}")
        print(f"    MISS and absent even at k={K_DEEP}       : {len(unfindable)} {unfindable}")

    best_mode = max(MODES, key=lambda m: len(reachable[m]))
    best = len(reachable[best_mode])
    print(f"\n=== VERDICT vs the bar of {BAR}, registered before this ran ===")
    print(f"  best mode = {best_mode}: {best} reachable {reachable[best_mode]}")
    print(
        f"  RESULT: {'CLEARS' if best >= BAR else 'BELOW'} bar -> "
        f"{'probe HyDE' if best >= BAR else 'NULL, arc ends -- no probe, no production code'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
