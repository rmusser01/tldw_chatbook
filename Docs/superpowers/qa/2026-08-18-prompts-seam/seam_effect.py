"""TASK-18255: what does wiring the harness prompts seam actually change?

Run::

    RAG_EVAL=1 PYTHONPATH=$(pwd) <venv>/bin/python \
        Docs/superpowers/qa/2026-08-18-prompts-seam/seam_effect.py

Reports, in `plain` (the only mode that uses the Library's four-seam fan-out):

* whether `_search_prompts` reports itself AVAILABLE at all -- the distinction
  the metrics table cannot show, and the one TASK-17855 misread;
* per query, hit/miss for all five prompt goldens (an aggregate cell of 0.200
  means one of five, which would hide four misses);
* the COST the harness comment predicted: the seam appends rows to every
  plain fan-out, so non-prompt queries can move too;
* whether the run logged "prompts seam failed." -- with a service wired, an
  exception still returns `(True, [])`, so a zero would otherwise be
  ambiguous between no-match and threw.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

K = 10
SEAM_FAILED_MARKER = "prompts seam failed."


def main() -> int:
    """Measure the seam's availability, per-query hits and fan-out cost.

    Returns:
        0 on a complete run; 1 if any query errored or the seam logged a
        failure (either makes a zero unreadable).

    Raises:
        SystemExit: ``RAG_EVAL`` is not set to ``"1"``. This builds a real
            index over the full fixture corpus, so it never runs by accident.
    """
    import tempfile

    if os.environ.get("RAG_EVAL") != "1":
        raise SystemExit("refusing to run without RAG_EVAL=1 (this builds a real index)")

    import tldw_chatbook
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
        SeamState,
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

    # Capture loguru so a seam exception cannot hide behind a warning nobody reads.
    from loguru import logger

    seam_failures: list[str] = []
    sink_id = logger.add(
        lambda m: seam_failures.append(str(m)) if SEAM_FAILED_MARKER in str(m) else None,
        level="WARNING",
    )

    corpus, golden = load_fixtures()
    rt = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rt = build_eval_runtime(corpus, tmp)

            # 1. Is the seam AVAILABLE? `(False, [])` and `(True, [])` render
            #    the same downstream; ask the seam directly.
            seam = LibraryLocalRagSearchService(rt.app)
            svc = getattr(rt.app, "prompt_scope_service", None)
            print(f"\nPROBE PROOF: app.prompt_scope_service = {type(svc).__name__}")
            state, rows = rt.run(seam._search_prompts("prompt", K))
            print(f"PROBE PROOF: _search_prompts state = {state} "
                  f"(UNAVAILABLE/FAILED both mean the rows are meaningless)")
            print(f"PROBE PROOF: smoke-query rows = {len(rows)}")
            # TASK-18903: `if not state` would be ALWAYS FALSE -- every Enum
            # member is truthy. Compare with `is`, or this guard goes inert.
            if state is not SeamState.AVAILABLE:
                print("  SEAM UNAVAILABLE -- nothing below would be meaningful.")
                return 1

            lookup = slug_lookup_from(rt.slug_to_source)
            cfg = rt.service.config.search
            cfg.default_search_mode = "plain"

            errors: list[str] = []
            results: dict[str, tuple[bool, int, int]] = {}
            for q in golden:
                scope = build_query_scope(rt.slug_to_source, q)
                try:
                    res = rt.run(
                        seam.search(q.query, SOURCE_TYPES, "rag", top_k=K, scope=scope)
                    )
                    rws, _b, err = _extract_rows(res)
                except Exception as exc:                       # noqa: BLE001
                    rws, err = [], f"{type(exc).__name__}: {exc}"
                if err:
                    errors.append(f"{q.id}: {err}")
                    continue
                docs = rows_to_doc_ids(rws, lookup)
                n_prompt = sum(
                    1 for r in rws
                    if (r.get("provenance") or {}).get("source_type") == "prompt"
                )
                results[q.id] = (
                    any(s in set(docs) for s in q.relevant_slugs), len(rws), n_prompt
                )
    finally:
        # The runtime owns SQLite connections, a Chroma store and the event
        # loop its pools are bound to; the scratch dir is about to vanish, so
        # they must be released first. Every harness caller closes it -- this
        # probe forgot to, which leaks handles into the rest of the process.
        if rt is not None:
            try:
                rt.close()
            except Exception as exc:                          # noqa: BLE001
                print(f"NOTE: runtime.close() failed after the run: {exc!r}")
        logger.remove(sink_id)

    by_id = {q.id: q for q in golden}
    print(f"\nPROBE PROOF: queries measured = {len(results)}, errors = {len(errors)}")
    for e in errors:
        print(f"    !! {e}")
    print(f"PROBE PROOF: '{SEAM_FAILED_MARKER}' logged = {len(seam_failures)} time(s)")
    if errors or seam_failures:
        print("NO VERDICT CLAIMED -- a zero here would be unreadable.")
        return 1

    print("\n=== the five prompt goldens, PER QUERY (an aggregate hides misses) ===")
    prompts = [q for q in golden if q.category == "prompt"]
    for q in prompts:
        hit, n, n_p = results[q.id]
        print(f"  {'HIT ' if hit else 'MISS'} {q.id:22s} rows={n:2d} prompt-rows={n_p:2d}"
              f"  target={q.relevant_slugs[0]}")
    hits = sum(1 for q in prompts if results[q.id][0])
    print(f"  -> {hits}/{len(prompts)} (aggregate cell would read {hits/len(prompts):.3f})")

    print("\n=== the COST the harness comment predicted ===")
    others = [q for q in golden if q.category != "prompt"]
    with_prompt_rows = [q.id for q in others if results[q.id][2] > 0]
    print(f"  non-prompt queries whose plain fan-out now contains prompt rows: "
          f"{len(with_prompt_rows)} of {len(others)}")
    for qid in with_prompt_rows[:15]:
        hit, n, n_p = results[qid]
        print(f"      {qid:30s} [{by_id[qid].category:20s}] rows={n:2d} "
              f"prompt-rows={n_p} hit={hit}")
    print(f"  non-prompt queries currently HITTING: "
          f"{sum(1 for q in others if results[q.id][0])} of {len(others)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
