"""Qodo PR-1801 finding 1: order-sensitive metrics (MRR/NDCG) can change
without any CUT. So the real question is not "is the window full" but:
is there a query whose merged list has >=2 rows AND a relevant document
whose RANK a reordering could move?
"""
import tempfile
import pathlib
from Tests.RAG_Eval.harness.goldenset import load_fixtures
from Tests.RAG_Eval.harness.ingest import build_eval_runtime
from Tests.RAG_Eval.harness.runner import run_eval

corpus, golden = load_fixtures()
tmp = pathlib.Path(tempfile.mkdtemp(prefix="tierorder-"))
runtime = build_eval_runtime(corpus, tmp)
try:
    for k in (10, 20):
        rep = run_eval(runtime, golden, k=k, modes=("plain",))
        qs = rep.modes["plain"].queries
        multi = [q for q in qs if q.rows_returned > 1]
        # A reordering can only move a SCORE if the query has ground truth.
        scorable = [q for q in multi if q.relevant_slugs]
        print(f"k={k}: >1 row: {len(multi)}   of those WITH ground truth: {len(scorable)}")
        for q in multi:
            rel = set(q.relevant_slugs or ())
            hits = [i for i, d in enumerate(q.retrieved_doc_ids, 1) if d in rel]
            print(f"   {q.query_id:26s} rows={q.rows_returned} relevant={len(rel)} "
                  f"relevant_at_ranks={hits or 'NONE'}")
finally:
    runtime.close()
