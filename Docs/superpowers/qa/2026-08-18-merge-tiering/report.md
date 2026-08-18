# Four-seam merge tiering: the failure mode cannot occur here (TASK-17955)

Date: 2026-08-18 · measured on the gated fixture (172 docs, 60 golden
queries), plain path, current dev. No network, no spend.

## What tiering would change, precisely (CORRECTED after review)

**The first version of this report argued from CUTS and that was wrong.**
It said tiering matters only when `rows_returned >= top_k`, so a corpus that
never fills the window is safe. Qodo (PR #1801, finding 1) pointed out the
hole: **MRR and NDCG consume ORDER, not membership.** Moving a relevant row
from rank 1 to rank 3 changes both, with nothing cut. A cut-based argument
cannot establish that ordering is unobservable, and a `rows >= top_k` review
trigger would have left tiering closed through exactly the corpus change that
made it matter.

## The correct question, and the measurement

Reordering can move a score only for a query that has **≥2 rows** AND **at
least one retrieved row that is relevant** — otherwise every permutation
scores identically.

| k | queries with >1 row | of those, with ground truth | with a RELEVANT row retrieved |
|---|---|---|---|
| 10 | **1** | 1 | **0** |
| 20 | **1** | 1 | **0** |

The single multi-row query is `ng-mains-supply` (6 rows, the one TASK-17755's
fallback rescued). It *has* a relevant document — and **that document is not
among the rows retrieved** (`relevant_at_ranks: NONE`). So every ordering of
its six rows scores exactly the same, and no other query has more than one
row to order at all.

**Tiering is therefore unobservable on this corpus** — not because nothing is
cut, but because there is no relevant row whose rank an ordering could move.

Reproduce with `tier_observability_census.py` in this directory:

```
RAG_EVAL=1 PYTHONPATH=<repo> .venv/bin/python \
    Docs/superpowers/qa/2026-08-18-merge-tiering/tier_observability_census.py
```

It prints, per depth, the >1-row queries, whether each has ground truth, and
the ranks at which relevant rows actually landed.

## Why the fixture was NOT extended (AC#1, deliberately not taken)

AC#1 asks for a query with a multi-row primary seam and a rescued seam. To
make the comparison mean anything, that query would also have to **fill the
window** — ≥10 rows on a path whose current maximum is 6.

Authoring it means choosing its shape, and its shape decides which ordering
wins. That is the trap TASK-16072's kill condition named six days ago in this
same programme: *a class invented to give a feature something to show
measures the class*. It applies with equal force to a class invented to give
a comparison something to distinguish. `Tests/RAG_Eval/README.md`'s admission
protocol admits only what today's pipeline is **measured** to fail, and
nothing here fails.

## Decision (AC#3): untiered is the measured choice; TASK-16071's note retires

Not "untiered is fine because nobody proved otherwise" — **untiered is
indistinguishable from tiered on every query this instrument contains, for a
structural reason that is cheap to re-check**: the plain path does not fill a
retrieval window.

What a future arc would need, stated so nobody re-derives it: a corpus where
the plain four-seam path returns, **for at least one query, ≥2 rows including
a relevant one**. That is the condition under which an ordering becomes
observable at all — and it is a property of the corpus and the construction
together, not something a merge change can create. The script above is the
check; the earlier `rows >= top_k` formulation was wrong and would have
under-triggered.
