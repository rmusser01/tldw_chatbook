# Four-seam merge tiering: the failure mode cannot occur here (TASK-17955)

Date: 2026-08-18 · measured on the gated fixture (172 docs, 60 golden
queries), plain path, current dev. No network, no spend.

## What tiering would change, precisely

Tiering makes a primary-form row outrank a fallback-form row instead of
interleaving with it by position. That changes what a consumer sees **only
when the merged list is CUT** — i.e. when `rows_returned >= top_k`. If the
window is never full, every row survives whatever the order, and the two
merges are observationally identical to any downstream metric.

## The measurement

| k | queries whose window is FULL (`rows >= k`) | max rows seen |
|---|---|---|
| 10 | **0 of 60** | 6 |
| 20 | **0 of 60** | 6 |

Row distribution on the plain path today: **37** queries return 0 rows,
**22** return exactly 1, **1** returns 6 (`ng-mains-supply` — a negation
query, and the single query TASK-17755's fallback rescued into multi-row
territory).

**So the displacement failure mode cannot occur on this corpus at all.**
This is a stronger statement than TASK-17755's review made. That review found
the two orderings *coincide* (no query mixes primary and fallback rows). The
measurement above shows something more basic: **no row is ever cut**, so the
order cannot change what any consumer receives, whatever the mix.

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
the plain four-seam path returns **≥ top_k rows for at least one query**.
That is a property of the corpus and the construction together, not something
a merge change can create — and the same census in this report (`rows >= k`,
two depths) is the one-command check for whether it has become true.
