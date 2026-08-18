# Four-seam AND-strictness: what it costs, and what the alternatives buy (TASK-3997)

Date: 2026-08-18 · Branch `docs/task-3997-and-strictness` off dev `59300e234`
Measured on the gated fixture corpus (172 docs) and golden set (60 queries,
53 with ground truth; 7 negatives excluded from scoring). No network, no spend.

## AC#1 — the baseline: what AND-strictness costs today

`build_fts_match_query` (`Library/library_fts_query.py`) OR-groups a term's
spelling variants but joins **every group with AND**, so one absent term
zeroes the seam.

On the golden set, the Library's plain four-seam path returns:

| | queries |
|---|---|
| **zero rows** | **39 of 60** (32 of the 53 ground-truthed) |
| exactly 1 row | 21 |
| more than 1 row | **0** |

The original filing measured "2 of 15" on the smaller P1 set. On the current
instrument it is **two thirds of the query set returning nothing**.

## AC#2 — the alternatives, measured rather than argued

Three constructions, same corpus, same queries, MRR over the 53 scored:

| construction | MRR | zero-row | what it does |
|---|---|---|---|
| **AND-strict** (shipped) | 0.396 | 32/53 | one absent term ⇒ nothing |
| **pure prefix OR** | **0.261** | 0/53 | every query returns 20–30 rows |
| **`and_then_prefix`** (the engine's shipped shape) | **0.423** | 25/53 | AND primary; prefix **only** where the primary found nothing |

**The naive fix is measurably worse.** Replacing AND with OR/prefix rescues
every zero-row query and *halves* ranking quality — 20–30 loosely-matching
rows per query on a 172-document corpus bury the answers that AND was getting
right. Recall bought at that price is not recall worth having.

**`and_then_prefix` is better than both**, and its shape is why: the 21
queries whose AND primary already returns a row are **untouched by
construction**, so the only queries it can change are the ones currently
returning nothing. It rescues **7 of 32** zero-row queries with the *right*
document, for **+0.027 MRR**, and cannot regress the precise answers.

## The second argument, which is not about metrics

The Library screen's plain **Search** mode is this path; **RAG Answer** on the
same screen is the engine leg, which has shipped `and_then_prefix` since
TASK-15700. A user switching between the two tabs on one screen gets **two
different matching rules** — an inflection miss ("guy tension" vs
"tensions") answers in one and returns nothing in the other. Adopting
`and_then_prefix` on the four-seam path collapses that divergence to zero as
a side effect of the better-measuring option.

## Method notes (so the numbers can be trusted or refuted)

- The first A/B run reported **identical** results for both arms. That was a
  monkeypatch that never took: `library_local_rag_search_service` does
  `from ... import build_fts_match_query`, binding the name at import, so
  patching the defining module changed nothing. Re-patched at the consumer
  namespace, with a **call counter proving the arm ran** (173 calls). An
  intervention that silently does nothing produces a perfect-looking null.
- `and_then_prefix` is computed per query from both runs (AND result where it
  returned rows, prefix result otherwise), which is exactly the engine's
  fallback rule; it is not a third live run.
