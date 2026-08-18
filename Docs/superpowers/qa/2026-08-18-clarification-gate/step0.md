# Step 0 — the clarification gate's premise, answered from the corpus (TASK-16072 AC#1)

Run 2026-08-18 against the shipped fixture (172 docs, 60 golden queries).
Bar and kill condition were registered first, in `bar.md`, unchanged since.

## The census (CORRECTED after review — see the method note below)

Condition 2 asks whether a query's interpretations point at **different
documents in this corpus**. The first version of this census answered that
with the count of RELEVANCE LABELS, which was wrong: a label set encodes what
the fixture intends, not what the corpus contains. Corrected measure — for
each query, how many corpus documents contain every content word (plausible
readings), versus how many are labelled relevant:

| | queries |
|---|---|
| corpus holds an unlabelled alternative reading (`match > rel`) | **1** |
| no alternative reading in the corpus | 59 |

The full per-query table for all 60 queries is
`per-query-census.md`; the script is `census.py` and prints a probe-proof
line (`docs with non-empty text: 172 of 172`) before its table.

**The one qualifying query is `negation`:** *"which outstation does not take a
standard maintenance record"* — 3 corpus documents contain its content words,
1 is labelled relevant. That is not gate-shaped either: the query is
**precise**, and the other two match because the corpus asserts what the
query excludes. A clarifying question cannot repair a negation the index
cannot express — which is the same reason `negation` sits at 0.000 across
every retrieval construction this programme has measured.

Qualifying under all three conditions: **0**. Under condition 2 alone: **1**.
Against a bar of 5, **the kill condition fires either way.**

## Two review claims, checked rather than accepted

- **"The census misses corpus ambiguity" — RIGHT, and it changed the method.**
  Label count ≠ corpus ambiguity. Re-measured against document text; the
  answer moved from 0 to 1, still far below the bar.
- **"`plant maintenance record` has industrial and botanical readings in the
  corpus" — FACTUALLY WRONG, verified in one query.** Exactly **one** corpus
  document mentions "plant" at all (`note-saltmarsh-hide`, the estuary/
  botanical one) and it **is** the labelled relevant document. There is no
  industrial-plant document to disambiguate against. The claim was plausible —
  "plant" is the textbook polyseme — which is why it was checked instead of
  believed.
- **"Every ≤3-word query has exactly one relevant document" — MY ERROR, now
  corrected.** `Nimbus-14 firmware rollback` is three words and has two.
  The corrected statement: of the 12 queries of ≤3 words, **11** have exactly
  one relevant document and one (`kw-nimbus-rollback`) has two — and its two
  are both correct, so it is a multi-target query, not an ambiguous one. My
  own script printed both facts and I did not cross-check them.

## Method note (the second time today this reflex paid)

The corrected census initially reported `match = 0` for nearly every query —
which would have made it look like a clean confirmation. It was reading
**empty strings**: `CorpusDoc`'s text field is `content`, and the probe asked
for `body`/`text`. A measure that silently reads nothing produces a
perfect-looking null. `census.py` now prints how many documents it actually
read before printing any result.

## Verdict: NULL — the arc ends here, per AC#3

No production code was written, and none should be. The finding is that this
corpus contains **no query a clarification gate could fire on usefully**, and
the reason generalises past this fixture: every failing class fails for a
reason a question cannot repair — `negation` because the corpus asserts what
the query excludes; `prompt` and the residual zero-row queries on **absent
content words**; `paraphrase` / `vocabulary_mismatch` in `plain` because the
query shares no content word with its target. No answer to any question puts
a missing document into an index.

Reaching the bar would have required **authoring ambiguity fixtures**, which
the pre-registered kill condition declared a kill rather than an invitation:
`Tests/RAG_Eval/README.md`'s admission protocol admits only what today's
pipeline is *measured* to fail, and a class invented to give a feature
something to show measures the class.

**This is the fifth P2c premise to die before it was built** — after
`expansion`, `acronym`, `compositional` and PRF (TASK-15965). Four of the
five died on measurement; this one died on a census, which is cheaper and
should be the first move for the remaining candidates.
