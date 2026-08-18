# TASK-18155 — granularity router: census, and the NULL it produced

**Verdict: NULL. No probe was run, no production code exists, the arc ends
here.** One query is rescuable against a pre-registered bar of five.

Reproduce: `RAG_EVAL=1 PYTHONPATH=$(pwd) <venv>/bin/python
Docs/superpowers/qa/2026-08-18-granularity-census/granularity_census.py`

## The bar, and an honesty note about when it was set

**I ran the census before registering the bar.** That is the wrong order —
AC#1 asks for the bar first, precisely so it cannot be chosen to fit the
number. To keep it non-circular I did not invent one: the bar is **≥5
qualifying queries, inherited verbatim** from the two prior P2c candidates
that used it — PRF's clause 1 ("≥5 of the 22 plain-failing queries reach
their target in the second pass's top-10", TASK-15965) and the clarification
gate (0 against a pre-registered bar of 5, TASK-16072). Its precedent is
what makes it defensible here, not my result.

The bar counts **rescues**, following PRF: a query already retrieving its
target cannot be rescued, only preserved or harmed.

## The premise has two mechanisms, and the cheap count only sees one

"Chunk vs document granularity" can change an outcome two ways:

1. **Direct** — the query's own relevant document is multi-chunk.
2. **Displacement** — some *other* document occupies several top-k slots.
   `canonicalize.py` states it: *"one document can occupy several of the
   top-k slots"*. The top-k cut happens at **row** level and rows collapse to
   documents only afterwards, so a 3-chunk document spends 3 of 10 slots and
   only 8 distinct documents can appear. Collapsing at retrieval would free
   those slots.

Mechanism 2 is the stronger form of the hypothesis and a corpus count cannot
see it, so this census ran the live index for it.

`plain` is excluded throughout: it returns whole items, never chunks, so it
is **already** document-granular and structurally cannot move.

## What the corpus is

Measured with the real `ChunkingService` at the harness profile's settings
(`hybrid_basic`, `chunk_size=384` words, `overlap=64`) — not inferred from
word counts:

| | |
|---|---|
| corpus documents | 172 |
| total chunks | **179** |
| documents yielding exactly 1 chunk | **168** |
| documents yielding >1 chunk | **4** |

The four: `note-fennimore-changeover` (3), `media-larkspur-turbine` (3),
`conv-drayton-conveyor` (3), `note-saltmarsh-hide` (2).

**For 168 of 172 documents a chunk IS the document**, so granularity is a
no-op for them by construction.

## Population 1 — direct: 4 of 60, and 3 of the 4 are already hits

All four are `keyword` category.

| query | semantic | hybrid |
|---|---|---|
| `kw-fennimore-changeover` | HIT | HIT |
| `kw-larkspur-turbine` | HIT | HIT |
| `kw-drayton-conveyor` | HIT | HIT |
| `kw-plant-maintenance-record` | **MISS** | HIT |

So the direct mechanism offers **one** rescuable query, in `semantic` only.

**And displacement is not even its mechanism.** `kw-plant-maintenance-record`
has **zero** duplicate slots in semantic — its document's chunks do not reach
the top-10 at all. This is the query the fusion arc already characterised:
*"plain rank 1, semantic ABSENT from top-10 (present ~rank 22 in the
index)"*. That is a similarity miss, not a slot-consumption one, and
**fusion weighting already rescued it** — which is why hybrid reads HIT.

## Population 2 — displacement: 0 qualifying, in both modes

| mode | queries with duplicate slots | of those, already HIT | qualifying |
|---|---|---|---|
| `semantic` | 11 | 9 | **0** |
| `hybrid` | **0** | — | **0** |

The two semantic misses are **structurally unrescuable**, and the census
excludes them with the reason recorded:

- `neg-honeybee-colony` is a **negative** query with no relevant document.
  Retrieving nothing is the *correct* outcome — not a failure to rescue.
- `pm-incident-timeline` is a **prompt** query, and prompt targets have **no
  vector index at all** (B2 gave prompts an FTS sub-leg deliberately without
  one). No freed slot can admit a document that is not in the index.

**`hybrid` — the shipped default mode — has zero duplicate slots on all 60
queries.** A granularity router would have nothing whatsoever to act on
there.

### A defect this census had, and how it was caught

The first run reported 2 qualifying queries. Both were artifacts: a
`negative` query has an empty `relevant_slugs`, so the hit test is False **by
construction** and all 7 negatives register MISS whether retrieval behaved or
not. That is the same species as this programme's recent instrument failures
— a value that means "not applicable" rendered identically to one that means
"failed". The exclusions above are now explicit in the script rather than
implicit in the reader.

## Verdict

| | semantic | hybrid |
|---|---|---|
| direct, currently missed | 1 | 0 |
| displacement qualifying | 0 | 0 |
| **rescuable** | **1** | **0** |
| exposure: currently-HIT queries with duplicate slots a reorder could only move DOWN | **9** | 0 |

**1 against a bar of 5 → BELOW. NULL.**

The exposure column is the same asymmetry PRF failed on: nothing to gain, and
nine currently-correct semantic queries whose slots would be re-ordered. The
one rescuable query is already fixed in the shipped mode by a cheaper
mechanism.

## Relationship to the retired knobs and to `expand_document` (AC#5)

This matters because a router would be the **third** surface over one
capability:

1. `include_parent_docs` and siblings — **retired by TASK-16174** for being
   inert.
2. `expand_document` — the gated, pull-based tool that replaced them, which
   fetches surrounding context *after* retrieval, on demand.
3. A granularity router — would decide *before* retrieval.

The task required a router's census to clear a **higher** bar than "it might
help", precisely to avoid re-adding surface 1 under a new name. It cleared
nothing: the mechanism has no population in the shipped mode, and the single
candidate elsewhere is a similarity miss that fusion already handles. Where a
user genuinely needs whole-document context, surface 2 exists and is
demand-driven, which is strictly better than a router guessing per query.

## What review changed (PR #1812)

Two substantive findings, both accepted; neither moved the verdict, and the
second is worth recording because it is the *third* instance of one species
in this programme:

**Production parity in the chunk count.** The census chunked raw
`CorpusDoc.content`, but conversations are indexed as a transcript with the
sender prepended (`conversation_document` builds `f"{sender}: {content}"`),
so the text measured was one word shorter than the text that exists in the
index. Real defect: a document sitting on a boundary would be counted wrong.
**Re-measured under parity across all 31 conversation fixtures: 0 chunk
counts change and the multi-chunk set is identical.** The fix is in the
script regardless, because a future corpus would not be so forgiving.

**An errored query used to shrink the population silently.** The first
version printed the error and continued, then reported NULL from whatever
survived — so a failed search could have hidden a qualifying query and ended
the investigation below a bar it never actually faced. The census now
collects errors and, if any occurred, **claims no verdict at all** and exits
non-zero. This run reports `errors: 0 -- population COMPLETE (60 queries x 2
modes)`, which is now printed evidence rather than my assurance.

That is the same failure shape as the negatives bug above and as TASK-18255
one arc earlier: **a number that means "could not measure" rendered
identically to one that means "measured, found nothing".** Three instances in
two arcs is a pattern, not a coincidence.

**Pins added.** `Tests/RAG_Eval/test_granularity_census.py` (13 tests) covers
the classification logic, including the negatives case. Mutation-verified:
disabling the negative exclusion reds
`test_negative_query_is_excluded_not_qualifying` and restoring it greens —
so the pin proves the repair rather than merely coexisting with it.

## What would reopen this

The census is a property of **this corpus**, and it is honest to say so: 168
of 172 documents fit in one chunk. A corpus of long documents would move
every number here. Re-run the script if the fixture gains substantial
long-form content — the reachable population, not the idea's appeal, is what
should decide.
