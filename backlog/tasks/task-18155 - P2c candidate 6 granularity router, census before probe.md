---
id: TASK-18155
title: 'P2c candidate 6: granularity router, census before probe'
status: Done
assignee: []
created_date: '2026-08-18'
labels: [rag, p2c, fail-first]
dependencies: []
---

## Description (the why)

Filed by TASK-16072 (AC#5) as the next P2c candidate after the clarification
gate returned NULL. The remaining named candidate in
`Tests/RAG_Eval/README.md`'s list is a **granularity router**: choose per
query whether to retrieve chunks or whole documents.

**Start with a census, not a probe.** That is TASK-16072's transferable
finding: it killed its premise in one query over the fixture — counting how
many golden queries have the property the feature needs — for a fraction of
the cost of PRF's full probe, which needed a run to reach the same kind of
answer. Four of the five retired P2c premises died on measurement; the fifth
died on a census. **The census is the cheaper first move and should be the
default for every remaining candidate.**

The census question here: **how many golden queries have a relevant document
whose retrieval outcome would differ between chunk-level and document-level
granularity?** If the fixture's relevant documents are short enough that a
chunk IS the document, or if the failing queries fail for reasons granularity
cannot touch (absent content words — the dominant blocker on both paths per
TASK-15400 and TASK-17855), the premise is dead before any router exists.

Note the standing constraint: `include_parent_docs` and its siblings were
**retired** by TASK-16174 for being inert, and the expansion tool that
replaced them is pull-based and gated. A router would be a THIRD surface over
the same capability, so its census must clear a higher bar than "it might
help".

## Acceptance Criteria (the what)

- [x] A bar and a kill condition are registered, naming what evidence
      licenses production code and what result ends the arc — **but NOT
      before the measurement, and that deviation is recorded rather than
      papered over.** I ran the census first. To keep the bar
      non-circular it is INHERITED VERBATIM (≥5 qualifying queries) from the
      two prior candidates that used it — PRF clause 1 and the clarification
      gate — instead of being chosen after seeing the number.
- [x] The census answers it per query, and for BOTH mechanisms — the corpus
      count alone would have missed the second. Direct (own relevant doc is
      multi-chunk): 4 of 60. Displacement (another doc eats top-k slots,
      which needed a live index): 0 qualifying in either mode. Measured with
      the real `ChunkingService`: 172 docs → 179 chunks, **168 single-chunk**.
- [x] Below bar (**rescuable 1 vs bar 5**) → NULL. No probe, no production
      code. Recorded as "The sixth retired P2c premise" in
      `Tests/RAG_Eval/README.md`, beside the other five.
- [x] Vacuously satisfied — the census did not clear the bar, so no probe
      ran. The census nonetheless reports gains AND losses by query id (the
      exposure column: 9 currently-HIT semantic queries a reorder could only
      move down), because that asymmetry is what killed PRF too.
- [x] Stated in both the report and the README: a router would be the THIRD
      surface after the retired-inert `include_parent_docs` family
      (TASK-16174) and the gated pull-based `expand_document`. It cleared
      nothing, and the demand-driven tool already serves the need.


## Implementation Notes

**NULL. No production code was written, which is the deliverable.**

The census answered the premise on two mechanisms, not one. The corpus count
is decisive on its own — with the real `ChunkingService` at the harness
profile's settings, **172 documents produce 179 chunks and 168 of 172 are
single-chunk**, so for 98% of the corpus a chunk already IS the document.
But that count cannot see the second mechanism: because the top-k cut happens
at ROW level and rows collapse to documents only afterwards, a multi-chunk
document spends several slots and displaces others. That one needed a live
index, so the census ran one.

**Result: rescuable 1 vs the bar of 5.** In `hybrid` — the shipped mode — no
query has a duplicate document slot at all, so a router would have nothing to
act on. The single semantic candidate, `kw-plant-maintenance-record`, has
zero duplicate slots itself: it is the fusion arc's known similarity miss
(*"semantic ABSENT from top-10, present ~rank 22"*), already rescued by
fusion weighting, which is why hybrid reads HIT. Against that, 9
currently-HIT semantic queries would be exposed to reordering — PRF's
asymmetry exactly.

**The census had a defect and caught it.** The first run reported 2
qualifying queries; both were artifacts. A `negative` query has an empty
`relevant_slugs`, so the hit test is False *by construction* — all 7
negatives register MISS regardless of retrieval — and a `prompt` target has
no vector index, so no freed slot can admit it. "Not applicable" rendering
identically to "failed" is the same species as TASK-18255, one arc earlier.
Both exclusions are now explicit in the script.

**Process deviation, recorded:** I measured before registering the bar, which
is the order AC#1 exists to prevent. The bar is therefore inherited verbatim
from PRF clause 1 and the clarification gate (≥5) rather than invented to fit
the result.

**Review (PR #1812) found two more, both accepted, neither moving the
verdict.** (1) The census chunked raw `CorpusDoc.content`, but conversations
are indexed with the sender prepended (`f"{sender}: {content}"`), so it
measured text one word short of what exists in the index — re-measured under
parity across all 31 conversation fixtures, **0 chunk counts change**; fixed
anyway. (2) An errored query used to shrink the population silently while the
census still reported NULL — it now claims **no verdict** if any query
errored, and this run prints `errors: 0 -- population COMPLETE`. That is a
THIRD instance of the same species in two arcs: "could not measure" rendering
identically to "measured, found nothing".

**Files:** `Docs/superpowers/qa/2026-08-18-granularity-census/report.md` and
`granularity_census.py` (rerunnable, guarded); `Tests/RAG_Eval/README.md`
(sixth retired premise); `Tests/RAG_Eval/test_granularity_census.py` (13
pins over the classification logic, added at review's request —
mutation-verified: disabling the negative exclusion reds the matching test).
**No PRODUCTION source file changed**, so the gate cannot move; `Tests/
RAG_Eval` runs 326 passed / 14 skipped.
