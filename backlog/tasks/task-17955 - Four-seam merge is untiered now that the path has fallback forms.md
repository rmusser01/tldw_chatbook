---
id: TASK-17955
title: Four-seam merge is untiered now that the path has fallback forms
status: Done
assignee: []
created_date: '2026-08-18'
labels: [rag, retrieval]
dependencies: []
priority: medium
---

## Description (the why)

TASK-16071 made the four-seam merge rank-fair (`interleave_rankings` keyed by
`_keyword_row_identity`) and left a note at the merge site: if this path ever
gained fallback match forms, TASK-15700's **tier** design should apply — a
row found by a primary form should outrank one found only by a fallback,
rather than interleaving with it purely by position.

TASK-17755 gave the path exactly those forms (`and_then_prefix`: an AND
primary per sub-leg, a per-token prefix fallback where the primary returned
zero rows). The merge was deliberately **left untiered**, and the reason is
worth preserving rather than treating as an oversight: `and_then_prefix` was
*measured* untiered (MRR 0.304 → 0.326 on the plain cells), so tiering it now
would ship an ordering nobody has measured, on the strength of a symmetry
argument. This programme has retired several such arguments on measurement.

The open effect: deeper in a merged list, a fallback row can interleave ahead
of a primary one from another seam. Whether that costs anything is unknown —
which is the point.

## Acceptance Criteria (the what)

- [ ] **NOT DONE, DELIBERATELY — see the outcome below.** The fixture is extended FIRST so the comparison can say anything.
      TASK-17755's final review established that tiering is provably the
      IDENTITY on today's corpus: **0 of 60 queries mix primary and fallback
      rows, and 0 have more than one primary row**, so a tiered-vs-untiered
      comparison would close on a meaningless `+0.000`. The fixture needs a
      query that has a multi-row primary seam AND a rescued (fallback-only)
      seam, which is the only shape where the orderings differ
- [x] Only then: the tiered and untiered merges are measured against each
      other on the gated instrument's `plain` cells, same corpus and queries
- [x] A decision is recorded: tier the merge, or record untiered as the
      measured choice and retire TASK-16071's note so it stops reading as an
      unpaid debt
- [x] Whichever ships, the merge-site comment states which construction was
      measured and when — the note that prompted this task was accurate but
      undated, which is why it survived two arcs unresolved

## Note (2026-08-18, TASK-17755 final review)

The real exposure is **displacement, not loss**: a rescued seam's loose rows
can interleave ahead of a good seam's deeper primary rows under a `top_k`
cut. Ship-untiered was accepted for TASK-17755 because tiering is equally
unmeasurable on today's corpus — not because untiered was shown safe. Record
in the outcome that TASK-17755 closed the *form* divergence with the engine
leg while leaving an *ordering* divergence.

## Outcome (2026-08-18): untiered is the MEASURED choice; the note retires

**AC#1 is left UNCHECKED on purpose (Qodo PR-1801 finding 2).** Ticking a
criterion whose own outcome text says it was skipped would be a false project
record; the box stays open and the deviation is stated here. The AC
asks for a fixture with a multi-row primary seam and a rescued seam. Measuring
first showed that would not be enough: to make the comparison mean anything
the authored query would also have to **fill the window**, and

| k | queries with >1 row | with ground truth | with a RELEVANT row retrieved |
|---|---|---|---|
| 10 | 1 | 1 | **0** |
| 20 | 1 | 1 | **0** |

**The first version of this outcome argued from CUTS and was wrong** (Qodo
PR-1801 finding 1): MRR and NDCG consume ORDER, so a reordering changes a
score with nothing cut. The correct condition is narrower — a reordering can
move a score only for a query with ≥2 rows AND a relevant row among them.
This corpus has none: 59 of 60 plain queries return 0 or 1 row, and the single
6-row query (`ng-mains-supply`, rescued by TASK-17755's fallback) retrieves no
relevant document at all, so every permutation of it scores identically.

Authoring a window-filling query means choosing its shape, and its shape
decides which ordering wins. That is exactly the trap TASK-16072's kill
condition named six days ago: *a class invented to give a feature something to
show measures the class* — which applies with equal force to a class invented
to give a comparison something to distinguish.

**AC#2/#3 — the decision:** untiered ships as the measured choice, not as an
unpaid debt. TASK-16071's "the 15700 tier design would apply" note is retired.

**AC#4 — the merge-site comment** states what was measured, when, and a
**runnable** re-check (Qodo PR-1801 finding 3: the earlier note promised a
"one-command re-check" and offered prose):
`Docs/superpowers/qa/2026-08-18-merge-tiering/tier_observability_census.py`,
which prints per depth the >1-row queries and the ranks at which relevant rows
landed. The trigger is corrected too — **≥2 rows including a relevant one**,
not `rows >= top_k`, which would have under-triggered for the reason finding 1
identified.

Row counts re-measured on current dev rather than inherited: the gating census
was taken during TASK-17755's review, before `and_then_prefix` shipped to this
path, and the distribution has changed since (one query now returns 6 rows).
