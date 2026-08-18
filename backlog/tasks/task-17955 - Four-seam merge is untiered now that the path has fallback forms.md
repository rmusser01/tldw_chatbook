---
id: TASK-17955
title: Four-seam merge is untiered now that the path has fallback forms
status: To Do
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

- [ ] **The fixture is extended FIRST so the comparison can say anything.**
      TASK-17755's final review established that tiering is provably the
      IDENTITY on today's corpus: **0 of 60 queries mix primary and fallback
      rows, and 0 have more than one primary row**, so a tiered-vs-untiered
      comparison would close on a meaningless `+0.000`. The fixture needs a
      query that has a multi-row primary seam AND a rescued (fallback-only)
      seam, which is the only shape where the orderings differ
- [ ] Only then: the tiered and untiered merges are measured against each
      other on the gated instrument's `plain` cells, same corpus and queries
- [ ] A decision is recorded: tier the merge, or record untiered as the
      measured choice and retire TASK-16071's note so it stops reading as an
      unpaid debt
- [ ] Whichever ships, the merge-site comment states which construction was
      measured and when — the note that prompted this task was accurate but
      undated, which is why it survived two arcs unresolved

## Note (2026-08-18, TASK-17755 final review)

The real exposure is **displacement, not loss**: a rescued seam's loose rows
can interleave ahead of a good seam's deeper primary rows under a `top_k`
cut. Ship-untiered was accepted for TASK-17755 because tiering is equally
unmeasurable on today's corpus — not because untiered was shown safe. Record
in the outcome that TASK-17755 closed the *form* divergence with the engine
leg while leaving an *ordering* divergence.
