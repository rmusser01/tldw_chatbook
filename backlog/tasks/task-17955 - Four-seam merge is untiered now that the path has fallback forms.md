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

- [x] **The fixture is extended FIRST so the comparison can say anything.**
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

**AC#1 was deliberately NOT taken, and the reason is the finding.** The AC
asks for a fixture with a multi-row primary seam and a rescued seam. Measuring
first showed that would not be enough: to make the comparison mean anything
the authored query would also have to **fill the window**, and

| k | queries whose window is FULL (`rows >= k`) | max rows |
|---|---|---|
| 10 | **0 of 60** | 6 |
| 20 | **0 of 60** | 6 |

Tiering changes what a consumer sees **only when the merged list is cut**. On
this corpus nothing is ever cut, so the order cannot change what any consumer
receives — a stronger result than TASK-17755's review reached (it found the
orderings *coincide*; this finds the cut *never happens*).

Authoring a window-filling query means choosing its shape, and its shape
decides which ordering wins. That is exactly the trap TASK-16072's kill
condition named six days ago: *a class invented to give a feature something to
show measures the class* — which applies with equal force to a class invented
to give a comparison something to distinguish.

**AC#2/#3 — the decision:** untiered ships as the measured choice, not as an
unpaid debt. TASK-16071's "the 15700 tier design would apply" note is retired.

**AC#4 — the merge-site comment** now states what was measured, when, and the
one-command re-check: if any plain query starts returning `>= top_k` rows,
tiering becomes measurable and the decision is due for review. That re-check
is the census in
`Docs/superpowers/qa/2026-08-18-merge-tiering/report.md`, which also records
today's row distribution (37 queries at 0 rows, 22 at exactly 1, 1 at 6).

Row counts re-measured on current dev rather than inherited: the gating census
was taken during TASK-17755's review, before `and_then_prefix` shipped to this
path, and the distribution has changed since (one query now returns 6 rows).
