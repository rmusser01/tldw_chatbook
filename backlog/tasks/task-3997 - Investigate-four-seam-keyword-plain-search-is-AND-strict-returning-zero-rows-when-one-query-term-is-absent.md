---
id: TASK-3997
title: >-
  Investigate: four-seam keyword (plain) search is AND-strict, returning zero
  rows when one query term is absent
status: Done
assignee: []
created_date: '2026-08-09 05:17'
updated_date: '2026-08-11 18:06'
labels:
  - rag
  - retrieval
  - investigation
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness (TASK-3894). build_fts_match_query (Library/library_fts_query.py, near L94-101) joins every query term group with AND, so a single term with no match anywhere in the corpus zeroes the entire query result for that seam. On the P1 golden query set this produced 0 rows for 2 of 15 keyword-category queries and only a partial match (1 of 2 relevant documents) for a third, despite the corpus containing documents that satisfy most of the query intent. Whether to move to OR-with-ranking, a configurable AND or OR strictness, or keep AND-strict behavior as an intentional precision-over-recall choice for the plain four-seam mode is a product decision, not a mechanical fix; this task is scoped to investigating and proposing an approach, not implementing one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The current AND-strict behavior effect on the P1 golden query set is documented (which queries return zero or partial results and why) as a baseline for the decision.
- [x] #2 At least one alternative (for example OR-with-ranking, or configurable strictness) is proposed with its precision-versus-recall tradeoffs for the four-seam keyword path.
- [x] #3 A product decision on whether and how to change the AND-join behavior is recorded, in this task or a follow-up task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cross-reference (2026-08-11, TASK-15020/B2): TASK-15400 is the ENGINE-leg half of this same AND-strictness question. This task covers the Library's four-seam path (build_fts_match_query); 15400 covers RAGService._escape_fts5_query, where the effect is far larger (zero rows for 40 of 60 golden queries). 15400's AC#8 overlaps this task's AC#3 — the product decision on AND-join behaviour should be taken once for both paths, or one should explicitly defer to the other. 15400 also carries the measured comparison table and two constraints (the vector-blind fixture kw-plant-maintenance-record, which OR-of-tokens loses; and the load-bearing per-token quoting).

OUTCOME OF THE OTHER HALF (2026-08-12, TASK-15400 CLOSED — this task is NOT closed by it, and 15400's AC#8 was answered by DEFERRING to this one): the engine leg now ships `and_stopword_trim` (AND over the CONTENT tokens; function words dropped). The four-seam path is UNCHANGED and still AND-joins every term group. The divergence is deliberate and documented in Tests/RAG_Eval/README.md's known-defects list, for a measured reason rather than a scheduling one: the engine's construction was chosen by a hybrid-FUSION measurement — every constraint that decided it (the vector-blind rescue; the scoped 1.000 -> 0.429 collapse) is about rows competing inside a fused top-k — and the four-seam path has no fusion, no ranking, and no leg whose rank another sub-leg's row count can displace. So its construction has to be decided on its own evidence. Two numbers to start from, both measured during 15400's sweep: reusing `build_fts_match_query` on the ENGINE leg rescues 1 of the 40 zero-row golden queries (it is AND-joined too), and a stopword-trimmed AND rescues 1 — i.e. on that corpus the two constructions are the same order of magnitude, and the dominant blocker on both paths is absent CONTENT words, not function words. Also relevant to the product judgment: the Library screen's plain "Search" mode is THIS path (confirmed live on 2026-08-12 — a function-word query that the engine leg answers returns nothing in Search mode), so a user switching between "Search" and "RAG Answer" on the same screen gets two different matching rules today.

UPDATE (2026-08-13, TASK-15700 closed): the ENGINE leg's construction moved again — it now ships `and_then_prefix` (FULL AND over every token as the primary — function words INCLUDED — with per-token prefix matching as a fallback for a sub-leg whose primary returned zero rows). So the divergence between the two paths grew rather than shrank: the engine leg now answers an inflection miss ("guy tension" against a document saying "tensions") that the four-seam path still cannot, and the stopword trim that this note describes as the engine's construction now lives only in the engine's FALLBACK. The product judgment this task owns is unchanged and is now worth more: a user switching between "Search" and "RAG Answer" on the same screen gets two matching rules that differ in two dimensions, not one.
<!-- SECTION:NOTES:END -->

## Investigation outcome (2026-08-18)

**AC#1 — the baseline.** The four-seam path returns **zero rows for 39 of 60**
golden queries (32 of the 53 ground-truthed) and exactly one row for the other
21; never more than one. The original filing's "2 of 15" was the smaller P1
set — on the current instrument the effect is two thirds of the query set.

**AC#2 — the alternatives, measured.** Over the 53 scored queries:

| construction | MRR | zero-row |
|---|---|---|
| AND-strict (shipped) | 0.396 | 32 |
| pure prefix OR | **0.261** | 0 |
| `and_then_prefix` (the engine's shape) | **0.423** | 25 |

The naive fix is **measurably worse**: OR rescues every zero-row query and
halves ranking quality, because 20–30 loosely-matching rows on a 172-document
corpus bury the answers AND was getting right. `and_then_prefix` beats both,
and its shape is why — the 21 queries whose primary already returns a row are
untouched by construction, so it can only change queries that currently return
nothing. It rescues **7 of 32** with the right document.

Full report + method notes:
`Docs/superpowers/qa/2026-08-18-four-seam-and-strictness/report.md`. One method
note matters: the first A/B reported identical arms because the monkeypatch
never took (`from ... import` binds at import time), which is how an
intervention that does nothing produces a perfect-looking null; re-run at the
consumer namespace with a call counter proving the arm executed.

**AC#3 — the decision (owner, 2026-08-18): ADOPT `and_then_prefix`.** Filed
as **TASK-17755** (implementation, with the untouched-primaries property as a
pinned AC). The owner also asked for the deeper recall question to be carried
forward: **TASK-17855** investigates the 25 residual zero-row queries, whose
blocker is absent CONTENT words — a different question that no token
rearrangement can answer, and where a null is an acceptable outcome.

This also answers the divergence TASK-15400's AC#8 deferred here: adopting the
engine's construction collapses the two matching rules on the Library screen
to one.

## CORRECTION (2026-08-18, from TASK-17755's final review)

**The "7 of 32 rescued" figure in this task's AC#2 table was wrong, and the
owner's decision was taken partly on it.** The delivered implementation
rescues **1** zero-row query with the right document (`kw-thimble-relay`),
not 7. A second query (`ng-mains-supply`, a negation) gains 6 rows, none
relevant.

**Why this task over-counted.** Its `and_then_prefix` arm was not the shipped
construction. It was computed *analytically* from two whole-query runs — AND
result where the query returned rows, prefix result otherwise — whereas the
shipped construction falls back **per sub-leg**, and its prefix form is the
engine's (stopword-trimmed), not this probe's crude `OR` of quoted prefixes
over the raw query string. The probe's arm was an approximation of the real
thing and it over-estimated. **The harness was never committed**, which is
why the number could not be reconciled by inspection; the repo's own
construction matrix scores this construction on the engine leg at 3 rescues,
so 7 was always the outlier.

**The decision still stands, on better evidence than the number that
prompted it.** The gate measured the shipped change: overall MRR
0.304 → 0.326, NDCG 0.296 → 0.318, keyword-category cells +0.062, with every
`semantic` and `hybrid` cell holding at +0.000 — verified independently by
the reviewer, including a control mutation that reproduces the OLD baseline
bit-for-bit, proving attribution. What changed is the *magnitude* of the win,
not its direction or its risk profile: the untouched-primaries property that
made this the low-risk arm is pinned on construction, not on the rescue
count.

**The methodological lesson, which is this task's real residue:** an
analytically-composed arm is a different construction from the one you will
ship, and a probe that is not committed cannot be reconciled later. Compose
arms by running the real code path, and commit the harness.
