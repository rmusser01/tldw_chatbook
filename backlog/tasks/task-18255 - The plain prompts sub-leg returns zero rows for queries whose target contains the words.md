---
id: TASK-18255
title: >-
  The plain prompts sub-leg returns zero rows for queries whose target contains
  the words
status: To Do
assignee: []
labels: [rag, retrieval]
dependencies: []
priority: medium
---

## Description (the why)

Found by TASK-17855's census of residual zero-row queries. On the gated
fixture, the `prompt` category behaves like this:

| mode | queries returning rows | queries finding the target |
|---|---|---|
| `plain` | **0 of 5** | 0 |
| `semantic` | 5 of 5 | 0 |
| `hybrid` | 5 of 5 | 1 |

**The plain path returns nothing at all for any prompt query** — and that is
not explained by AND-strictness, which is what TASK-3997 and TASK-17755 were
about. Four of the five queries have high lexical overlap with their target,
and one has **every content word present**: `"saved prompt for chasing a
supplier about a late order"` against `prompt-vendor-chaser`, whose name is
*"Saved prompt: chasing a late order"* and whose body names the supplier, the
order and the chase.

`prompts_fts` indexes five columns including `name`, so the words are
indexed. A construction that reaches nothing when every term is present
points at the sub-leg — its scoping, its query shape, or whether it runs at
all on this path — rather than at how the MATCH expression is built.

This is a **defect candidate, not a recall-broadening candidate**, which
matters: broadening was measured at a 34% MRR cost (TASK-3997), while fixing
a seam that returns zero rows for exact-term queries has no precision cost by
construction.

## Acceptance Criteria (the what)

- [ ] The cause is named from evidence: whether the plain prompts sub-leg
      runs, what MATCH expression it issues, and against which columns —
      not inferred from the construction
- [ ] `"saved prompt for chasing a supplier about a late order"` returns
      `prompt-vendor-chaser` on the plain path, or the reason it cannot is
      recorded with the same specificity
- [ ] The other four prompt queries are reported with it, since they share
      the seam and three of them also have partial overlap
- [ ] The gated suite still reads `PASSED: No regression. 105 metric(s)`;
      note that `plain`'s `category.prompt.*` cells are currently 0.000, so
      a fix should MOVE them — if it does, that movement is the deliverable
      and the baseline is re-stamped deliberately
