---
id: task-17755
title: Adopt and_then_prefix on the four-seam keyword path
status: To Do
assignee: []
created_date: '2026-08-18'
labels: [rag, retrieval]
dependencies: []
priority: medium
---

## Description (the why)

TASK-3997 investigated the four-seam (plain Search) path's AND-strictness and
the owner took the decision on 2026-08-18: **adopt `and_then_prefix`** — the
construction the engine leg has shipped since TASK-15700. That task was scoped
to investigate and propose; this one implements.

The measured case (`Docs/superpowers/qa/2026-08-18-four-seam-and-strictness/report.md`,
172-doc corpus, 53 ground-truthed golden queries):

| construction | MRR | zero-row queries |
|---|---|---|
| AND-strict (shipped) | 0.396 | 32 of 53 |
| pure prefix OR | 0.261 | 0 |
| **`and_then_prefix`** | **0.423** | 25 |

Two properties make this the low-risk arm. The 21 queries whose AND primary
already returns a row are **untouched by construction** — the fallback only
fires where the primary returned nothing — so there is no regression path to
the answers AND currently gets right. And pure OR was measured *worse* than
the status quo, which kills the naive alternative rather than leaving it as a
plausible-sounding option.

It also ends a live divergence: the Library screen's **Search** mode is this
path while **RAG Answer** is the engine leg, so one screen has two matching
rules today (an inflection miss answers in one and returns nothing in the
other).

## Acceptance Criteria (the what)

- [ ] The four-seam keyword path applies an `and_then_prefix` construction:
      the existing AND-of-variant-groups stays the primary, and a per-token
      prefix form runs **only** for a sub-leg whose primary returned zero rows
- [ ] A query whose AND primary returns rows produces byte-identical results
      to today — pinned by a test, since this is the property that makes the
      change low-risk
- [ ] The zero-row rescue is demonstrated on the golden set, and the measured
      MRR does not fall below the AND-strict baseline (0.396 on the corpus in
      TASK-3997's report)
- [ ] Plain Search and RAG Answer answer the same inflection-miss query on the
      same corpus — the divergence TASK-3997 documented is gone
- [ ] The gated retrieval suite reads `PASSED: No regression. 105 metric(s)`;
      note that its `plain` cells CAN legitimately move here, unlike in
      reranking arcs — if they do, the move is the deliverable and the
      baselines are re-stamped deliberately with the reason recorded
