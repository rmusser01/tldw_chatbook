---
id: TASK-18155
title: 'P2c candidate 6: granularity router, census before probe'
status: To Do
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

- [ ] A bar and a kill condition are registered BEFORE any measurement,
      naming what evidence licenses production code and what result ends the
      arc
- [ ] The census runs first and answers, per query and from the corpus,
      how many golden queries could change outcome under a different
      retrieval granularity
- [ ] A below-bar census ends the arc with a recorded null — no probe, no
      production code — and is recorded beside the other retired premises in
      `Tests/RAG_Eval/README.md`
- [ ] If the census clears the bar, the probe follows TASK-15965's shape:
      gains AND losses by query id, guard populations derived at probe time,
      no control read as a pipeline property unless the variables it holds
      fixed are named
- [ ] The relationship to TASK-16174's retired parent-inclusion knobs and to
      `expand_document` is stated, so a third surface over one capability is
      a deliberate choice rather than an accident
