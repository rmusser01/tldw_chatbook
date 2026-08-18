---
id: TASK-18255
title: >-
  Wire a prompts seam into the eval harness so plain-mode prompt cells measure
  something
status: To Do
assignee: []
labels: [rag, retrieval]
dependencies: []
priority: medium
---

## Description (the why)

**CORRECTED 2026-08-18, before any work: this task was filed as a production
defect and it is not one.** TASK-17855's census observed that the `plain`
mode returns zero rows for all five `prompt` golden queries — including one
whose target contains every content word — and concluded the Library's
prompts sub-leg was broken. It is not.

The cause is in the **instrument**, and the harness says so in its own
comment (`Tests/RAG_Eval/harness/ingest.py`, the fake app's construction):

> `prompt_scope_service=None` … *"Leaving it None means the harness's plain
> column reports 0.000 for prompts while the shipped app's plain mode does
> find them."*

`_search_prompts` returns `(False, [])` when that attribute is absent — the
seam reports itself **unavailable**, which is not the same as matching
nothing. Production wires it (`app.py:5682`,
`build_prompt_scope_service`). So the plain `category.prompt.*` cells are
**vacuous by construction**, and TASK-17855's reading of them as a retrieval
failure was wrong.

**What remains is the work the harness comment defers**, and it is worth
doing for the reason the wrong conclusion demonstrates: a 0.000 cell that
means "not measured" is indistinguishable, to every reader, from one that
means "measured and found nothing". This programme has now mis-read it once.

The harness comment also names the cost, which is why it was deferred rather
than smuggled in: the seam appends its rows to **every** plain fan-out, so
wiring it moves plain-mode numbers for non-prompt queries too.

## Acceptance Criteria (the what)

- [x] **The cause is named from evidence** (done at filing-correction time,
      2026-08-18): the harness's fake app sets `prompt_scope_service=None`,
      so `_search_prompts` returns `(False, [])` — seam UNAVAILABLE, not
      seam matching nothing. Production wires it at `app.py:5682`.
- [ ] The harness wires a real prompts seam, so `plain`'s
      `category.prompt.*` cells measure retrieval rather than absence
- [ ] The cost the harness comment predicts is measured, not assumed: the
      seam appends rows to EVERY plain fan-out, so non-prompt plain cells
      may move too — report which moved, by how much, before deciding
- [ ] Whichever way the numbers go, the gate is re-stamped DELIBERATELY with
      the reason recorded — this changes what the instrument can see, so a
      moved cell is the deliverable rather than a regression
- [ ] TASK-17855's report and this programme's README stop describing the
      plain prompt cells as a retrieval result
