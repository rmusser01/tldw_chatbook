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
defect on evidence that cannot support the claim.** TASK-17855's census
observed that the `plain` mode returns zero rows for all five `prompt`
golden queries — including one whose target contains every content word —
and concluded the Library's prompts sub-leg was broken. **That conclusion is
withdrawn as unsupported.** Whether the sub-leg is defective is unknown and
untested; what is established is only that the measurement could not have
shown otherwise.

The cause of the zeros is in the **instrument**, and the harness says so in
its own comment (`Tests/RAG_Eval/harness/ingest.py`, the fake app's construction):

> `prompt_scope_service=None` … *"Leaving it None means the harness's plain
> column reports 0.000 for prompts while the shipped app's plain mode does
> find them."*

`_search_prompts` returns `(False, [])` when that attribute is absent — the
seam reports itself **unavailable**, which is not the same as matching
nothing. Production wires it (`app.py:5682`,
`build_prompt_scope_service`). So the plain `category.prompt.*` cells are
**vacuous by construction**, and TASK-17855's reading of them as a retrieval
failure was wrong.

**A second collapse sits one branch further down, and it survives wiring the
seam.** `_search_prompts` ends `except Exception: return True, []` — a seam
that *threw* is reported as available-and-empty, indistinguishable in the
metrics from one that searched and matched nothing. So "wire the service and
watch the cell" is not on its own a sufficient test: a zero afterwards would
still have three possible causes (no match / threw / genuinely absent). The
`logger.warning("… prompts seam failed.")` line is the only tell, and the
run must be checked for it.

Note what this does and does not settle: the defect claim is **withdrawn as
unsupported, not disproven.** No arc has exercised the plain prompts sub-leg
against a real `PromptScopeService`, so whether it retrieves is still an open
question — which is a second, independent reason to wire the seam.

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
- [ ] **Reported PER QUERY, not as a category average.** All five prompt
      goldens are listed with hit/miss individually — an aggregate cell of
      0.200 means one of five succeeded and four still miss, which would let
      this task close while the behaviour it exists to settle is still open.
      `prompt-vendor-chaser` is named explicitly: its query contains every
      content word of the target, so if any query retrieves, it must
- [ ] **The run is checked for `"prompts seam failed."` in the log.** With
      the service wired, an exception still returns `(True, [])`, so a zero
      would remain ambiguous between no-match and threw. A clean run must
      show the warning absent; if present, the exception is the finding
- [ ] The `(False, [])` / `(True, [])` collapse is addressed at the metrics
      layer or recorded as accepted: an unavailable seam should not render
      as `0.000`, and this problem belongs to EVERY optional seam, not just
      prompts (per the reviewer's suggestion on PR #1807)
- [ ] The cost the harness comment predicts is measured, not assumed: the
      seam appends rows to EVERY plain fan-out, so non-prompt plain cells
      may move too — report which moved, by how much, before deciding
- [ ] Whichever way the numbers go, the gate is re-stamped DELIBERATELY with
      the reason recorded — this changes what the instrument can see, so a
      moved cell is the deliverable rather than a regression
- [x] TASK-17855's report and this programme's README stop describing the
      plain prompt cells as a retrieval result (done in PR #1807: the report
      carries a correction block, and `Tests/RAG_Eval/README.md` now flags
      the `plain` cell as vacuous in the category table, the P2c-targets
      paragraph, and the keyword-limits section — where the stated REASON was
      also wrong, blaming the absent vector index for a mode that never uses
      one)
