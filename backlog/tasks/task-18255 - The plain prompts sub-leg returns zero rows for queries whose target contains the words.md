---
id: TASK-18255
title: >-
  Wire a prompts seam into the eval harness so plain-mode prompt cells measure
  something
status: Done
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
- [x] Wired via `build_prompt_scope_service(prompt_db=prompts_db,
      server_service=ServerPromptService(client=None))`. Availability proven
      directly rather than inferred: `_search_prompts` returns `True` with 6
      rows on a smoke query, not `(False, [])`.
- [x] All five reported individually. **`pm-vendor-chaser` HITS** — the query
      the AC named, and the one whose target contains every content word. The
      other four MISS, consistent with TASK-17855's surviving finding (their
      targets lack the content words). 1 of 5, which is the 0.200 aggregate
      stated plainly rather than hidden behind it.
- [x] A loguru sink captures the marker; it fired **0 times**.
      `seam_effect.py` refuses a verdict if it ever fires, so this cannot
      silently regress.
- [x] **Recorded as accepted, with the reason.** Fixing it properly means a
      distinct "unavailable" value at the metrics layer for every optional
      seam — larger than this task, and exactly the defect shape that produced
      TASK-17855. Documented as the residual in the report;
      `_search_prompts`'s `except Exception: return True, []` still collapses
      threw-vs-empty, which is why the log check above is mandatory.
- [x] Measured, and **it did not materialize**: **0 of 55** non-prompt
      queries have a prompt row in their fan-out, and no non-prompt CATEGORY
      moved. The seam appends nothing where nothing matches. Worth recording,
      since this predicted cost was the stated reason the wiring was deferred.
- [x] Re-stamped deliberately. **10 of 105 metrics moved, all up, all in
      `plain`**: the five `category.prompt.*` cells 0.000 -> 0.200 and the
      five `plain overall.*` cells +0.022 (arithmetic — a vacuous category now
      contributes). Gate reads `PASSED: No regression. 105 metric(s)`. Every
      `semantic` and `hybrid` metric is +0.000.
      **Disclosed:** the fingerprint's `sentence_transformers` moved 5.7.0 ->
      5.4.1 (this machine's venv). Nothing else in it moved, and the change is
      demonstrably retrieval-neutral here — all **70** semantic+hybrid metrics
      are byte-identical, which is precisely where an embedding-library change
      would show.
- [x] TASK-17855's report and this programme's README stop describing the
      plain prompt cells as a retrieval result (done in PR #1807: the report
      carries a correction block, and `Tests/RAG_Eval/README.md` now flags
      the `plain` cell as vacuous in the category table, the P2c-targets
      paragraph, and the keyword-limits section — where the stated REASON was
      also wrong, blaming the absent vector index for a mode that never uses
      one)


## Implementation Notes

**The seam is wired, and it settled the question the arc was named for.**

One line of harness wiring (`build_prompt_scope_service`), passing
`server_service=ServerPromptService(client=None)` **explicitly** rather than
letting `app_config` resolve it — the default path runs
`derive_configured_server_binding`, and a harness consulting ambient config
could bind to the developer's own server, the same hazard the `*_db_path`
overrides exist to prevent.

**`pm-vendor-chaser` retrieves.** PR #1807 withdrew TASK-17855's defect claim
as *unsupported* while explicitly declining to assert the opposite, because
nothing had exercised the sub-leg against a real `PromptScopeService`. Now
something has: the claim is **disproven**, not merely withdrawn. The other
four prompt goldens still miss, for the reason 17855 established and which
survives — absent content words.

**The deferred cost did not bind.** The harness comment predicted the seam
would move plain numbers for non-prompt queries too. Measured: 0 of 55. That
prediction was the stated reason for deferring the wiring, so its falsity is
worth the record.

**Re-stamp:** 10 of 105 cells, all up, all `plain`. Disclosed separately: the
`sentence_transformers` fingerprint moved 5.7.0 -> 5.4.1 on this machine, and
its retrieval-neutrality is shown by 70 unchanged semantic+hybrid metrics
rather than asserted.

**Residual, accepted not fixed:** `_search_prompts`'s
`except Exception: return True, []` still renders a *thrown* seam as
available-and-empty. The run logged the failure marker 0 times and the probe
refuses a verdict if it ever fires, so these numbers are clean — but the
general fix (a distinct "unavailable" value for every optional seam) is
larger than this task and is exactly the defect shape that produced
TASK-17855.

**Files:** `Tests/RAG_Eval/harness/ingest.py` (the wiring),
`Tests/RAG_Eval/baselines/*.json` (deliberate re-stamp),
`Tests/RAG_Eval/README.md`, `Docs/superpowers/qa/2026-08-18-prompts-seam/`
(report + `seam_effect.py`). No production source changed.
