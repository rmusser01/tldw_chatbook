---
id: TASK-16072
title: 'P2c candidate 5: clarification gate, probed before built'
status: To Do
assignee: []
created_date: '2026-08-14 01:52'
labels:
  - rag
  - p2c
  - fail-first
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The next candidate in the approved P2c cost order after PRF is a
**clarification gate**: detect that a query is ambiguous or underspecified
and ask the user a question instead of retrieving on a guess. It gets the
same treatment PRF got in TASK-15965 and that retired the expansion,
`acronym` and `compositional` premises before it — **the premise is probed
and priced before any production code exists**, under a bar and a kill
condition registered before the probe runs.

**The premise, stated honestly and pessimistically up front, because that is
the point of filing it this way.** It is not clear this candidate has any
measurable cell on this instrument:

- Every golden query is **well-specified by construction**. The fixture
  classes were authored to fail retrieval for reasons a clarifying question
  cannot repair: `negation` fails because the corpus asserts what the query
  excludes; the four residual `prompt` misses fail on absent CONTENT words;
  `paraphrase` / `vocabulary_mismatch` fail in `plain` because the query
  shares no content word with its target. A gate has nothing to ask about any
  of them, and asking would not change what is in the index.
- The gate's value is a **user-interaction** outcome (fewer wrong answers,
  a faster second query), and this harness scores single-shot retrieval
  against fixed relevance labels. There is no interaction model in it. Any
  simulated "user answers the clarifying question" is an oracle feed, and an
  oracle feed measures the fixture, not the gate — the trap TASK-15965's own
  control fell into (see Lessons: a control that holds a second variable
  fixed measures the pair).
- Which leaves an honest possibility this task must be free to reach: **the
  probe is a paper analysis rather than a run.** If Step 0 shows the corpus
  contains no query the gate could fire on, the deliverable is that finding
  plus a recorded null — not a fixture set authored until something moves.
  Authoring ambiguity fixtures is itself governed by the admission protocol
  in `Tests/RAG_Eval/README.md` (admit only what today's pipeline is
  *measured* to fail), and a class invented to give a feature something to
  show is a measurement of the class.

**The machinery to start from, so this does not get re-invented**
(TASK-15965 built it and it survived a review round):

- `Tests/RAG_Eval/harness/prf_probe.py` — the pure-function half: term
  derivation and expression composition, engine helpers **imported** rather
  than re-implemented (`_FTS5_STOPWORDS`, `_quote_fts5_token`,
  `_fts5_query_tokens`), exact `Fraction` weights so a ranking key cannot
  inherit float/doc-order dependence, plus `ProbeQueryResult` whose
  before-columns are always the SHIPPED pass. 28 always-on pins in
  `Tests/RAG_Eval/test_prf_probe.py`, no gate and no model needed.
- `Tests/RAG_Eval/test_prf_probe_run.py` — the **idiom** worth copying more
  than the code: a module-level `pytestmark = harness_gate()` in its OWN
  module (never a directory-level mark, which would gate the always-on pins),
  then STEP 0 fireability census FIRST, controls that say how many cells an
  effect could have been observed in at all, the grid, guard populations
  derived from a fresh baseline pass at probe time (never hardcoded), and a
  verdict computed clause-by-clause against the pre-registered bar with an
  assertion that a non-pre-registered variant can never reach it.
- The precedent documents, all committed: spec
  `Docs/superpowers/specs/2026-08-13-rag-p2c-prf-fail-first-design.md`, the
  recorded null in `Tests/RAG_Eval/README.md` ("The fourth retired P2c
  premise"), and TASK-15965's Implementation Notes. The arc's full run report
  (`.superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/task-2-report.md`) is an
  untracked SDD record; the run itself is reproducible from the gated module.

A recorded null is a success outcome of this arc, exactly as it was for PRF.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Step 0 runs first and answers, from the corpus rather than from argument, whether any golden query carries an ambiguity or underspecification signal a gate could fire on — and its per-query answer is reported before any design or code
- [ ] #2 The admission bar and the kill condition are written down BEFORE the probe runs, including what evidence would license production code and what result ends the arc
- [ ] #3 If the gate's value cannot be measured on this instrument without user-interaction fixtures, that is recorded as the finding and the arc ends there — a paper analysis is an acceptable deliverable; an unmeasurable improvement claim is not
- [ ] #4 Any measurement reports gains AND losses by query id, with guard populations derived from a baseline pass at probe time, and no control is read as a property of the pipeline unless the variables it holds fixed are named
- [ ] #5 No production code exists before the verdict; a below-bar result is recorded beside the retired premises in Tests/RAG_Eval/README.md with the next candidate filed
<!-- AC:END -->
