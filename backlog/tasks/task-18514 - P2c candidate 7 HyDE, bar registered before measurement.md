---
id: TASK-18514
title: 'P2c candidate 7: HyDE, bar registered before measurement'
status: In Progress
assignee: []
created_date: '2026-08-18'
labels: [rag, p2c, fail-first]
dependencies: []
priority: medium
---

## Description (the why)

**HyDE is the last of the five named P2c candidates never investigated.**
The list at `Tests/RAG_Eval/README.md` line 53 is: query expansion, HyDE,
PRF, a clarification gate, a granularity router. Four are retired — query
expansion to a *cell* (`vocabulary_mismatch` reads 1.000 in both vector
modes), PRF to a probe (TASK-15965), the clarification gate to a census
(TASK-16072), the granularity router to a census (TASK-18155).

HyDE generates a hypothetical answer document with an LLM and embeds *that*
in place of (or alongside) the query, on the theory that an answer's
embedding sits closer to a real answer than a question's does.

**This task is written and committed BEFORE any measurement**, which is the
correction to TASK-18155's recorded process deviation: there I ran the census
first and had to inherit a bar to keep it non-circular. Here the bar exists
in git history before a single number does.

**What is different about HyDE, and why it was deferred this long**: every
prior candidate could be probed offline against the pinned venv. HyDE needs
an LLM call per query. That is now available locally (a llama.cpp server on
`localhost:9099` serving a Gemma-class GGUF, OpenAI-compatible), so the cost
objection is gone — but the dependency is real and must be recorded in any
result, because a HyDE number is a number about *that generator*, not about
HyDE in the abstract.

## Pre-registered bar and kill conditions (REGISTERED BEFORE MEASUREMENT)

Inherited verbatim from the three prior candidates that used it, so it is not
tuned to this arc:

1. **Rescue bar: ≥5 golden queries that currently MISS their target must be
   reachable.** Below that, the arc ends NULL — no probe, no production code.
2. **Harm gate: zero currently-hitting queries may lose their target.** PRF
   died on exactly this clause (10 of 21 hitters lost); HyDE has the same
   shape of risk and is judged the same way.
3. **Structural exclusions, named now so they cannot be argued in later**:
   `negative` queries have no target (a miss is the correct outcome) and
   `prompt` targets have **no vector index at all**, so HyDE — which acts
   only on the semantic leg — cannot reach them by construction.
4. **A generator-specific null does not retire the premise.** If HyDE fails
   because this local model writes poor hypothetical documents, that is
   recorded as a bound on the measurement, not as evidence against HyDE.
   Distinguishing the two is part of the deliverable.

## Acceptance Criteria (the what)

- [ ] This bar is committed before any measurement exists (verifiable in git
      history, not asserted in prose)
- [ ] The census counts, per query and from a live run, how many queries
      currently miss a **vector-reachable** target — the only population HyDE
      can act on
- [ ] A below-bar census ends the arc with a recorded null, no probe and no
      production code, recorded beside the other retired premises in
      `Tests/RAG_Eval/README.md`
- [ ] If the census clears the bar, the probe reports gains AND losses by
      query id, and the harm gate is measured rather than assumed
- [ ] Any result names the generator (model, endpoint, decoding settings) and
      states whether the outcome is a property of HyDE or of this generator
- [ ] The instrument proves it measured: row counts, error counts, and the
      generator's own output are shown, not summarized — three arcs running,
      this programme's dominant defect is a value meaning "could not measure"
      rendering identically to "measured, found nothing"
