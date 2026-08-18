---
id: task-17855
title: Investigate the remaining zero-row queries — absent content words
status: To Do
assignee: []
created_date: '2026-08-18'
labels: [rag, retrieval, investigation]
dependencies: []
priority: low
---

## Description (the why)

Filed at the owner's request alongside TASK-3997's decision (2026-08-18).

Adopting `and_then_prefix` on the four-seam path (TASK-17755) rescues 7 of
32 zero-row golden queries. **Twenty-five still return nothing**, and
TASK-15400's sweep already identified why on both paths: the dominant blocker
is **absent CONTENT words**, not function words or inflection. No construction
that rearranges the tokens a user typed can fix a term the corpus does not
contain.

That makes this a different question from AND-strictness, and it is
deliberately an INVESTIGATION rather than a fix: the candidate answers
(synonym/expansion, a vocabulary bridge, falling back to the semantic leg when
the keyword legs return nothing) are each their own measured arc, and this
programme has already retired expansion, acronym and compositional
query-formulation candidates on measurement. **A null result here — "the
remaining 25 are not reachable by keyword construction at all" — is a
publishable outcome and is sufficient to close this task.**

## Acceptance Criteria (the what)

- [ ] The 25 residual zero-row queries are characterised: for each, whether
      the relevant document is reachable by ANY keyword construction over the
      terms the user typed, or only by semantic retrieval
- [ ] At least one candidate is proposed with its measured or estimated
      precision cost, judged against the finding that pure OR HALVED MRR —
      recall bought by broadening is not free and must be priced
- [ ] A decision is recorded: pursue a specific candidate as its own arc, or
      record that keyword construction has reached its ceiling on this corpus
      and the remainder is the semantic leg's job
