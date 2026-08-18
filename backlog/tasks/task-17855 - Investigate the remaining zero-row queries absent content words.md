---
id: TASK-17855
title: Investigate the remaining zero-row queries — absent content words
status: Done
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

- [x] The 25 residual zero-row queries are characterised: for each, whether
      the relevant document is reachable by ANY keyword construction over the
      terms the user typed, or only by semantic retrieval
- [x] At least one candidate is proposed with its measured or estimated
      precision cost, judged against the finding that pure OR cut MRR by a
      THIRD (0.396 -> 0.261, -34%) —
      recall bought by broadening is not free and must be priced
- [x] A decision is recorded: pursue a specific candidate as its own arc, or
      record that keyword construction has reached its ceiling on this corpus
      and the remainder is the semantic leg's job

## Outcome (2026-08-18): the residual splits in two, and neither half wants broadening

**AC#1 — characterised, from the corpus.** 30 residual zero-row queries carry
ground truth (the count moved from the filed 25 because TASK-17755 shipped
`and_then_prefix` to this path in between; re-measured rather than inherited).
The test is lexical, because a construction can only rearrange the terms the
user typed: does the target contain ANY content word of the query?

| | queries |
|---|---|
| **UNREACHABLE** — no shared content word | **19** (63%) |
| lexically reachable | **11** (37%) |

**AC#2 — the candidate, and it is not the one this task expected.** The
reachable 11 are dominated by `prompt`, which is not behaving like a
construction problem at all:

| mode | prompt queries returning rows | finding the target |
|---|---|---|
| `plain` | **0 of 5** | 0 |
| `semantic` | 5 of 5 | 0 |
| `hybrid` | 5 of 5 | 1 |

The plain path returns **nothing** for any prompt query — including one whose
target contains **every content word** and whose name is indexed by
`prompts_fts`. A sub-leg returning zero rows when every term is present is a
defect, filed as **TASK-18255**.

Costs, since AC#2 asks for them priced:

| candidate | reaches | precision cost |
|---|---|---|
| fix the prompts sub-leg (TASK-18255) | up to 5 of the 11 | **none** — terms already present and indexed |
| broaden the construction | some of the other 6 | **−34% MRR**, measured (TASK-3997) |
| semantic/vocabulary work | the 19 unreachable | the only route for them |

**AC#3 — the decision.** Keyword construction has **not** reached its ceiling,
but the headroom is not in broadening. The 19 unreachable queries are closed
as the semantic leg's business permanently. The reachable 11 go to TASK-18255.
**Broadening is explicitly not recommended**: paying 34% MRR to reach queries
whose target already contains the words — when the seam should be matching
them — is buying with the wrong currency.

Artifacts: `Docs/superpowers/qa/2026-08-18-residual-zero-row/report.md` and
`reachability_census.py` (runnable; prints its document count before any
result).

### CORRECTION (2026-08-18)

**The prompt-seam finding above was wrong.** `plain` returns zero rows for
prompt queries because the eval harness sets `prompt_scope_service=None`, so
the seam reports itself UNAVAILABLE (`_search_prompts` → `(False, [])`) —
not because it matched nothing. Production wires it (`app.py:5682`), and the
harness comment says so explicitly: *"the shipped app's plain mode does find
them."* I read the instrument's numbers without checking whether the
instrument could see the thing.

The defect claim is **withdrawn as unsupported, not disproven**: whether
the sub-leg actually retrieves against a real `PromptScopeService` remains
untested, and TASK-18255 exists to settle it.

**What survives:** the 19-unreachable / 11-reachable split, a lexical
property of the corpus, independent of any seam. **What does not:** the claim
that the prompts sub-leg is defective, and the cost table's implication that
fixing it would reach 5 queries. TASK-18255 is re-scoped from "fix the seam"
to "wire the seam into the harness so the cells measure something".
