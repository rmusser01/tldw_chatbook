# Step 0 — the clarification gate's premise, answered from the corpus (TASK-16072 AC#1)

Run 2026-08-18 against the shipped fixture (172 docs, 60 golden queries).
Bar and kill condition were registered first, in `bar.md`, unchanged since.

## The census

Condition 2 of the bar — do a query's interpretations point at **different
documents in this corpus** — is the one the fixture answers directly, and it
is binding: a query with a single relevant document has nothing for a gate to
disambiguate *between*, however vague its surface.

| | queries |
|---|---|
| more than one relevant document | **2** |
| exactly one relevant document | 51 |
| zero (negative controls) | 7 |

Categories: keyword 16, paraphrase 13, vocabulary_mismatch 9, negative 7,
scoped 7, prompt 5, negation 3.

## The two candidates fail condition 3, and they fail it badly

| query | docs | text |
|---|---|---|
| `kw-nimbus-rollback` | 2 | "Nimbus-14 firmware rollback" |
| `kw-calyx-limiter` | 2 | "Calyx-77 torque limiter slipping" |

Both are `keyword` queries whose **two documents are both relevant**. A
clarifying question here would not choose between competing interpretations —
it would ask the user to discard one of two correct answers. Firing a gate on
these would make retrieval *worse*, not ambiguous-then-better.

So the qualifying count under all three conditions is **0**, against a bar of
5. **The kill condition fires.**

## Twelve short queries are not twelve gate cases

For completeness (condition 1 alone): 12 queries are ≤3 words, e.g.
`'asset tag QX-8842'`, `'plant maintenance record'`, `'gum disease treatment'`,
`'pump chamber inspection'`. Every one has **exactly one** relevant document.
They are terse, not ambiguous — the corpus gives each a single target, so a
question would add a round trip and change nothing about what is retrieved.
This is the distinction the bar was written to force: surface vagueness is not
an ambiguity signal unless the corpus supplies a second thing to mean.

## Verdict: NULL — the arc ends here, per AC#3

No production code was written, and none should be. The finding is that this
corpus contains **no query a clarification gate could fire on usefully**, and
the reason generalises past this fixture: every failing class fails for a
reason a question cannot repair — `negation` because the corpus asserts what
the query excludes; `prompt` and the residual zero-row queries on **absent
content words**; `paraphrase` / `vocabulary_mismatch` in `plain` because the
query shares no content word with its target. No answer to any question puts
a missing document into an index.

Reaching the bar would have required **authoring ambiguity fixtures**, which
the pre-registered kill condition declared a kill rather than an invitation:
`Tests/RAG_Eval/README.md`'s admission protocol admits only what today's
pipeline is *measured* to fail, and a class invented to give a feature
something to show measures the class.

**This is the fifth P2c premise to die before it was built** — after
`expansion`, `acronym`, `compositional` and PRF (TASK-15965). Four of the
five died on measurement; this one died on a census, which is cheaper and
should be the first move for the remaining candidates.
