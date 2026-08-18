# Clarification gate: the bar and the kill condition (TASK-16072 AC#2)

**Written before Step 0 runs.** Registered so the verdict cannot be
renegotiated once numbers exist — the discipline TASK-15965 established and
the cross-encoder arc reaffirmed.

## What a gate would have to fire on

A clarification gate detects that a query is **ambiguous or underspecified**
and asks the user a question instead of retrieving on a guess. For it to have
any measurable effect on this instrument, the corpus must contain golden
queries where:

1. the query admits **more than one defensible interpretation**, AND
2. those interpretations point at **different documents** in this corpus, AND
3. a user's answer to a clarifying question would **change which document is
   retrieved**.

All three are required. A query that is vague but has exactly one plausible
target is not a gate case — asking would be noise. A query whose target is
absent from the corpus is not a gate case either: no answer to any question
puts a missing document into the index.

## Step 0's question (AC#1)

For each of the 60 golden queries, answered **from the corpus** rather than
from argument: does it carry a signal a gate could fire on, under the
three-part test above?

## The admission bar (what licenses production code)

**≥ 5 of 60 golden queries** must satisfy all three conditions — enough that
a gate could move a scored cell rather than a rounding artifact. Below that,
any measured "gain" would rest on one or two queries and could not be
distinguished from fixture noise, which is the failure mode that retired
`acronym` and `compositional`.

## The kill condition (what ends the arc)

**Fewer than 5 qualifying queries ⇒ the arc ends with a recorded null**, per
AC#3: a paper analysis is the deliverable and no production code is written.

Two specific outcomes are pre-declared as **kills, not invitations to author
fixtures**:

- If the qualifying count is zero because every failing class fails for a
  reason a question cannot repair (absent content words, corpus asserting
  what the query excludes, no shared content word), that is the finding.
- If reaching the bar would require **authoring new ambiguity fixtures**,
  the arc ends instead. `Tests/RAG_Eval/README.md`'s admission protocol
  admits only what today's pipeline is *measured* to fail; a class invented
  to give a feature something to show measures the class, not the feature.

## What is NOT admissible as evidence

- A simulated user answering the clarifying question. That is an oracle feed,
  and an oracle feed measures the fixture (Lessons: a control that holds a
  second variable fixed measures the pair).
- Any interaction-level claim — fewer wrong answers, faster second query.
  This harness scores single-shot retrieval against fixed labels and contains
  no interaction model. Such a claim would be unfalsifiable here, and an
  unmeasurable improvement claim is exactly what AC#3 forbids.
