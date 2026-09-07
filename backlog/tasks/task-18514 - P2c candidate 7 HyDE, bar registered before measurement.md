---
id: TASK-18514
title: 'P2c candidate 7: HyDE, bar registered before measurement'
status: Done
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

- [x] Bar committed in `724f28951` **before** the census script existed —
      verifiable in git history, not asserted in prose.
- [x] Census run live: **11 reachable in semantic** (37 hitting, 7 negative
      and 5 prompt excluded structurally, 0 unreachable even at k=200), 3 in
      hybrid. **The first P2c candidate to CLEAR its census.**
- [x] Not applicable — the census cleared, so the probe ran. The NULL came
      from the probe instead, and is recorded as the **seventh** retired
      premise in `Tests/RAG_Eval/README.md`.
- [x] Probe reports both by query id: **GAINS 2** (`ng-mains-supply`,
      `sc-valve-pit-access`), **LOSSES 0**. The harm gate was measured, not
      assumed, and **passed** — the clause PRF died on.
- [x] Generator named in full (llama.cpp `localhost:9099`, Gemma-class GGUF,
      temp 0 / 220 tok / `enable_thinking=False`) and the outcome is split:
      **8 of 11 are HyDE-bound** (lexical traps a better generator makes
      WORSE, since HyDE increases topical specificity away from an
      incidentally-matching document); at most 3 are generator-bound. **3 < 5,
      so the null does not depend on the generator.**
- [x] Census: 15,820 rows retrieved, 0 errors. Probe: 60 generations, **0
      empty**, mean 71 words, 0 retrieval errors; both scripts refuse a
      verdict on any error. All 60 passages committed verbatim in
      `generations.json` rather than summarized. The `enable_thinking=False`
      discovery is itself an instance of the family: without it this
      reasoning model returns `content=""` while spending every token in
      `reasoning_content` — a full budget producing an empty string that
      reads exactly like a refusal.


## Implementation Notes

**NULL — HyDE is not admitted. No production code was written.** This
exhausts the five named P2c candidates.

**First candidate to clear its census** (11 reachable vs bar 5), so unlike
the clarification gate and the granularity router this one earned its probe.
The probe then returned **2 rescues, 0 losses**: it FAILS the gating rescue
clause and PASSES the harm clause that killed PRF.

**The failure is mechanistic, not a generator shortfall, and that is the
finding.** All 11 reachable targets contain every content word of their
query — I first read them as semantically unrelated from a 190-character
truncation and had it backwards. They miss because the words appear in a
*different sense, incidentally*, inside a document about something else (a
`pump chamber inspection` query whose target is an icehouse conservation
note; `plant maintenance record` whose target is a bird hide with a
**botanical** plant list). HyDE makes the query embedding MORE topically
specific, which moves it further from such a document — so a better
generator makes these strictly worse. Its true population here is the 3
`negation` queries, below the bar before generation begins.

**Process:** the bar was committed in `724f28951` before the census script
existed, correcting TASK-18155's recorded deviation.

**Wrong turn worth recording:** I hypothesised the 7 scoped misses were a
scope-application gap and that semantic mode ignored the allowlist.
Measured: scope IS applied in both modes (allowlist 100, all 10 returned
rows in-scope) — hybrid hits those targets and semantic does not, because
the keyword leg finds the incidental term match. The hypothesis was wrong
and the measurement corrected it before it reached the report.

**Files:** `Docs/superpowers/qa/2026-08-18-hyde-census/` (report,
`hyde_census.py`, `hyde_probe.py`, `generations.json`);
`Tests/RAG_Eval/README.md` (seventh retired premise). No production source
changed, so the gate cannot move.
