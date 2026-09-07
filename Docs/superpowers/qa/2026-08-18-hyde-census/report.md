# TASK-18514 — HyDE: census cleared, probe returned NULL, and the mechanism explains why

**Verdict: NULL. HyDE is not admitted.** 2 rescues against a pre-registered
bar of 5 — but it **passed the harm gate with zero losses**, which no prior
candidate did, and the failure has a mechanistic explanation rather than
being a bare number.

**The bar was registered before any measurement existed**, in commit
`724f28951` — the correction to TASK-18155's recorded process deviation.

Reproduce: `hyde_census.py` and `hyde_probe.py` beside this file;
`generations.json` holds all 60 generated passages verbatim.

## Generator, named as required (AC#5)

| | |
|---|---|
| endpoint | `http://localhost:9099/v1/chat/completions` (llama.cpp) |
| model | `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` |
| decoding | `temperature=0`, `max_tokens=220`, `enable_thinking=False` |
| output | 60 passages, **0 empty**, mean 71 words, 108s total |

`enable_thinking=False` is load-bearing: this server fronts a **reasoning**
model, and without it the whole token budget goes to `reasoning_content` and
`content` comes back empty — a generation that looks like a model refusing
rather than a field being unread.

## The census cleared the bar — 11 reachable

HyDE acts only on the semantic leg, so its population is queries that miss
their target *and* whose target is vector-indexed. Excluded structurally, as
registered in advance: 7 `negative` (no target) and 5 `prompt` (no vector
index at all).

| | semantic | hybrid |
|---|---|---|
| hitting at k=10 | 37 | 46 |
| miss, but found by k=200 — **HyDE's case** | **11** | 3 |
| miss and absent even at k=200 | 0 | 0 |

11 ≥ 5, so the probe was licensed. This is **the first P2c candidate to clear
its census**; the clarification gate scored 0 and the granularity router 1.

## The probe: 2 rescues, 0 losses

Measured in `semantic` at k=10 over the 53 target-bearing queries. Baseline
37 hits → HyDE 39.

| | |
|---|---|
| **GAINS (2)** | `sc-valve-pit-access` [scoped], `ng-mains-supply` [negation] |
| **LOSSES (0)** | — |

| pre-registered clause | measured | result |
|---|---|---|
| 1. rescues ≥ 5 | **2** | **FAIL** |
| 2. zero currently-hitting queries lose their target | **0** | **PASS** |

Clause 1 is gating, so the verdict is NULL. **Clause 2 passing is the
interesting half**: PRF died on exactly this clause, losing 10 of 21 hitters.
HyDE is *safe* here and merely insufficient — a different failure, and worth
recording as such.

## Why it failed, mechanistically — and it is not the generator

The generations are good: coherent, on-topic, none empty. The failure is
**structural**, and reading the 11 targets shows it.

**Every one of the 11 targets contains every content word of its query.** I
initially read them as "semantically unrelated" from a 190-character
truncation and had that backwards; checked against the full text, the match
rate is 3/3 terms on all of them. What makes them miss is that the words
appear in a **different sense, incidentally, in a document about something
else**:

| query | target document | how the words appear |
|---|---|---|
| `pump chamber inspection` | Icehouse and courtyard, conservation note | a real pump chamber under the flagstones; an annual *inspection of the fabric* |
| `meter box key` | Staff allotment strip, standing note | the meter box key hangs in the tool shed |
| `plant maintenance record` | Estuary bird hide, wardens' note | a **botanical** plant list; *maintenance* of the hide; a *record* of each visit |
| `sample point sign` | Boundary fence artwork | the group worked from a *sample point* on the fence; a *sign* at the stile |

**This is why HyDE cannot help.** HyDE's mechanism is to make the query
embedding *more topically specific* — it writes a passage about pump-chamber
inspection procedure. That moves the vector **further** from an icehouse
conservation note, not closer. On this corpus the semantic misses are
**lexical coincidences, not semantic near-misses**, and HyDE is
anti-correlated with that failure mode. It is the right tool for a problem
this corpus does not have.

The two rescues confirm the reading rather than contradicting it:

- **`ng-mains-supply`** — HyDE wrote *"uniquely equipped with an integrated
  solar-battery array, precluding the need for a traditional mains
  connection"*; the target is Skellow Isle, which *"draws everything it needs
  from a solar array on the shed roof, with a battery bank"*. The hypothetical
  document **is** the answer, which is HyDE working exactly as designed — and
  on `negation`, the one category at 0.000 in every mode.
- **`sc-valve-pit-access`** — not a coincidence either: that target genuinely
  contains a passage about valve-pit access (*"the slab was cast with a
  lifting panel and the access cover beneath it stays reachable"*).

So the rule is clean: **HyDE rescues where the target really is a topical
answer, and cannot where the query's words are present only incidentally.**

Counted honestly, that splits the 11 as **4 topical / 7 lexical-trap** — the
3 `negation` queries *plus* `sc-valve-pit-access`, whose target does contain
a real passage about valve-pit access. (An earlier draft of this report said
3/8 while simultaneously arguing `sc-valve-pit-access` was a genuine topical
match; a reviewer caught the contradiction. The corrected split is 4/7.)

**4 is still below the bar of 5**, so HyDE's reachable population falls short
before a single token is generated — but the margin is one query, not two,
and the report should say so.

## Generator-bound vs HyDE-bound (AC#5)

Registered in advance: *"a generator-specific null does not retire the
premise."* Splitting it:

- **HyDE-bound, not fixable by a better model**: the 7 lexical-trap queries.
  A better generator writes a *better* passage about pump-chamber inspection,
  which is further still from an icehouse note. More capability makes this
  worse, not better.
- **Possibly generator-bound**: the 2 unrescued `negation` queries.
  `ng-surfaced-approach`'s passage (*"unpaved tracks, seasonal trails, or
  alternative transport such as air or water"*) is genuinely close to a target
  *"reached only by boat"*, and `ng-three-panel-head` produced *"single-panel
  head assembly"* against a target saying *"a single dish"*. A stronger
  generator might close those.

**Even granting both, HyDE reaches 4 of 53 — still below the bar of 5.** The
null does not depend on the generator, which is what makes it safe to record
as a property of the premise on this corpus. It is worth being precise that
the margin is **one query**: a corpus with one more genuinely-topical miss
would have flipped this arc from NULL to a probe-clearing result.

## What review changed (PR #1815)

**The topical/lexical split was internally contradictory.** I argued
`sc-valve-pit-access` was a genuine topical match and then counted only the 3
negation queries as topical. Corrected to **4 topical / 7 lexical**; the
conclusion holds (4 < 5) but the margin is **one query**, not two, and the
report now says so.

**An empty generation was scored as a HyDE miss** — latent, since 0 came back
empty, but it would have converted an *unmeasurable* query into a **LOSS** and
corrupted the harm gate, the one clause HyDE passed. Empty generations are now
excluded from scoring and reported separately. This is the programme's
recurring defect in its purest form: "could not measure" rendering identically
to "measured, and it got worse".

**Pins added.** `Tests/RAG_Eval/test_hyde_probe.py` (15 tests) over
`score_arms` and the endpoint/model validation. **Mutation-verified**: scoring
empty generations as misses reds both empty-generation tests; restoring greens
them. Endpoint validation now fails at startup rather than surfacing later as
"the generator returned nothing" — the failure mode this programme keeps
misreading.

The census's bucketing was also extracted to a pure `classify` and pinned —
it produced the "11 reachable" that licensed the probe, so an over-count
would probe a population that does not exist and an under-count would kill a
real candidate silently. **Mutation-verified**: disabling the prompt
exclusion (prompts have no vector index, so no query-vector rewrite can reach
them) reds `test_prompt_excluded_even_when_found_deeper`.

**22 pins total.** Both scripts reproduce their numbers exactly after the
refactor: census 15,820 rows / 0 errors / 11 reachable; probe 2 gains, 0
losses, NULL.

## What would reopen this

HyDE is mis-matched to *this corpus*, whose hard cases are deliberate lexical
traps. A corpus whose failures were true vocabulary gaps — the query and
target saying the same thing in different words — is the shape HyDE is built
for, and `negation` is the only category here with that character. **If the
negation category is ever expanded**, re-run `hyde_probe.py` against it
specifically: this arc's evidence is that HyDE moves that category and only
that category.
