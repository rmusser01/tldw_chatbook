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
Of the 11, only the 3 `negation` queries are the former, which caps HyDE's
reachable population at 3 — below the bar before a single token is generated.

## Generator-bound vs HyDE-bound (AC#5)

Registered in advance: *"a generator-specific null does not retire the
premise."* Splitting it:

- **HyDE-bound, not fixable by a better model**: the 8 lexical-trap queries.
  A better generator writes a *better* passage about pump-chamber inspection,
  which is further still from an icehouse note. More capability makes this
  worse, not better.
- **Possibly generator-bound**: the 2 unrescued `negation` queries.
  `ng-surfaced-approach`'s passage (*"unpaved tracks, seasonal trails, or
  alternative transport such as air or water"*) is genuinely close to a target
  *"reached only by boat"*, and `ng-three-panel-head` produced *"single-panel
  head assembly"* against a target saying *"a single dish"*. A stronger
  generator might close those.

**Even granting both, HyDE reaches 3 of 53 — still below the bar of 5.** The
null does not depend on the generator, which is what makes it safe to record
as a property of the premise on this corpus.

## What would reopen this

HyDE is mis-matched to *this corpus*, whose hard cases are deliberate lexical
traps. A corpus whose failures were true vocabulary gaps — the query and
target saying the same thing in different words — is the shape HyDE is built
for, and `negation` is the only category here with that character. **If the
negation category is ever expanded**, re-run `hyde_probe.py` against it
specifically: this arc's evidence is that HyDE moves that category and only
that category.
