# Research Report Self-Eval Baseline

Date: 2026-08-15 · Task: task-16327 · Scorer: `tldw_chatbook/Evals/research_report_scorer.py`

## What is scored

Every completed deep-search research run produces a verification payload
(`citation_verification` on the pipeline's final answer; persisted in the
run's `verification_summary.json` artifact — tasks 16319/16322). The
research-report self-eval scores that payload deterministically — no LLM is
consulted — so pipeline changes are measurable run-over-run.

| Metric | Definition |
|---|---|
| `citation_accuracy` | resolved `[n]` markers / total markers (0.0 when nothing was cited) |
| `quote_grounding` | verbatim-verified quotes / checked quotes (0.0 when nothing was quoted) |
| `claim_support_rate` | supported claims / claims (falls back to marker accuracy when no per-claim detail exists) |
| `cited_sentence_ratio` | cited sentences / all sentences |

## Recorded baseline (synthetic)

The baseline below is computed from `BASELINE_VERIFICATION_PAYLOAD` in the
scorer module (10 markers, 8 resolved, 3/4 quotes verified, 3/4 claims
supported, 6 uncited sentences). It pins the metric definitions — a scorer
regression moves these numbers — and stands in until a live baseline is
recorded the same way:

| Metric | Value |
|---|---|
| `citation_accuracy` | 0.80 |
| `quote_grounding` | 0.75 |
| `claim_support_rate` | 0.75 |
| `cited_sentence_ratio` | 0.625 |

## Recorded live baseline (2026-08-15)

Configuration: academic lane (arXiv, phrase-quoted queries; Semantic Scholar
429-rate-limited from this network), web lane blocked (DuckDuckGo/Baidu bot
challenge, no keyed engines configured), relevance + synthesis LLMs =
local llama.cpp Qwen3.8-27B at `127.0.0.1:52864`. Bounded: 5 results/query,
no sub-query fan-out, 3 questions.

Command:
`python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --llm-base-url http://127.0.0.1:52864/v1`

| Metric | Live (n=2 scored runs) | Notes |
|---|---|---|
| `citation_accuracy` | **1.00** | 20/20 `[n]` markers resolved across scored runs |
| `quote_grounding` | 0.00 | model emitted no quoted spans (0 checked) — untested, not failing |
| `claim_support_rate` | 1.00 | every cited claim verified |
| `cited_sentence_ratio` | 0.68 | runs 0.44 / 0.93 |

Observations worth keeping:

- The weak link is the RELEVANCE GATE, not citation integrity: 1 of 3
  questions produced no synthesis because the strict "answers the question
  comprehensively" bar (temp 0.7) rejected application papers. With the
  local 27B model, gate pass-rate varies between identical runs.
- arXiv phrase-quoting was required: token queries surfaced off-topic
  papers (cognitive augmentation, image generation) that the gate then
  correctly rejected (fixed in `academic_providers.search_arxiv`).
- Local OpenAI-compatible providers previously returned raw response dicts
  that broke every string consumer; fixed at the `chat_api_call` seam
  during this live verification.

## Post gate-hardening re-measurement (2026-08-15, task-16333)

Same configuration, same command, after hardening the relevance gate
(usefulness-based prompt, 0.1 judgment temperature, zero-relevant flagged
fallback) and priming `max_tokens=16384` for the thinking model:

| Metric | Before (n=2 of 3 scored) | After (n=3 of 3 scored) |
|---|---|---|
| `citation_accuracy` | 1.00 (20/20 markers) | **1.00 (49/49 markers)** |
| `quote_grounding` | 0.00 (no quotes emitted) | **0.67** (quotes emitted in 2/3 runs; all verified) |
| `claim_support_rate` | 1.00 | 1.00 |
| `cited_sentence_ratio` | 0.68 | 0.63 |
| `gate_pass_rate` | whole runs lost to the gate | **0.93** (question 3: 4/5 passed) |

The measured weak link moved: before, the gate silently killed whole runs
(1 of 3 questions produced nothing); after, all three questions produced
verified reports. Remaining failure mode observed en route: a thinking
model can exhaust the default 4096 `max_tokens` on reasoning alone and
return an empty completion -- the baseline script now primes 16384 for
local endpoints.

## Biomedical-lane live baseline (2026-08-16, task-16794)

Same runner, `--providers biomedical` (PubMed lane) over the default
question set — local llama.cpp Qwen3.8-27B at `127.0.0.1:9191`, duckduckgo
web lane, bounded as above:

Command:
`python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --providers biomedical --llm-base-url http://127.0.0.1:9191/v1`

| Metric | Academic lane (n=3) | Biomedical lane (n=3) |
|---|---|---|
| `citation_accuracy` | 1.00 (49/49 markers) | **1.00 (36/36 markers)** |
| `claim_support_rate` | 1.00 | 1.00 |
| `gate_pass_rate` | 0.93 | **0.93** |
| `cited_sentence_ratio` | 0.63 | 0.51 |
| `quote_grounding` | 0.67 | 0.00 (no quotes emitted) |

Citation integrity holds on the biomedical lane: every marker resolved
across all three runs. The noted failure mode from the academic-lane runs
repeats (thinking model + question-shaped topics): one run's evidence mix
drove cited_sentence_ratio down, and no quotes were emitted (quote
grounding untested, not failing). En route the script gained the
`autonomy_mode="autonomous"` fix — checkpointed (the service default since
task-16482) would park every run at plan review and produce no report.

## Category-lane live baselines (2026-08-16, task-17385)

Three more lanes measured with the same runner/config (local Qwen3.8-27B on
:9191, duckduckgo web lane, bounded):

| Metric | repositories (n=3) | open_research_graph (n=3) | biomedical stress (n=3) |
|---|---|---|---|
| `citation_accuracy` | **1.00 (73/73)** | **1.00 (85/85)** | **1.00 (62/62)** |
| `claim_support_rate` | 0.97 | 1.00 | 1.00 |
| `gate_pass_rate` | **0.29** | 0.72 | 0.53 |
| `cited_sentence_ratio` | 0.52 | 0.69 | 0.73 |
| `quote_grounding` | 0.00 (no quotes) | 0.00 (no quotes) | 0.00 (no quotes) |

The biomedical stress run used the new `--question-set biomedical` domain
questions (CRISPR off-target effects, tau aggregation, gut microbiome) —
PubMed held perfect citation integrity under domain-specific vocabulary.

**Lane-specific finding**: the repositories lane (Zenodo/Figshare/OSF) has
the LOWEST gate pass rate (0.29) — repository records (datasets, figures)
frequently fail the "answers the question comprehensively" bar even when
on-topic, confirming the relevance gate's strictness hits
non-paper sources hardest. Citation integrity is unaffected (every kept
source verifies).

Live-verification fixes en route (task-17385): OSF intermittently 301s —
httpx does not follow redirects by default, which yielded empty bodies;
the OSF client now follows redirects and sends the server-parity
`Accept: application/json` header. Malformed payloads (the OSF 301 HTML)
previously escaped the lane's typed degradation catch as a raw
JSONDecodeError, killing the OTHER providers' results — all provider
JSON parsing now raises `AcademicProviderError` so one bad payload
degrades that provider only. The script also gained
`--question-set {default,biomedical}`.

## Gate source-type re-measurement (2026-08-17, task-17066 follow-up)

The repositories lane re-run with the source-type-aware gate note (same
runner/config/questions as the recorded 0.29 baseline):

| Metric | Before (strict prompt) | After (source-type note) |
|---|---|---|
| `gate_pass_rate` | 0.29 | **0.42 (+45% relative)** |
| `citation_accuracy` | 1.00 (73/73) | **1.00 (72/72)** — integrity held |
| `claim_support_rate` | 0.97 | **1.00** |
| `cited_sentence_ratio` | 0.52 | **0.75** |
| `quote_grounding` | 0.00 | 0.33 (one run quoted; all verified) |

Honest read: the note meaningfully improves admission of repository records
without costing verification integrity — but repositories still pass at less
than half the paper rate (0.93). That residual is partly genuine: many
repository records ARE marginal evidence for general-purpose questions
(a dataset about topic X is supporting material, not an answer). Fully
closing the gap would need either a per-kind relevance threshold or
category-tuned question sets — recorded as the follow-up lever, with the
fallback (top-3 flagged) still covering the remainder.

## Decomposition measured on the repositories lane (2026-08-17, task-17370)

Same runner, same three questions, same judge (local Qwen3.8-27B on
`:9191`), same 5-results bound as the 0.29 and 0.42 arms; the only change is
`--max-queries 3`, which turns on phase-1 sub-question generation:

Command:
`python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --providers repositories --llm-base-url http://127.0.0.1:9191/v1 --max-queries 3 --max-iterations 1 --deadline-s 900`

| Metric | 1 query, strict | 1 query + source-type note | **3 queries + note** |
|---|---|---|---|
| `gate_pass_rate` | 0.29 | 0.42 | **0.38** |
| `citation_accuracy` | 1.00 (73/73) | 1.00 (72/72) | 1.00 (70/70) |
| `claim_support_rate` | 0.97 | 1.00 | 1.00 |
| `cited_sentence_ratio` | 0.52 | 0.75 | 0.70 |
| `quote_grounding` | 0.00 | 0.33 | 0.33 |

Per-question `gate_pass` in the fan-out arm: 0.54, 0.40, 0.20.

**The result is negative: giving the gate narrower facets did not admit more
repository evidence.** 0.38 against 0.42 is flat to slightly down, and this
doc already records that gate pass-rate varies between identical runs on this
model, so the honest reading is "no measurable improvement", not "a
regression". Citation integrity again held (70/70 markers resolved).

What this arm did and did not test, because it decides how far the result
generalizes:

- **Tested:** the gate's context. `websearch.result_relevance_eval` takes
  `sub_questions` as a required placeholder, so the earlier arms rendered it
  as an empty list and this one rendered real facets. That is a genuine
  change in what the judge was asked.
- **Not tested:** retrieval. The web lane returned zero results throughout
  (the DuckDuckGo bot challenge this doc notes elsewhere), and at the time the
  academic lane only searched round 1's `[question]`, so the repository records
  under judgment were the SAME set as the 0.42 arm. Fan-out changed how they
  were judged, not which ones existed. task-17372 has since made the lane
  search the generated facets, so a re-run of this arm now exercises retrieval
  as well -- the number above does not.

So the case for decomposition on this lane now rests on retrieval rather than
on gate context: multi-hop rounds >= 2 do drive retrieval (measured
separately below), and fan-out reaching the paper providers does not happen
today at all (task-17372).

One bound on the non-gate rows of every table in this document, including
this arm's: per-result summarization was silently failing for llama.cpp
(task-17382), so the synthesis was built from titles and gate reasoning with
an error string where each source body belonged. `gate_pass_rate` is
unaffected — the gate judges scraped content before summarization — but
`cited_sentence_ratio`, `claim_support_rate` and `quote_grounding` were
graded on reports written without their sources, and should be re-measured
after that fix.

## Multi-hop measured on the repositories lane (2026-08-17, task-17370)

The same arm plus `--max-iterations 2`, so gap analysis gets to spend a second
round. Read this table per question, NOT by its mean: the three runs did three
different things.

| | Arm B: 1 round | Arm C: 2 rounds |
|---|---|---|
| Q1 RAG | gate 0.54, markers 33/33, cited 0.92 | gate **0.71**, markers **0/0**, cited 0.00 |
| Q2 MoE | gate 0.40, markers 24/24, cited 0.77 | gate 0.37, markers **39/39**, cited **0.95** |
| Q3 GNN | gate 0.20, markers 13/13, cited 0.42 | gate 0.07, markers 1/1, cited 0.02 |
| mean `gate_pass_rate` | 0.379 | 0.381 |

**Multi-hop does what fan-out could not: it changes retrieval.** Gap analysis
produced well-formed follow-ups rather than restatements (for Q1: "how vector
databases support semantic search in RAG", "RAG evaluation metrics faithfulness
relevance hallucination", "advanced RAG variants self-RAG GraphRAG hybrid
retrieval reranking"), and round-2 queries DO reach the paper providers,
because the academic lane loops `round_queries` -- `[question]` in round 1, the
gap list afterwards. Search calls: 3 in Arm B (one per question), 12 in Arm C
(Q1 1+5, Q2 1+4, Q3 1+0).

Per question, what actually happened:

- **Q2 is the clean datapoint, and it is positive.** Same gate rate (0.37 vs
  0.40) but 39 resolved markers against 24, and citation density 0.95 against
  0.77 -- multi-hop admitted more usable evidence and the report used it.
- **Q1 retrieved the most and cited nothing**, which is not a gate result but a
  synthesis failure: it is the ONLY question where map-reduce chunking engaged
  (54 chunk operations, 5 MAP calls), and under task-17382 every chunk summary
  was the provider error string the caller's guard failed to recognize. The
  model was handed five copies of an error message and had nothing to cite.
  Q2, which fit in a single pass, was unaffected. That defect is now fixed, so
  this run needs redoing before Q1's number means anything.
- **Q3 is not a multi-hop datapoint at all**: gap analysis returned no gaps, so
  only round 1 ran. Its 0.07 reflects a round-1 pool of 9 judged results (2
  admitted), not the effect of iteration.

So the mean `gate_pass_rate` moving 0.379 -> 0.381 states nothing. The
measurable claims from this arm are: multi-hop retrieval works and reaches the
academic lane; where the synthesis path held together it produced markedly more
cited evidence; and the pipeline's bottleneck under a larger evidence pool was
the SYNTHESIS path, not the relevance gate.

`gate_pass_rate` also needs reading as the ratio it is: round 2 adds to the
denominator, so pulling in more marginal repository records can lower the rate
while admitting more good evidence -- Q1 (0.71) and Q2 (39 markers) are the
same mechanism seen from two sides.

## What every number above was measured with (task-17370)

Both of the pipeline's decomposition mechanisms were OFF for every baseline
recorded above — the synthetic one, the academic and biomedical lanes, the
category lanes, and the 0.29 -> 0.42 gate re-measurement:

| Mechanism | Where it lives | Why it was off |
|---|---|---|
| phase-1 sub-question fan-out | `WebSearch_APIs.generate_and_search` (`subquery_generation`) | the recorder hard-coded `subquery=False, max_queries=1` as a spend bound |
| gap-driven replanning (multi-hop) | `local_research_engine._execute_phases` (task-16324) | the recorder launched runs with no `limits_json`, so `max_iterations` fell to the engine default of 1 |

That matters for how the repositories residual is read. The relevance gate is
prompted with the run's sub-questions, so a single-query run asks the gate to
judge every result against one broad question with an EMPTY sub-question
list — a repository record (a dataset, a figure) has no narrower facet it
could be relevant to. The recorded reading that the residual is "partly
genuine" was therefore not falsifiable from these runs: the mechanism whose
whole purpose is to give the gate narrower facets had never run.

Two things make the pending measurement a clean experiment on the
repositories lane specifically:

- Sub-questions DO reach the gate: the engine passes
  `merged_sqd = {"sub_questions": all_sub_questions, ...}` to the analyze
  phase, so generated facets change how ALL evidence is judged.
- Fan-out does NOT change what the academic lane retrieves: the paper lane
  loops over `round_queries`, which is `[question]` in round 1, so phase-1
  sub-queries never reach Zenodo/Figshare/OSF (filed as task-17372). Only
  multi-hop rounds >= 2 change retrieval.

So a `--max-queries N --max-iterations 1` re-run judges the SAME repository
evidence set with a non-empty sub-question list, isolating the gate effect
from any change in retrieval.

The recorder now takes `--max-queries`, `--max-iterations` and `--deadline-s`
(task-17370), and every emitted aggregate states the decomposition settings
it ran under, so no future baseline can silently be read as measuring
something it did not. **The decomposition-on arm is not yet measured** — it
needs the same judge model as the recorded arms (local Qwen3.8-27B) for the
comparison to mean anything.

## What decomposition is worth: the verdict (2026-08-17, task-17370)

The question this whole arm answered: the recorded repositories residual was
read as "partly genuine" -- repository records simply ARE marginal evidence for
a broad question -- and the counter-claim was that this is precisely what
sub-question generation and multi-hop exist to fix. Both mechanisms had been
off in every recorded baseline. They are now measured separately, because they
do different things and only one of them helped.

| Arm | Config | Q1 gate / markers | Q2 gate / markers | Q3 gate / markers |
|---|---|---|---|---|
| recorded | 1 query, 1 round | 0.42 mean over 3 (0.29 pre-note) | | |
| B | 3 queries, 1 round | 0.54 / 33 | 0.40 / 24 | 0.20 / 13 |
| C | 3 queries, 2 rounds | 0.71 / **0** | 0.37 / **39** | 0.07 / 1 |
| D | as C, guard fixed | 0.59 / 23 | -- | -- |
| E | as C, endpoint fixed | 0.63 / 17 | -- | -- |

> **Scope note (task-17372).** Every fan-out number below was measured while
> generated sub-questions could not reach the paper providers, so it measures
> the GATE-CONTEXT half of fan-out only. That limitation is now fixed -- the
> academic lane searches the facets too, bounded by the same query cap and the
> search ledger -- so fan-out's retrieval value is untested rather than
> disproven, and needs its own arm before the "no measurable benefit" reading
> can be extended to it.

**Fan-out (gate context): no measurable benefit.** The relevance prompt takes
`sub_questions` as a required placeholder, so the recorded arms rendered an
empty list and Arm B rendered real facets -- a genuine change in what the judge
was asked. Mean `gate_pass_rate` went 0.42 -> 0.38, i.e. flat within this
model's run-to-run variance. On this lane fan-out cannot change retrieval
either, because the paper providers only ever see round 1's `[question]`
(task-17372).

**Multi-hop (retrieval): positive where the pipeline let it through.** Gap
analysis produced real follow-up queries, round-2 queries DO reach the paper
providers, and search calls went 3 -> 12. On Q2 -- the one question whose
synthesis path was unaffected by task-17382 -- it held the gate rate while
taking markers from 24 to 39 and citation density from 0.77 to 0.95. On Q1
every 2-round arm (0.71 / 0.59 / 0.63) beat the 1-round arm (0.54).

**What had been hiding it was the synthesis path, not the gate.** Q1 retrieved
the most evidence of any run and cited NOTHING, because it was the only
question large enough to trigger map-reduce chunking, and under task-17382
every chunk summary was a provider error string the caller failed to recognize.
Fixing that took Q1 from 0/0 markers to 23/23. A negative result on the
strongest-retrieval run was an artifact.

So the reading of the residual is amended: it is not established that
repository evidence is genuinely marginal. The gate half of the argument does
not hold; the retrieval half does, and the measurement that appeared to refute
it was measuring a bug.

### The caveat that outlives this arm

No baseline in this document has EVER measured the pipeline with per-result
summarization working. It failed in about a millisecond (wrong config section),
then with a 404 (base URL posted raw), then with an unparseable payload
(OpenAI endpoint, native shape parsed), and once all three were fixed it timed
out at exactly the shipped 30s per call on a local 27B. Each time the pipeline
fell back to raw source content -- correct degradation, and invisible in the
metrics, because a report built from source text still resolves markers and
verifies quotes. Every number above therefore describes source-text evidence.
`--llm-timeout-s` now exists so that is measurable rather than assumed.

### First run with summarization actually working (arm F)

Bounded deliberately (1 question, 3 results/query, 1 query, 1 round) to isolate
one question: does the pipeline work at all when summaries are allowed to
finish? With `--llm-timeout-s 240`:

| Metric | Value |
|---|---|
| per-result summarizations | **7 attempted, 7 succeeded** |
| their durations | 42s, 94s, 131s, 64s, 81s, 115s, 93s (mean 88.5s) |
| over the shipped 30s timeout | **7 of 7** |
| `citation_accuracy` | 1.00 (15/15 markers) |
| `claim_support_rate` | 1.00 |
| `cited_sentence_ratio` | 0.65 |
| `gate_pass_rate` | 0.54 |

**The shipped `relevance_llm_timeout_s` of 30s cannot succeed on this model at
all** -- the fastest summary took 42s. That default is calibrated for hosted
providers; against a local 27B it guarantees the fallback path, which is why no
recorded baseline in this document ever measured a summarized evidence pool.

Residual: map-reduce CHUNK summarization still failed (2 of 2) with "No choices
in response data" while per-result calls on the same path in the same run all
succeeded, so it is input-size dependent -- a success status whose body carries
none of the parsed shapes. Filed as task-17384. The caller falls back to the
chunk's source text and marks it not-generated, so evidence stays real; the cost
is a wasted call and summarization quality on exactly the large evidence pools
multi-hop produces.

### The shipped default that came out of this (task-17371)

Local research runs are **multi-hop by default** as of this measurement:
`DEFAULT_MAX_ITERATIONS = 2` in `local_research_engine.py`, overridable per
install via `[SearchSettings] research_max_iterations` and per run via
`limits_json.max_iterations`.

Chosen from the numbers above rather than from caution: the second round is the
half of decomposition that changed retrieval, and on the one question whose
synthesis path was intact it held the gate's pass rate while taking resolved
markers 24 -> 39 and citation density 0.77 -> 0.95. Fan-out stays OFF by
default, because it measured flat (0.42 -> 0.38) and on this lane it cannot
change retrieval at all while task-17372 stands.

The cost is real and multiplicative: one extra search per gap, each with its own
per-result relevance and summarization calls, plus another synthesis and gap
analysis per round -- the measured arm went 3 -> 12 search calls over three
questions and roughly tripled wall-clock. Recorded baselines are unaffected
because the recorder passes `max_iterations` explicitly (default 1), which is
the property that keeps them reproducible byte-for-byte.

Two known interactions worth stating, since this default increases exposure to
both: multi-hop enlarges the evidence pool, and (a) each new source needs its
own summarization call, which cannot complete inside the shipped 30s timeout on
a local model (task-17382 chain), and (b) larger pools are what trigger
map-reduce chunking, where chunk summarization still fails against a local
endpoint (task-17384). Both degrade to real source text rather than to
nonsense, so the default is safe -- but a local-model install will feel it as
latency, not as better summaries.

### Reading gate_pass_rate

It is a ratio, and multi-hop adds to its denominator. Pulling in more marginal
repository records lowers the rate even when more good evidence is admitted --
Q1's 0.71 with zero citations and Q2's 39 markers are the same mechanism seen
from opposite ends. Read it beside the marker count, never alone.

## Recording a (fresh) live baseline

1. Configure `[SearchSettings]` (`relevance_analysis_llm`,
   `final_answer_llm`) and run several research runs from the Research
   window (task-16322 engine) across representative questions.
2. For each completed run, read `verification_summary.json` →
   `citation_verification` via the run bundle.
3. Feed each payload as a sample (`metadata.verification`) to a task config
   with `category: research` (or `task_type: research_report`) through the
   standard Evals runner; aggregate the four metrics.
4. Replace the synthetic table above with the live values and the date.
