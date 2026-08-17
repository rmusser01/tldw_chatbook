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

## Category-lane live baselines (2026-08-16, task-16812)

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

Live-verification fixes en route (task-16812): OSF intermittently 301s —
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
  (the DuckDuckGo bot challenge this doc notes elsewhere), and the academic
  lane only searches round 1's `[question]` (task-17372), so the repository
  records under judgment were the SAME set as the 0.42 arm. Fan-out changed
  how they were judged, not which ones existed.

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
