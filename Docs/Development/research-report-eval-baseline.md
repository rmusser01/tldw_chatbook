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
