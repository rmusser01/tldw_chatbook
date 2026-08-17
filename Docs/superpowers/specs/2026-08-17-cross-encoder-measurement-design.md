# cross_encoder: implement and measure, or retire the name (TASK-16965)

Date: 2026-08-17
Programme: RAG server-port (sixteen merged; last: 17065 dispatch repair #1767)
Worktree: `.worktrees/rag-16965-crossencoder`, branch `feat/rag-16965-cross-encoder`, off dev `50a0b49ed`.

## The question, and why it is finally answerable

Reranking's retrieval value has never been measured here. TASK-3502 declared
it out of scope for a good reason — the three shipped strategies are all
LLM-driven, and the gated instrument is local, deterministic and unpriced. A
**local cross-encoder** is the strategy that can be measured inside the gate:
no spend, no network, reproducible.

**It is not a new dependency.** `sentence-transformers` already ships in the
`embeddings_rag` extra (`pyproject.toml`, two entries) — the same extra every
eval venv installs. A cross-encoder is a new *model artifact*, not a new
library, which is what makes this arc cheap enough to be worth doing.

## The ceiling must be established BEFORE the measurement

This is the arc's central discipline, and it comes from TASK-16071's finding:
**every one of the 60 golden queries returns ≤1 row under the shipped plain
pass**, and reordering a list of length ≤1 is the identity function. Reranking
is an *order-only* change. So before measuring anything, the arc must publish
the **row census per mode** (`semantic`, `plain`, `hybrid` — `runner.MODES`):
how many queries return ≥2 rows, i.e. how many rows a reranker could even
touch.

That census is what makes a null result interpretable. Without it, "no cell
moved" is ambiguous between *reranking does not help* and *there was nothing
to reorder*. With it, the null is precise. **If the census shows the
reorderable population is negligible on all three modes, that alone is a
publishable answer and grounds to retire** — no model download required, and
the arc ends early with a measured reason rather than an opinion.

## Pre-registered decision rule (fixed before any run)

Only if the census shows a non-trivial reorderable population do we implement
and measure. Then, decided in advance:

- **Helped** = the census-eligible subset improves on the ranking metrics the
  instrument already reports (MRR/NDCG/P@k), on at least one mode, beyond the
  gate's own tolerance, with no category regressing beyond it.
- **Null** = no metric moves beyond tolerance on any mode. Reported as the
  answer, per AC#3, and sufficient grounds to retire the strategy name.
- **Harmed** = any regression beyond tolerance → retire, and say so plainly.

The rule is written into the plan before the model is ever loaded. The
programme has shipped a pure null arc before (PRF/TASK-15965); a null here is
a result, not a failure.

## Shape of the work

- The measurement is a **PROBE**, not a new gated cell —
  `Tests/RAG_Eval/harness/` already holds `prf_probe.py` and `fusion_sweep.py`
  as env-gated informational runs, with `test_prf_probe_run.py` as the
  invocation precedent. **The always-on gate must NOT acquire a cross-encoder
  model dependency**: the committed baselines are environment-fingerprinted,
  and making the 105-metric gate depend on a second model download would tax
  every future arc for one question's sake.
- If implemented, `cross_encoder` becomes a real strategy in `reranker.py`
  alongside the three LLM ones, selected by the same config, requiring no
  provider and no credential (AC#5 falls out).
- Whichever arm ships, `config_profiles.py:346-352`'s not-implemented comment
  and Hybrid Full's `pointwise` substitution stop being described as a stopgap
  (AC#4) — either the substitution becomes a documented permanent choice, or
  it is replaced by the implemented strategy.
- AC#6: the gate reads `PASSED: No regression. 105 metric(s)` on the shipped
  state either way. Since the probe is informational and the strategy is
  opt-in, the shipped default retrieval path must not move.

## Out of scope
- LLM-strategy quality (unmeasurable here, by TASK-3502's reasoning).
- TASK-17265 (system prompt never reaching anthropic/google) and TASK-17365
  (cloned profiles' `include_reasoning`).
- Any change to fusion, merge or the four-seam path.

## Plan-phase verification
1. **The row census per mode** — the gating fact above. Produce it first; it
   may end the arc.
2. Whether a cross-encoder model can be fetched and cached in this
   environment at all, and where the eval harness expects model artifacts
   (`conftest.py` un-sandboxes the model cache dir for env-gated tests).
3. Whether `RerankingConfig`'s existing fields (`top_k_to_rerank`,
   `max_retries`, `include_reasoning`) are meaningful for a local strategy, or
   need explicit no-op semantics.
4. How `enhanced_rag_service_v2` selects a reranker, so a credential-free
   strategy does not trip the provider-shaped paths (the degraded/skipped
   tagging from TASK-3502 assumes provider failures).
