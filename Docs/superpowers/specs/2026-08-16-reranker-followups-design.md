# Reranker follow-ups: honest controls, honest disclosure, safe diagnostics (TASK-3502)

Date: 2026-08-16
Status: draft-pending-user-review
Programme: RAG server-port (thirteen merged + the 13214 guard repair; dev `2b1d1817f`)
Worktree: `.worktrees/rag-3502-reranker`, branch `feat/rag-3502-reranker-followups`.

## What this arc is — and is not

TASK-3170 made a reranking-enabled profile actually construct and run a
reranker (a double-strategy TypeError had meant reranking silently never
activated). This arc closes what that fix exposed: the reranker is now
REAL, so its controls, costs and failure disclosures must be too.

**It is NOT the quality-measurement arc.** The LLM reranker (pointwise:
one provider call per candidate) cannot be measured on the gated
instrument — the instrument is local and deterministic; the reranker is
remote and priced. Measuring reranking's retrieval value needs a local
deterministic reranker (`cross_encoder`, explicitly unimplemented —
`config_profiles.py` says so in a comment). Declared out of scope; the
close-out will file one task for "implement-or-retire cross_encoder,
with a gated-instrument measurement" so the question has an owner.

## The six items (four ACs + the two final-review notes)

**AC#1 — provider/model selection.** Settings ▸ RAG's Reranking fold
today: `settings-library-rag-enable-reranking` + `reranker-model` +
`reranker-top-k` ids exist in the fold maps (`settings_screen.py:1132-
1134`) — the plan verifies what these actually render; the gap as filed
is that enabling creates a bare `RerankingConfig` defaulting to
`openai`/`gpt-3.5-turbo` (`RAG_Search/reranker.py:52-53`) with no
provider choice. Ship: a provider select + model input following the
fold's existing form idioms, persisted wherever the fold's other
reranking fields persist, and the default made visible rather than
implicit.

**AC#2 — cost disclosure BEFORE enabling.** The fold states, adjacent to
the toggle and visible before commit: pointwise reranking issues ONE
provider call per candidate (up to the configured rerank top-k, so up to
N calls per search), at the selected provider's prices. Static honest
text, not a live estimator — an estimator would need pricing tables this
repo does not own.

**AC#3 — copy-not-mutate under Pairwise/Listwise.** A regression test
drives the REAL `PairwiseReranker` and `ListwiseReranker` (reranker.py
:458/:596) through the `reranking_degraded` path
(`enhanced_rag_service_v2.py:330`) and asserts no cached `SearchResult`
is poisoned — the copy semantics differ from Pointwise's, which is the
whole reason the residual was filed.

**AC#4 — the counter race.** `BaseReranker.last_rerank_failures/_total`
(reranker.py:99-107) are instance state on a shared singleton, racy
under concurrent `search()`. Fix by SCOPING, not locking: return the
per-call counts to the caller (or compute them locally in the disclosure
site) so one search's failures cannot be attributed to another's tag;
the instance attributes become derived/legacy or are removed. Lock-based
"fixes" on a diagnostic path are the clever-unstable thing the owner has
ruled against.

**Note-(a) — disclosure tags have ZERO UI consumers.** `reranking_
skipped`/`reranking_degraded` are metadata-only, so a Hybrid Full user
with a dead credential sees normal-looking results. Ship the smallest
real consumer: the Library RAG results surface renders the tag as a
visible notice line when present (the recovery/notice vocabulary the
panel already has), with a test. No new subsystem.

**Note-(b) — the "| reranked" over-claim.** Partial pointwise failure
stamps `rerank_score = original_score` on failed rows, so a 14/15-failed
rerank renders "| reranked" on rows never rescored. Fix at the stamping
site: failed rows do not claim the reranked kind (they keep their
original score kind), and `library_rag_score_kinds.py`'s contract
comment is updated. Direction stays conservative — no fabricated scores.

## Out of scope (declared)
- `cross_encoder` implementation and any quality measurement (filed at
  close-out as its own task).
- Reranking defaults in profiles; the fusion/merge layers; anything the
  gated instrument covers — and the gate must hold **105/105 (+0.000)**
  regardless, since the disclosure/stamping changes touch the semantic
  service path. If a cell moves, STOP.
- Live provider spend: every test uses fakes at the provider seam (the
  reranker's LLM calls are mockable at `chat_api_call`); NO live LLM
  calls in this arc.

## Testing
- AC#3/#4: engine-level tests against the real reranker classes, fake
  provider seam; the race scoped by design, pinned by a concurrency test
  only if it can be made deterministic (else the scoping is pinned
  structurally: the disclosure site no longer reads shared state).
- AC#1/#2 + note-(a): Settings/panel tests in the existing harness
  style; counts read.
- Gate + `Tests/RAG_Search Tests/Library Tests/UI` (targeted files)
  batteries.

## Plan-phase verification (before tasks are cut)
1. What the existing `reranker-model`/`reranker-top-k` ids actually
   render, and where their values persist (the fold's save path).
2. The disclosure-tag plumbing: where `reranking_skipped/_degraded`
   would surface in the Library panel's row/notice model, and the
   existing notice vocabulary to reuse.
3. The stamping site for `rerank_score` on failed rows (PointwiseReranker
   `_apply_scores`) and what `library_rag_score_kinds.py` keys off.
4. Whether `RerankingConfig` is constructed anywhere besides the
   Settings toggle (profiles set reranking fields — enumerate).
5. The provider seam the reranker calls through (mock point for tests).
