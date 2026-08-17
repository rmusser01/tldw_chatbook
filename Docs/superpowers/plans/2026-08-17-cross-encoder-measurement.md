# TASK-16965: cross_encoder — Implement and Measure, or Retire

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer whether reranking helps retrieval in this repo, using the one strategy the gated instrument can see — a local cross-encoder — and let the measured answer decide whether `cross_encoder` ships or the name is retired.

**Architecture:** Three tasks. T1 implements `CrossEncoderReranker` (local, credential-free) behind the existing strategy factory. T2 runs the pre-registered measurement as an env-gated PROBE and reports its verdict. T3 acts on the verdict — ship-with-docs or retire — and closes the task. **The verdict rule is fixed in this plan and must not be renegotiated after seeing numbers.**

**Spec:** `Docs/superpowers/specs/2026-08-17-cross-encoder-measurement-design.md` (census amended at `af068c383`).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-16965-crossencoder`, branch `feat/rag-16965-cross-encoder` (off dev `50a0b49ed`). Venv EXISTS (pinned; provenance verified).
- **EVERY Bash block starts with its own `cd <worktree>` and echoes `pwd` + branch before any mutating git op.** Two cwd incidents in this programme; the second was caught only by a permission classifier. A cleanup block that leaves the worktree must be the LAST of its group.
- **After ANY agent/tool dies abnormally, run `git status` before trusting a test result** — a dead reviewer's abandoned mutation reads exactly like a real regression (TASK-17065 incident).
- Never `git stash`; Edit-based restores; RED-first; do NOT run `Tests/UI/test_library_shell.py`. ruff via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`.
- **No network during measurement, no provider spend, ever** (AC#5). The model is cached already (see below); probe runs under `HF_HUB_OFFLINE=1`.
- Commits reference TASK-16965 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; push after every task.

## Verified this session — do not re-derive

- **The census (the gating fact).** `plain`: **0/60** queries return ≥2 rows (39 return 0, 21 return exactly 1) — reranking is provably the identity there, so a moved plain cell is a STOP, not a win. `semantic`: **60/60** return a full window (10 at k=10, 20 at k=20). `hybrid`: **60/60**. So the reorderable population is large on two of three modes and a null cannot be dismissed as nothing-to-reorder.
- **No new dependency.** `sentence_transformers.CrossEncoder` imports from the installed `sentence-transformers` 5.7.0 (already in the `embeddings_rag` extra).
- **The model works offline.** `cross-encoder/ms-marco-MiniLM-L-6-v2` fetched + loaded in 17.9 s and separates correctly (relevant **+8.719** vs irrelevant **−11.14**); ~0.3 s for a 2-pair predict. It is now in the local HF cache. **Do NOT use `mixedbread-ai/mxbai-rerank-large-v2`** — the cached copy is a 20 MB partial with no weights file and raises `OSError` offline.
- **The selection seam** is `reranker.py:~783` (`create_reranker`), a plain `if/elif` on `config.strategy` ending in `raise ValueError(f"Unknown reranking strategy: {strategy}")`. Adding a branch is the whole wiring job; `config_profiles.py:346-352` holds the not-implemented comment to update.

## THE DECISION RULE — pre-registered, binding

Measured on `semantic` and `hybrid` only (plain is the identity by census). Compare the instrument's ranking metrics (MRR / NDCG / P@k) reranked vs not, over the same run:

- **HELPED** — at least one of MRR/NDCG/P@k improves beyond the gate's tolerance (0.05) on at least one mode, AND no category regresses beyond tolerance. → ship the strategy (T3 arm A).
- **NULL** — nothing moves beyond tolerance on either mode. → **report as the answer and RETIRE the name** (T3 arm B). Per AC#3 this is sufficient grounds; the programme has shipped a pure null before.
- **HARMED** — any regression beyond tolerance. → retire, and say so plainly.

Ties, partial movement, or a mixed picture resolve to **NULL** — the burden is on the strategy to show a gain, not on the reviewer to disprove one.

---

### Task 1: `CrossEncoderReranker` — local, credential-free

**Files:** Modify `tldw_chatbook/RAG_Search/reranker.py`; Test `Tests/RAG_Search/test_cross_encoder_reranker.py` (new).

- [ ] **Step 1:** `backlog task edit 16965 -s "In Progress"` + `--plan`. Read `BaseReranker` and one concrete strategy end-to-end first; note in the report which base behaviours (the `RerankOutcome` contract, the `scored` flag, the result cache) a local strategy must honour.
- [ ] **Step 2 (RED):** tests, with a FAKE model (no download in unit tests — a stub exposing `.predict(pairs) -> list[float]`):
  - `test_cross_encoder_reorders_by_model_score` — rows come back ordered by the stub's scores, and `rerank_score` is stamped on scored rows.
  - `test_cross_encoder_needs_no_credential` — construction and `rerank()` succeed with no api key configured and **no `chat_api_call` import reachable**; monkeypatch `chat_api_call` to raise and assert it is never called.
  - `test_cross_encoder_honours_top_k_to_rerank` — only the top-k window is scored; the tail keeps its original order and carries no reranked stamp.
  - `test_cross_encoder_model_failure_degrades_like_the_others` — a model that raises yields `scored=False` rows and a degraded outcome, not an exception (the TASK-3502 contract).
  - `test_create_reranker_dispatches_cross_encoder` — the factory returns the new class instead of raising `Unknown reranking strategy`.
  - `test_provider_shaped_fields_are_no_ops` — `max_retries` / `include_reasoning` do not change behaviour for this strategy (they are provider concepts); assert explicitly rather than leaving it implied.
- [ ] **Step 3:** Implement `CrossEncoderReranker(BaseReranker)`: lazy model load (module-level cache keyed by model name so a probe run loads once), `model_name` taken from `config.model_name` with the ms-marco default when it names an LLM, batch `predict` over `(query, document)` pairs, `RerankOutcome` with per-row `scored`. Register in `create_reranker`. **No `chat_api_call`, no credential read.**
- [ ] **Step 4:** GREEN; `Tests/RAG_Search/` counts READ; ruff. Commit `feat(rag): local cross-encoder reranking strategy (TASK-16965)` + trailer. Push.

---

### Task 2: The measurement probe (env-gated, offline)

**Files:** Create `Tests/RAG_Eval/harness/cross_encoder_probe.py` + `Tests/RAG_Eval/test_cross_encoder_probe_run.py`; follow `prf_probe.py` / `test_prf_probe_run.py` exactly for structure, env-gating and reporting.

- [ ] **Step 1:** Probe: build the eval runtime as `test_harness_run.py` does; for `semantic` and `hybrid`, run every golden query, then rerank each result list with the real cross-encoder and re-score the same metrics. Emit a per-mode before/after table plus the **VERDICT** line, exactly as the PRF probe emits its own.
- [ ] **Step 2:** **The probe must also print the census it relies on** (rows ≥2 per mode) so the verdict is self-justifying in its own artifact.
- [ ] **Step 3:** Run it under `RAG_EVAL=1 HF_HUB_OFFLINE=1`. Record wall-clock and confirm zero network. The output table + verdict are the deliverable, whatever they say.
- [ ] **Step 4:** Commit the probe and a short `Docs/superpowers/qa/2026-08-17-cross-encoder/report.md` carrying the table verbatim. Push. **STOP and report the verdict before T3.**

---

### Task 3: Act on the verdict, and close

- [ ] **Arm A (HELPED):** keep the strategy; update `config_profiles.py:346-352`'s comment (it is no longer unimplemented); decide explicitly whether Hybrid Full switches from `pointwise` to `cross_encoder` — and if it does, note the behaviour change; add the strategy to user-facing docs with the measured gain.
- [ ] **Arm B (NULL/HARMED):** **retire the name** — remove `cross_encoder` from the strategy vocabulary and the config comment, keep the probe and its report as the record of why, and document Hybrid Full's `pointwise` as the permanent choice rather than a stopgap (AC#4). The implementation from T1 is reverted or kept behind no vocabulary — state which and why.
- [ ] **Both arms:** gate reads `PASSED: No regression. 105 metric(s)` on the shipped state (AC#6); close TASK-16965's six ACs against evidence; Implementation Notes carry the census, the verdict, and the rule that was fixed beforehand. Push.

## Self-review (plan time)
- AC#1→T3 (a recorded decision either way); #2→T2 with the rule above fixed in T-minus; #3→the NULL arm being pre-authorised to retire; #4→T3 arm B; #5→cached model + `HF_HUB_OFFLINE=1`, no provider path in T1's implementation; #6→T3.
- The riskiest failure mode is renegotiating the rule after seeing numbers; it is therefore written here, before Task 1.
