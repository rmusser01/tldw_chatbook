# TASK-16965 Task 2 — the cross-encoder measurement, and its verdict

Date: 2026-08-17
Branch: `feat/rag-16965-cross-encoder` (worktree `.worktrees/rag-16965-crossencoder`)
Probe: `Tests/RAG_Eval/test_cross_encoder_probe_run.py` +
`Tests/RAG_Eval/harness/cross_encoder_probe.py` (mechanism) +
`Tests/RAG_Eval/test_cross_encoder_probe.py` (always-on tests for the rule)

Invocation (the whole run, reproducible):

```
RAG_EVAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv/bin/python -m pytest Tests/RAG_Eval/test_cross_encoder_probe_run.py -s -q
```

**Wall clock: 25.7 s** for the whole measurement (runtime build, the shipped
three-mode baseline, five reranked passes, 3,621 rows scored by the model;
17.2 s of that inside `rerank()`). **Zero network: confirmed three ways** —
`Tests/conftest.py`'s autouse `_no_network_io` guard blocks sockets and fails
the test at teardown on any blocked attempt (none recorded), the run asserts
`huggingface_hub.constants.HF_HUB_OFFLINE is True`, and `chat_api_call` —
the seam the other three reranking strategies bill through — was
monkeypatched to raise for the duration. **Zero provider spend.**

## VERDICT: HARMED

Per the rule pre-registered in
`Docs/superpowers/plans/2026-08-17-cross-encoder-measurement.md` before
`CrossEncoderReranker` was written, and implemented as a tested pure
function (`cross_encoder_probe.arm_verdict`) before this run produced a
number: **HARMED — any regression beyond tolerance.** Both arms regress:
`paraphrase` and `vocabulary_mismatch` lose MRR/NDCG beyond the 0.05 band on
both `semantic` and `hybrid`. The rule sends HARMED to T3 arm B — retire.

**The one reading of the rule that would change the verdict, stated rather
than buried.** The rule's gain clause is written at MODE level ("improves
... on at least one mode") and its regression clause at CATEGORY level ("no
category regresses"), which is what the probe implements. Read at the
OVERALL-ROW level only, nothing moved beyond tolerance in either arm on
either mode (largest overall move: semantic MRR −0.046 in arm B), and the
verdict would be **NULL**. Both readings land on the same T3 arm: retire.
The category reading is the one the plan's own words specify, so it is the
one reported, and it was fixed in code and pinned by tests before the run.

## What actually happened, in one paragraph

The cross-encoder is not inert — it reordered 442–1,032 rows per pass and
moved a correct document on 3–15 queries per arm — and its effects are
**strongly bimodal by category**. It produced very large gains exactly where
retrieval was weak: hybrid `scoped` MRR 0.163 → 0.929 (+0.766) and NDCG
0.348 → 0.947, hybrid `prompt` MRR 0.022 → 0.200, semantic `negation` NDCG
0.000 → 0.105 in arm B. It produced losses exactly where retrieval was
already perfect: `paraphrase` (13 queries) and `vocabulary_mismatch` (9
queries) both sat at MRR 1.000 before reranking — the target was at rank 1 on
every one — so the only movement available to them was downward, and three
queries lost rank 1 to rank 2 (`pr-requisition-freeze`,
`pr-photovoltaic-roof`, `vm-nearsightedness`, plus `pr-standup-slot` in arm
B). Those two ceilings are what trip the regression clause. On the averaged
overall row the two effects very nearly cancel: arm B buys recall@10 +0.022
on both modes and pays MRR −0.046/−0.026 for it. **Nothing in that picture
is a case for shipping the strategy as a default, which is what the rule was
written to decide.**

## Findings a reader should not have to derive

1. **P@k could not have moved in arm A, and the rule's P@k clause was
   therefore vacuous there.** `precision_at_k`, `recall_at_k` and `f1_at_k`
   are set functions of `retrieved_ids[:k]`; permuting a list of ≤ k
   documents leaves that set identical. Only MRR and NDCG read rank
   position. This is why the probe declares a second arm (retrieve at 20,
   rerank, score the first 10) — the only configuration in which reranking
   can promote a document into the top ten and therefore the only one where
   P@k/recall/F1 are live at all. Both arms were declared before the run,
   and both are reported in full.
2. **The measurement was one monkeypatch away from a fabricated NULL.**
   `Tests/conftest.py` sandboxes `HOME` at collection, and
   `huggingface_hub.constants.HF_HUB_CACHE` is computed from `expanduser("~")`
   at import — so under pytest the hub cache resolves into an empty temp
   directory and `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
   raises `OSError` ("couldn't connect ... and couldn't find them in the
   cached files") on a machine where the model is very much cached. Measured
   directly before the probe was written. `CrossEncoderReranker` degrades
   rather than raises (TASK-3502 contract), so every window would have come
   back in its original order and every metric would have read a clean 0.000
   delta: a perfect, entirely fake NULL. The probe repoints the constant and
   then **asserts `rows_failed == 0` and `rows_scored > 0`**, so the failure
   mode cannot return silently.
3. **The census reproduced exactly, from this run's own retrievals** —
   `plain` 0/60 queries with ≥ 2 rows (39 zero, 21 exactly one), `semantic`
   and `hybrid` 60/60 full windows at both k=10 and k=20. The null-adjacent
   parts of this result therefore cannot be explained away as "there was
   nothing to reorder". `plain` was run as a STOP guard and moved nothing:
   0 rows moved, every metric delta exactly 0.000.
4. **Arm A's before-column is the shipped measurement, not a re-implementation
   of it.** All 15 metric cells match `run_eval`'s own three-mode report to
   1e-9 (printed in the cross-check table below).

## The shipped gate is untouched by this measurement

The probe is informational and the strategy is opt-in; no production default
moved. Verified on this branch state, same session:

```
RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/test_harness_run.py -q
[rag-eval baselines] PASSED: No regression. 105 metric(s) within 0.05 of baseline.
3 passed in 10.31s
```

Always-on suite for the directory, with no env var set:
`Tests/RAG_Eval/` — **313 passed, 14 skipped** (the gated modules, including
this probe, skip with their own reason).

## THE PROBE'S OWN OUTPUT, VERBATIM

```text
==============================================================================
TASK-16965 — CROSS-ENCODER MEASUREMENT PROBE
model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, offline, no credential)
config: strategy=cross_encoder top_k_to_rerank=20 combine_original_score=True original_score_weight=0.3 score_scale=(0.0, 1.0)
metrics @k=10; tolerance 0.050 (the gate's own FAIL_BAND); verdict metrics mrr/ndcg/precision on semantic/hybrid
==============================================================================

CENSUS — how many queries a reranker could even reorder (measured in THIS run)
mode        depth  queries    >=2 rows   0 rows   1 row  full window
--------------------------------------------------------------------
semantic       10       60 60 (100.0%)        0       0           60
plain          10       60    0 (0.0%)       39      21            0
hybrid         10       60 60 (100.0%)        0       0           60
semantic       20       60 60 (100.0%)        0       0           60
hybrid         20       60 60 (100.0%)        0       0           60
A mode with 0 reorderable queries is the IDENTITY under any reranker; movement there would be a STOP, not a result.

INSTRUMENT CROSS-CHECK — arm A's before-column vs run_eval's own report
mode      metric          run_eval  arm A before   equal
--------------------------------------------------------
semantic  mrr             0.804348      0.804348     yes
semantic  ndcg            0.804348      0.804348     yes
semantic  precision       0.089096      0.089096     yes
semantic  recall          0.804348      0.804348     yes
semantic  f1              0.159673      0.159673     yes
plain     mrr             0.304348      0.304348     yes
plain     ndcg            0.295938      0.295938     yes
plain     precision       0.304348      0.304348     yes
plain     recall          0.293478      0.293478     yes
plain     f1              0.297101      0.297101     yes
hybrid    mrr             0.809179      0.809179     yes
hybrid    ndcg            0.817436      0.817436     yes
hybrid    precision       0.089130      0.089130     yes
hybrid    recall          0.847826      0.847826     yes
hybrid    f1              0.160738      0.160738     yes

ARM A (retrieve at k=10, rerank that window, re-score) — a permutation of the returned set
mode        depth  rows scored  failed  empty text  rows moved  queries reordered  docs reordered  predict s
------------------------------------------------------------------------------------------------------------
semantic       10          600       0           0         442                 60              58        3.2
plain          10           21       0           0           0                  0               0        1.0
hybrid         10          600       0           0         457                 59              59        2.3

mode      metric          before     after     delta  beyond tol  note
----------------------------------------------------------------------
semantic  mrr              0.804     0.772    -0.033          no  
semantic  ndcg             0.804     0.780    -0.024          no  
semantic  precision        0.089     0.089    +0.000          no  invariant under permutation
semantic  recall           0.804     0.804    +0.000          no  invariant under permutation
semantic  f1               0.160     0.160    +0.000          no  invariant under permutation
plain     mrr              0.304     0.304    +0.000          no  
plain     ndcg             0.296     0.296    +0.000          no  
plain     precision        0.304     0.304    +0.000          no  invariant under permutation
plain     recall           0.293     0.293    +0.000          no  invariant under permutation
plain     f1               0.297     0.297    +0.000          no  invariant under permutation
hybrid    mrr              0.809     0.798    -0.011          no  
hybrid    ndcg             0.817     0.810    -0.007          no  
hybrid    precision        0.089     0.089    +0.000          no  invariant under permutation
hybrid    recall           0.848     0.848    +0.000          no  invariant under permutation
hybrid    f1               0.161     0.161    +0.000          no  invariant under permutation

ARM A — per-category regression guard
mode      category              metric        before     after     delta  beyond tol
------------------------------------------------------------------------------------
semantic  keyword               mrr            0.938     0.938    +0.000          no
semantic  keyword               ndcg           0.938     0.938    +0.000          no
semantic  keyword               precision      0.117     0.117    +0.000          no
semantic  negation              mrr            0.000     0.000    +0.000          no
semantic  negation              ndcg           0.000     0.000    +0.000          no
semantic  negation              precision      0.000     0.000    +0.000          no
semantic  paraphrase            mrr            1.000     0.923    -0.077        LOSS
semantic  paraphrase            ndcg           1.000     0.943    -0.057        LOSS
semantic  paraphrase            precision      0.100     0.100    +0.000          no
semantic  prompt                mrr            0.000     0.000    +0.000          no
semantic  prompt                ndcg           0.000     0.000    +0.000          no
semantic  prompt                precision      0.000     0.000    +0.000          no
semantic  scoped                mrr            0.000     0.000    +0.000          no
semantic  scoped                ndcg           0.000     0.000    +0.000          no
semantic  scoped                precision      0.000     0.000    +0.000          no
semantic  vocabulary_mismatch   mrr            1.000     0.944    -0.056        LOSS
semantic  vocabulary_mismatch   ndcg           1.000     0.959    -0.041          no
semantic  vocabulary_mismatch   precision      0.103     0.103    +0.000          no
hybrid    keyword               mrr            0.944     0.950    +0.006          no
hybrid    keyword               ndcg           0.956     0.962    +0.005          no
hybrid    keyword               precision      0.113     0.113    +0.000          no
hybrid    negation              mrr            0.000     0.000    +0.000          no
hybrid    negation              ndcg           0.000     0.000    +0.000          no
hybrid    negation              precision      0.000     0.000    +0.000          no
hybrid    paraphrase            mrr            1.000     0.923    -0.077        LOSS
hybrid    paraphrase            ndcg           1.000     0.943    -0.057        LOSS
hybrid    paraphrase            precision      0.100     0.100    +0.000          no
hybrid    prompt                mrr            0.022     0.200    +0.178        GAIN
hybrid    prompt                ndcg           0.060     0.200    +0.140        GAIN
hybrid    prompt                precision      0.020     0.020    +0.000          no
hybrid    scoped                mrr            0.163     0.929    +0.766        GAIN
hybrid    scoped                ndcg           0.348     0.947    +0.599        GAIN
hybrid    scoped                precision      0.100     0.100    +0.000          no
hybrid    vocabulary_mismatch   mrr            1.000     0.944    -0.056        LOSS
hybrid    vocabulary_mismatch   ndcg           1.000     0.959    -0.041          no
hybrid    vocabulary_mismatch   precision      0.100     0.100    +0.000          no

ARM A — per-query MRR movement
semantic: 3 of 53 scored queries changed MRR (all of them, gains first):
    pr-requisition-freeze               1.000 ->   0.500  (-0.500)
    pr-photovoltaic-roof                1.000 ->   0.500  (-0.500)
    vm-nearsightedness                  1.000 ->   0.500  (-0.500)
hybrid: 12 of 53 scored queries changed MRR (all of them, gains first):
    sc-pump-chamber-inspection          0.111 ->   1.000  (+0.889)
    sc-intake-screen-survey             0.111 ->   1.000  (+0.889)
    sc-meter-box-key                    0.111 ->   1.000  (+0.889)
    sc-sample-point-sign                0.111 ->   1.000  (+0.889)
    sc-duty-board-notice                0.111 ->   1.000  (+0.889)
    pm-vendor-chaser                    0.111 ->   1.000  (+0.889)
    sc-valve-pit-access                 0.250 ->   1.000  (+0.750)
    sc-storm-overflow-record            0.333 ->   0.500  (+0.167)
    kw-plant-maintenance-record         0.111 ->   0.200  (+0.089)
    pr-requisition-freeze               1.000 ->   0.500  (-0.500)
    pr-photovoltaic-roof                1.000 ->   0.500  (-0.500)
    vm-nearsightedness                  1.000 ->   0.500  (-0.500)

ARM B (retrieve at 20, rerank, score the first 10) — the only arm in which P@k/recall/F1 are live
mode        depth  rows scored  failed  empty text  rows moved  queries reordered  docs reordered  predict s
------------------------------------------------------------------------------------------------------------
semantic       20         1200       0           0        1019                 60              60        5.6
hybrid         20         1200       0           0        1032                 60              60        5.1

mode      metric          before     after     delta  beyond tol  note
----------------------------------------------------------------------
semantic  mrr              0.808     0.762    -0.046          no  
semantic  ndcg             0.804     0.776    -0.028          no  
semantic  precision        0.085     0.087    +0.002          no  
semantic  recall           0.804     0.826    +0.022          no  
semantic  f1               0.153     0.157    +0.004          no  
hybrid    mrr              0.812     0.787    -0.026          no  
hybrid    ndcg             0.817     0.805    -0.012          no  
hybrid    precision        0.089     0.091    +0.002          no  
hybrid    recall           0.848     0.870    +0.022          no  
hybrid    f1               0.161     0.165    +0.004          no  

ARM B — per-category regression guard
mode      category              metric        before     after     delta  beyond tol
------------------------------------------------------------------------------------
semantic  keyword               mrr            0.938     0.938    +0.000          no
semantic  keyword               ndcg           0.938     0.938    +0.000          no
semantic  keyword               precision      0.106     0.106    +0.000          no
semantic  negation              mrr            0.049     0.072    +0.023          no
semantic  negation              ndcg           0.000     0.105    +0.105        GAIN
semantic  negation              precision      0.000     0.033    +0.033          no
semantic  paraphrase            mrr            1.000     0.872    -0.128        LOSS
semantic  paraphrase            ndcg           1.000     0.905    -0.095        LOSS
semantic  paraphrase            precision      0.100     0.100    +0.000          no
semantic  prompt                mrr            0.000     0.000    +0.000          no
semantic  prompt                ndcg           0.000     0.000    +0.000          no
semantic  prompt                precision      0.000     0.000    +0.000          no
semantic  scoped                mrr            0.019     0.190    +0.171        GAIN
semantic  scoped                ndcg           0.000     0.214    +0.214        GAIN
semantic  scoped                precision      0.000     0.029    +0.029          no
semantic  vocabulary_mismatch   mrr            1.000     0.944    -0.056        LOSS
semantic  vocabulary_mismatch   ndcg           1.000     0.959    -0.041          no
semantic  vocabulary_mismatch   precision      0.100     0.100    +0.000          no
hybrid    keyword               mrr            0.944     0.945    +0.001          no
hybrid    keyword               ndcg           0.956     0.957    +0.001          no
hybrid    keyword               precision      0.113     0.113    +0.000          no
hybrid    negation              mrr            0.049     0.078    +0.029          no
hybrid    negation              ndcg           0.000     0.111    +0.111        GAIN
hybrid    negation              precision      0.000     0.033    +0.033          no
hybrid    paraphrase            mrr            1.000     0.872    -0.128        LOSS
hybrid    paraphrase            ndcg           1.000     0.905    -0.095        LOSS
hybrid    paraphrase            precision      0.100     0.100    +0.000          no
hybrid    prompt                mrr            0.022     0.200    +0.178        GAIN
hybrid    prompt                ndcg           0.060     0.200    +0.140        GAIN
hybrid    prompt                precision      0.020     0.020    +0.000          no
hybrid    scoped                mrr            0.201     0.929    +0.728        GAIN
hybrid    scoped                ndcg           0.385     0.947    +0.563        GAIN
hybrid    scoped                precision      0.100     0.100    +0.000          no
hybrid    vocabulary_mismatch   mrr            1.000     0.944    -0.056        LOSS
hybrid    vocabulary_mismatch   ndcg           1.000     0.959    -0.041          no
hybrid    vocabulary_mismatch   precision      0.100     0.100    +0.000          no

ARM B — per-query MRR movement
semantic: 8 of 53 scored queries changed MRR (all of them, gains first):
    sc-valve-pit-access                 0.050 ->   1.000  (+0.950)
    sc-storm-overflow-record            0.083 ->   0.333  (+0.250)
    ng-surfaced-approach                0.083 ->   0.125  (+0.042)
    ng-three-panel-head                 0.062 ->   0.091  (+0.028)
    pr-photovoltaic-roof                1.000 ->   0.500  (-0.500)
    pr-standup-slot                     1.000 ->   0.500  (-0.500)
    vm-nearsightedness                  1.000 ->   0.500  (-0.500)
    pr-requisition-freeze               1.000 ->   0.333  (-0.667)
hybrid: 15 of 53 scored queries changed MRR (all of them, gains first):
    sc-pump-chamber-inspection          0.111 ->   1.000  (+0.889)
    sc-meter-box-key                    0.111 ->   1.000  (+0.889)
    pm-vendor-chaser                    0.111 ->   1.000  (+0.889)
    sc-intake-screen-survey             0.200 ->   1.000  (+0.800)
    sc-sample-point-sign                0.200 ->   1.000  (+0.800)
    sc-duty-board-notice                0.200 ->   1.000  (+0.800)
    sc-valve-pit-access                 0.250 ->   1.000  (+0.750)
    sc-storm-overflow-record            0.333 ->   0.500  (+0.167)
    ng-surfaced-approach                0.083 ->   0.143  (+0.060)
    ng-three-panel-head                 0.062 ->   0.091  (+0.028)
    kw-plant-maintenance-record         0.111 ->   0.125  (+0.014)
    pr-photovoltaic-roof                1.000 ->   0.500  (-0.500)
    pr-standup-slot                     1.000 ->   0.500  (-0.500)
    vm-nearsightedness                  1.000 ->   0.500  (-0.500)
    pr-requisition-freeze               1.000 ->   0.333  (-0.667)

ARM A VERDICT: HARMED
    semantic/paraphrase mrr: 1.000 -> 0.923 (-0.077)
    semantic/paraphrase ndcg: 1.000 -> 0.943 (-0.057)
    semantic/vocabulary_mismatch mrr: 1.000 -> 0.944 (-0.056)
    hybrid/paraphrase mrr: 1.000 -> 0.923 (-0.077)
    hybrid/paraphrase ndcg: 1.000 -> 0.943 (-0.057)
    hybrid/vocabulary_mismatch mrr: 1.000 -> 0.944 (-0.056)
ARM B VERDICT: HARMED
    semantic/paraphrase mrr: 1.000 -> 0.872 (-0.128)
    semantic/paraphrase ndcg: 1.000 -> 0.905 (-0.095)
    semantic/vocabulary_mismatch mrr: 1.000 -> 0.944 (-0.056)
    hybrid/paraphrase mrr: 1.000 -> 0.872 (-0.128)
    hybrid/paraphrase ndcg: 1.000 -> 0.905 (-0.095)
    hybrid/vocabulary_mismatch mrr: 1.000 -> 0.944 (-0.056)

VERDICT: HARMED — arm(s) A, B regressed beyond tolerance
(pre-registered rule, fixed before implementation: metrics mrr/ndcg/precision on modes semantic/hybrid, tolerance 0.050)

wall clock: 25.7s
```

---

## Addendum — what the owner decided, and what shipped (2026-08-17, Task 3)

Everything above is the measurement and is unchanged. This records the
disposition, because the body of this report sends the verdict to "T3 arm B —
retire" and that is **not** what happened.

Asked explicitly, with the trade-offs on the table — the pre-registered rule
says retire; the numbers also say the strategy is a large win exactly where
retrieval is weak (hybrid `scoped` MRR 0.163 → 0.929) and loses only rank-1 →
rank-2 in two categories that had no headroom left — **the owner ruled: "KEEP
THE CODE, RETIRE THE PROMISE."**

So, concretely:

- `CrossEncoderReranker` stays implemented and stays selectable as
  `reranking_strategy = "cross_encoder"`. The name is **not** retired from the
  strategy vocabulary.
- Every claim or implication that it *helps* is gone, and the verdict above now
  travels with it: `RAG_Search/reranker.py` (module docstring, the `strategy`
  Literal, the class docstring's numbers), `RAG_Search/config_profiles.py`,
  `config.py`'s config template, `Helper_Scripts/rag_config_examples/rag_v2_example.toml`,
  `Config_Files/rag_pipelines.toml`, `Docs/User_Guide/settings/rag.md`,
  `Docs/Development/RAG/RAG-DESIGN.md` (it was sitting under *Future
  Enhancements*), `Docs/Development/RAG/RAG-Documentation.md`, and
  `Tools/document_expansion_tool.py` (whose docstring claimed it was
  unimplemented).
- It is the default of nothing, the recommendation of nothing, and the strategy
  of no profile. **Hybrid Full keeps `pointwise`** — now documented as a
  permanent, measured choice rather than the task-3170 stopgap it started as.

The rule was not renegotiated: the verdict stands as HARMED, and nothing in
this repo recommends the strategy. What the ruling changed is whether working,
measurable code gets deleted for producing an unwelcome number — it does not,
so long as the number travels with it.
