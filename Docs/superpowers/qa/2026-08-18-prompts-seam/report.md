# TASK-18255 — wiring the harness prompts seam, and what it settled

**The seam is wired. `pm-vendor-chaser` now retrieves. The gated baseline was
re-stamped deliberately: 10 of 105 metrics moved, all up, all in `plain`.**

Reproduce: `seam_effect.py` beside this file (`RAG_EVAL=1`).

## What this settles, beyond the re-scoped task

TASK-17855 reported a production defect: `plain` returned zero rows for all
five `prompt` goldens, including `pm-vendor-chaser`, whose target contains
every content word. PR #1807 withdrew that claim as **unsupported** — the
harness set `prompt_scope_service=None`, so `_search_prompts` returned
`(False, [])`, the seam reporting itself *unavailable* — while being careful
not to assert the opposite, because nothing had exercised the sub-leg
against a real `PromptScopeService`.

**Now something has.** With the seam wired:

```
PROBE PROOF: app.prompt_scope_service = PromptScopeService
PROBE PROOF: _search_prompts availability = True
PROBE PROOF: smoke-query rows = 6
```

and `pm-vendor-chaser` **HITs**. The defect claim moves from *withdrawn as
unsupported* to *disproven*: the sub-leg works, and the 0.000 was the
instrument all along.

## The five prompt goldens, per query

An aggregate of 0.200 means one of five, which is exactly the kind of number
that hides four misses — so the AC required per-query reporting:

| query | result | rows | prompt rows | target |
|---|---|---|---|---|
| `pm-shift-summary` | MISS | 0 | 0 | `prompt-shift-summary` |
| `pm-incident-timeline` | MISS | 0 | 0 | `prompt-incident-timeline` |
| **`pm-vendor-chaser`** | **HIT** | 1 | 1 | `prompt-vendor-chaser` |
| `pm-meeting-actions` | MISS | 0 | 0 | `prompt-meeting-actions` |
| `pm-glossary` | MISS | 0 | 0 | `prompt-glossary-builder` |

**1 of 5.** The four misses are consistent with TASK-17855's surviving
finding: their targets do not contain the queries' content words, which no
seam wiring can fix. `pm-vendor-chaser` was the one query whose target *did*,
and it is the one that hits.

## The predicted cost did not materialize

The harness comment warned that wiring the seam "would move plain-mode
numbers for NON-prompt queries too, since the seam appends its rows to every
plain fan-out." Measured:

**0 of 55 non-prompt queries have a prompt row in their fan-out.**

The seam appends nothing where nothing matches, so the fan-out is unchanged
for every non-prompt query. The warning was reasonable and turned out not to
bind on this corpus — which is worth recording, because it was the stated
reason the wiring was deferred.

## The re-stamp: 10 of 105 metrics, all up, all in `plain`

`[rag-eval baselines] PASSED: No regression. 105 metric(s) within 0.05 of
baseline.`

| metric | before | after |
|---|---|---|
| `plain category.prompt.{precision,recall,mrr,ndcg,f1}` | 0.000 | **0.200** |
| `plain overall.{precision,recall,mrr,ndcg,f1}` | 0.315–0.326 | **+0.022** |

**No category other than `prompt` moved anywhere.** The `overall` movement is
arithmetic: a category that was vacuously 0.000 now contributes a real value
to the average. Every `semantic` and `hybrid` metric is unchanged at +0.000,
which is the expected shape — `plain` is the only mode that uses the
Library's four-seam fan-out.

## Disclosure: the environment fingerprint moved, and why that is safe here

The re-stamp changed one fingerprint field:

```
- "sentence_transformers": "5.7.0"
+ "sentence_transformers": "5.4.1"
```

The committed baselines were stamped on a machine with 5.7.0; this one has
5.4.1. Nothing else in the fingerprint moved — python, platform, torch,
transformers, chromadb and the embedding model are all unchanged.

**The library difference is demonstrably retrieval-neutral on this corpus**:
every one of the **70 metrics across `semantic` and `hybrid` is byte-identical**
(+0.000). If the version difference affected embeddings, those two vector
modes are where it would show, and it does not. So the 10 moved cells are
attributable to the seam, not to the library.

This is disclosed rather than hidden because a future arc reading these
baselines needs to know the environment block moved and why. Re-stamping on a
5.7.0 machine would restore it; the metric values should not change.

## Residual, recorded not fixed: `(True, [])` still collapses

`_search_prompts` ends `except Exception: return True, []`, so a seam that
**threw** is still reported as available-and-empty — indistinguishable in the
metrics from one that searched and matched nothing. This run logged the
warning **0 times**, so these numbers are clean, and `seam_effect.py` checks
for it and refuses a verdict if it fires.

Fixing the collapse properly means giving the metrics layer a distinct
"unavailable" value for every optional seam, not just prompts. That is a
larger change than this task, and it is the shape of the defect that produced
TASK-17855 in the first place — recorded here as the accepted residual.
