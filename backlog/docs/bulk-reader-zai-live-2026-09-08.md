# Bulk-reader ZAI live pilot — 2026-09-08

**Decision: keep the preset available for explicit use; do not add automatic delegation.** The corrected run did not establish cheaper-reader quality or savings. All four delegated arms failed to run the named reader. Direct reading delivered the core requested facts in three of four cases, with one minor unsupported claim noted below.

The user delegated model selection and supplied a temporary credential. The experiment used ZAI `glm-5.3` as main and `glm-5.3-flash` as reader. Both attempts used the same pinned corpus, questions, prompts, fenced-tool protocol and limits. The credential stayed in process memory/environment; both runs confirmed the original config was unchanged and the key was absent from artifacts.

## Corrected comparison

| Case | Direct result | Delegated result | Direct USD | Delegated USD |
| --- | --- | --- | ---: | ---: |
| TTL exceptions | Correct core facts and citations | Direct fallback after rejected spawns | 0.005775 | 0.013361 |
| Cache ordering | Correct core facts and citations | No answer; repeated spawn calls | 0.005373 | 0.005381 |
| Late transcript correction | No answer; call/step budget exhausted | No answer; repeated spawn calls | 0.016616 | 0.004866 |
| Absent vendor and price | Correct core absence facts; minor tense overstatement | Malformed tool-call JSON returned as answer | 0.003210 | 0.001362 |

The repeat made **35 main-model calls and zero worker-model calls**, costing an estimated **$0.055944** in total. Direct arms cost $0.030974; attempted delegated arms cost $0.024970. Those totals cannot demonstrate worker savings: the lower-cost failures often delivered no answer.

The TTL fallback retained the four core facts but incorrectly described three constants as four. The direct absent-answer response correctly withheld a vendor and price, with accurate citations, but its closing claim that proposals “were gathered” is stronger than the source’s plan to collect them. The cache answer’s statement that no fallback mechanism exists is supported only within the supplied files. These are assistant source-review judgments, not human-validated scores.

## What the live run changed

The initial run exposed two concrete defects, now fixed and integrated:

- ZAI reasoning/control deltas became repeated synthetic “unsupported response shape” messages after private reasoning was removed. These polluted agent history and could consume the retained-output limit. ZAIStream now explicitly represents empty visible content while preserving native tool fragments and private terminal metadata.
- The evaluator missed configured prices because normalized `load_settings()` keeps pricing under `COMPREHENSIVE_CONFIG_RAW`. The live entry point now accepts that shape and direct raw configuration.

The initial run made 41 calls (35 main, 6 worker). Its raw report remains unchanged. Separately repricing its recorded provider usage gives **$0.089019**; its polluted answers are unsuitable for model-quality conclusions. **Estimated spend across both attempts: $0.144963.**

## Evidence and limits

- [Corrected raw comparison](../../Docs/Examples/agents/bulk-reader/results/2026-09-08-zai/comparison.json), [assistant review](../../Docs/Examples/agents/bulk-reader/results/2026-09-08-zai/review.json), and [run settings and rates](../../Docs/Examples/agents/bulk-reader/results/2026-09-08-zai/manifest.json).
- [Initial raw comparison](../../Docs/Examples/agents/bulk-reader/results/2026-09-08-zai/initial.json) and [separate pricing correction](../../Docs/Examples/agents/bulk-reader/results/2026-09-08-zai/initial-cost-correction.json).
- [Pinned source corpus](../../Docs/Examples/agents/bulk-reader/corpus.json).

Both new regressions failed on the original behavior. After the fixes, 72 targeted checks passed, 11 native-tool loopback checks passed, and 21 evaluator/ZAI regression checks passed in the integrated checkout. Scoped code review found no Critical/Important issues. Existing unrelated ZAI lint findings are unchanged. A broader provider-contract check still has a pre-existing fixture omitting the required `native_tools` argument; its test and constructor signature are unchanged by these fixes. No full suite was run.

This small synthetic run used fixed execution order and bounded calls/output. It is not a statistical comparison, and its failures must not be removed from cost reporting. Before a larger efficiency trial, verify reliable named-reader invocation; a controlled native-tool variant is a possible next experiment, not an established remedy.

Prices were verified against [ZAI’s pricing page](https://docs.z.ai/guides/overview/pricing) on 2026-09-08. Flash promotional rates end September 9 at 16:00 UTC. Estimates include recorded main and worker calls; they are not invoice totals.
