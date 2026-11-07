# Scoped PR dependency audit

The authorized scope is **85 of 94 local-only commits**, covering **968 feature paths**: native phased shared-worktree workflows and the subsequent native Console UI coroutine fix. Use pinned upstream `3c39215749a96d5c1b8c05972fdd2aeb7588b002`, local source `e08627ac38392fb890984169376b1c242c3a2f3e`, and the parent of design commit `8f989edc9fc33998108a0ba093a44f784700180b` as the feature-delta base. Exact commit/path lists and dependency decisions are in `scope-audit.json`.

## Exclusions and shared metadata

Exclude the first seven radio/mobile design commits (`8447cb870d` through `1bb7c50723`) and **both** interleaved prompt-mixin commits: `13ded5544d` (design) and `235f783e78` (plan). Their 46 exclusive paths include ADR-242/244/286, radio/mobile/prompt-mixin specs/plans/reviews, and their Backlog tasks. These are exclusions from the local delta, never instructions to delete equivalent existing upstream files.

`backlog/decisions/README.md` is the sole path shared between excluded and included commits. Include only the local ADR-285 row and preserve current upstream rows. Keep feature task TASK-38493, TASK-38523–38533, TASK-38614, ADR-285, amendments to ADR-084/094/148/163/181, both feature/fix plans, and the relevant architecture/user/lesson updates. No new ADR is required for this read-only audit.

Keep all 146 production paths and 102 test paths in the feature delta. Shared ordinary Console, hooks, MCP, RawCLI, SSH, first-send persistence, Change Review/Undo, storage/recovery and native activity ownership changes were introduced by the feature commits and are necessary to the feature’s existing ownership contract. They are not unrelated mobile work merely because their owner is shared. Keep the 685 frozen QA records, including source-qualified adverse results, archives and the separate coroutine follow-up; rewriting those would destroy provenance.

## Staged dependency assessment

All 61 originally staged added files exist upstream: 44 are byte-identical and 17 differ. **No required staged local-only API delta was identified. Do not copy any staged file into the PR.** The 17 differing files are older versions, documentation-only deltas, or canonical import relocations. Upstream adds optional arguments/helpers or preserves the existing public methods; it also improves Unicode exclusion handling and retains current prompt-mixin/capture safety behavior. Exact per-file observations are in the JSON.

One integration-sensitive signature is `console_first_send_modules.preimport_first_send_modules(app, on_complete)` upstream, versus the local one-argument form. Preserve the current upstream callback and app caller. Existing schema/prompt/card additions use optional parameters. Preserve upstream `Provider_Contract.thinking_blocks`, `state.settings_config_models`, `LLM_Provider_Catalog.provider_model_resolution` and public recovery participant imports. Reverting these dependencies would remove unrelated delivered behavior and may break current callers.

## Recommendation and evidence limits

Apply only the feature delta over pinned upstream. Do not overlay complete local HEAD monoliths: upstream has since split controller/app/preparation code and added independent behavior. Resolve shared files with both upstream refactors and the feature’s actual changes intact; regenerate CSS from integrated source modules. The original staged index and dirty working tree remain authoritative user work and must stay untouched.

This audit inspected committed diffs, dependency bytes and Python signatures/import references only. It did not repeat runtime review, run tests, access the network, alter Git/index/refs, or touch profiles. Only the requested audit Markdown/JSON were written. Historical dirty-source provider/test receipts cannot prove the integrated tree; root must run its targeted integration checks and review gates before presenting a clean PR claim.
