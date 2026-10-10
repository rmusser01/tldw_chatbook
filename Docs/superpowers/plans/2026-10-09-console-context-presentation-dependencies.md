# Context presentation dependencies

Task: TASK-34601, OPT10. Status: correction qualified locally; no whole-Send speed benefit established.

ADR required: no new ADR; dependency correction within ADR226.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md; preserves ADR225/126.
Reason: correct the existing disposable presentation memo inputs without adding a cache, owner, service or freshness policy.

## Evidence and contract

The stock context_control_presentation_inputs reads the held compaction echo but not active_run_id or run status. The existing key tracks the latter two and omits the former. Actual _resume_compaction_hold changes PAUSED to READY, clearing the echo, then awaits provider resolution before a run-status write. Preparation transitions do not bump payload/display/context/settings revisions. Ordinary hold entry already changes status and ordinary cancellation changes payload; those do not establish the missed dependency.

Use the existing stock_native_methods checker and two flat definition-time function/code anchors for the public presentation reader and held-echo getter. For this exact stock contract, use the held-echo ID instead of status/run ID. Custom or replaced readers/getters retain conservative lifecycle invalidation. Key shape must distinguish stock and custom routes, including source drift across an awaited read. Preserve every existing owner/repository/workspace/revision/config fence and one-second TTL. No dispatch/action read uses this memo.

## Ownership and sequence

1. Root is the integration/native timing owner and owns this plan, task, ADR and optimization ledger. No native test or timing runs overlap.
2. Baseline verification owns only Tests/Chat/test_console_context_presentation_reads.py. Establish original-source RED through real persisted lineage and original DB reads: lifecycle-only changes avoid rereads; actual compaction resume invalidates despite otherwise unchanged keys; an in-flight old echo result cannot publish. Preserve custom reader/getter lifecycle behavior and existing expiry/owner tests. Do not replace the reader being qualified or increase TTL/deadlines to obtain a pass.
3. Controller/provider lane owns only definition-time reader anchors in Chat/console_chat_controller.py. Shared preparation owns only the key change in UI/Console_Modules/console_spend_projection.py. Agree one anchor name and shape before edits; defer both implementations until original-source RED is recorded.
4. Root integrates both implementations, runs targeted context/compaction/publication/lifetime controls after both are ready, reviews and lints the patch. Custom callback/source changes must not reuse an earlier stock result.
5. Measure whole-Send sequentially against unchanged a017 source after resolving the separate image experiment. Report actual read-count savings separately from wall-clock results. The held-echo regression can justify the minimal correctness fix independently; do not claim an unmeasured speed improvement or completion of the latency target.


## Original-source evidence

context-dependencies-red-1 runs all15 presentation cases against unchanged a017 production:9 passed/6 failed in23.50s, with frozen sources. The six failures are intended: status and optional run-ID changes reread twice instead of once; actual compaction resume reads once instead of twice; the old in-flight echo result publishes True instead of refusing; replacing either public reader or echo getter after warm reuses the old result. Both custom-before-warm lifecycle controls and all seven existing cases pass. XML SHAd8412306489b7ff9cc19704da8a24ae91cb12bd360f632036f5c084406e06f26.

The resume cases execute real submit/hold/PAUSED-to-READY transition and stop at the original provider-resolution await, after its network-facing double delegates normally. Status and every existing mutable key fact remain unchanged while the echo becomesNone. The actual version read delegates before its optional hold and is released/gathered before database teardown. No TTL or timeout changes. This establishes the missed dependency; ordinary hold entry/cancel remain covered by their prior revision changes.

Implementation is one definition-time two-reader tuple plus the tagged _key dependency. Exact stock controller and original bound function/code checks qualify echo-based reuse; unsupported/custom routes keep lifecycle invalidation. Existing post-await publication comparison, all other key fields,1sTTL, native read and action/dispatch bodies are unchanged. Two independent lane reviews report no findings. Projection/test lint and formatting pass. Controller has the same60 lint findings and pre-existing formatting debt on baseline and candidate; no new findings, unrelated reformatting avoided.


## Integrated qualification and timing

After both product lanes were ready, root ran57 integrated cases in226.01s: all15 presentation cases, original native read/host-return/cancellation controls, real compaction actions/runtime resume, mounted context publication, shared configuration/source/owner/refusal controls, six repaired maintenance cases and both original Enter/button message-pump controls. All pass with unchanged source/HEAD. XML SHA2a26a0a53524c231db74da227a60feea00282166546d58db793fb65a9a9c26e7. The image experiment is absent from this source.

Separate final rendered-feedback checks pass2/2 in25.96s. Enter Preparing/input mutation/input frame are30.648/2.162/6.414ms; button44.346/1.241/7.497ms. These are supplied headless compositor frames under an actual held native reader, not physical-terminal latency or consistent percentiles. Existing earlier feedback misses remain in the image experiment record. XML SHA01b27c7c3ed6ab4d68311018fab446bf89b31b82d774407536d928c32cbb8279.

Quiet timing uses original a017 versus these two product files, sequential A1/B1/B2/A2. The integrated targeted checks occurred between A1 and B1; this gap is retained as a comparison limitation. Each fresh full-profile process saves three replies/traces/links with zero checkpoints, F/F/T streaming, unchanged source/HEAD and no detected overlap. Three additional loaded-source byte differences are line-ending-only; normalized text/AST match. No native census, stack or detail probe was enabled for these four timings.

| Run | Cold seconds | Warm1 seconds | Warm2 seconds |
| --- | ---: | ---: | ---: |
| A1 original |4.043252|3.578406|2.906646|
| B1 candidate |4.410993|3.496861|5.334612|
| B2 candidate |9.350818|7.188220|8.207069|
| A2 original |21.415890|6.653487|3.293971|

Observed warm means worsen4.108127 to6.056690s (47.43%); no speed improvement is established. The unchanged baseline's cold4.043-to21.416s change spans every measured stage, demonstrating broad uncontrolled variation with unknown cause. Candidate excess is principally saved-to-trace (+1.325s of +1.949s warm total). These samples establish neither an attributable47.43% code regression nor absence of a performance regression. Preserve the complete comparison rather than repeating runs until favorable. Receipt context-dependencies-comparison.json SHA892e1486c5e0226977c0af1f945f2c41f1d30cf3834ca86709be5909f8a6dfc9.

Retain the minimal dependency correction for its independently reproduced stale-result/source-drift bug. Lifecycle-only original read count drops2 to1 in the causal controls; that is a work-count result, not a whole-Send timing claim. Under-one-second overhead and consistent physical-feedback acceptance remain open. Separate original-function diagnostics on both arms investigate the remaining work; their instrumented timing is excluded from this table.


## Current original-function diagnostic

The same existing83-code DetailSpans observer ran sequentially on the candidate and baseline, each with3 saved replies/traces/links and42 stages, source/bindings current and monitoring retired. This preimports selected owners and is diagnostic-only. Pre-provider logical version reads are baseline2/5/5 and candidate2/6/6. Critical reads stay one dispatch per Send plus one preaccept per warm Send; all five original version calls total1.90ms baseline and3.44ms candidate. Extra candidate calls are presentation reads, with differing elapsed/TTL windows; no causal invalidation regression follows.

Warm saved-to-trace is baseline1.129/1.904s versus candidate2.022/1.872s. Recurring postcommit hook_admission_reason is0.183/0.237s versus0.331/0.358s, with nested locked_hooks_config_snapshot entry-to-first-yield0.088/0.125s versus0.171/0.191s. These are original inclusive wall intervals, not removable costs. Each Send retains six complete admission first-yield pairs. Context records have no gaps; hook/raw ancestry has92 depth misses per arm and seven unpaired ordinary starts (exception unwinds were not observed). Both generator-label unfinished counts are0. No blanket observer-completeness or speed claim.

A host snapshot during the candidate diagnostic shows54% aggregate CPU across12 logical processors; this isolated observation cannot explain the earlier quiet timing. Analysis and full limits: context-dependencies-detail-comparison.{json,md}, JSON SHA19c40b940b5a4b862514cfdc5daee8917a0719db466ac8737155b4521b47902c. Source reviewed at committed7d4039f6ac; all50 files captured by integrated qualification match its frozen bytes (context-dependencies-committed-source.json).

No new safe snapshot/L3/L4 contraction was found. Ordinary raw config/hook/MCP scopes normally carry one selected-parent pin, so replacing its scalar named stat with the existing forward/reverse multi-path proof would roughly double ancestor opens. Multi-pin directory-creation operations are a separate unmeasured case with error/close-contract differences. OPT17/25 remain deferred. Current evidence prioritizes OPT41's hook/config read declarations and OPT06's explicitly owned auxiliary prompt-history ordering for source review; no new authority or persistence contract is implemented here.
