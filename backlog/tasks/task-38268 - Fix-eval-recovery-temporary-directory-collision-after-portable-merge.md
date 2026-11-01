---
id: TASK-38268
title: Fix eval recovery temporary directory collision after portable merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-31T23:49:00Z'
updated_date: '2026-11-01T07:44:10Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged portable bundle introduced an autouse private temporary directory that collides with the existing launch-environment regression on Ubuntu and macOS. Restore meaningful coverage and audit the remaining merged PR CI results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three native temporary selectors reach and pass their launch-environment assertions with the autouse private fixture active.
- [x] #2 The targeted eval-recovery workflow and remaining PR checks are verified or their unresolved results are explicitly recorded.
- [x] #3 Windows recovery tests and their child processes use owned APPDATA and LOCALAPPDATA sandboxes so ambient installed pet state cannot enter a test backup.
- [ ] #4 The required workflow executes its unchanged gate on the owner-approved Ubuntu 24.04 runner and current-head verification is resolved before leaving draft.
- [x] #5 All seven newly inventoried modal classes have individually justified policy classifications, and the existing complete reverse census passes without changing production geometry or weakening its coverage.
- [x] #6 The setup guidance keyboard test covers every displayed action and verifies that refresh preserves the action reached by Tab navigation.
- [x] #7 An edited Prompt-cache snapshot Keep count remains marked unsaved after Console quit review returns to Providers & Models, with the edited value and rail marker retained.
- [x] #8 The Console stylesheet meets the existing 2,000-active-line ceiling through a cohesive vocabulary split while preserving selectors, declarations and visual behavior.
- [x] #9 Every lazy first-send module observed by the send guard is warmed after the screen routes, while modules resident at UI-ready stay classified as resident and the 94-module budget remains unchanged.
- [ ] #10 All four project-chat UAT variants explicitly enable the full agent tool surface they exercise and retain real project-binding, disclosure, file-evidence and wrong-family refusal assertions.
- [ ] #11 The targeted Windows style-governance checks accept native diagnostic paths and canonical generated text on publication and fresh checkout within the unchanged boot-byte budget, with complete CSS token parity preserved.
- [x] #12 The provider choice list keeps its existing 40vh cap through a defined semantic design token and passes the unchanged zero-literal dimension contract.
- [x] #13 The exact visible-shell navigation census includes Kanban and Plugins and still requires every declared destination to resolve to its intended screen class.
- [ ] #14 Application lookup/private-name, layer-direction, runtime import-footprint and affected module-size guards pass through explicit service ownership and cohesive shutdown composition, with no raised pins or allowlist expansion.
- [x] #15 The provider key hint retains its computed width through an exact existing ID selector and the unchanged 270 ancestor-type-rule ceiling passes.
- [x] #16 The arrival and runtime-custody tests use the current next-request estimate and pressed-input witness contracts while preserving their unknown-window, lifetime-only and privacy assertions.
- [x] #17 Notes focus verification waits through the temporary absence of the replaced New button and still requires focus on its actual replacement within the existing deadline.
- [x] #18 The Plugins list has a visible focused border in all existing theme contrast checks without changing geometry, thresholds or debt records.
- [x] #19 Setup-step errors preserve their finite wrapper-indent alignment through token-backed classes, including repeated errors and step transitions, without inline padding writes.
- [x] #20 Successful pressed-input custody acceptance starts the existing send-start window through provider resolution, preserving received-intent, expiry and durable-commit controls.
- [x] #21 The two local-model preflight regressions verify fenced schema disclosure and minimal plain fallback through the actual prepared request, preserving native/fenced request accounting and the light tool set.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing ADR-103; no duplicate ADR).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: owner-approved explicit Linux runner selection for the required workflow, preserving its gate and capacity contracts. Existing ADR-126 still governs unchanged recovery authority.
1. Preserve both recovery fixture fixes and the clean rebase onto dev 77718ebebc.
2. Record the observed ubuntu-latest queue and the owner's approved ubuntu-24.04 trial in ADR-103 before changing the workflow.
3. Update only existing Derived Artifacts runner assertions, observe them fail against the floating label, then pin all nine workflow jobs to ubuntu-24.04 and verify them.
4. Preserve test selectors, shard counts/parallelism, dependencies, permissions, timeout budgets and fail-closed aggregation; run the existing workflow-contract tests and relevant static guards.
5. Push the qualified change, retire only this PR's superseded runs, and inspect all current-head CI/review feedback before marking ready. No merge or auto-merge.
6. Repair the hosted UI modal-registry failure under existing ADR-150/161/162 and TASK-36382/36522 policy: reproduce the reverse census, record seven individually verified existing-geometry exemptions, and run its existing static contracts plus scoped lint. No production layout or census changes.
7. Diagnose and reproduce the three hosted UI-shard-5 failures on the newly rebased tree. Preserve actual action traversal and unsaved-preference review semantics; correct stale tests or product logic according to evidence. Split Console CSS by existing component vocabulary, rebuild through build_css.py, and retain all current design-token, stylesheet-size and bundle contracts. Existing ADR-150/161 govern unchanged UI contracts; no new ADR for routine corrections.
8. Under existing ADR-033, reproduce the diagnosed Internal Prompts callback overwriting a dirty Providers & Models inspector. Replay that real callback deterministically in the retained snapshot/quit test, then move only selected status/banner updates into the existing active-category guard; retain rail and category-owned updates.
9. Repair first-send inventory drift under existing ADR-097 boot budgets, ADR-198 and ADR-230: remove the already-resident dispatch entry, add the two observed safe lazy modules, and run strict residency/order/send, cycle and unchanged payload-budget guards.
10. Adapt the actual-app project UAT's preconfigured test environment to explicitly enable the full tool mode required by its find/load/fs scenarios under ADR-104's accepted light-surface amendment. Reproduce fenced failure, then qualify all four cases and unchanged receipt/negative-evidence contracts; do not alter product defaults or permissions.
11. Repair two observed native style-check assumptions: make the existing atomic generated-text writer publish canonical UTF-8/LF text on every OS, and use an escaped native relative path in the synthetic hex-floor assertion. Rebuild through the same builder, rerun the two failing guards plus relevant reproducibility checks, and retain token parity and the 608,090-byte ceiling. Routine cross-platform corrections under existing ADR-150/161; no new ADR.
12. Repair the newly reported UI13/UI12 contract gaps: define the existing 40vh provider-choice cap as a semantic setup token in core/_variables.tcss, use it in the wizard source and rebuild without changing effective geometry; reproduce and update the complete navigation expectation with KanbanBoardScreen and PluginsScreen. Preserve the zero-literal floor, screen resolution and all numeric budgets. Existing ADR-150/161 and navigation policy apply; no new ADR.
13. Repair the primary architecture failures under the amended existing ADR-036/250/146 contracts: replace new consumer app reflection with declared/injected services, preserve Notes non-constructing peeks and current MCP reads, put plugin caching in its existing public owner, and declare selected-tool count state. Move the service-drain phase plus cohesive worker-settlement body into a presentation app_shutdown_services owner, retaining lifecycle patch points and lowering measured size pins. Reverse the route-suffix mapping dependency while preserving one owner. Run strict architecture and relevant behavioral tests; never raise a frozen count or add an exception to hide a violation.
14. Re-key the provider hint's only Static rule to its existing exact ID, preserving its declaration and computed width; run the actual 270-rule ratchet, bundle sync and byte guard.
15. Reproduce and update the stale arrival fake and custody request field census for the accepted next-request and same-view pressed-input witnesses. Preserve all existing behavioral/ownership checks. Repair only the Notes polling predicate's transient missing-widget gap, retaining old-node detachment, real replacement identity and the existing wait limit.
16. Pay down the measured 95-to-94 first-send closure by deferring Character avatar helper imports until their existing explicit-avatar branches, preserving the backend patch point and validation/authorization order. Verify avatar service behavior and all unchanged warmup guards.
17. Repair the remaining observed UI25/UI26 failures using existing design vocabulary: use an existing high-contrast semantic token for the Plugins focus edge; replace SetupStep's finite inline error-padding states with token-backed classes. Keep all contrast thresholds, layout geometry, style floors and debt records unchanged. Reproduce the exact failures and verify contrast plus representative focused rendering and repeated-error/step-transition geometry. Existing ADR-150/161 apply; routine repairs require no new ADR.
18. Repair the missing successful pressed-fallback acceptance marker under existing ADR-230/231 and TASK-33620.15.4: call the same note_send_accepted owner used by received-intent acceptance before the transcript-poll handoff. Keep captured-draft custody, the skipped duplicate commit, provider-entry marker and three-second expiry unchanged. Reproduce the existing quiet-gap node and run all seven quiet-gap cases plus the durable-commit window contracts. No new ADR or authority boundary.
19. Keep the current MCP reader within the frozen runtime envelope by moving the cohesive recover_console_trace_calls implementation into its existing console_trace_settlement leaf, retaining the original lazy delegate/signature, patch points, call-time imports and timestamp/grace/repository behavior. Run the direct recovery and grace controls; lower only genuinely reduced size rows. Existing lifecycle/import/trace contracts apply; no new ADR.
20. Stabilize the quiet-gap manual-poll precondition after the observed replay-ownership refusal: wait for both the existing sync-in-progress flag and whole-sync replay ownership to settle within the same30s deadline. Keep all exact projection/count/hold assertions and production gates unchanged.
21. Preserve canonical generated LF bytes through native Git checkout as well as publication: add a narrow top-level CSS eol=lf attribute following existing exact-byte artifact patterns, and verify actual checkout filters return exact LF blob bytes. Keep the boot-byte ceiling and source semantics unchanged; recheck final committed artifacts after integration.
22. Correct the two observed stale llama.cpp native-tools assertions in the existing gated context-meter module. Reuse the existing complete-request capture and schema-disclosure helpers to assert empty native rows plus exact fenced light schemas for roomy/reasoning resolutions, and retain the unverified-window minimal no-tool fallback. Verify those two nodes and existing light-native/light-fenced representation controls. Existing ADR-104 amendment governs protocol accounting; no production planner, limits or permissions change and no new ADR.
23. Qualify the public plugin presentation cache independently of the existing Darwin-only managed-runtime fixtures: add one process-free test using the actual app mixin property and PluginOperations facade over preconfigured inert services, requiring one retained owner/service and draft state per app. Keep all real runtime/platform/permission guards unchanged; record13 capability skips rather than claiming mounted-host success. Existing ADR-036/162 apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Recovery tests now use distinct temporary names and private native AppData roots. TEMP/TMP/TMPDIR still reach real selector discovery; generic restart filtering excludes ambient AppData. Production recovery checks and deadlines are unchanged (ADR-126).

The owner-approved runner trial pins nine Derived Artifacts jobs to Ubuntu 24.04, preserving selectors, capacity, permissions and fail-closed aggregation (ADR-103 amendment). All prerequisites executed on head 56a934ad44; this does not establish the cause of the earlier ubuntu-latest queue.

Integrated repairs:

- Keep inactive Settings callbacks from overwriting the selected dirty inspector/banner; restore the successful pressed-send start marker and retain durable-commit polling semantics.
- Match warmup inventory to actual imports and defer optional avatar helpers after validation; retain the backend patch point and 94-module ceiling.
- Use declared service readers, controller-owned session counts and the existing public plugin cache. Move shutdown/worker settlement and trace recovery into their cohesive owners; keep patch points, cancellation, primary errors and grace. Setup facts re-export the runtime-owned route mapping. Existing ADR-036/250/146 amendments document these boundaries.
- Split cohesive Console CSS, publish/check out canonical LF, adopt existing semantic geometry/focus tokens and preserve actual layout. Correct native-path, navigation, custody, modal and fenced-schema test expectations.

Targeted Windows controls pass: first-send/cycles 14 at 94/94; optional-owner 4; supported Notes 15; pure plugin cache/browsing 3; archive 14; shutdown/TTS 15; endpoints/trace 234; quiet-gap/durable-window/Notes-focus 11. Counts/MCP has 83 unaffected passes plus five focused passes after two stale test assertions were repaired. Styling has 104 existing passes plus three final geometry/artifact/byte passes. Focus contrast is 4.69–5.79:1; boot CSS is 595,085/608,090 bytes; ancestor count 270. Formatting is AST-equivalent; lint has 121 baseline findings and zero introduced findings. Independent applied-tree review has no outstanding material findings. No full repository sweep.

Explicit limits: the broad guard run has 199 passes and three unchanged baseline failures: chat-store 22361/22338, command-provider 1133/1123 (neither required selector), and Windows-only windows_files absent from the shared runtime snapshot. Four POSIX Notes setups require the configured Ubuntu lane; thirteen managed-plugin fixtures retain their Darwin guard. No platform capability or permission was faked. Native recovery remains timing-sensitive at the unchanged 30/90-second limits; the timing-only observer and clean hosted Windows control passed, and host pet-state hashes remain unchanged.

Historical head 56a934ad44 recovery passed on Ubuntu 85, macOS 85 and Windows 84 plus one unsupported-FIFO skip: https://github.com/rmusser01/tldw_chatbook/actions/runs/41270189726. The explicit full-agent opt-in passes the real fenced UAT journey (1 passed, 208.48s); all four integrated variants, negative evidence control, committed checkout-byte proof and current-head CI remain required before leaving draft. Live final evidence belongs in PR #3342; no merge or auto-merge is requested.

<!-- SECTION:NOTES:END -->
