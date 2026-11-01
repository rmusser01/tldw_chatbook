# Eval recovery CI repair implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; verify the existing regression before and after the smallest fix.

**Goal:** Repair PR #3287's temporary-directory collision and audit its unfinished checks.
**Architecture:** Preserve the private recovery fixture and every production guard. Give the selector regression a distinct test-owned directory.
**Tech Stack:** Python 3.12, pytest, native Windows fixture runner, GitHub Actions.
**Spec:** backlog/tasks/task-38268 - Fix-eval-recovery-temporary-directory-collision-after-portable-merge.md

ADR required: no
ADR path: N/A; existing ADR-126 applies.
Reason: test-only namespace correction with unchanged recovery policy.

## Constraints and review focus

Keep all TEMP/TMP/TMPDIR cases, actual tempfile discovery, secret filtering, and native fixture ownership. Run the exact nine-file eval-recovery selection; no full-suite sweep. Report pending CI explicitly. Do not merge the follow-up automatically.

## Task 1: Restore the selector regression and audit CI

Files: Tests/Backup_Recovery/test_profile_launch_environment.py; task record; lessons-testing-evidence.md.

- [x] Reproduce the three FileExistsError failures with the autouse fixture active.
- [x] Change only the test-owned directory from private-temp to selected-temp.
- [x] Run the five-case module, then the exact workflow's nine test files with --timeout=180.
- [x] Run scoped Ruff/format and diff checks; review the full change.
- [x] Record results, inspect every remaining PR #3287 check, and publish a follow-up PR.

## Added scope: Native AppData fixture isolation

The full native test discovered that HOME redirection leaves ConfigFileStorage's Windows APPDATA selector pointing at ambient pet state. Isolate APPDATA and LOCALAPPDATA in the existing per-test recovery fixture. Production recovery policy is unchanged; existing ADR-126 applies.

- [x] Add a parameterized regression requiring both native selectors to name pinned private paths under tmp_path while generic restart filtering continues to exclude them; observe it fail first.
- [x] Create two private fixture-owned native application-data roots and redirect the selectors with monkeypatch.
- [x] Verify all seven launch-environment cases, the complete backup/reopen case, and the full CI selection. Keep host profile data outside test archives.

## Verification and remaining gates

- Source e2ec8715e3850551a75361cf68531403bb82bd57: hosted Ubuntu 85 passed; macOS 85 passed; Windows 84 passed, 1 unsupported-FIFO skip. [Matrix run](https://github.com/rmusser01/tldw_chatbook/actions/runs/41257072705).
- Final native local nine-file selection: 81 passed, 4 platform capability skips, 1061.94 seconds; actual host pet-state bytes unchanged.
- Seven focused environment cases, scoped Ruff/format/whitespace and independent review passed. Production recovery checks/timeouts remain unchanged. Full repository suite was not run.
- [Draft follow-up #3342](https://github.com/rmusser01/tldw_chatbook/pull/3342) remains In Progress pending 34 queued checks and a base update when preparing to merge. Original PR #3287 still has 36 queued checks. No merge/auto-merge performed.
- The verification-record commit changes only documentation; all tested source bytes match e2ec8715e3.

## Rebase and ready-for-review continuation

The owner requested rebasing PR #3342 onto dev and resolving outstanding CI before leaving draft. Rebased both commits without conflicts onto dev 77718ebebc; git range-diff reports both original patches unchanged. Focused native environment tests: 7 passed. The earlier three-platform results above remain historical evidence for the pre-rebase source; current-head CI must qualify the integrated tree.

Keep the PR in draft while reviewing the normal pull_request CI runs and available review threads. Retire only superseded runs for this PR if their always-run final gate holds its concurrency slot; never dispatch the required check by hand. No merge or auto-merge is authorized. Live final CI results will be recorded in the PR description so recording them does not create another source head.

## Owner-approved explicit runner trial

ADR required: yes (existing ADR-103 amendment).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: owner approved testing the already-executing Ubuntu 24.04 image for the required workflow's queued Linux jobs. Keep all gate/capacity contracts intact.

- [x] Change the five existing Derived Artifacts runner assertions and observe RED against ubuntu-latest (7 intended failures, 82 passed).
- [x] Pin the workflow's nine runs-on values to ubuntu-24.04 and verify the existing contract suite (150 passed, 186.59s); parsed workflow otherwise identical, independent review clean.
- [ ] Confirm current-head hosted allocation, all required checks and available review feedback before leaving draft.

The runner trial passed the existing workflow-contract suite, 1,242-entry UI census and both Backlog guards. Scoped Ruff lint passes; the five assertion literals introduce no formatting drift relative to the already-unformatted baseline. Dev advanced to 2561efd658 during verification; rebase the qualified patch onto that base before publication and use current-head hosted CI for the integrated result.

## Added scope: modal policy inventory repair

Hosted UI shard 1 failed the complete reverse modal census after rebase: seven classes have no tier anchor or policy classification. Source audit confirms ConsoleCodeReviewModal uses uncapped 95% geometry, four plugin-shell dialogs use uncapped 90% geometry, PluginAuthorityDialog inherits the exempt masked PasswordDialog, and NewCardModal is a title-only prompt with default geometry. Preserve those existing behaviors with individual reasoned registry entries; do not blanket-skip a module or change the census.

ADR required: no new ADR.
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md; backlog/decisions/162-managed-agent-plugins.md. TASK-36382 and TASK-36522 define the inventory policy.
Reason: test-inventory correction documenting current UI geometry; no production behavior change.

- [x] Reproduce the existing reverse census failure on the current source (one failed, five passed; exact seven classes).
- [x] Add only the seven individually justified exemptions to Tests/UI/modal_wide_tier_registry.py (+9 lines).
- [x] Run the existing six static contracts (six passed, 62.66s), scoped Ruff/format and whitespace checks; independent review before pushing.

## Added scope: remaining UI-shard-5 contracts

Hosted shard 5 has three failures: setup-card reverse focus traversal, an unsaved snapshot-preference assertion, and the Console stylesheet's 2,001 active lines exceeding the existing 2,000 ceiling. Investigate each against dev db7e04cf48 before changing source; the old hosted snapshot predates the latest setup/Console merges.

ADR required: no new ADR for routine test corrections or a stylesheet refactor preserving behavior.
ADR path: existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md; use the governing feature ADR if a product fix proves necessary.
Reason: retain the current keyboard, preference review and component vocabulary contracts.

- [ ] Reproduce each exact failing case on the rebased tree and identify stale test assumptions versus product faults.
- [ ] Repair the smallest proven cause without weakening focus, dirty-state, size, timeout or census checks.
- [ ] Split one cohesive Console CSS vocabulary into its proper source module; rebuild generated CSS and prove selector/declaration equivalence.
- [ ] Verify the affected cases, relevant style guards, scoped lint and independent review before the next push and fresh current-head CI.

## Proven status ownership and later CI failures

The metadata trace proves the restored Providers & Models draft is dirty and its inspector initially says Unsaved changes. An asynchronous Internal Prompts count callback then updates that shared inspector/banner for the inactive category. Enforce the accepted ADR-033 contract by putting those two selected-surface writes inside the existing active-category guard. Preserve category rail and category-owned descendant updates. Replay the real callback in the existing retained-edit test and assert the dirty model, edited value, inspector, selected banner and rail remain consistent; a wait cannot substitute for the repair.

ADR required: no new ADR; direct repair of accepted behavior.
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md.

Hosted UI shard16 also found warmup inventory drift: console_received_dispatch is now resident at UI-ready, while console_plugin_context and Agents.hook_permissions remain lazy send imports. Update the production/test inventories only, preserving route ordering, guards and the94-module ceiling. Governing ADRs: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/198-gc-policy-freeze-boot-heap.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md. No new ADR.

The second admission lane's project-chat UAT expects deferred full-tool discovery and project-binding disclosure, but its preconfigured tools section omits the new opt-in full-agent mode. Reproduce its fenced case, explicitly enable the existing mode in this test's saved config, describe that substitution, and qualify all four variants plus receipt and negative-evidence checks. Preserve product defaults, permissions and every model/file-evidence assertion. Existing backlog/decisions/104-token-budgeted-agent-tool-disclosure.md and its light-surface amendment govern the mode; no new ADR.

The CSS split's native guard run found two existing portability assumptions: the atomic text writer translates canonical LF content to CRLF on Windows, taking the physical boot census over608,090bytes, and a synthetic diagnostic assertion expects a literal POSIX separator. Publish generated UTF-8 text with explicit LF through the existing atomic writer, and assert the escaped native relative path in that existing test. Rebuild and rerun both observed failures plus reproducibility checks; retain the exact45,315-token stream and all numeric ceilings. These are routine representation/test-path corrections under the existing artifact and ADR-150/161 contracts; no new ADR.

Hosted UI13 found one raw dimension: #setup-provider-choice's40vh cap. Add$ds-setup-provider-choice-max-height:40vh beside existing setup geometry and reference it from that selector; an unrelated Messages token must not couple these features. Preserve resolved max_height at multiple viewport sizes, run the existing zero-literal/definition/bundle/budget guards, and retain the earlier exact token-stream proof for the separate Console extraction. Existing ADR-150/161 govern this token adoption; no existing token value changes.

Hosted UI12 found two added visible destinations absent from an otherwise exact expected class map: kanban/KanbanBoardScreen and plugins/PluginsScreen. Reproduce and add only those two entries, retaining the complete equality and real lazy route resolution. UI15 repeats the already repaired keyboard action-order failure.

The native dimension-floor run also exposed a pre-existing path-key assumption: core\\_variables.tcss did not match the canonical core/_variables.tcss exemption. Canonicalize _sheet_sources keys with relative_to(CSS).as_posix(); preserve the exact token-file exemption and every zero floor, then rerun only the failed dimension case because the other three guards already pass.

## Primary architecture and remaining contract repairs

ADR required: yes, clarify existing ownership decisions before implementation.
ADR paths: backlog/decisions/036-application-service-composition-lifecycle.md; backlog/decisions/250-application-services-bundle.md; backlog/decisions/146-console-custom-endpoint-registry.md.
Reason: preserve established service and runtime boundaries while repairing new reflection, upward imports and import-footprint growth. No new permission or resource owner is introduced.

- Replace the identified new app-reflection sites with declared/current service references or explicit borrowed owners/readers. Notes must retain a cached non-constructing peek plus off-loop safe read; MCP switch checks stay fresh and fail closed. Plugin caching stays in the existing public app owner; selected-tool counts get an explicit publication owner. Preserve required fake-host contracts, lower any genuinely reduced census rows, and remove stale private-name entries only when their last real lookup disappears.
- Move the complete220-line unmount service phase and its cohesive47-line worker-settlement body into app_shutdown_services.py, classified by the existing app_* presentation rule. Retarget the sole lifecycle caller and source-inspection test, keep public mixin delegates/patch points, grace constants and cancellation/primary-error semantics. Lower measured source-size rows; no shim or allowlist addition. Verify the actual size guard: app_lifecycle currently2150 against2135, Utils app_shutdown776.
- Put the shared suffix mapping in provider_endpoint_contract and have setup facts re-export it. Verify route behavior, single ownership and unchanged runtime footprint.
- Resolve the first-send fixture's intermittent fromlist anti-vacuity observation by invoking the real safe voice-chip restore from the test gateway under an asserted real send frame, before delegating. Do not change frame recognition, start audio, relax assertions or raise the94-module ceiling. Find a legitimate eager dependency paydown for the remaining95→94 gap before claiming GREEN.
- Re-key .setup-key-hint Static to the sole existing #setup-provider-key-hint-text ID, keeping width:$ds-width-fill and the270 performance ceiling; prove matched-target/computed-width equivalence.
- Update only the stale no-overhead context fake and explicit custody request field census for current immutable input witnesses. The witness fields are repr-hidden and generation-fenced; controller/runtime lifetime checks remain exact.
- Guard the Notes replacement query with a presence check during recompose; preserve exact detached-old/new-focus conditions and the10-second limit.

First-send residual proof: the two necessary lazy entries leave95 cold modules against94. The legitimate paydown is the optional Character avatar edge: character_tool_service imports character_avatar solely for an explicit generate-avatar summary and the validated avatar-save branch. Keep the service warm; defer only those helper imports until those existing branches. Preserve the service-level image_backend_configured patch point with a typed lazy forwarding function and keep avatar resolution after validation. Run the full character-tool service tests with strict first-send/cycle/payload guards. This follows the existing ADR-097 lazy-feature boundary; no budget or runtime-authority change.

## Final observed UI contracts

ADR required: no new ADR; existing ADR-150/161 and ADR-230/231 govern these direct repairs.

- UI25: reproduce all Plugins focus-edge contrast failures, replace only that component edge's token with an existing contrast-safe semantic foreground token, and verify every theme plus representative focused geometry. Preserve thresholds, debt records and layout.
- UI26: replace SetupStep.show_step_error's three finite inline padding states with token-backed classes; verify repeated errors and step transitions preserve 2/4/6-column alignment and the style floor stays unchanged.
- UI31: successful pressed-input fallback acceptance skips the old captured-draft commit callback, so it also lost note_send_accepted. Restore that marker in the successful custody branch before polling. Keep custody, the received-intent path, provider-entry marker and three-second expiry. Verify the existing failing node, all seven quiet-gap cases and durable-commit window contracts.

Stale-contract GREEN: nine targeted arrival, custody and actual lazy-navigation tests passed in26.84s. Complete equality and lifetime/privacy assertions remain intact.

Architecture source-size detail before apply: the current MCP reader adds three lines to the frozen Console runtime. Move its existing37-line recover_console_trace_calls implementation into the already-owning Chat/console_trace_settlement leaf, retaining the exact lazy delegate/signature and call-time imports in console_runtime. Preserve recovery timestamp, grace interval, repository behavior and patch points; run direct recovery/grace controls. This is an existing-boundary extraction, not comment/whitespace compression or a budget increase. Existing ADR-036/097 and trace-settlement contract apply; no new ADR.

Quiet-gap covering test investigation before the additional test edit: the paused timer's task is sleeping, but its whole-sync replay still owns presentation. Minimal callback snapshots reproduced the unchanged exact first-poll count failure with sync_in_progress=False, replay_whole_sync=True and sync_scheduled=True at callback entry and return; the production gate correctly leaves a request and returns without projection. Extend only this test's existing settled predicate to wait for replay_whole_sync to clear as well, keeping30s and every exact first-sync/held-tick/count assertion. The separately selected Notes creation control requires an explicit collection-time bootstrap-profile probe because its unmarked mixed-profile fixture correctly refuses raw_source_selection_changed before UI; with that disclosed harness substitution, the real New/Blank-note assertions passed1/24.88s. No production gate/authority changes.

UI25 compositor check confirms the inherited solid outline masks a border-only token repair: perimeter ratios stay2.563/1.196/1.591 on textual-dark/pastel_dreams/cloudy_day at80x24 and140x40. Set this component's focused outline to the same existing ds-active-fg token, preserving solid thickness, geometry and contrast thresholds. This is the already-approved visible-focus repair, not a global focus-token change.

Independent architecture review found one compatibility gap before tracked apply: ProviderStep still accepts wizard/config=None, so its direct app-config read must retain an explicit absent-owner branch. Add that conditional and focused constructor-based optional-owner coverage in the existing provider guard tests. Keep declared-owner reads direct and all frozen counts unchanged. Other reviewed ownership/extraction semantics had no material finding; actual applied-tree tests remain pending.

Canonical LF checkout boundary before edit: Git reports no text/eol attribute for the generated root CSS. Its real checkout filter emits20,255CRLF lines/443,333bytes for the HEAD main bundle, versus the current canonical generated423,398bytes; main-bundle smudging alone exceeds the unchanged13KB remaining boot headroom. After the active CSS run exits, add a narrow tldw_chatbook/css/*.tcss text eol=lf attribute, following existing exact-byte artifact attributes. Verify Git checkout filters emit exact LF blobs, then validate the committed integrated artifacts through the same filters. Generated publication and fresh Windows checkout must both preserve canonical bytes. Existing ADR-150/artifact reproducibility applies; no new ADR or runtime behavior.

Counts/MCP actual run:83passed/2failed838.72s; all new ownership/current-reader controls passed. The two failures are existing gated llama.cpp tests that inspect only native tools rows, while current preview deliberately renders fenced schemas into the system message for that native-incapable resolution. Existing light-fenced prepared-vs-service-request control passes. Correct the two tests using existing _capacity_request and actual disclosed_schemas helpers: require no native rows, exact LIGHT fenced names for roomy/reasoning requests, and retain the unverified-window minimal plain fallback. Keep native OpenAI control and both complete request-representation controls. No production planner/accounting/limit change; existing ADR-104 applies. Native baseline/new fixture source control required before marking this green.

Plugin qualification boundary: all13 selected real managed-runtime fixtures retain their existing64-bit Darwin guard, so the Windows run is13skipped, not mounted-host success. Add one process-free cache-ownership regression to the already-selected test_plugins_browse.py, using the real app mixin property and actual PluginOperations facade over an inert service. Verify repeated public reads retain one presentation owner and existing draft state over the app-owned service, while separate app owners remain isolated; no managed runtime, capability flag or permission is substituted. This tests the changed cache/borrow seam without claiming portable managed execution. Existing ADR-036/162 apply; no new ADR or production behavior.

Pre-integration opt-in proof: the real fenced project-chat UAT now passes (208.48s), including workspace creation, instruction review, actual file evidence, persisted/painted answer and clean shutdown. All four variants and the missing-README negative control still qualify the final integrated head. Backlog readability/ID guards pass across6,016 tasks; final publication must also retain the complete UI census and current-head hosted checks.

Final integration base f34510953e: rebase completed with one size-table conflict, preserving upstream console_display_state2908 and lowering actual combined lifecycle2113 from2147. Range-diff keeps the other five commits identical and changes only that numeric resolution/context in the repair patch. Independent integration review confirms upstream quit-loss and Console changes intact. Builder regeneration reproduces the merged artifacts. Workflow parsed equality proves only nine runner labels differ; UI census1253 and6032 Backlog files pass.

Actual integrated targeted group:60passed/4failed345.90s. One failure is the already-proven unchanged Windows footprint module. New dev growth puts _console_panels at2035active lines, adds four broad Button rules (274>270), and exposes another Windows-only synthetic diagnostic assertion using POSIX separators. Before-code repair: move the exact cohesive attachment-block source into an adjacent module with both source inventories updated; retain exact flattened token stream for the extraction. Re-key its three state rules to existing exact Button IDs at equal specificity, and give every real Terminal control a semantic component class for its padding rule. Preserve actual matched targets and computed geometry at representative sizes. Correct only the synthetic diagnostic expected native relative path; keep return-code and no-baseline-write controls. Existing ADR-097/150/161 apply; no new ADR or budget change. Recheck the three observed failures plus artifact/byte/style closure, without repeating the entire345-second already-passing group.

Attachment split refinement before apply: the cohesive38-active-line block sits between a1318-line prefix and679-line tail. A global token-sequence requirement would force two extra fragments unrelated to the actual vocabulary boundary. Use one attachment module after panels instead: preserve the block's exact declarations/token sequence and all other rules' relative order, prove that crossed tail selectors do not match its actual controls, and compare resolved styles/geometry (including relevant focus states) at representative sizes. Re-keying is a separate intentional selector change. This is a semantic cascade-preserving refactor under the already-authorized repair; no added tail module, value change or user approval gate is needed.

Integrated CSS closure:12targeted checks passed132.56s, final sync passed13.54s, and the private/native-owner actual-widget proof passed29.84s. All13Terminal controls retain styles/regions across27projection/focus cases; crossed tail selectors match no attachment nodes, and equal attachment specificity is retained. Normalizing only the isolated move/four re-keys restores45,467baseline tokens. Panels1997/newattachment41active lines; physicalboot596,521/608,090bytes; ancestorlimit270unchanged. Scopedlint/format/whitespace pass. One discarded preliminary proof imported ambient config before isolation; final proof imports only inside the selected private child, starts no TldwCli/services/DB, and host-pet SHA remains unchanged. No budget/allowlist changes.

Final publication checks on the integrated source: first-send/cycle/payload14passed46.16s at94/94; native launch environment7passed1.19s. Actual staged Git blobs, checkout filters and working bytes agree as LF for all16rootCSS paths; the legacy unused main.tcss needed only worktree EOL normalization with no Git content delta. The CSS repair has independent review with no material findings. Final UAT variants/negative control and normal pull_request CI still qualify the published head before ready-for-review.


Current-head hosted follow-up before implementation: UI shard6 has three receipt-open failures because its minimal running ingest host lacks the declared peek_notes_sync_runtime interface. Update stand-ins explicitly with their existing absent-runtime semantics; preserve production direct-owner reads and audit related callers. Wrong-family UAT reached one asynchronously mounting approval row before its header descendants existed; wait for that structural precondition within the existing deadline, then retain exact header/one-row/request/approval assertions. Perf attempt2 passed unchanged latency but exposed UI-ready1032>1031 and two undeclared Console boot workers. Trace their requirements under ADR-097 and related feature tasks, reproduce focused controls, and defer/shed only proven unnecessary boot work without raising budgets, hiding workers or changing authority. ADR required: no new ADR; routine fixture fixes and existing ADR-097/lifecycle policy apply. All final UAT and current-head CI still gate leaving draft.

Attachment-readiness follow-up before code: UI shard3 reports Workbench/composer disabled=(True,False) after vision becomes blocked. Exact native RED also fails this test, sometimes earlier with the staged-image update still unpainted. Capture only failure-time checked-sync return, replay ownership and cached composer state in the existing node before selecting a fix. Preserve all original state expectations and production gates. Boot census policy clarification: finite managed attach reconciliation is mandatory for answerability under TASK-35593/35614; classify that one exact worker with its feature rationale under the census explicit reviewed-membership policy, retaining console-sync and all stagger limits. Avoid the genuinely empty recovered-image worker, and defer only the sole-use compaction-hook import off UI-ready. ADR required: no new ADR or numeric-budget exception.

Actual CI checkout4150e8b22c combined f477 with newer dev621b8f1550, exposing incoming provider JSON fallback. Rebased once onto pinned current devcf5fda70d3fe75e442078b37aa8234a83b7aaf9b; all7 own patches remain range-diff identical (HEAD51eb666733), no conflicts, owned WIP restored from preserved stash9d81cc3. Collect current-source RED before supplying concrete response Content-Type fixtures. Wizard validation remains unproven until original snapshot/builder/binder exception location is observed. Close failure has three stale stored-title assertions; require exact positional Tab 2 plus nondisclosure while preserving all recovery checks. Attachment failure metadata proved checked sync returnedFalse with a scheduled replay; use the existing until helper default15s to settle in-progress/replay/scheduled owners before exact state assertions, without changing production code or expected payload states.

Wizard observer actual RED: all3 original bindings restored; ValueError is in ProviderSetupDraft.__post_init__303 via shared_draft500. Resolver687 intentionally returns (None,None,env_var) for an unset declared/default variable. The builder maps missing key_source to none but forwards that metadata, contradicting the strict source contract. Forward resolved_env_var only for actual environment source; leave validator and resolution precedence unchanged. Existing unset/stored/inline/environment/rejection tests plus actual listed-model integration qualify the correction.

Post-rebase verification: attachment4 pass49.20s; actual dispatcher12 pass4.53s; wizard credential contracts17 pass2.05s and listed-model integration1 pass6.85s. Image6 supported controls pass while10 untouched unmounted mocks returnNone instead of a Worker; exact nonempty source is unchanged. Eleven actual-command compaction controls reach the existing win32 refusal (ADR-163); add a real no-command lifecycle control without capability substitution. Native close grouping still refuses earlier at1179, before changed privacy assertions; hosted privacy RED remains the direct evidence. Boot group26 pass/4capability skips/2native limits: UI-ready1032 includes Windows-only windows_files while compaction_hooks is successfully absent; first-use subprocess refuses recovery_scope_uncertain. Boot-worker gate passes; first-send13 pass75.52s at unchanged94 and reconciliation17 pass54.77s. Qualify Linux1031 and full close workflow on hosted CI; no limit or recovery exception. UI26 native RED matches hosted /rules timeout exactly; bounded original-code observation precedes repair. Format compaction with exact-None assignment expression, same await/finally and4217 lines.

UI26 concrete source correction before code: full4-case module reproduces the original /rules timeout in history-size0 (3otherpass), proving it is not size1-specific. Exact WorkerFailed traceback is session.py5684 reading the legitimately closed OLD visible chat from a matching switch snapshot, before the existing protected old-only draft write. Runtime.dispose does not explain that deletion. Expand only the existing old-session absence boundary; preserve active/current reads and live-old revision/conflict/suffix behavior, and avoid restoring the closed undo bank. Current module6981/6982 allows a cohesive one-line expansion without pin change. Require deterministic real-state RED and closed/live old GREEN plus original4-case flow; no new wait/profiler/deadline or authority exception. Fixture disposal order is separate ownership hygiene, not yet a causal fix.

Closed-old regression qualification:2actual closed cases fail exact original5684 lookup before fix; live revision passes. Corrected live undo setup uses ordinary per-character coalesced typing (two multi-character inserts legitimately create two undo entries), then unchanged-product baseline1passes15.93s. Applied old-only boundary fix: all4new controls +3saved-projection controls +6982pin pass;3older unmarked cases refuse config profile selection before app construction, not a product failure. Original rules4-case flow now3pass/1earlier native-preparation close refusal, with no old UnknownSession or rules timeout in reached cases. Final firstsend14passes79.51s at94/94. Native close contract explicitly requires physical drain: stock context_repository.read_context_presentation_inputs registers controller-only, while runtime close drains its own set and finalize checks controller set; its currentness also lacks close fence. Add a deterministic held original stock-read control before implementing existing observer/fence pattern, preserving all real ownership/grace/guard behavior; no caller settle wait or broad exception suppression. Specific saved E2E lease not yet identified, so do not claim attribution before validation.

Context-read qualification: new held-read/close/fence tests are RED (3 failures); the actual runtime observer set is empty and one close reaches the existing native-preparation refusal. Independent review identified a missing captured-runtime currentness check in the proposed fix. Verify nullable runtime identity plus reverse controller/store ownership consistently with configuration/received readers, and require stale refusal after a genuine held first probe during attach/rebind. The product patch remains unapplied pending that test. No physical-retirement, close-grace, finalization or custom/direct contract changes.

Context/current/close qualification:33pass101.16s, including all6new and4native-close controls. Three existing context-presentation fixture cases also fail with only the product patch absent (24.13s): custom-reader fallback expectation and two raw-memory lineage expectations. Those are unchanged baseline limits, not green or fixed here. Original rules4 + full close/race/privacy journey now running uninstrumented; final current-head CI and UAT still required.

Final uninstrumented rules history/ancestry at both sizes:4pass; complete close/race/privacy group still fails earlier at unchanged enrichment-entry precondition1179, before changed privacy assertions. Combined selection220.55s. No original UnknownSession, /rules timeout or native-preparation close refusal in the four rules cases. Preserve the native precondition as a limit and require actual Ubuntu complete-journey CI; no capability or deadline substitution. Final applied-tree lint78 inherited/0introduced across16Python files, no new whole-file format debt, diff whitespace clean; Session6982 and compaction4217 pins unchanged.
