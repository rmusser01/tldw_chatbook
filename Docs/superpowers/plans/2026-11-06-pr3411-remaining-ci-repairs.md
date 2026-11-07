# PR3411 remaining CI repairs

> For agentic workers: use `superpowers:executing-plans` after root explicitly hands off each serial task. This draft does not dispatch workers.

ADR required: no new ADR for the selected routine fixes.
ADR path: existing backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md, 135-fleet-completion-delivery-and-crash-recovery.md, 227-native-console-goal-runs.md, 285-native-phased-agent-workflows.md and 286-console-prompt-mix-ins.md.
Reason: restore the existing contracts; reassess before a diagnosis changes a boundary.

Prerequisite plan gate: SPEC PASS / QUALITY APPROVE in feedback-task14-plan-rereview1.md. Implementation follows the preceding Task9–13 gates.

**Goal:** Close the remaining exact hosted failures and separate proven Goal baseline defect without weakening their original behavioral or custody assertions.

**Architecture:** Use existing native preparation, submission, receipt and lifecycle owners. Correct proven fixture mismatches in their shared fixtures; diagnose unresolved original-owner failures before choosing any product change. The labels below partition evidence and exact node ownership; root chooses cohesive dispatch/review batches around the actual shared helper or core owner, without requiring one implementation/gate per symptom.

**Tech stack:** Existing Python3.12 private-profile Source runner, pytest, original Textual/SQLite/MCP owners; no new dependency.

**Spec/evidence:** `feedback-current-ci-triage.{md,json}`, `feedback-current-ci-supplement.{md,json}`, `feedback-frontier-late-f9.json`, `feedback-tested-merge-parity.json`, `feedback-ui-baseline-probe.md`, `feedback-consent-baseline-probe.md`, `feedback-note-baseline-probe.md`, and approved `feedback-goal-baseline-design.md` / `feedback-goal-design-review.md` / `feedback-goal-brief.md`. Paths without a prefix below are relative to Source; evidence files are in PREP. Root owns checkpoints, task hygiene and publication.

## Global constraints and inventory

- ROOT is protected `dev` plus its existing index/294-file work. Source is the ignored integration export, **not a Git worktree**. Execute this reviewed plan only after the preceding serial gates. Root owns Git, Backlog and publication; no dependency change or broad test sweep is selected.
- Hosted bytes were synthetic merge `f5471860` of published head `f9b59ea342132bd111ab69ca91550fca2a12b49f` into `b2c618185f58ace553d4a6f15dfc636310f458ff`. Their tree equals bare f9 (`fa9a038243886d5491dd5fdbcff45513dea6ce96`, zero path differences). This proves code parity, not Linux/macOS or process/profile equivalence. Current Source includes later repairs and must get a frozen root-owned epoch before execution.
- Ledger below assigns **85 hosted nodes exactly once**: Task8b=1, Task9=5, Task13=16, Task14a–p=63. Task14q is the separately reproduced Goal baseline (#86 including it, not hosted). Private-child duplicate reports and the expected fenced-UAT negative are excluded. The late frontier records22 failed/28 successful CheckRuns,3 skipped,1 cancelled,1 neutral and1 queued (plus successful CodeRabbit StatusContext); it is a snapshot, not a fresh current-head CI result. Its raw count does not add any untriaged exact-node claims to this ledger.
- Task8b is locally CLOSED C0. Do not reopen Task8b/cold, Task9/schema, Task13/architecture ownership here. Reconcile their gated integrated result against their exact rows. Coordinate hook overlap with Task12; first-send/config changes with Task11. A task passing its rows cannot certify another group's downstream oracle.
- No blanket fixture weakening, synthetic ready roots/acceptance, original guard bypass, persisted historical-USER workaround, deletion/xfail of failing tests, cap/pin increase, timeout inflation, hidden module manipulation or schema waiver. Keep existing native/foreign/stale-source and real physical-return controls.
- Before each group: root sole-writer handoff; read the current receiving owner and grep all changed helpers' callers; record source hashes and original RED/adverse evidence. Freeze Source writes through its focused run. Use ROOT/`.venv/bin/python3.12`, existing private-profile Source runner and PREP/`run-bounded-source-check.py`; unique group receipts, isolated config/HOME/keyring/offline providers. No full suite without the user's explicit opt-in.
- For every group: exact ledger nodes are the required failing selectors. Add only the named controls below. Run focused lint/format on changed paths; preserve failed runs, process exit and physical retirement separately. End each coherent helper/core-owner repair batch with independent spec/quality review and root checkpoint; no next writer while Important/Critical findings remain. Related diagnostic labels may share one batch only when original-path evidence establishes their shared repair, with every assigned node/oracle still checked. A diagnostic may close with an identified predicate and reviewed minimal correction plan, but its hosted blocker stays open until original behavior and clean completion are proved.
- All fixture-only/diagnostic groups below have **ADR required: no; ADR path: N/A** because no boundary change is selected. Existing ADR285 original-source/finite-ownership constraints apply; add/amend and independently review an ADR **before** a diagnosis leads to a storage, authority, runtime, process or cross-module contract change. UI changes, if any are actually needed, first read `backlog/docs/design-language.md` and `component-patterns.md`; no UI redesign is included.

## Task14a — compiled tab publication fixture (21 hosted nodes; confirmed missing query)

**Reach:** Modify only `Tests/Performance/test_console_tab_sync_source_ownership.py`'s generated `CompiledSourceScreen`/`make_screen` fixture to represent the installed workflow-line query dependency. Read original `_sync_console_native_session_tabs` and its workflow query in `UI/Screens/chat_screen.py`, and `UI/Console_Modules/config_sync.py` replay. Their production callers/exports/config remain unchanged. The generated class is intentionally not installed ChatScreen and must retain its cold-import control.

- [x] Confirm the current extracted original method still requires `query`; add the smallest truthful empty/current workflow-row fixture, rather than suppressing the query or production path.
- [x] Run the whole existing `test_console_tab_publication_source_ownership` parameterization: ledger's21 failures **plus3 existing passing controls** (24 cases; published f9, the preflight reference and current `_CONTROLS` match; no xfail), preserving active/session/store/membership/surface replacements, cancellation/replay/drain oracles and import absence.
- [x] Gate on real original method completion and unchanged publication ownership assertions; downstream timeouts are not separately proved regressions before this correction.

ADR required: no. ADR path: N/A. Existing ADR285 source provenance and original UI publication contract apply.

## Task14b — managed-native attention/picker admission qualification (6; confirmed legacy path)

**Reach:** `Tests/UI/test_console_visible_turn_attention.py` `_run_visible_turn` **and all three direct background admissions**, its `_console_app`/`_reach_console` navigation fixture; `Tests/ProductionApp/test_provider_selection_ownership.py` real model-picker test's post-Apply configuration observation. Read shared `Tests/UI/test_console_native_chat_flow.py` received/native fixture, `UI/Console_Modules/wiring.py::_admit_console_turn_to_runtime`, `session.py::_build_console_turn_execution_context`, and `Chat/turn_admission.py::capture_authority/authority_for`. Search every caller before adapting a test helper. No production legacy fallback, API/export/config change.

- [x] Keep real picker/search/Apply and actual visible/modal/background/off-screen journeys; route the test operation/config observation through the existing received/native owner used by the managed Console.
- [x] Run all6 ledger nodes, plus `Tests/Performance/test_console_managed_received_acceptance.py::test_received_managed_demand_uses_one_catalog_and_real_cold_initializer`, `Tests/Chat/test_console_received_admission.py::test_foreign_store_or_session_cannot_consume_claim` and `::test_promotion_refuses_changed_session_witness`.
- [x] Review that all four attention admission sites were reached and notification-once/none oracles execute, rather than merely eliminating the setup exception. Retain system-prompt/provider-selection assertions and final original cleanup.

ADR required: no. ADR path: N/A. Existing managed native admission is restored; no alternative authority is introduced.

## Task14c — compaction fixture's real skill-trust publication (2; confirmed unready original source)

**Reach:** Two ledger tests in `Tests/UI/test_console_compaction_forecast.py`; their existing `_ask_console`/`_held_console` in `Tests/UI/test_console_compaction_hold_flow.py`, standard source-enabled mount in `Tests/UI/test_console_native_chat_flow.py`. Read `Chat/console_configuration_preparation.py::_ready_attribute/_source_references` and `remember_pending_mcp_capture`; original publication owners are `app_skill_wiring.py` and `app_service_wiring.py`. Check both callers of any shared setup change. No config/exports change.

- [x] Await original `ensure_local_skill_trust_service` publication **before** the measured snapshot/forecast window; assert the published stock backing identity without nulling its active factory or replacing snapshot capture.
- [x] Run both ledger nodes and `Tests/Performance/test_console_skill_trust_stock_owned.py::test_stock_console_skill_setup_has_original_owned_lifetime`, `Tests/Performance/test_original_skill_trust_builder_cancellation.py::test_original_skill_trust_builder_retains_native_callback`, and `Tests/Chat/test_console_configuration_preparation.py::test_prepared_skills_without_selected_source_refuses_before_capture`.
- [x] Require actual overhead/cap forecast and bound-skill ask/regenerate commit assertions; budget and original-source refusal stay intact. Reconcile Task13 marker changes first.

ADR required: no. ADR path: N/A. Existing lazy-publication contract only.

## Task14d — HeldMcpRead observes original native finite permission read (11; local C0 qualified)

**Reach:** Shared `HeldMcpRead` in `Tests/UI/test_console_send_admission_off_pump.py:56–82`; all its callers in `test_console_send_resend_guard.py` (the11 ledger failures) and any direct helper users found by caller search. Read `Tests/UI/test_console_send_acknowledgement.py::build/ready_console`, `Chat/console_configuration_capture.py::capture_mcp_definition_maximum/_capture_mcp_definition_selection`, and `MCP/console_snapshot.py::_original_method/standard_console_sources/read_sources`. No production callable, schema/config/export change selected.

Confirmed: fixture replaces `service.get_kill_switch`; original callable identity rejects it. Its immediate-caller-name predicate is independently obsolete: delegation moved the public kill-switch call, while the native path reads `captured.read_permission_payload`. Trust was actually ready in the baseline representative. Do not recategorize this as an unknown readiness failure.

- [x] Replace only the shared test hold mechanism with a scoped in-place observer of the **original native finite permission-payload read and exact stack**, leaving service/store bindings and executable identities intact; restore observer in finally and retain original finite10s release/30s entry budget.
- [x] First run `test_a_draft_sent_after_a_non_append_edit_says_it_was_sent` through actual entered hold, original finally release, physical worker return, test teardown and process exit. Existing baseline probes were SIGKILL adverse receipts; they proved rejection, not this success/retirement.
- [x] Then run all11 ledger nodes plus `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused`, `::test_standard_catalog_refuses_changed_class_descriptor_before_getter`, `::test_private_bundle_reader_preserves_original_native_load`, and every changed helper caller. Require physical key delivery, draft-clear/tab/hidden-edit/no-resend/status-line assertions. A native read hold reached in one node does not certify the other10.

ADR required: no. ADR path: N/A. Preserve ADR285 native source identity and finite ownership; no production guard needs relaxing on current evidence.

Final Task14d scope and gate (2026-11-07): four existing test files; the final
same-epoch run passes all31 affected selectors with normal exit0 in175.545s,
zero failures/errors/skips/warnings and unchanged production. Source core
df4d122d0936d527cbfc578f887ccdcdd89a93e3ed0c404d5e7370a59e16abad. Native permission and post-hook profile
body/callback/Future/read retirement, actual pending/OPAQUE custom callback,
completed nonstandard getter refusal and real pre-entry caller cancellation are
qualified. R52 preserves the inner empty selection while requiring the outer
owner-change refusal; its inherited changed-callback counter is an independent
witness for two of five mutations only. The supported custom-manifest positive
explicitly replaces the unsupported getter positive, with a separate real getter
refusal control. The busy-name case explicitly uses the legacy API; it does not
claim native voice/B-Enter legacy-gate coverage. Separate native integration
controls pass in the final run. Cooperative exclusive observer assumptions and
permanently stuck native ownership remain bounded limitations. All earlier
adverse and separate-epoch records remain. Whole-D SPEC PASS / QUALITY APPROVE /
C0: feedback-task14d-final-review.md; final18-entry evidence inventory
5d929ed42164d89c7e1e15b9474911f6de040d607fcb1ab5dc57bb81f04cbbf3.
Root full-tree/mode/preservation gate precedes checkpoint and Task14e.

## Task14e — real consent readiness (6; diagnosis required on failing environment)

**Reach:** `_console` in `Tests/UI/test_console_session_admitted_roots.py`, reused by `test_console_environment_refresh_off_loop.py`; original real setup in `test_console_environment_bound_registry.py` and `test_console_environment_consent_revoked_hop.py`. Read `Workspaces/change_review_consent.py::_schedule/_run_work/_complete/admit_turn/capture_turn/wait_closed`, `change_review_ownership.py::_InitializationSource`, `change_turn_tracker.initialize_shadow_root`, and snapshot-ticket return. Registry/binding config remains real; no fixture/export change is selected.

The original one-root reader representative **passed** bare-f9/macOS through genuine initialization, READY, queue.join, all three readers, consent worker closure and process exit0. Hosted Linux failure is not reproduced; tree parity does not identify its initializer/refusal reason. Current setup already uses real admit_turn+queue.join; the old SimpleNamespace diagnosis is stale.

- [ ] No failing Linux/local environment is assumed available. Use the intermediate hosted diagnostic checkpoint below when local evidence cannot identify the failure: on the exact targeted hosted rerun, observe body-free original initializer outcome/error, source identity, native operation/ticket owner, PREPARING/FAILED/READY publication and queue.join continuation. Start with `test_the_three_readers_answer_from_one_helper` and independently `test_a_bound_workspace_lands_through_the_real_consent_service`; keep unresolved selector ACs open.
- [ ] Only after an actual failed predicate is observed select the smallest owner/fixture correction; obtain a design/ADR ruling if it changes custody/readiness. Do not synthesize roots or add a readiness sleep to make the local passing case greener.
- [x] Run all6 ledger nodes; preserve root equality, off-loop/no-full-config-reader assertions, bound hop/revocation/no-gh checks and worker closure. Retain `Tests/UI/test_console_session_admitted_roots.py::test_narrow_admission_normalizes_paths_without_granting_metadata_authority`; `Tests/Workspaces/test_change_review_consent.py::test_disabled_admission_never_schedules_initializer`, `::test_external_revision_change_rejects_old_initializer` and `::test_shutdown_is_bounded_and_late_completion_reads_no_registry`. If a root reader itself changes, also enumerate its existing empty/damaged/excluded-root negatives before implementation.

ADR required: no for diagnostics. ADR path: N/A; boundary changes require their own prior ruling under ADR285.

Local Task14e qualification (2026-11-07): all10 exact original selectors pass
unchanged on macOS; the final single run returns0 in61.132s with nine matching
private-child reports and unchanged df4d122d/prod8ed maps. The original wrapper
does not forward outer CLI warning flags. A separately validated inherited
PYTHONWARNINGS builtin/module filter makes unawaited-coroutine/unraisable warnings
fail at the installed pytest emitter; final child logs have no invalid-option or
warning markers. Initial passes and the failed external-category startup preflight
are retained. Local SPEC PASS / QUALITY APPROVE / C0 does not identify the hosted
Linux cause; those diagnosis/public ACs remain open for current-head CI. No Source
correction, synthetic readiness, cap increase, Docker install or Linux equivalence
is claimed. Root takes the already authorized intermediate integration/publication
route next, with fresh dev, actual union review and affected verification.

## Task14f — Temporary recovery shelf capture (4; diagnosis required)

**Reach:** `Tests/UI/test_console_quit_loss_guard.py::_app_with_quit_recorder/_mounted_console/_temporary_chat_with_parked_turn`, `_build_test_app`, `_mounted`, `_configure_native_ready_console`; original controller send result and `Chat/console_runtime.py::_record_turn_recovery/recoveries_for_session/_turn_recovery_offered`, plus close-impact/quit readers in `UI/Screens/chat_screen.py` and `app.py`. Check the exact live recording branch and all helper callers before edit. No new config/export/Temporary semantics selected.

- [ ] Run `test_ctrl_q_counts_a_temporary_chat_with_a_parked_unsent_turn` with original accepted/refused stage and recovery capture reason/owner observation; determine why enabled Temporary send produces no record before touching privacy/quit logic.
- [ ] If qualification is wrong, correct its existing shared native setup; if real capture failed, propose the smallest diagnosed producer repair. Preserve durable/original record ownership and finite cleanup.
- [ ] Run all4 ledger nodes plus `test_quit_totals_equal_the_sum_of_every_tabs_close_impact` and `test_ctrl_q_with_only_a_composer_draft_asks_and_stay_keeps_it`. Require actual shelf record, close count, Stay behavior and short/renamed-title privacy assertions.

ADR required: no for diagnosis. ADR path: N/A. **These four hosted nodes are unrelated by evidence to the separately assigned ordinary child-Thread.start custody risk.**

## Task14g — acknowledged/heap send reaches genuine provider validation (2; separate symptoms, shared fixture review gate)

**Reach:** `Tests/UI/test_console_send_acknowledgement.py` `HeldAdmission`, `HeldGateway`, `build/ready_console`; `Tests/UI/test_console_idle_heap_refreeze.py` caller and recording collector. Read `chat_screen.py` visible dispatch, original hook-permission snapshot, received preparation and controller submit; `UI/Console_Modules/transcript.py` heap busy/refreeze is the downstream intended oracle. No heap production change, config or export selected.

- [ ] Observe the exact original refusal after hook hold release separately for each ledger node. The heap case already entered its hook hold; it is not proven the same root as acknowledgement or HeldMcpRead.
- [ ] Qualify the shared standard source fixture only if its actual predicate is established. Then run both ledger nodes and `test_an_acknowledged_enter_dispatches_once_its_status_line_is_laid_out` / `test_a_refused_acknowledged_enter_repaints_its_refusal`.
- [ ] Keep actual compositor-frame acknowledgement, gateway resolve/provider ordering, per-loop busy sampler and no-refreeze-while-busy assertions. Require process/worker cleanup; do not infer a heap defect from unentered validation.

ADR required: no for diagnosis/fixture correction. ADR path: N/A.

## Task14h — image-only native submission (1; diagnosis required)

**Reach:** Exact ledger node and `_ready_console`/`_build_console_send_test_app`/`_select_llamacpp_console` in `Tests/UI/test_console_send_ready_attachment.py` / `test_console_native_chat_flow.py`; original staged attachment owner, controller accepted-submit/commit and provider request. No attachment UI/API/config/export change selected.

- [ ] Observe original controller-submit refusal and standard-source identity with image staged; do not treat staged retention as image loss/duplication.
- [ ] Make only the diagnosed setup/producer repair; run ledger node plus `test_attachment_readiness_waits_for_checked_config_projection` (its failure/refusal parameterizations), retaining provider image content and accepted-only staged commit/refusal-retains-image assertions.

ADR required: no for diagnosis. ADR path: N/A.

## Task14i — first-send and provenance-recovery harness (2; diagnosis required)

**Reach:** `Tests/UI/test_console_blocked_send_recovery.py` shared mounted harness, `_persist_console_provider_config`, `ConsoleHarness`/`app_factory`; original received configuration/dispatch, controller recovery and recovery action route. Coordinate Task11 first-send durable config/copy and Task8 source gates; no config/export/API change assumed.

- [ ] Observe original refusal/source predicate separately for the system-prompt-first-send and retry-recovery nodes. Neither provider nor actual recovery-callout oracle is reached in hosted evidence.
- [ ] Correct only diagnosed shared qualification or producer; run both ledger nodes plus `test_provenance_failure_is_visible_everywhere_the_turn_is_shown` and `test_a_send_refused_because_the_chat_was_archived_keeps_its_reason`.
- [ ] Require real first provider/system prompt and actual provenance failure→callout→Retry reply, with draft/reason/cleanup intact; do not manufacture the callout or force acceptance.

ADR required: no for diagnosis. ADR path: N/A.

## Task14j — spoken-send second completion (1; diagnosis required)

**Reach:** `Tests/UI/test_console_send_resend_guard.py` exact ledger node's speech bridge, Enter-hold helper, completed-turn counter; `test_console_send_acknowledgement.py` native fixture and original dictation/dispatch/controller finish owners. Search hold/speech helper callers first; coordinate14d only if this node actually consumes its helper. No config/exports change.

- [ ] Observe original send acceptance/receipt/result and physical key/speech ordering until `_turns_done(...,2)` fails; distinguish refusal, missing completion and duplicate/resend defect.
- [ ] Run this node and the same-file affected spoken-before-clear/committed-meanwhile controls after their14d qualification. Keep exact provider/turn counts, no duplicate resend and original key delivery; another hold-entry repair does not automatically close this node.

ADR required: no for diagnosis. ADR path: N/A.

## Task14k — second session's provider start gate (1; diagnosis required)

**Reach:** `Tests/UI/test_console_send_start_window.py::StagedGateway` and exact ledger test; `_select_llamacpp_console`/source-enabled app in `test_console_native_chat_flow.py`; `Chat/console_chat_controller.py::submit_draft`, `UI/Console_Modules/config_sync.py` send-start delegation, `Chat/console_send_start.py` original owner. No config/exports change.

- [ ] While A is genuinely held at its first provider gate, record B's received/config/authorization identities and original submit refusal. A already reached provider and durable commit; do not call this a no-send-at-all failure or treat its teardown failure as the original cause.
- [ ] Run ledger node plus `test_a_received_sends_start_ends_at_its_own_real_provider_request`, `test_custody_task_completion_retires_only_its_send_start` and `test_exceptional_submission_retires_only_its_send_start`. Preserve two actual session gates and per-send start/provider-entry ownership plus custody cleanup.

ADR required: no for diagnosis. ADR path: N/A.

## Task14l — applied Anthropic settings handler entry (1; diagnosis required)

**Reach:** Exact ledger node in `Tests/UI/test_console_settings_hidden_fields.py`; its direct `_build_test_app`, file-backed CharactersRAGDB and `_ConsoleFlowHarness`/`_drain_settings_tasks` in `test_console_provider_apply_defaults_flow.py`; real settings modal Apply, native received dispatch and `Chat/Chat_Functions.py::API_CALL_HANDLERS[anthropic]`. No parameter/schema/export change selected.

- [ ] Observe original refusal/readiness/source identities after real Apply; compare current native source-enabled fixture before changing the direct harness. The modal and committed hidden-fields checks passed; actual handler kwargs omission/temperature assertions have not run.
- [ ] Keep that modal and handler interception; qualify an actual accepted send through the same original configured Anthropic path, run ledger node and `Tests/UI/test_console_provider_apply_defaults_flow.py::test_apply_closes_and_changes_only_later_send_context` / `::test_apply_lifecycle_stages_persists_resumes_and_promotes`. Preserve hidden-field omission/temperature and refusal/source-change controls, offline fake provider and clean DB/worker closure.

ADR required: no for diagnosis/fixture correction. ADR path: N/A.

## Task14m — native repair original storage source (1; diagnosis required)

**Reach:** `Tests/Chat/test_response_rules_runtime.py::native`/exact ledger test, `Tests/Chat/response_rules_fixtures.py`, `_controller_rig` in `test_console_fleet_wake.py`, bridge replacement in the test; original `Chat/console_agent_sources.py` config-generation and bridge/controller/runtime/store/permission binding guards, original repair route. No guard/config/export change selected.

- [ ] Use an in-place observer to distinguish config-generation from exact bound object/path mismatch; the short hosted RuntimeError does not identify it and replacing bridge alone is not root-cause proof.
- [ ] Correct only diagnosed qualification/owner; run ledger node plus `test_real_agent_service_reply_uses_the_owned_rule_boundary`, `test_real_agent_remaining_budget_limits_native_repairs` and `test_source_changes_revoke_late_learning_activation`.
- [ ] Retain completed real tool work, genuine native repair, remaining budget and source-change refusal. Review against Task8/13 changed source records, rather than weakening them.

ADR required: no for diagnosis. ADR path: N/A.

## Task14n — missing external Stop marker (1; diagnosis required; coordinate Task12)

**Reach:** `Tests/Chat/test_response_rules_external_hook.py` exact node, `native` imported from `test_response_rules_runtime.py`; original hook configuration/admission/selection/launch/result and Stop/native-repair coalescing owners. Existing hook files and process fixtures changed by Task12 remain its sole implementation ownership; no hook config/export/custody redesign here.

- [ ] Record original hook reason enums through selection/admission/launch/result for the exact node; missing invocation marker after accepted result does not identify which step failed and is not process-group proof.
- [ ] Apply only a fixture/selection correction supported by that sequence; hand production lifecycle/custody findings to Task12's existing design/review gate. Run exact node with `Tests/Chat/test_console_run_hooks_regressions.py::test_stop_is_per_accepted_turn_even_while_queue_owns_notifications`, `::test_hook_refusal_releases_preparation_for_retry` and Task12's real descendant-return controls from its already reviewed design (`Tests/Agents/test_native_legacy_hooks.py`, `test_run_hooks.py` affected selectors). Task12 remains responsible for selecting those existing process-specific controls based on its actual fix.
- [ ] Require actual marker plus exactly one coalesced follow-up and native violation behavior; retain original hook/group physical cleanup. Passing this test cannot substitute for Task12 PGID proof.

ADR required: no for diagnosis. ADR path: N/A. Any custody change uses Task12's required prior ADR285 amendment.

## Task14o — run-start store loop turns (2; diagnosis required)

**Reach:** `Tests/UI/test_console_run_start_yields.py::LoopTurns/_in_composition`, shared `_Console/_run_start/_with_console` and autouse `trust_new_files` from `test_console_mcp_catalog_reuse.py`; `Chat/console_run_start_tools.py`, controller `_compose_mcp_provider/_compose_local_provider`, original unified MCP/permission/local store fingerprint and catalog readers. No caller-supplied bypass/config/export change selected.

- [ ] Observe original function entry with scoped in-place profiling plus loop ticks, avoiding replaced bindings where identity is part of the path. Separate wrapper-induced observation, required native kill-switch recheck and a real scheduling regression.
- [ ] Only then select fixture observer correction or minimal existing-reader scheduling fix; no removal of required recheck or relaxed loop oracle to silence it.
- [ ] Run both ledger nodes, same-file `test_the_fingerprint_reads_its_first_store_in_a_loop_turn_of_its_own`, and catalog reuse controls `test_run_start_reads_the_kill_switch_once_for_mcp_and_local_tools`, `test_an_unchanged_catalog_is_reused_until_the_kill_switch_changes`, `test_a_store_write_during_composition_is_never_reused`. Require catalog equality/content, live store-change invalidation and original guards alongside distinct turn reads.

ADR required: no for diagnosis. ADR path: N/A. Existing scheduling/source contracts retained; any selected contract change needs prior ruling.

## Task14p — Library rail-return autosave and post-teardown limit (1; diagnosis required)

**Reach:** `Tests/UI/test_library_note_autosave_recovers.py` return predicate/current body lookup/`_armed_editor` setup; `UI/Screens/library_screen.py` retained-note selection/schedule/save fences, `UI/Library_Modules/library_note_autosave.py`, `library_notes_controller.py` EditorReady/body-change/timer handlers, and `Library/library_notes_session.py` coordinator/port persistence. No file sync/schema/config/exports change selected.

Hosted failure is final actual DB equality after existing10s autosave bound. Corrected immutable run reached all assertions and teardown with full persisted edit, saved=true/dirty=false/saving=false, **then supervisor SIGKILL at55.035s**; process exit/drain/retirement is unproved. The first two attempts failed in an extra probe audit, not the app. Return's widget-exists+0.2s lacks initial `_armed_editor` barrier, a concrete synchronization gap **not a proved failure cause**.

- [ ] Preserve the existing passing local observation and adverse exit; do not repeat merely to obtain a green exit or change the product on this evidence.
- [ ] Use the intermediate hosted diagnostic checkpoint below if the unresolved persistence failure needs hosted evidence; no local/Linux reproduction environment is assumed. On that exact rerun observe original generation, selected-session/editor equality, armed/presentation state, actual body mutation, save outcome and final equality/dirty/saving booleans. Keep unresolved selector ACs open; select only the diagnosed fix and return it through this group's focused gates, preserving final-row assertions and timing.
- [ ] Scope any lifecycle follow-up separately to the original post-teardown wait frame and complete provenance inventory captured **before** supervisor retirement. Do not equate teardown_passed with cleanup success.
- [ ] Verification after a justified correction: ledger node plus `test_typing_after_keep_editing_on_a_refused_quit_is_autosaved`; actual original DB row and clean process outcome required before closure.

ADR required: no for diagnostics. ADR path: N/A. Any lifecycle/storage boundary correction requires a prior ADR check.

## Task14q — accepted fresh Goal/wake human-receipt applicability (separate baseline; approved design)

**Reach:** Only the human mixin composition/receipt condition inside `Chat/console_chat_controller.py::_stream_assistant_response_inner`; original seven call sites in `_submit_draft_body`, `resume_durable_postcommit`, `retry_dispatch_recovery`, `_retry_message_body`, `_continue_from_message_body`, `regenerate_message`, `edit_and_resend_message`; `Chat/console_automatic_submission.py` token resolution/later source check; `console_native_commit.py` distinct chat-start receipt, `console_prompt_mixins.py` service and `DB/console_prompt_mixins.py` exact durable ownership guard. Tests/fixtures: `test_console_goal_dispatch.py`, `test_child_run_scope_ordering.py::_survivor_bridge`, `test_console_fleet_wake.py` actual survivor wake, mixin admission/retry and chat-start controls below. No config, API, schema or exports change.

- [ ] Implement the approved RED extension: no-USER, existing transient-old-USER and durable-old-USER Goal history. Durable case must expose **false newly produced assistant receipt/marker**, not just successful outcome; historical legitimate receipts may remain. Retain exact Goal outcome/chain/allowance/fresh-provider-history/SYSTEM/no-new-USER checks.
- [ ] Extend the original `test_a_survivor_settle_wakes_the_supervisor_with_a_machine_notice` with historical human input; assert **actual memory-owner** receipt/marker for its produced wake assistant (fixture is memory-only), real child completion, original accepted token, one request, SYSTEM/no-new-USER/no composer callback, delivery/accounting. Do not demand a nonexistent wake SQLite table. The Goal durable case proves the durable false receipt.
- [ ] Minimal predicate at existing block: `current_user is not None and not (route is ConsoleRequestRoute.FRESH and wake_authorization is not None)`. Preserve existing block contents and later `automatic_source_conversation` validation; no early stream return. Original accepted Goal/wake object is forwarded without reconstruction; do not broaden by WorkOrigin.AUTOMATIC, snapshot emptiness, message metadata or a new flag.
- [ ] Run ledger Goal node plus exact Goal controls `test_goal_authorization_cannot_be_forged_or_moved_to_another_session`, `test_failed_acceptance_never_calls_preparation_or_provider`.
- [ ] Run exact human controls from approved design: `Tests/Chat/test_console_prompt_mixin_admission.py::test_memory_database_acceptance_persists_exact_receipt` (empty/nonempty); in `Tests/Chat/test_console_prompt_mixin_retry.py`, `test_snapshot_write_failure_blocks_initial_adapter_and_preserves_draft`, `test_retry_original_regenerate_current_and_clean_history`, `test_hook_user_row_is_never_the_mixin_anchor`, `test_continue_replay_composes_only_synthetic_current_user`, `test_current_choice_edit_continue_preview_and_clean_history`. All these controls exist in those two inspected files; preserve their existing parameterizations and no synthetic replacement controls.
- [ ] Run `Tests/Chat/test_console_chat_start.py` exact `test_native_start_uses_both_receipts_and_literal_machine_request`, `test_live_machine_retry_starts_manual_work_without_reassigning_old_allowance`, `test_reopened_machine_retry_is_explicit_and_retains_provenance`; augment initial chat-start with exact empty mixin receipt/marker at its own durable USER/ASSISTANT anchors. Its automatic WorkOrigin must remain applicable.
- [ ] Run `Tests/DB/test_console_prompt_mixins.py` exact `test_foreign_anchors_refused_and_fork_survives_source_purge`, `test_acceptance_uses_explicit_native_map_and_rolls_back`, `test_empty_attempt_and_lost_marked_record_are_distinct`. Keep fail-closed receipt storage and explicit manual Retry/current-choice/Continue/Edit semantics.

ADR required: no new ADR. ADR path: existing `backlog/decisions/227-native-console-goal-runs.md`, `286-console-prompt-mix-ins.md`. Reason: approved bounded correction restores existing machine/human separation; no storage contract amendment. Follow approved Goal brief and independent gate, not a fresh architecture review.

## Task14r — ordinary invoked child-start custody (additional qualified defect; no hosted node)

**Evidence:** `feedback-child-start-qualification.md` is final:3 original-owner cases completed with released/joined exact threads and exit0 at frozen Python/SQL epoch `fe165d39ea1b4cf174ddc38bc92ce2eaa34d37c0381e56ebc54197cc8888511c`. Stock-start-return faults before target and during actual provider entry, plus an explicitly opaque queued-preident control, produced positive child drain/removal while the exact target could still execute. Task8 workflow/route/wrapper gates stay CLOSED C0. This is separate from14f Temporary failures, outside85 hosted+Goal count; do not redo the research.

**Reach:** Repair shared `Agents/agent_service.py::_launch_prepared_child_result` and ordinary `_launch_prepared_child/_launch_fleet_child/_spawn/_resume_retained_child` disposition/refund callers once. Preserve `run_child_owned`, `_finish_execution`, `_settle_fleet` exact-thread tracking, `Agents/execution_capacity.py`, fleet/automatic runtime and `DB/automatic_work.py` ledger custody; `Chat/console_ordinary_tree.py` original attach/host-return owner and existing worktree admission/retirement. Fixtures: `Tests/Chat/test_native_admission_return.py` with `ordinary_host/reference_host`, `Tests/Agents/test_automatic_child_scope.py`, `test_fleet_runtime.py`, `test_fleet_continuation.py`, code-review and workflow callers named in the final report. No new configuration/export/schema/dependency.

- [ ] Retain the exact stock thread and child custody before invoking start. Any invoked start exception preserves owner, outstanding reservation, original route/source/storage and cancellation/tracking until positively established exact target/physical return; no absent-ident/is_alive/terminal-row inference. Preserve separate proven stock construction-before-invocation cleanup and truthful public tuple/raising/uncertain reporting. Check earlier fleet/DB/route/refund/notification and outer spawn allowance, not only the final `native_thread_started` branch.
- [ ] Add focused regressions through existing original Console tree and real worktree fixtures for queued/entered start fault, ambiguous BaseException/cancellation, late target return/original join, and genuine stock construction failure. The direct qualification had no Console tree/provider operation or real admitted checkout; these stronger retention/retirement claims remain to prove. Unknown never-starts retain custody, while late proven return must still retire it.
- [ ] Recheck exact `Tests/Agents/test_automatic_child_scope.py::test_proven_prelaunch_failure_refunds_shared_reservation[construct/start/review]` with invoked-start expectation corrected conservatively; retain `test_captured_context_reaches_raw_child_and_tool_threads`, `test_three_generations_share_six_actual_launches_and_manual_is_uncharged`, `test_finished_child_continuation_cannot_bypass_shared_launch_limit`.
- [ ] Run `Tests/Agents/test_fleet_runtime.py::test_worktree_thread_start_failure_retains_checkout_and_retires_routing`, `::test_worktree_creation_is_durable_until_exact_execution_owner_drains`, `::test_cleanup_unproven_stays_uncertain_after_exact_owner_drains`; same-file `::test_thread_start_failure_is_contained_and_the_turn_still_finalizes`, `::test_thread_start_and_transient_terminal_status_failure_are_both_contained`, `::test_thread_start_and_persistent_terminal_failure_reconciles_on_reopen`, `::test_thread_start_refusal_survives_atomic_and_status_fallback_failure`, `::test_shared_child_launcher_binds_one_native_owner_row_and_lifecycle`; `Tests/Agents/test_fleet_continuation.py::test_send_to_agent_to_a_finished_child_starts_a_resumed_seeded_run`, `::test_a_resume_consumes_a_spawn_slot_and_refuses_at_the_budget`; `Tests/Chat/test_native_admission_return.py::test_original_native_child_and_finalization_boundaries` and `::test_ambiguous_start_before_ident_retains_same_target_until_positive_return`.
- [ ] Add **one** invoked-start uncertainty regression through the existing original retained-continuation fixture in `Tests/Agents/test_fleet_continuation.py`: fault the resumed child's invoked start, retain that exact resumed thread, execution owner, outstanding spawn slot and uncommitted reservation until positive original source/storage scopes return and exact original join. Assert truthful uncertainty, no duplicate resumed execution or resume admission, no duplicate retained transcript/steering delivery and no duplicate charge. Preserve the named normal seeded-resume and budget controls; no new broad matrix.
- [ ] Preserve review/no-bridge compatibility through `Tests/Agents/test_code_review_dispatch.py::test_fleet_thread_start_refusal_retries_only_when_definitely_unstarted`; `test_agent_service_review_containment.py::test_thread_start_refusal_survives_a_refused_launch_retirement` / `::test_thread_start_refusal_keeps_a_scheduled_attempt_launchable`; `test_workflow_child_launch.py::test_failed_native_thread_start_refunds_only_reserved_launch` / `::test_task_owned_native_log_finalizes_once_after_settlement[thread]`; `test_workflow_run_service.py::test_real_thread_start_failure_retains_original_receipt_until_checkpoint`. Update only unsupported invoked-start retry/refund assumptions, retain Task8 entry-bound controls and existing exact-thread joins.

ADR required: no new ADR for implementing existing custody. ADR path: existing `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `135-fleet-completion-delivery-and-crash-recovery.md`, `285-native-phased-agent-workflows.md`. Reassess before any new long-lived contract; no second registry/executor/issuer is selected.

## Task14s — pre-existing completed-history startup and off-screen return (additional qualification; no hosted node)

**Reach:** Preserve Task10's exact pre-seeded completed-history reproducer and pre-status `ConsoleWorkspaceTree.render_label` component-style exception. Read `tldw_chatbook/UI/Console_Modules/left_rail.py::ConsoleLeftRail.compose`, its synchronous `sync_projection` before yield, and `tldw_chatbook/Widgets/Console/console_workspace_tree.py`. The actual `Tests/UI/test_console_workflow_command.py::mounted_workflow` imports `ConsoleHarness` from `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py:89`, which explicitly selects `APP_STYLESHEETS` at line93. The similarly named workbench-contract harness is distinct and its missing extra sheets are not this reproducer's cause. The ready component selector exists in the app-tier Console stylesheet and generated bundle. Actual supported production reach is not yet established.

- [ ] Record the actual original script, stack, loaded stylesheet sources and original runtime/history state. Distinguish the initial 60-second timeout, the separate pre-status style failure, and Task10's later mounted-first removal journey; none substitutes for the others.
- [ ] Qualify first composition with real pre-existing completed native history, then separately native completion while off-screen followed by return, using the stock harness and the existing production CSS configuration. Show whether the normal supported app lifecycle can reach the reported failure; synthetic projection or a minimally styled widget alone is insufficient. Preserve exact original native publication/source and physical-return evidence.
- [ ] If a genuine defect is reproduced, use the smallest existing mount/sync owner correction. Enumerate every caller of that owner and focused controls before edits. Do not add fallback style values, suppress KeyError, change global CSS or introduce an authority/callback framework. If the cause is only a fixture's missing app-tier styles, repair that setup without claiming a product lifecycle fix.
- [ ] Verify initial empty/normal sync, pre-existing history, off-screen return and token-backed ready-label style under production CSS; preserve the separately assigned Task14a publication ownership and Task14b visible/background/modal/off-screen attention controls. Original numeric budgets, native/foreign/stale-source guards, lifetime and physical cleanup remain binding. Obtain focused independent implementation review before closing the item.

ADR required: no new ADR for a routine demonstrated lifecycle/fixture correction. ADR path: existing `backlog/decisions/150-design-token-system-and-design-language.md`, `backlog/decisions/161-component-pattern-library.md`, and `backlog/decisions/285-native-phased-agent-workflows.md`. Reassess before any actual contract change. This item stays outside the85 hosted-selector ledger and separate from Goal, ordinary child-start custody and Temporary recovery. Plan gate: `feedback-task14-ui-startup-plan-review.md` and its harness-provenance correction in `feedback-task14-ui-startup-plan-rereview1.md`, SPEC PASS / QUALITY APPROVE; execution and production reach remain open.

## Intermediate hosted diagnostic checkpoint — unresolved selectors stay open

- [ ] After known local repair batches have passed their existing focused and independent gates, root may obtain evidence for14e,14p or any other unreproduced diagnosis-only group through one bounded hosted diagnostic publication. No local Linux environment is assumed or required for this route; it is not conditional on all14a–s groups already qualifying.
- [ ] First freshly sync the proposed branch against then-current `dev` under the queue rules below. Freeze the concrete publication/diagnostic delta and obtain **scoped independent review of that delta**, including body-free original-outcome observers and their source/lifecycle/privacy limits. This checkpoint is not the final whole-branch review.
- [ ] Root alone pushes the reviewed concrete checkpoint and requests the exact unresolved ledger selectors/necessary controls on the hosted environment. Retain actual original reason/outcome/readiness/custody observations, code identity and targeted rerun receipts, including failing/adverse results. Do not infer original behavior from a replaced guarded callable or emit bodies, credentials or private path values.
- [ ] Keep every unresolved selector's ACs **OPEN**. Intermediate publication/rerun grants no final-green claim, CI waiver, full-suite waiver, auto-merge or merge authorization. Route any diagnosed repair back through its existing group's focused tests, lint/format, independent review and root checkpoint; use only a further bounded diagnostic delta if an unresolved original predicate still needs hosted evidence.

## Final root-owned integration gate

- [ ] Task8 and Task8b are now locally CLOSED C0; preserve those gates and complete serial9–13 gates first; root reconciles all ledger nodes with the integrated epoch. After the coherent14a–r repair gates, run only combined affected selectors/controls, focused static checks and current architecture/startup budgets required by changed owners. Preserve unrelated red/adverse receipts honestly; do not invoke a full sweep without user opt-in.
- [ ] Freshly sync the proposed branch against then-current `dev` using the applicable queue rules. Root alone owns Git/export/checkpoints/publication. Queue on: do not update/rebase an armed PR by hand; disable auto before further push, pull --rebase because queue may have rebased, use --force-with-lease. Queue off/dry: rebase only the PR being prepared for merge, never merge dev into it.
- [ ] **Only after all groups qualify**, root performs the final whole-branch independent review, including every runtime/DB/source-identity/hook/Goal interaction, task/ADR/doc/AC notes, and original physical retirement claims. Include all reviewer corrections in the new integrated identity. The intermediate diagnostic checkpoint does not replace this review.
- [ ] Root makes the final push after that review and requires CI on the actual published current head, Qodo posted on that head, every review thread resolved, and strict up-to-date protection. A conflict-free queue rebase may reuse review under repo rules, but new threads still block. Do not use workflow_dispatch as a replacement for required pull_request checks. Check current queue mode before any eventual merge decision. **This plan does not arm auto-merge or merge.**

## Exact node disposition ledger

Each row is both the original failing selector and the required targeted recheck for its sole owner. Additional positive/negative controls are defined above or in the already reviewed owner plans. No row is claimed fixed by this draft.

### Owner Task8b — 1 hosted node(s)

- `Tests/Agents/test_fleet_message_tools.py::test_core_imports_keep_progress_runtime_lazy_and_public_schemas_available`

### Owner Task9 — 5 hosted node(s)

- `Tests/DB/test_agent_orchestration_dev_migration.py::test_v15_upgrade_preserves_upstream_records_and_authority[False]`
- `Tests/DB/test_agent_orchestration_dev_migration.py::test_v15_upgrade_preserves_upstream_records_and_authority[True]`
- `Tests/DB/test_agent_run_budget_accounting.py::test_legacy_usage_stays_unknown_across_migration_and_reopen`
- `Tests/DB/test_automatic_runtime_owner.py::test_v17_to_v18_migration_preserves_history_and_does_not_recover[False]`
- `Tests/DB/test_automatic_runtime_owner.py::test_v17_to_v18_migration_preserves_history_and_does_not_recover[True]`

### Owner Task13 — 16 hosted node(s)

- `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[Agents]`
- `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[Chat]`
- `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[MCP]`
- `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[Workspaces]`
- `Tests/Architecture/test_layer_direction.py::test_no_new_upward_import`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Chat/console_trace_maintenance_runtime.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/DB/private_sqlite.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Media/local_media_reading_service.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Personal_Context/repository.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/change_review_screen.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_lifecycle.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_service_wiring.py]`
- `Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/config.py]`
- `Tests/CI/test_app_module_entry_contract.py::test_nothing_at_module_level_follows_app_main_block`
- `Tests/test_config_load_settings_table_guard.py::test_default_config_template_bytes_are_unchanged`

### Owner Task14a — 21 hosted node(s)

- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[active_change_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[blank_creation_call_through]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[coalesced_completion]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[cold_full_schedules_its_pending_replay]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[drain_without_whole_boolean]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[ensure_owner_replacement]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[established_settings_reuse]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[fresh_after_serial_wait]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[live_error]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[missing_settings_call_through]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[ordered_membership_change_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[ordered_resume_no_creation]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[publication_cancellation]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[queued_cancellation]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[replay_cancellation_retires_one_count]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[replay_keeps_drain_owned]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[same_id_session_replacement_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[store_replacement_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[surface_replacement_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[teardown_during_surface_await]`
- `Tests/Performance/test_console_tab_sync_source_ownership.py::test_console_tab_publication_source_ownership[teardown_error]`

### Owner Task14b — 6 hosted node(s)

- `Tests/ProductionApp/test_provider_selection_ownership.py::test_real_console_change_model_command_opens_real_picker`
- `Tests/UI/test_console_visible_turn_attention.py::test_a_turn_finishing_in_a_background_tab_notifies_once_while_console_shows`
- `Tests/UI/test_console_visible_turn_attention.py::test_a_turn_finishing_under_a_modal_over_console_raises_no_hidden_notice`
- `Tests/UI/test_console_visible_turn_attention.py::test_a_turn_finishing_while_console_is_off_screen_still_notifies_once`
- `Tests/UI/test_console_visible_turn_attention.py::test_turns_finishing_in_the_visible_tab_raise_no_hidden_notice[explore_home]`
- `Tests/UI/test_console_visible_turn_attention.py::test_turns_finishing_in_the_visible_tab_raise_no_hidden_notice[start_chatting]`

### Owner Task14c — 2 hosted node(s)

- `Tests/UI/test_console_compaction_forecast.py::test_automatic_forecast_matches_real_preflight_with_overhead_and_cap`
- `Tests/UI/test_console_compaction_forecast.py::test_bound_skill_ask_gate_uses_its_overhead_before_regenerate_commits`

### Owner Task14d — 11 hosted node(s)

- `Tests/UI/test_console_send_resend_guard.py::test_a_draft_cleared_while_it_is_sent_draws_no_was_sent_notice[backspaces]`
- `Tests/UI/test_console_send_resend_guard.py::test_a_draft_cleared_while_it_is_sent_draws_no_was_sent_notice[ctrl_u]`
- `Tests/UI/test_console_send_resend_guard.py::test_a_draft_sent_after_a_non_append_edit_says_it_was_sent`
- `Tests/UI/test_console_send_resend_guard.py::test_a_draft_sent_from_a_tab_left_during_admission_does_not_come_back`
- `Tests/UI/test_console_send_resend_guard.py::test_a_hidden_chat_whose_draft_was_replaced_draws_no_was_sent_notice`
- `Tests/UI/test_console_send_resend_guard.py::test_a_press_held_on_a_non_append_edit_of_a_sent_draft_is_not_sent`
- `Tests/UI/test_console_send_resend_guard.py::test_a_send_refused_while_another_tab_sends_names_that_tab`
- `Tests/UI/test_console_send_resend_guard.py::test_an_enter_never_resends_a_draft_a_spoken_send_committed_meanwhile`
- `Tests/UI/test_console_send_resend_guard.py::test_an_enter_never_resends_a_draft_a_spoken_send_took_before_a_clear`
- `Tests/UI/test_console_send_resend_guard.py::test_enter_after_a_tab_round_trip_during_admission_sends_the_draft_once[round_trip]`
- `Tests/UI/test_console_send_resend_guard.py::test_enter_after_a_tab_round_trip_during_admission_sends_the_draft_once[typed_in_b]`

### Owner Task14e — 6 hosted node(s)

- `Tests/UI/test_console_environment_bound_registry.py::test_a_bound_workspace_lands_through_the_real_consent_service`
- `Tests/UI/test_console_environment_consent_revoked_hop.py::test_consent_revoked_after_a_worker_re_check_drops_the_landing`
- `Tests/UI/test_console_environment_refresh_off_loop.py::test_rail_open_completion_and_poll_never_build_a_turn_configuration`
- `Tests/UI/test_console_environment_refresh_off_loop.py::test_the_narrow_root_read_equals_the_full_snapshots_roots`
- `Tests/UI/test_console_session_admitted_roots.py::test_the_three_readers_answer_from_one_helper`
- `Tests/UI/test_console_session_admitted_roots.py::test_the_three_readers_return_the_same_string_roots`

### Owner Task14f — 4 hosted node(s)

- `Tests/UI/test_console_quit_loss_guard.py::test_ctrl_q_counts_a_temporary_chat_with_a_parked_unsent_turn`
- `Tests/UI/test_console_quit_loss_guard.py::test_ctrl_q_from_home_keeps_short_and_renamed_titles_private[False]`
- `Tests/UI/test_console_quit_loss_guard.py::test_ctrl_q_from_home_keeps_short_and_renamed_titles_private[True]`
- `Tests/UI/test_console_quit_loss_guard.py::test_tab_close_reports_a_parked_unsent_turn`

### Owner Task14g — 2 hosted node(s)

- `Tests/UI/test_console_idle_heap_refreeze.py::test_console_refreezes_after_painting_a_conversation_and_after_a_run_settles`
- `Tests/UI/test_console_send_acknowledgement.py::test_enter_paints_a_sending_row_before_admission_and_validation`

### Owner Task14h — 1 hosted node(s)

- `Tests/UI/test_console_send_ready_attachment.py::test_a_ready_send_with_only_an_image_staged_sends_the_image`

### Owner Task14i — 2 hosted node(s)

- `Tests/UI/test_console_blocked_send_recovery.py::test_each_trigger_chat_sends_and_renders_the_reply[system-prompt-before-first-send]`
- `Tests/UI/test_console_blocked_send_recovery.py::test_provenance_failure_recovery_actions_work[retry]`

### Owner Task14j — 1 hosted node(s)

- `Tests/UI/test_console_send_resend_guard.py::test_a_held_enter_never_resends_a_draft_a_spoken_send_took`

### Owner Task14k — 1 hosted node(s)

- `Tests/UI/test_console_send_start_window.py::test_a_sends_start_ends_at_its_own_provider_request_not_another_sessions`

### Owner Task14l — 1 hosted node(s)

- `Tests/UI/test_console_settings_hidden_fields.py::test_applying_on_anthropic_submits_none_of_the_hidden_fields`

### Owner Task14m — 1 hosted node(s)

- `Tests/Chat/test_response_rules_runtime.py::test_real_completed_tool_work_is_retained_during_native_repair`

### Owner Task14n — 1 hosted node(s)

- `Tests/Chat/test_response_rules_external_hook.py::test_real_external_stop_and_native_violation_commit_one_followup`

### Owner Task14o — 2 hosted node(s)

- `Tests/UI/test_console_run_start_yields.py::test_a_full_composition_reads_each_store_in_its_own_loop_turn`
- `Tests/UI/test_console_run_start_yields.py::test_a_reused_catalog_checks_each_store_in_its_own_loop_turn`

### Owner Task14p — 1 hosted node(s)

- `Tests/UI/test_library_note_autosave_recovers.py::test_a_note_left_dirty_by_a_rail_switch_is_autosaved_on_return`

### Owner Task14q — separate Goal baseline, not hosted

- `Tests/Chat/test_console_goal_dispatch.py::test_goal_native_dispatch_is_automatic_fresh_and_returns_exact_native_outcome`

Ledger self-check:85 distinct hosted selectors; owner counts={'8b': 1, '9': 5, '13': 16, '14a': 21, '14b': 6, '14c': 2, '14d': 11, '14e': 6, '14f': 4, '14g': 2, '14h': 1, '14i': 2, '14j': 1, '14k': 1, '14l': 1, '14m': 1, '14n': 1, '14o': 2, '14p': 1}; separate Goal1. Source, product/tests, Git, Backlog and dependencies unchanged.

Skipped: execution, tests, implementation, remote refresh/posts, root/source/Git/Backlog mutation and duplicate child-start research. Risk: diagnosis-required groups and clean exit/physical retirement remain open; only the approved Goal scope and confirmed fixture mechanisms justify the specific repairs stated above.


Task14a inventory correction (2026-11-07): published f9, preflight0539d72e and current fixture all contain the same24case names and sole parameterized test. All21assigned hosted failures are present; the other3are maintenance_gate, replay_gate and dead_before_lock. The earlier27/5positive+1xfail claim was a plan inventory error, not missing or removed coverage. Preserve all24cases, all original oracles and all85failure-ledger assignments; no invented filler controls or new xfail. Static proof feedback-task14a-inventory-correction.json; Root ruling recorded before final gate.


Task14a local gate: SPEC PASS/QUALITY APPROVE in feedback-task14a-review.md. Only the existing generated fixture adds owner.query returning the truthful empty workflow-line tuple; all24actual cases pass after the21assigned originals fail for missing query. Original function/source/cold-import/session/store/member/surface/cancel/replay/drain checks and1second per-control cap remain. Rootfull-tree/checkpoint evidence is recorded with the gate; no product or dependency change.

Task14b preflight clarification before implementation: reuse existing unchanged Tests/UI/test_console_store_continuity.py::_admit_prepared_console_turn for ALL FOUR attention admission sites, retaining actual Home/Console/modal/off-screen/tab journeys and their exact notice/marks oracles. It proves managed-native custody, not that every turn used received-intent fastpath; the separately required managed-demand received control qualifies that route. Replace BOTH picker before/after legacy configuration observations with existing wiring._capture_console_configuration_async; preserve realpalette/search/Apply and assert the seeded system prompt as well as original provider/model/one-notice. Production turn_admission lives UI/Console_Modules, not Chat. Only two existing testfiles selected; no sharedhelper/production changes. Add ordinarylocaltrustservice readiness only if original currentRED identifies it, using existing service/identity; do not prewarmMCP or fake roots. Original sixledgernodes+threeextraselectors expand11cases, caps/privateprofile/fulloriginalcleanup retained, independentgate before nextwriter. ADRno/N/Aexisting285.


Task14b required-control repair, before extending the test edit scope: current original dispatch calls HookPermissions.authority_read(), which calls the unchanged snapshot(). The managed-demand Performance observer still requires snapshot's immediate parent to be dispatch's nested snapshot; it misses hook_ready and causes the three observed KeyErrors/unarmed hold. Read-only source and saved trace/facts identify this fixture mismatch; no production/readiness change is indicated. Extend this batch to ONE existing file, Tests/Performance/test_console_managed_received_acceptance.py, after its exact pre-edit baseline. Pin the original authority_read definition/code/globals and class slot through the existing observer; qualify only the exact snapshot -> authority_read -> dispatch snapshot stack, same original hook owner/captured source.permissions/runtime and original received record/claim/selection/task. Keep current callback events, the direct hook_ready assertions/error collection, genuine cold backing/factory, one catalog/builder/provider, actual held native Future/read/producer lifetime, phase order, release/drain/source cleanup and all original limits. No generic ancestor walk, monkeypatch, extra read, factory nulling, prewarm or ignored KeyError. Fresh same-selector 240s-child/10s-prerequisite/360s-outer verification and independent gate required. ADR required no, path N/A; existing ADR230 read contract and ADR285 custody apply.

Task14b local gate (2026-11-07): independent SPEC PASS / QUALITY APPROVE C0
in `feedback-task14b-review.md`. All eleven unique required cases have passing
Source receipts across their documented runs. The final unmodified managed-demand
control passed in 9.36 seconds, with genuine held initializer, original phase/order
and cleanup assertions; the separate diagnostic pass is explicitly instrumented.
Only three test files changed; core f7163a38, production 8ed71e67 unchanged.
The earlier corrected Source attempt took an apparently different route and logged
a RuntimeError whose exact predicate/origin remains unexplained. The later pass
does not establish reliability or erase that material risk. Root's full-tree and
checkpoint evidence belongs to this local gate; public readiness remains open.

Task14c handoff clarification: the existing source-ready flag is selected in
native-chat setup but implemented by `ConsoleHarness.on_mount` in the core-loop
screen-adaptation fixture. Opt in only in the existing compaction `_held_console`,
then assert the original app/local backing and factory in `_ask_console` before
controller/provider/forecast work. The helper reaches nineteen expanded cases
across forecast and the three hold-family files; the required stock-owned,
builder-cancellation and source-refusal controls add twenty-three, for forty-two
focused cases. Preserve each original profile, cold control, cap and cleanup.
Only `Tests/UI/test_console_compaction_hold_flow.py` is selected for edits; no
shared-harness or production change. ADR required no, path N/A; existing ADR285.

Task14c additional original-readiness barrier (before code, 2026-11-07): both
forecast nodes pass after stock trust publication, but fifteen shared callers
still fail before their behavioral oracles. Qualified R2 return observations
locate the first receiver decline at `received_send_held(screen)`, followed by a
normal refused fallback and no observed custody. They do not identify the
predicate's underlying subcondition or fallback reason. The fixture currently
waits only for a composer widget. In the same existing `_ask_console`, reuse the
original attachment-reconciliation helper and await the original off-loop
`emergency_stop_hold.refresh` under the existing five-second stop-readiness
budget, then assert its real published path and inactive cached state. This
establishes the required real precondition; it must fail on an actual held or
unknown state. Never clear a stop, assign readiness/path/copy fields, replay the
received guard, change admission route, or infer that the prior path was absent.
Keep `_send`, provider/runtime choices, original test bodies, forecasts, rollback,
numeric Send/child caps and cleanup unchanged at this step. Qualify one original
cancel case first without diagnostics, then affected callers only after that
barrier reaches the intended behavior. Preserve all fifteen failures and R0/R1
setup-only diagnostics; R2 stays instrumented. ADR required no, path N/A; existing
original readiness and ADR285 ownership apply. Task14c remains open until every
affected case and the twenty-three remaining source/lifetime controls qualify.

Task14c canonical native MCP fixture (before code, 2026-11-07): the original
readiness assertions now complete, and the reused qualified diagnostic passes
the earlier stop branch. Its next original return is False from
`standard_console_sources(refs.mcp)`; no database condition was reached. The
factory's default fresh user-data directory feeds MCP constructors through its
app-global patches, whereas `mcp_source_participants.bind` installs only sources
at the selected configuration's canonical profile paths. The existing native
send builder already reconciles this distinction. In `_held_console` only, pass
`user_data_dir=config_module.get_user_data_dir(), isolate_databases=True` to the
existing `_build_test_app`. This aligns real MCP paths while retaining separate
SQL files per app. Keep the explicit compaction chat DB, provider/runtime mode,
mount flag, original barriers and every behavioral oracle. In `_ask_console`,
assert the original resident `standard_console_sources` predicate before Send;
it must qualify without constructing, rebinding or replacing any source. No
production path/guard, manual binding, synthetic owner, MCP payload prewarm or
shared-factory change. The diagnostic identifies the failed MCP guard group;
its exact internal branch was not observed. First run the original cancel node
without instrumentation and pause on any new predicate. ADR required no, path
N/A; this applies existing ADR285 native-source fixture rules, not new authority.

Task14c native task completion (before code, 2026-11-07): canonical MCP and all
real readiness assertions now pass. The original log reaches received
`awaiting_review` and controller submission while the helper's accepted-task
list remains empty. The original receiver uses `accept_received_intent`, whereas
the fixture records only `accept_turn`. A direct call outside Textual's worker
returns before received custody completes; zero adapter calls at that instant
do not establish a budget hold. Extend only existing `_send`: capture the
runtime's current custody IDs and target session before its unchanged visible
action; afterward add only newly created, same-session received records' actual
tasks to the existing captured-task list and await them through its original
shielded thirty-second loop. Keep the original ten-second action bound, older
admission capture, route, provider/runtime choices and final UI sync. Use actual
existing records/tasks, never a synthetic result, replacement method, new
worker/executor or polling delay. The later RuntimeError origin remains
unproved; awaiting the original task must surface real failures rather than
interpret pending work as a hold. First qualify the unmodified original cancel
case, then complete all affected callers/controls and independent review.
ADR required no, path N/A; existing runtime custody and ADR285 apply.

Task14c final caller refinements (before code, 2026-11-07): seventeen affected
cases now pass; two fail. The tooltip builder's original forecast branch always
says `Estimated next send pauses for compaction approval`; align the one stale
case-sensitive expectation with that exact qualified copy, retaining all its
hold/cancel/status/label assertions. The queued case completes its first turn,
then the next visible action returns before provider entry. Its caught exception
origin is unobserved. Successful first-send completion schedules a separate owned
configuration publisher; its real metadata fence can refuse the next programmatic
capture. After `_send`'s original turn-task waits, await its existing non-None
`runtime._first_send_task` through a shield under the existing native first-send
ten-second bound before final UI sync. This establishes real publisher quiescence,
not a claim about the failed invocation's exact flag. Never clear state, premark
onboarding, disable the fence, change config or fabricate completion. No new
production or shared owner. First run the two failed original cases; on success
requalify all remaining affected cases because the helper changed, then the
twenty-three original controls and independent gate. Preserve all prior failures.
ADR required no, path N/A; existing ADR285 first-send ownership applies.

Task14c queued follow-up caller (before code, 2026-11-07): the tooltip case now
passes, and the queued case reaches both original provider entries. Its direct
follow-up action reports received `awaiting_review`, then the immediate queue
snapshot has no committed entry and a held reservation. That caller does not use
the corrected native-completion helper. Extend Source edit scope to exactly one
existing caller in `Tests/UI/test_console_compaction_hold_cardless_sends.py`:
replace its follow-up load/direct-action block with the already imported `_send`
helper on `_FOLLOW_UP`. Keep the live second provider held until the original
queue-count assertion, original twenty-second release limit, all queue/pause/
rollback/transcript/recovery/frame assertions and original cleanup. No production
or other caller change, polling delay, fabricated entry or weaker snapshot check.
This waits the follow-up's actual new received task through existing ten-second
action/thirty-second completion bounds. Capture the second file's exact original
baseline before editing. Qualify this original queued case, then the seventeen
remaining affected cases at final helper bytes and twenty-three controls. The
tooltip pass remains tied to its unchanged bytes. ADR required no, path N/A;
existing runtime custody and ADR285 apply. Task14c remains open.

Task14d caller-purpose clarification (before its code): Root Ruling42 resolves
the preflight's explicit caller conflict. Keep the existing intentionally custom
getter hold as `HeldCustomMcpRead` in the same off-pump test file and import that
symbol under the old local name only in `test_console_fallback_pressed_draft.py`.
Preserve both nonstandard-source assertions and their reauthored-draft/session
oracles; update only the stale original custom-capture caller predicate and
scoped restoration. `HeldMcpRead` serves the fifteen native construction sites
with original-Code observation of the actual finite permission payload boundary.
All sixteen sites require scoped cleanup, including error paths. Scope is three
existing UI test files (off-pump, resend guard, fallback), no production/shared
readiness helper or new framework. The preflight inventories twenty direct cases
plus seven expanded original MCP source-control cases. First qualify the original
non-append representative through entered hold, physical return and normal exit;
then all twenty-seven unique cases with original ten/thirty-second hold/entry
caps. No silent source retargeting, native/custom conflation or ready-root bypass.
ADR required no, path N/A; existing ADR285 original-source/custody rules apply.

Task14c local closing gate (2026-11-07): SPEC PASS / QUALITY APPROVE C0 in
`feedback-task14c-review.md` (5de044e4). All forty-two exact collected cases have
qualifying original Source positives: seventeen final helper cases, twenty-three
original stock/builder/refusal controls, one queued case and the unchanged tooltip
case from its explicitly preserved earlier epoch. The mixed parent exit-one
receipt remains adverse, with only its passing tooltip node counted. Final core
0e9a211c, unchanged production8ed71e67, exactly two test files; all original limits
and behavior/rollback/custody assertions retained, with one exact tooltip-copy
expectation and one queued caller deliberately adapted. Sealed129-entry evidence
inventory4f52ce2b and final gate report92935d1b retain every earlier failure and
diagnostic limit. Root's full-tree/preservation/checkpoint gate precedes Task14d.
No full-suite, reliability, hosted or public-readiness claim follows.

Task14d spoken-commit capture barrier (before further code, 2026-11-07): the
native representative and nine further cases pass. The next spoken-send case
times out waiting for its pending Enter capture, then retains the original native
hold timeout as a separate cleanup error. Its paint-only gate can be skipped by
the original busy-session acknowledgement path. This does not identify the failed
invocation's exact busy operand. Reuse the existing same-file `_start_send`
reached/release barrier from the later spoken-clear case, with a tiny shared
in-file helper if useful, in place of only the obsolete paint-delay fixture.
Keep native permission observation intact. Require the actual pending capture's
session and `y` text, original spoken Worker completion and committed `y`/empty
composer while Enter is still delayed, then release Enter and retain the original
single-user/single-stream/no-resend assertions. Keep all releases/restoration and
scoped physical cleanup in finally. This is the existing UI scheduling test seam,
not a replacement of a guarded permission reader or receiving method. Preserve
original numeric limits and verify both users of a shared extraction. Run the
original failed case first, then requalify affected cases and all unrun custom/
source controls. No deleted pending assertion, custom-route substitution, fake
capture or timeout increase. ADR required no, path N/A; existing ADR285 applies.

Task14d R48 review follow-up before code (2026-11-07): the first corrected spoken-commit case passes normally. Preserve that epoch and its limit: `_idle` does not itself wait for the delayed start task. Add the same-pending identity assertion immediately before release and wait for the original pending capture to retire after release, before idle and final no-resend assertions. Retain all original checks/limits and qualify the original case first at the corrected bytes, then all affected and unrun custom/source controls. ADR required no; path N/A, existing ADR285.

Task14d fresh diagnostic handoff before diagnostic code (2026-11-07): final0cf83e50/prod8ed is frozen with17passes,3distinctfailures,7unrun. Four remaining native cases pass; the busy-copy test holds native preparation although its asserted notice belongs to the legacy hook gate. No third custom-route edit is selected: existing custom-visible control refuses before its hold. The first MCP retarget control returns RecoveryRequired; cleanup re-await may obscure the primary stage. A fresh owner may write only bounded PREP in-place original-production-Code diagnostics for the actual custom refusal and, separately if needed, retarget stage. Use the existing private runner, original limits, exact owners, body-free finite rows and selected original raise/catch lines/true final returns; no global trace, guarded wrapper, replay, raw-pinned rewritten test helper, fake readiness or Source edit. Diagnostic import timing remains a limit. First custom-visible case360s; at most one separately needed retarget360s. Preserve every earlier epoch and physical outcome. Root and reviewer must approve the observed-predicate correction and ADR check before implementation. ADR required no for diagnostics; path N/A, existing285. Task14e remains HOLD; public ACs open.

Task14d R51 diagnostic correction before code (2026-11-07): preserve firstcustom diagnostic limited facts; actual preparationowner is the selected UI session module, not its screen. One new PREP-only observer may retain and validate that exact owner/app/session and observe the five pinned original fixed unavailable raise sites in wiring/turn_admission. One original custom-visible360s run, unchanged application/test limits; no guard replay, wrapper, broad trace or Source edit. Require the actual selected raise before attributing origin. Separate MCP diagnostic proves selection returned empty after retarget, then outer prepared-source currentness raised; its exact OR operand is unmeasured. No source-control oracle or product contract change is selected yet. ADR required no for diagnostics; path N/A, existing285.

Task14d MCP retarget fixture correction before Source code (2026-11-07): independent review approves adding only existing Tests/MCP/test_console_snapshot_source_contracts.py to the three-file test scope. In its existing admitted-bundle probe, observe only the same-service original selection true final typed return as count/empty boolean, excluding coroutine yields and retaining no result/frame. Keep both original four-second physical-entry waits and each mutation; require exact outer RecoveryRequired owner-change refusal, exactly one empty inner selection, unchanged changed-callback non-invocation, nonempty real leases all retired, and original finally settlement. Static source covers five standard-source mutations; runtime initially proves only the first outer currentness branch, with its exact OR operand unmeasured. Run that original case first360s, then remaining four kinds plus probe descriptor caller and private-bundle control1860s. No production/source guard/timeout change. ADR required no, path N/A; existing285. Custom compatibility and busy-copy diagnoses remain open; Source stays frozen during each command.

Task14d R53 explicit contract correction before Source code (2026-11-07): review approves using the existing supported custom-manifest seam for both mounted positive reauthoring cases. Keep every revision, hidden-draft privacy, USER/provider and final-draft oracle; observe actual deferred pending/OPAQUE ownership, original manifest return and native read retirement. Assert original standard service and unknown metadata. This explicitly replaces the obsolete nonstandard-getter positive premise, not equivalent coverage. Preserve that counterexample as a separate physically sent, completed managed-refusal test with counted getter non-invocation, no USER/provider, original draft and restoration/disposal. Scope only existing fallback test and removal of its unused old custom hold helper after caller search; native hold behavior unchanged. First visible360s, then hidden/negative and two existing custom-ordinary/managed-refusal controls1260s. No production/guard/limit changes; existing ADR285 applies, no new ADR. The legacy busy-copy correction is separate and not yet selected. Preserve prior mixed-epoch evidence and open public ACs.

Task14d R54 legacy API correction before Source code (2026-11-07): R53 independent gate and full Root audit pass atc0501a15/prod8ed. Limit edits to the existing resend test module: explicitly describe A/B wired original legacy hook-dispatch Workers, physical tab navigation and unchanged busy-name/draft outcomes; this is not native voice/B-Enter coverage. Hold the original post-hook profile/rules finite read by exact profile/lambda/invoke/WorkItem Code/globals, original app/controller/session/workspace/read registries and running Future. Reuse unchanged native observer enter/exit with all lifecycle fields; no generic framework, wrappers, synthetic readiness or production changes. Require complete original body/callback/invoke/Future/read and Worker/runtime return, original10/30caps, finally release and error preservation. B counting/delegating real queue continuation must never run; prove typed BREFUSED/Alpha, A queue acceptance plus exact USER/provider and B privacy. First busy360s, then actual native spoken-commit and tab-left controls660s; stop on a new predicate. Independent gate/full Root audit precede Task14d closure. ADR required no; path N/A, existing285. Public ACs stay open.

Task14d R55 error-path cleanup before Source code (2026-11-07): preserve R54 final three positive receipts; review blocks its unbounded Worker waits when actual hook review precedes profile entry. Limit changes to the same resend module and one new real cancellation control. Shared cleanup releases the hold, validates/cancels only unfinished actual nonthread Worker tasks and uses bounded non-cancelling waits, reports pending ownership rather than retirement, consumes expected cancellation only after actual completion, and bounds actual runtime disposal with original read/review retirement. Preserve primary and cleanup errors. The new control opens real ASK review through existing hook_file/owners, cancels the genuine subject task during its unchanged30s entry wait, and requires CancelledError only after actual Worker/read/runtime/review/host return and no USER/provider/queue/draft loss. No fake snapshot/event/readiness or timeout increase. First cancellation360s, then original busy and two native controls960s. Independent review and full audit still required. ADR required no; path N/A, existing285. Public ACs remain open.

Intermediate exact-union accounting (Root Ruling56,2026-11-07): accept only actual combined caps controller29726, gateway8633, settings35770, personas16828, Personal Context4908, registry4314 and ChatScreen24722/744. The last two explicitly correct old private one/four-line understatements plus required upstream/logging union growth; no spare, new method allowance, trace exception, lazy edge, cold/call/query/operation/timeout or product budget change. Both code and metadata proposals passed static review. Actual final ratchets, inventory privacy and runtime checks remain required. Three stale diagnostic rows are reviewed by statements widened to publishedf9 before canonical regeneration. Public ACs remain open.

### Intermediate identity-governance amendment after R56

Intermediate identity-governance prerequisite after native75P+CI1P and original3ratchet failures: ADR required yes, amend existing ADR285; path backlog/decisions/285-native-phased-agent-workflows.md. RootRuling57 and independent design gate intermediate-identity-governance-design-review.md explicitly amend TASK36944.9 whole-Chat rule, preserving historicalR56. All-reference scanner unchanged; exact24named-owner equality table226/129/156 plus tightened ordinarypool equality19/10/17 totals245/139/173, zeroheadroom, missing/nonzero/zero dimensions included. All scopes counted; count-preserving within-owner/ordinarypool substitutions and dynamic introspection remain outside census proof. Only existing Tests/Architecture/test_chat_code_identity_ratchet.py may change: minimal accounting and actual addition/reduction/missing/zero/retirement controls, no production edits/scanner bypass/newissuer/relocation. Run actual3ratchets+fullrelevanttestmodule and focusedstatic/format, fresh implementation gate; reuse unchanged-production75native/coldpositives as recorded, notnewpolicy passes. Update QA and archive all57Rulings before reviewed intermediate push. All public ACs remain open; no merge/autoarm.

Intermediate local completion before actual-tree publication: Intermediate locallyqualified currentdevunion: sourcea0+e635/fc129, 9code/7metadata reviewed; diagnostic3rowcanonical/censuspass; corrected8expectedclassfields; realretention50P18.151s anddiagnostic/default/consent28P29.348s sixprivatechildrenstrict; governance24P1historyskip +remaining183P3Windowsskip/25deselections, prior203module+4screenpositive. Native75P+CI1P includes allunchangedcold/physical/sourcecontrols; original3identitycensusfailures retained. Ruling57exactADR24ownerequality+ordinary19/10/17/all245/139/173 expresslyamendsoriginal56/27/43; newfulltestmodule25P4.309s with independentSPEC/QUALITYPASS (no productionchange). Root57literalarchivepublished/alloldprefixesunchanged. InheritedSyntax/headroom/static/format andcountoffset/wrapper/local-v-hostlimits retained. QA note records lifecyclefakeapp-v-fulltimer distinction. Rootactualtree/preservation/review/soledevparent/privateCAS/push next, remaining14f–s andpublic5ACsOPEN; no merge/autoarm.
