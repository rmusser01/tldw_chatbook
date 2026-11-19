# Five perf guard failures: causal diagnosis and proposed scope

Read-only repository investigation; no source, budget, snapshot, manifest, index or HEAD edits. The three exact five-node arms are COMPLETE and exec sessions40814/85296/4836 have returned exit1. No running current-worktree test remains.

## Paired native evidence

Same original five pytest nodes, interpreter and conftest/private-profile bootstrap. Observation wrappers only serialize returned census values; original assertions remain unchanged.

| Check | HEAD022e3e5435 | dev3412c2b6a3 | dev88af9f9379 | Limit |
|---|---:|---:|---:|---:|
| Warm UI-ready tldw modules |1036|1033|1033|1031|
| Screen preimport modules |553|553|553|558|
| Screen preimport LOC |426322|426191|426189|426199|
| First-send warmup modules |99|94|116|94|
| First-send warmup LOC (diagnostic) |66593|62932|81908|none|
| Boot reconcile worker absent from inventory |yes|yes|yes|exact reviewed set|
| Recovery CSS fixture buttons |8 vs7|8 vs7|8 vs7|fixture mount assertion|

Results: HEAD5 failed31.30s; dev3412=3 failed2 passed42.79s; dev88af=4 failed1 passed41.62s. These are failures with causal data, not qualification passes.

## Attribution

1. UI-ready: exact added set versus BOTH devs is only Chat.console_prompt_mixins, Widgets.Console.console_prompt_mixins_view, Widgets.Prompts.prompt_mixin_editor. Nothing was shed. Two excess modules are inherited; the feature adds3. The latter two arrive through ConsolePromptsModal's eager view import; Widgets.Console.__init__ and UI.Console_Modules.prompts both eagerly import that modal. The core mixin values module is also needed by resident controller/store/configuration dataclasses and should not be hidden by a census exception.
2. Boot worker: exact same `_reconcile_console_after_attach`, group `console-attach-reconciliation`, starts on all three. Source matches dev. It performs required initial full reconciliation and source/attachment checks before the visit is usable; deferring it to arbitrary first interaction would change existing readiness/ownership semantics. This is a stale reviewed-worker inventory.
3. CSS fixture: actual current/dev recovery card has8 real actions, including TASK33621.4 `console-trace-change-policy`. The test still demands7 and fails BEFORE counting ancestor-scoped CSS rules. This is a stale complete-mount fixture, not proof the CSS rule budget is exceeded.
4. Preimport: branch growth is exactly131 LOC, ALL in the Library route; module sets are identical. Module attribution: library_prompt_browse_controller+5, UI.Library_Modules.library_prompts_state+2, library_screen+21, library_prompts_canvas+103. Dev3412 has only8LOC headroom. This is real feature growth, not a census mistake.
5. Warmup: branch's +5 versus3412 is exactly hook_permissions, console_plugin_context, console_pre_provider_stop, console_prompt_mixin_service, DB.console_prompt_mixins. First three were proven upstream lazy omissions by earlier UI16 evidence, but adding them consumes budget. Latest88af now includes those three itself, plus other warmup changes; branch-specific growth after rebase is the two mixin modules. The latest baseline's explicit console_agent_bridge warmup entry contributes17 modules/16243LOC even though native UI-ready census already has that bridge resident. New citation_key_first_use and console_send_start each add1. Latest warmup is116 before feature integration, so the likely merged count is118 before reductions; this is an estimate until rebased measurement, not a result.

## Minimal causal plan for root approval

No numeric limit increase, snapshot repin, timing relaxation or blanket import exclusion is justified.

- Correct the CSS mount fixture to assert the exact8 recovery action IDs and that all are mounted, then run its ORIGINAL selector-ratchet assertion. This retains anti-vacuity and catches missing/duplicate/unmounted controls.
- Add exactly the inherited attach-reconciliation name/group to the worker inventory with its initial-attach owner documented. Preserve worker execution, attachment generation and native-read retirement behavior. No general new worker permission.
- Repay UI-ready growth at the Prompts first-use seam: make ConsolePromptsModal a lazy Widgets.Console export; defer its coordinator imports to the actual modal/apply paths (TYPE_CHECKING for annotations, same concrete classes at runtime). This should remove modal/Browse/Mixins editor work until Prompts opens. Keep the cheap mixin value types on their actual native owners. Measure the complete UI-ready AND route payload afterward, because shared editor imports can move into the Library route rather than disappear. If shared editors migrate into route payload, defer their Library editor-only imports too; do not count cost moved between two guarded paths as saved. Preserve all real event routes and held Save/toggle/source ownership tests.
- For the131LOC Library breach, there is a small concrete first-use target: Widgets.Library.prompt_delete_confirmation_modal is238LOC, imported at module scope by the Library controller/screen and only needed for explicit deletion. Defer that modal import to the existing delete action, preserving exact request/items/confirmation and failure handling. This is a narrow actual payload reduction within the affected Prompts surface, not comment deletion to game LOC. Recheck the route guard jointly with the above prompt deferral before claiming sufficient reduction.
- Warmup needs measured cost removal after88af rebase. First remove the redundant explicit console_agent_bridge warmup entry only after pinning the observed native-ready residency (its17-module closure is already paid before the pass in production). This does not raise a guard or change ready timing. A further concrete first-use target is SSH transport: remote_session_registry currently eagerly imports RemoteSessionWorker, SessionStartError and TransportFailureKind solely needed by acquire/start; that pulls8 modules/11308LOC including the registry. Keep the registry/cleanup owner resident but defer transport/worker imports to acquire and remove the now-unnecessary explicit remote_session_worker warmup entry. Its cold closure is7 modules, enough in combination with the redundant bridge removal for estimated118-17-7=94. This is a proposed safe-boundary reduction, NOT yet tested or implemented. Keep public type compatibility, tombstones, cancellation, failure taxonomy, remote-session cleanup and native-tool advertisement unchanged; run exact registry/session controls plus actual first-send import guard/cycle guards. If the merged measured closure differs, stop and report it rather than ratcheting.

The SSH proposal is a named performance repayment outside the mixin source; approval is needed before that product edit. It should not broaden into general transport cleanup. The newer Automatic Library modules stay deliberately cold per upstream owner decision; do not re-add them to the warmup.

## Artifacts and ownership

- This report.
- /private/tmp/mixins-perf-probe.py (observation-only pytest driver).
- /private/tmp/mixins-perf-head022.log
- /private/tmp/mixins-perf-dev3412.log
- /private/tmp/mixins-perf-dev88af.log
- Corresponding pytest basetemps /private/tmp/mixins-perf-{head022,dev3412,dev88af}-tmp.
- Private source archives /private/tmp/mixins-perf-baseline3412 and /private/tmp/mixins-perf-dev88af. First archive command raced origin/dev update; its source was verified as88af via content comparison and renamed before ANY test ran. The3412 archive was then created with explicit immutable hash. Neither archive was source-edited.
-12 JSON observations under /private/tmp/mixins-perf-{head022,dev3412,dev88af}-*: UI-ready module list, boot workers/threads, preimport routes/modules/LOC, first-send entries/modules/LOC.

Exact five nodes: test_ui_ready_module_census::test_ui_ready_module_census_stays_at_the_pinned_size; test_boot_worker_census::test_boot_worker_and_thread_starts_stay_within_the_allowlist; test_textual_css_fastpath::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet; test_screen_preimport_payload_budget::{test_preimport_pass_payload_stays_within_budget,test_first_send_warmup_payload_stays_within_budget}, all under Tests/Performance.

Input hosted perf log is parent-owned. No new repository owned artifacts from this investigation.
