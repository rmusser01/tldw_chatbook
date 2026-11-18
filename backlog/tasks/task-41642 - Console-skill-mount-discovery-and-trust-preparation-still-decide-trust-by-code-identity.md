---
id: TASK-41642
title: "Console skill mount discovery and trust preparation still decide trust by code identity"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
- architecture
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/skill.py
- tldw_chatbook/Widgets/compact_model_bar.py
- Tests/Architecture/test_code_identity_ratchet.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** ADR-356 rule 4 says no authority comes from code identity. TASK-41319 (owner decision: the Send path only) removed the identity checks on the Console Send path; a wrapped or patched collaborator now records the same trust receipt as stock. These checks remain:
- the source-capsule definitions `_CONSOLE_SKILL_CONTEXT_SOURCE`, `_CONSOLE_SKILL_HELPER_SOURCE`, `_COMPACT_MODEL_CONFIG_SOURCE`, `_WIDGET_SOURCE` and `_CONSOLE_SKILL_CATALOG_SOURCE`;
- `_console_skill_source_current`, and the `_source_current`/`_function_current` helpers in `Widgets/compact_model_bar.py`;
- the identity checks in `UI/Console_Modules/skill.py` mount discovery (the `_CONSOLE_SKILL_ENTRY` identity, the capsules and `_DISCOVERY_SOURCE`);
- the controller-source check and the `workers`/`app`/`run_worker` class-method identity checks in `_capture_console_skill_trust_preparation`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skill mount discovery and trust preparation decide from the app's registrations, not code identity
- [x] #2 A wrapped collaborator gets the same result as stock, and a foreign one is still refused
- [x] #3 The code-identity ratchet is lowered by the removed reads
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR-356 rule 4 already decides this; the change applies TASK-41319's pattern (D2/D3: the app's registrations, slots and per-instance override refusals) to mount discovery and the discovery-side trust preparation.

Per site (`K` = `UI/Console_Modules/skill.py`, `S` = `app_skill_wiring.py`):

| Site | Decision |
|---|---|
| K `_fetch_console_skill_context`: `_CONSOLE_SKILL_ENTRY` identity, `_CONSOLE_SKILL_WIRING_SOURCE`/`_CONSOLE_SKILL_HELPER_SOURCE` capsules | REPLACE: call the wiring module's current preparation helpers (a wrapped helper runs); the discovery owner's `current()` and the preparation's own `current`/`source_current` decide |
| K `_capture_console_skill_discovery`: `_DISCOVERY_SOURCE` capsule, proof-helper code rows, popup partial function identity | REPLACE: the registrations decide (the popup partial is bound to a view of the bound type whose `_skill` is this controller, the app's live runtime is attached to that view at the same generation, closing flags, per-instance overrides); returns the `current` callable |
| K `bind_console_skill_discovery` capsule build | REPLACE: bind only the view type and the context classes; the first binding still wins |
| K `_capture_console_skill_context_read`: `_DISCOVERY_SOURCE`/`_CONSOLE_SKILL_CATALOG_SOURCE` capsules, loader `__code__`, policy factory/provider code and closure | REPLACE: the loader must be the app's registered callback (`_local_skills_callbacks[2]`, TASK-41319 D3); the policy keeps its types, per-instance refusals, the policy module's registry and its captured slots |
| S `_console_skill_source_current` | DELETE |
| S `_capture_console_skill_trust_preparation` controller-source check | DELETE (the parameter goes) |
| S `_capture_console_skill_trust_preparation` `workers`/`app`/`run_worker` class-attribute identity | REPLACE: per-instance refusals, `app.app is app`, the worker-manager slots; the preparation starts the worker with `app.run_worker` and keeps its post-start worker checks |
| Capsule definitions with no reader left: S `_CONSOLE_SKILL_CONTEXT_SOURCE`/`_WIRING_SOURCE`/`_ENTRY`/`_HELPER_SOURCE`/`_FUNCTIONS`; `Skills_Interop/local_skills_service.py` `_CONSOLE_SKILL_CONTEXT_SOURCE`/`_FUNCTIONS`/`_CATALOG_FUNCTIONS`/`_CATALOG_SOURCE`; `Skills_Interop/skills_scope_service.py` `_CONSOLE_SKILL_CONTEXT_SOURCE`/`_FUNCTIONS`; `app_service_wiring._STOCK_DISABLED_BUILTINS_READER` | DELETE |
| `Widgets/compact_model_bar.py` `_source_current`/`_function_current`, `_WIDGET_SOURCE`, `config._COMPACT_MODEL_CONFIG_SOURCE` | The skill path stops reading them. They still back the compact model bar's own setup read (`_sources_current`), not skill discovery or trust; that route's tests pin code-identity refusals (`custom_body`, `body`, `malformed`), so it is reported as a follow-up rather than changed here |

Steps: 1) a test drives real mount discovery with each collaborator wrapped by `functools.wraps` (red on base where it was declined) and with a foreign/per-instance one (refused); 2) apply the table; 3) retarget tests that pin a code-identity decline to runtime-state changes (`test_console_skill_discovery_boundaries.py`, the genuine-capture `popup_contract`/`held_*` regressions, the stock-owned helper-name pins); 4) lower the ratchet by the measured removed reads; covering files base vs head, `Tests/Architecture` serially, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Skill mount discovery and its trust preparation now decide from the app's registrations and current state (ADR-356 rule 4), TASK-41319's pattern; no ADR exception was needed and no ADR changed.

- **Discovery (`UI/Console_Modules/skill.py`).** `_capture_console_skill_discovery` returns the `current()` callable. It keeps every registration/state check (the controller's popup partial is bound to a view of the bound type whose `_skill` is this controller, the app's live runtime is attached to that view at the same generation, closing flags, loop/thread, per-instance overrides of the controller methods and of the view popup) and drops the `_DISCOVERY_SOURCE` capsule, the proof-helper code rows and the popup partial's function identity. `bind_console_skill_discovery(view_type)` binds only the view type and the context classes (first binding still wins).
- **Fetch.** `_fetch_console_skill_context` calls the wiring module's current `_capture_console_skill_trust_preparation`/`_prepare_console_skill_trust_service` (so a wrapped helper runs) instead of the `_CONSOLE_SKILL_ENTRY` identity and the wiring/helper capsules; the owner's `current()` and the preparation's own source decide before and after.
- **Context read.** `_capture_console_skill_context_read(app, service, wiring)`: the loader must be the callback the app registered for its stack (`_local_skills_callbacks[2]`, TASK-41319 D3); the policy keeps its exact types, per-instance refusals, the policy module's `CAPABILITY_REGISTRY` and its captured `_engine`/`_state_provider` slots. Code/closure reads of the provider and the catalog capsule are gone.
- **Trust preparation (`app_skill_wiring.py`).** `_capture_console_skill_trust_preparation(app, service, owner_current)` drops the controller-source parameter/check and the `workers`/`app`/`run_worker` class-attribute identity; it refuses those names in the app's `__dict__`, requires `app.app is app` and keeps the worker-manager slots. `_prepare_console_skill_trust_service` starts the worker with `app.run_worker` and awaits `worker.wait()`, keeping its post-start worker checks.
- **Deleted (no reader left).** `_console_skill_source_current`; in `_install_skill_source` the `_CONSOLE_SKILL_CONTEXT_SOURCE`/`_WIRING_SOURCE`/`_ENTRY`/`_HELPER_SOURCE`/`_FUNCTIONS` capsules and the captured Textual callables (only `_SkillWorkerManager` and the two slot tuples remain); `local_skills_service` `_CONSOLE_SKILL_FUNCTIONS`/`_CONTEXT_SOURCE`/`_CATALOG_FUNCTIONS`/`_CATALOG_SOURCE` (`_CONSOLE_SKILL_CATALOG_METHODS` stays); `skills_scope_service` `_CONSOLE_SKILL_FUNCTIONS`/`_CONTEXT_SOURCE`; `app_service_wiring._STOCK_DISABLED_BUILTINS_READER`.
- **Review round (2026-11-18).** The context read accepted any callable policy `_state_provider` and ran it off the UI loop (base refused a foreign one). `TldwCli.__init__` now builds the provider itself and records it as `_service_policy_state_provider` (the enforcer is constructed with it; same behaviour as `from_runtime_policy_context`), and the read accepts only that registered provider. New foreign case `foreign_policy_provider` (red on 33de5eaab: the foreign provider ran on the worker thread; green after). `app.py` grew by 4 lines; its size row is raised by exactly 4.
- **Not changed: the compact model bar's own setup.** `Widgets/compact_model_bar.py` `_source_current`/`_function_current`/`_WIDGET_SOURCE`/`_PROOF_HELPERS` and `config._COMPACT_MODEL_CONFIG_SOURCE` are no longer read by any skill path; they still gate the compact bar's own off-loop provider read (`_sources_current`), which is neither skill discovery nor trust. Its tests pin code-identity refusals (`test_compact_setup_ownership.py` outcomes `custom_body`, `body`, `queued_body`, `malformed`; `_compact_original_config_gate.py` pins the capsules), so it is reported as a follow-up.
- **Tests.** New `Tests/UI/test_console_skill_discovery_wrapped_collaborators.py` (subprocess, real app + ChatScreen): with the preparation helpers, the trust source, `_plain_fields`, a controller method, the popup function, `LocalSkillsService.get_context` or the disabled-builtins reader wrapped, discovery still builds trust and reads skills off the UI loop like stock (8 cases red on base: "trust built on the UI loop"/"skills read on the UI loop"); a controller or view per-instance override or an unregistered loader keeps the read on the UI loop and the foreign callback never runs off it. Retargeted (D2-style): `test_console_skill_discovery_boundaries.py` (capsule cases become owner/prepared-source changes; the second binding keeps the bound view), `test_console_skill_discovery_genuine_capture.py` (`held_*` become per-instance overrides of the view popup/controller methods plus a replaced `_skills_scope_service` slot for the held worker, as TASK-41319 did for `helper_body`; `popup_contract` overrides the popup on the screen instance), `test_console_skill_context_owned.py` (an in-place `get_context` body change becomes an instance override), signature updates in `test_console_skill_capture_config_rebuild.py`, the deleted helper dropped from `test_console_skill_trust_stock_owned.py`'s pins.
- **Mutation.** Dropping the view-popup instance check and the loader registration check fails `instance_view_popup` and `foreign_loader`.
- **Ratchet.** `test_code_identity_ratchet.py` PINS -109: `__func__` 8, `__code__` 14, `__defaults__` 11, `__kwdefaults__` 20, `__closure__` 21, `cell_contents` 9, `__globals__` 17, `__wrapped__` 9. Chat ratchet unchanged.
- Files: `UI/Console_Modules/skill.py`, `app.py` (provider registration), `app_skill_wiring.py`, `app_service_wiring.py`, `Skills_Interop/local_skills_service.py`, `Skills_Interop/skills_scope_service.py`, `UI/Screens/chat_screen.py` (bind call); tests above; `Tests/Architecture/test_code_identity_ratchet.py`; lesson in `backlog/docs/lessons-testing-evidence.md`.
- **New-test evidence.** `test_console_skill_discovery_wrapped_collaborators.py` against base product: 8 failed (every wrapped case: trust built and/or skills read on the UI loop), 4 passed (stock, 3 foreign); head 12 passed. Serial: 12 passed in 135 s at load ~52-66 (alone).
- **Evidence (2026-11-18, load 50-90, base = detached dev d0486978dd).** Covering fast set (17 files incl. both new ones, `-n 4`): base 28 failed/155 passed, head 29 failed/176 passed; the one head-only node was `test_a_refused_acknowledged_enter_repaints_its_refusal` (a class-level foreign gate, retargeted above; passes alone). Slow skill set (genuine capture, stock-owned, config rebuild, context owned, managed acceptance, `-n 4`): base 2 failed (`send_enter`/`send_button`, as TASK-41319 recorded)/28 passed; head after the test-API updates: the same 2 fail, `custom_affinity`/context-owned/`shutdown` pass on rerun (`-n 3`; `shutdown` had hit `sqlite_retirement_incomplete:AgentRuns` once under `-n 4`). Wider net (77 `Tests/UI/test_console_*` send/received/queue/skill/preparing/composer files + 6 more, `-n 4`): base 288 failed/1142 passed, head 290 failed/1161 passed (200 of the shared failures are `RecoveryRequired: raw_source_selection_changed` in `build_test_app_config`, the fixture's config participant, same on both). Head-only nodes: 8; rerun at `-n 4` 5 passed; `test_enter_after_a_tab_click_during_a_sync_pass_sends_to_the_clicked_chat` and `test_console_blocked_send_retains_exact_recovery_after_runtime_custody` fail on base too; `test_an_inspector_opened_after_chat_switches_matches_one_open_throughout` passes alone on head and base. `Tests/Architecture` serially: 4 failed/1475 passed; rerun of the 4 on base and head: the same 3 fail on both (`test_reviewed_diagnostic_changes_are_metadata_only`, two timer-path inventory tests), the 4th passed. `./scripts/preflight.sh` passed. Ruff: no new findings in touched files (base vs head diff empty).
<!-- SECTION:NOTES:END -->
