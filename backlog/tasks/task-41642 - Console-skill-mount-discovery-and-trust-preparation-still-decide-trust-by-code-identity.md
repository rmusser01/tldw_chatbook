---
id: TASK-41642
title: "Console skill mount discovery and trust preparation still decide trust by code identity"
status: In Progress
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
- [ ] #1 Skill mount discovery and trust preparation decide from the app's registrations, not code identity
- [ ] #2 A wrapped collaborator gets the same result as stock, and a foreign one is still refused
- [ ] #3 The code-identity ratchet is lowered by the removed reads
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
