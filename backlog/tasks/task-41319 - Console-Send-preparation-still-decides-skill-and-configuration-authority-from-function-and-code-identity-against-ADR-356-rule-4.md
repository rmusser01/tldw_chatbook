---
id: TASK-41319
title: "Console Send preparation still decides skill and configuration authority from function and code identity, against ADR-356 rule 4"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- architecture
- console
- agents
priority: medium
references:
- backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
- tldw_chatbook/Chat/console_configuration_preparation.py
- tldw_chatbook/app_skill_wiring.py
- tldw_chatbook/Agents/activation.py
- Tests/Architecture/test_code_identity_ratchet.py
- https://github.com/rmusser01/tldw_chatbook/pull/3539
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40719 (PR #3539) recorded this as out of its scope. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** ADR-356 rule 4: "Orchestration must not inspect function objects, code/defaults/closures, descriptors, caller stacks/locals or coroutine implementation to establish authority." TASK-40719 removed three such checks, and its whole-package ratchet (`Tests/Architecture/test_code_identity_ratchet.py`) caps the rest without removing them. The Send preparation path still has several. If any checked function is wrapped, reloaded or patched, skill trust or the turn's configuration capture is silently refused or not recorded. The user's Send then fails or loses its skills for no reason they can see.

**Remaining sites on the Send path.**
- `Chat/console_configuration_preparation.py`:
  - `_resident_skill_wiring` (`:447-486`) accepts the wiring only if each of `_capture_console_skill_trust_source`, `_console_skill_metadata_current` and `_console_skill_source_current` has its recorded `__code__`, `__globals__`, no `__defaults__`/`__kwdefaults__`/`__closure__`, and its recorded `__wrapped__` (`:473-479`).
  - `_function_unchanged` guards (imported as `callable_source_current` from `Agents/activation.py`, via `Chat/console_ordinary_tree.py:10`) at `:655`, `:891`, `:1024`, `:1134` and `:2145`.
  - Further `__code__` comparisons at `:187`, `:206`, `:210` and `:364`.
- `app_skill_wiring.py`:
  - `_console_skill_metadata_current` (`:73`) checks the identity of `compact_model_bar`'s module, spec and bindings, and `:124-128` checks function code, globals, defaults and closure.
  - `_capture_console_skill_trust_source` (`:175`) requires the app factory's recorded `__code__`, `__globals__`, no defaults and no closure (`:198-208`).
  - The lazy trust factory is checked by code, globals, defaults, free vars and closure cell (`:329-339`), and functions by `__code__`/`__globals__` (`:399-401`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Send whose skill-trust, skill-metadata or configuration-capture collaborators are wrapped (as a decorator or a test patch would) captures and records the same skill trust and configuration as the stock path; tests show this and are red on dev
- [ ] #2 Each listed check is replaced by a decision from runtime-owned registration and current state, or kept with a written ADR-356 exception naming the attack it blocks
- [ ] #3 The code-identity ratchet is lowered to the new counts and still fails on any increase
<!-- AC:END -->

## Implementation Plan

ADR required: no. ADR-356 rule 4 already decides that function, code, default, closure and descriptor identity establish no authority; this applies it to the Send-preparation sites TASK-40719 left, the way TASK-40719 did (publication + slot). No site needs a KEEP exception (see the table).

**Status: design round; waiting for the owner's decisions D1-D3 before any product change.**

Reproduced on the base tree (fbf1c9ea74 = dev 5e3c8b2bd4 product code): `Tests/Chat/test_console_send_wrapped_skill_collaborators.py` drives a real cold received Send with a managed skill. `stock` passes end to end (turn accepted, trust built, `_NativeSkillTrustPublication` recorded). Wrapping `_capture_console_skill_trust_source`, `_console_skill_metadata_current`, `_console_skill_source_current` (in `app_service_wiring`) or `console_configuration_capture._capture_skill_context_from_records` with `functools.wraps` fails all four the same way: `capture_console_received_configuration_preparation` returns None ("cold Send preparation declined"). By code reading, the received route then falls back to the queue dispatch, and the mounted precapture raises "Console preparation source unavailable; Send again." (`UI/Console_Modules/wiring.py:810`).

Per site (line numbers verified on this tree; P = `Chat/console_configuration_preparation.py`, S = `app_skill_wiring.py`):

| Site | What the identity check blocks | Decision |
|---|---|---|
| P:447-486 `_resident_skill_wiring` (rows :473-479, capsule call :485) | A replaced proof helper being called and forging a "stock" verdict, so the off-loop cold-trust build runs on an unvetted source | REPLACE: module presence (`sys.modules`, `ModuleType`) only; call the helpers. Trust is decided inside `_capture_console_skill_trust_source` by the app's registered slots and runtime state |
| P:530-536 `_skill_catalog_sources_current` capsule calls (not listed; needed for AC #1 `skill_capture`) | A replaced catalog/capture function projecting forged skill records | REPLACE: keep the instance-override check (`_CONSOLE_SKILL_CATALOG_METHODS` in `vars(local)`), drop the capsule calls |
| P:187 `_ready_published_plugin` property/getter code | A patched `plugin_service` property whose value differs from the `_plugin_service` slot | REPLACE: `type(local) is LocalSkillsService` + read the slot (the slot is the registration the getter returns) |
| P:205-210 plugin factory `__func__`/reader `__code__` | Invoking a foreign factory, or adopting a plugin service the app did not publish | REPLACE: factory slot identical to the one the app registered when it built the stack (D3), then read `vars(app)["_plugin_service"]` |
| P:364 disabled-builtins loader `func`/`__code__` | A foreign loader doing IO or reporting a different disabled set off-loop | REPLACE: loader slot identical to the app's registration (D3) |
| P:629-655 `_first_library_source_current` (`__wrapped__`, `__func__`, `_function_unchanged`) | A patched store/repository feeding a forged first durable library policy into the turn | REPLACE: keep module/type/instance-slot checks; the callers already require session/commit/continuation/fingerprint/effects state and `holder.snapshot is result.snapshot` equal to the frozen policy |
| P:891 `_first_library_coordinator_current` | A patched coordinator whose live read would differ from the receipt standing in for it | REPLACE: the coordinator's own registration (`_holders[session.id] is registered`, holder and conversation id current); drop descriptor + `_function_unchanged` |
| P:1024 `_trust_source_records_current` / `trust_functions` | In-place mutation of the trust source's two private closures | REPLACE: delete; `_ConsoleSkillTrustSource` is an immutable NamedTuple only the preparation holds |
| P:1134 `_skill_publisher_guard_current` (bindings + functions) | Swapping the trust getter/builder between build and receipt | REPLACE: keep instance fields + environment selector; drop class-dict bindings and `_function_unchanged`. The receipt is already gated on the active publication marker and the slots (TASK-40719) |
| P:2145 `_same_mcp_source_entry` | In-place code mutation of a captured MCP source callable | REPLACE: `value is current[0]` (the slot's current value) only |
| S:73-130 `_console_skill_metadata_current` | A reloaded/replaced `compact_model_bar` whose capsule checker always says "stock" | REPLACE: delete; with the capsule calls gone from this path nothing calls it |
| S:198-208, :417-424 app `_create_deferred_startup_task` code | A non-stock app startup racing the cold build | REPLACE: `type(app) is app_namespace["TldwCli"]` + runtime/app state already checked |
| S:221-222, :408-409 capsule calls over wiring/helper/scope/local/config sources (not listed; the wiring capsule's binding rows name the wrapped helpers) | Any replaced function in the skill chain | REPLACE (D1): drop on this path; the slot, runtime, config-identity, environment and thread checks stay |
| S:329-339, :494-499 lazy trust factory code/closure cell | `local.trust_service` served by a factory other than this app's trust getter | REPLACE (D3): factory slot identical to the app's registration; post-build `local.trust_service is` the app winner is already enforced by `finish_native_skill_trust_publication` |
| S:394-406 support-row helper identity | A replaced metadata/source helper called mid-build | REPLACE: delete with the capsule calls |

Steps:
1. Keep the new test (all 5 cases) as the AC #1 evidence; it is red on base for the reason above.
2. Apply the table; delete what becomes unused (`_CONSOLE_SKILL_TRUST_FACTORY_CODE`, `_STOCK_PLUGIN_SERVICE_PROPERTY`, `_STOCK_PUBLISHED_PLUGIN_READER`, `trust_functions`, `_console_skill_metadata_current`), keeping records other modules still read (`_CONSOLE_SKILL_APP_SOURCE`, the skill capsules used by `UI/Console_Modules/skill.py`).
3. Update tests that pin the old refusal: `test_resident_proof_rejects_changed_helper_before_invoking_it`, the `helper_binding`/`helper_body`/`metadata_body` cases of `test_console_skill_trust_stock_owned.py` (D2), the plugin-reader tests in `test_console_configuration_preparation.py`, `test_console_managed_received_acceptance.py:198`.
4. Lower `PINS` in `Tests/Architecture/test_code_identity_ratchet.py` to the measured counts.
5. Covering files base vs head, preflight, `Tests/Architecture` serially. The new test should join the `console-p0-gate` lane (owner adds it).
