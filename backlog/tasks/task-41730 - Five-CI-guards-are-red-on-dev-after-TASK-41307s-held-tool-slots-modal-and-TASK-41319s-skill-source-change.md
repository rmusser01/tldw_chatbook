---
id: TASK-41730
title: >-
  Five CI guards are red on dev after TASK-41307's held tool slots modal and
  TASK-41319's skill-source change
status: Done
assignee:
  - '@claude'
created_date: '2026-11-18 08:01'
labels:
  - console
  - ci
  - tests
  - regression
priority: high
references:
  - tldw_chatbook/Widgets/Console/console_held_tool_slots.py
  - tldw_chatbook/Widgets/Console/console_cleanup_release.py
  - tldw_chatbook/Widgets/Persona_Widgets/actor_pack_recovery_notice.py
  - tldw_chatbook/css/features/_console.tcss
  - Tests/UI/test_console_modal_dismissal.py
  - Tests/UI/modal_wide_tier_registry.py
  - Tests/UI/test_widget_css_consolidation.py
  - Tests/UI/test_component_pattern_governance.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-41307 (PR #3590, merged to dev by the 2026-11-18 owner bypass) added `ConsoleHeldToolSlotsModal` and an explanation-and-Release group for unproven cleanup records, and TASK-41319 (PR #3583) moved the skill source's authority from the wiring module's pointer to the app's slots. Five guards that run on every PR now fail on dev because of them. Four come from TASK-41307: the Console modal inventory (two nodes in PR Fast Lane admission-sensitive 2), the modal wide-tier coverage contract (UI Fast Lane), the class-level CSS allowlist (UI Fast Lane) and the HARD ZERO Python style ratchet (UI Fast Lane). The fifth comes from TASK-41319: the recovered-suites node `test_actual_stock_held_pointer_refuses_before_foreign_access[skill]` still swaps the pointer TASK-41319 stopped reading, so the build runs and reads the foreign pointer. The burn-down's integration run tested its own 50 files, not these guards, and its tip CI run (44225401445) showed them red next to failures that were already red on dev. Every PR's required check is red until they pass. The CSS allowlist node also names `ActorPackRecoveryNotice.DEFAULT_CSS` (TASK-41114, round 15), which needs the same one-word fix, so that node can pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two Console modal inventory nodes, the wide-tier coverage node, the class-level CSS allowlist node, the Python style ratchet and the held-pointer skill node pass, and each failed on dev before the fix
- [x] #2 The held tool slots modal and the release group look and behave as before: their own test files pass
- [x] #3 The CSS bundle reproduces from its sources (preflight passes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (guard registrations and a CSS move; no behaviour change).

1. Name `ConsoleHeldToolSlotsModal` in the Console root's inventory-only modal list (it adopts the safe-dismissal contract) and in the wide-tier skip list (a compact 76-column list).
2. Rename both class-level `DEFAULT_CSS` blocks to `BUNDLED_CSS` and rebuild the sheets.
3. Replace the release group's `styles.margin = 0` with a two-class CSS rule.
4. Retarget the held-pointer skill case to a foreign app slot, as TASK-41319's D2 did for its siblings.
5. Run the guards, both widget files and the held-pointer file at `-n 4`, then preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five guards pass. The held tool slots modal and the release group render from the same rules, now in the bundle.

- **Modal inventories.** `ConsoleHeldToolSlotsModal` joins `_ROOT_INVENTORY_ONLY_MODAL_TYPES` (it adopts `SafeModalDismissMixin`), and the reachable count goes 80 -> 81. It joins `MODAL_WIDE_TIER_SKIPPED` as a compact list (base width 76, like the 70-72 column dialogs there).
- **CSS.** `ConsoleHeldToolSlotsModal` and `ActorPackRecoveryNotice` (TASK-41114, round 15; the same node names it) use `BUNDLED_CSS`. The sheets are rebuilt.
- **Style ratchet.** The release group's `styles.margin = 0` became a second class with a two-class rule in `_console.tcss`, so the group keeps the protected-click class and sits flush.
- **Held pointer.** For the skill route, the test now replaces the app's `_local_skills_service` slot with the foreign object while the read is held. That slot is what TASK-41319 made decide (ADR-356 rule 4). The collections route still swaps the module pointer. Negative control: putting the original slot back fails the case.
- **Evidence (2026-11-18).**
  - **Red on dev, green on the parent.** Each node was run singly on dev 908e9840aa and on 45b1949ca3, the parent of the burn-down merges. The two modal-inventory nodes, wide tier, style ratchet and held-pointer skill node fail on dev and pass on the parent. The CSS node fails on both, for `ActorPackRecoveryNotice`.
  - **After the fix.** All 8 targets pass at `-n 4`: 335 passed. That covers the whole modal dismissal, wide tier, CSS consolidation, component governance, held tool slots, cleanup release and held-pointer files, plus the Personas recovery-notice node. Preflight passes.
- **Still red on dev, not from this stack** (red on 45b1949ca3 too): `test_layer_direction`, `test_console_semantic_mutation_inventory`, the Roleplay header and item-list files, `test_roleplay_stylesheet`, `test_personas_workbench`'s human-identity nodes, and the Widgets app-lookup census (TASK-41654).
<!-- SECTION:NOTES:END -->
