---
id: TASK-41800
title: "Dev is red on 25 nodes at 908e9840aa, introduced by round-15/16 merges"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 16:45
labels:
- ci
- tests
- follow-up
priority: high
references:
- Tests/UI/test_console_modal_dismissal.py
- Tests/UI/test_modal_wide_tier.py
- Tests/UI/test_component_pattern_governance.py
- Tests/UI/test_widget_css_consolidation.py
- Tests/UI/test_roleplay_header.py
- Tests/UI/test_roleplay_item_list_app.py
- Tests/Chat/test_console_semantic_mutation_inventory.py
- Tests/Architecture/test_layer_direction.py
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Performance/test_app_import_weight.py
- Tests/Performance/test_screen_preimport_payload_budget.py
- Tests/Performance/test_app_wiring_held_pointer_currentness.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several round-15/16 PRs merged with CI red (#3583, #3590, #3592, #3599, #3600, #3606), and dev's required check failed for every PR after them. On #3536's run against dev 97f8c58c0c, 27 of 28 failures were dev's own: the same nodes failed with identical messages on #3614, and the two Perf Guard rows also failed on fu/r16-filing. Twenty-five of them still failed locally on dev 908e9840aa. Fix each at its root, without raising any ratchet pin, budget or limit, so the required check can go green again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Console modal inventory, wide-tier, Python-style and CSS-consolidation nodes pass with ConsoleHeldToolSlotsModal registered and its CSS bundled
- [x] #2 The 12 Personas list-geometry nodes pass, and the Actor Pack recovery notice still shows only when an import is interrupted
- [x] #3 The three semantic-mutation inventory nodes pass, with Personal Context's dynamic SQL reviewed and no rule loosened
- [x] #4 Layer direction and the Widgets app-lookup census pass without raising the pin
- [x] #5 The boot-import, first-send warm-up and skill held-pointer nodes pass, with no limit raised and no snapshot refreshed
- [x] #6 Preflight and the module-size and code-identity ratchets pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Five independent fixes, one commit group each:

- **A, #3590 / TASK-41307.** The new `ConsoleHeldToolSlotsModal` was never registered.
  - It is now in the Console modal inventory (root-inventory-only, with safe dismissal) and in the wide tier (cap 120, like its sibling list modals).
  - `console_cleanup_release.py` uses a token-backed class in place of `styles.margin = 0`.
  - `DEFAULT_CSS` became `BUNDLED_CSS`, and the bundle was regenerated.
- **B, #3599 / TASK-41114.** The product was right; the fixture was stale.
  - `PersonasScreen` mounts the recovery notice when `blocked_intent_ids` is truthy. The shared `mock_app_instance` is a bare MagicMock, so that was always truthy and the 5-row notice sat above the list.
  - The fixture now states `()`. The notice's CSS is bundled.
- **C, 1534a6f3d4 / d32a546f54.** `repository_keys.py` split `_DROPPABLE` into a literal `_LOCAL_POINTERS` mapping and a frozenset. The inventory can then derive both dynamic SQL tables, and both sites gained review entries.
- **D.**
  - `keep_legacy_custom_2_endpoint` and its constants moved from `Chat/console_provider_endpoints.py` to the kernel module `Utils/legacy_custom_2_endpoint.py`, which `config.py` imports.
  - The notice's two `getattr(app, ...)` lookups became plain attribute reads.
- **E.**
  - `ChaChaNotes_DB.py` imports `chachanotes_sql_steps` inside the two migration steps that use it, so boot is now 687 of 688.
  - Five modules the built Console already holds left the first-send warm-up: lazy and TYPE_CHECKING imports in the Canvas modules and `console_ordinary_tree`, and `app_served` dropped from `FIRST_SEND_MODULES`. The warm-up is now 94 of 94.
  - `app_skill_wiring.source_current()` again refuses a replaced wiring pointer before any wiring read. #3583 had removed that check along with its code-identity capsules.

Not fixed here: `test_deep_link_library_route_lands_its_canvas_over_restored_state` (ScreenStackError at shutdown) fails 2 of 3 runs on dev and on this branch. Open PR #3621 already guards that exact read in `_library_emergency_return_eligibility`. QA `test_project_chat_uat[fenced]` failed once in CI and passes locally.
<!-- SECTION:NOTES:END -->
