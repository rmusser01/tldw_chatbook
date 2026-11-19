---
id: TASK-36231
title: >-
  Tests: the ADR-097 UI-ready snapshot's contents drift unchecked; re-pin it and
  flag membership drift
status: To Do
assignee: []
created_date: '2026-10-22 19:02'
labels:
  - performance
  - testing
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-33620.15.5 (2026-10-22). Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt (last written by 42a5106c41, 2026-10-16) lists 1022 modules, but test_ui_ready_module_census.py enforces only the count against MAX_TLDW_MODULES_AT_UI_READY, never the contents, so the list goes stale silently. On origin/dev e2cffbed60, importing tldw_chatbook.app and the initial route (UI.Screens.chat_screen) in a fresh interpreter loads 15 tldw_chatbook modules the snapshot does not list: Chat.approval_constants, Chat.console_configuration_capture, Chat.console_hook_review, Chat.console_preparation_reads, Chat.console_received_intent, Chat.console_received_turn, Chat.console_token_rate, LLM_Provider_Catalog.models_dev_catalog, UI.Console_Modules.attach_visit, UI.Console_Modules.context_spend, UI.Console_Modules.inspector_data, UI.Console_Modules.pricing_display (imported at module scope by the listed console_spend_projection), UI.Navigation.surface_swap_guard, app_collections_wiring, app_skill_wiring. Anything that reads the snapshot's contents inherits the gap: TASK-33620.15.5's harness send guard treated pricing_display as lazy and failed on correct code 1 run in 26 until it united the snapshot with a live import census. The ratchet's culprit-naming diff (ADR-097) also compares against these contents, so a stale list names the wrong culprits. (Five of the 15, plus UI.Console_Modules.response_rules, are also in Chat/console_first_send_modules.FIRST_SEND_MODULES: dictionary hits for the warm-up, harmless.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ui_ready_modules.txt is re-pinned with scripts/update_boot_budget_snapshots.py --only ui-ready on a current dev, and the count row is not raised
- [ ] #2 The census (or a sibling check on a PR-required lane) fails when a module resident at UI-ready is missing from the snapshot, or the snapshot lists one that is no longer resident, naming the modules
<!-- AC:END -->
