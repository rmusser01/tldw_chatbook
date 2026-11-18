---
id: TASK-41768
title: "Importing tldw_chatbook.app loads 688 modules, at the boot ratchet's limit, since ChaChaNotes_DB imports chachanotes_sql_steps eagerly"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- performance
- architecture
priority: medium
references:
- Tests/Performance/test_app_import_weight.py
- Tests/Performance/boot_budget_snapshots/boot_import_modules.txt
- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/DB/chachanotes_sql_steps.py
- backlog/decisions/097-boot-budget-ratchets.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size` fails at dev 908e9840aa (2026-11-18). It failed at 97f8c58c0c and 47a05413cb too: "import tldw_chatbook.app loaded 688 tldw_chatbook modules (ratchet limit 688)" (`assert 688 < 688`).

The pinned snapshot is `boot_budget_snapshots/boot_import_modules.txt`, last refreshed in 592425f087 (2026-11-08). Against it, the test lists four new eager modules and three shed ones. The new ones are:
- `Agents`;
- `Agents.agent_models`;
- `DB.chachanotes_sql_steps`;
- `app_services`.

The test passed at 45b1949ca3 and first failed at 97f8c58c0c. Between them, 3911d7df50 (TASK-41310 part A, 2026-11-17, merged with #3592) added `DB/chachanotes_sql_steps.py`, the newest of the four. `ChaChaNotes_DB.py` imports it at module top (`:150`).

ADR-097 says `MAX_TLDW_MODULE_COUNT` never rises. The cost must be deferred or shed, or the owner records an exception in the ADR's ledger. Related: TASK-21248 questions the count budget itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev with MAX_TLDW_MODULE_COUNT unchanged, or with an owner exception recorded in ADR-097's ledger
- [ ] #2 The boot snapshot is refreshed in the same change to match the resident set
- [ ] #3 The UI-ready module census does not rise
<!-- AC:END -->
