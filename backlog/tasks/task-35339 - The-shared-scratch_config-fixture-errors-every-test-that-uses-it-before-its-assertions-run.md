---
id: TASK-35339
title: The shared scratch_config fixture errors every test that uses it before its assertions run
status: To Do
assignee: []
created_date: 2026-10-17 11:34
dependencies: []
labels:
- testing
- tech-debt
- follow-up
priority: medium
references:
- Tests/Internal_Prompts/conftest.py
- Tests/private_profile.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
- backlog/decisions/126-complete-local-backup-and-recovery.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-33002.17 review, round 2 (2026-10-17). `scratch_config` in Tests/Internal_Prompts/conftest.py is the shared fixture that points the app at a throwaway config file. Tests/UI, Tests/Agents, Tests/RAG and Tests/Web_Scraping re-export it from their own conftests. Its `write()` sets `TLDW_CONFIG_PATH` to a new per-test file and then reloads the config. By then the config source is already bound to the process's first path, and ADR-126's admission refuses the change by design: `_register_raw_participant` in Backup_Recovery/raw_participants.py raises `RecoveryRequired('raw_source_selection_changed')` when an owner's selected path moves. So every test that asks for the fixture errors at setup and never reaches its assertions.

Measured at 85bba8bb41 (branch p9-w5-t14) under the repo's scratch-profile runner, xdist loadfile: 101 setup errors in 12 files, every one raised from the fixture's `write()`.
- Tests/UI: test_settings_image_gen_panel.py 32, test_internal_prompts_panel_editing.py 7, test_internal_prompts_panel.py 4, test_internal_prompts_search_survives_save.py 1. The 14 tests in these files that passed do not ask for the fixture.
- Elsewhere: Tests/Internal_Prompts test_resolver.py 16, test_document_generation_migration.py 10, test_authoring.py 7, test_summarization_migration.py 6, test_prompt_improvement_prompts.py 4; Tests/Agents/test_agents_internal_prompts.py 6; Tests/Web_Scraping/test_websearch_internal_prompts.py 5; Tests/RAG/test_reranker_internal_prompts.py 3.

The same run found 17 more failures with the same refusal in five files that re-point `TLDW_CONFIG_PATH` by hand rather than through the fixture: Tests/Evals/test_eval_orchestrator_db_path.py 8, Tests/Persona_Buddy/test_persona_buddy_config_projection.py 5, Tests/test_user_data_dir_isolation.py 2, Tests/Agents/test_agents_internal_prompts.py 1, Tests/Wizards/test_first_run_setup_wizard.py 1 (test_summary_footer_shows_the_effective_config_path).

No PR lane runs any of these files, so nothing turns red. Only test.yml's Core Tests (pushes to main, dispatch) and nightly-deep (`pytest ./Tests/`) reach them. TASK-33370 fixed this failure class for the DB, Personas and config suites at their shared seam; this fixture was not in its set. TASK-33002.17 converted one test of test_settings_image_gen_panel.py (test_backend_row_controls_visible_and_clickable_under_real_css) to `@private_profile_test`, with the real config writer storing its keys in the case's private profile. That is one working shape. The fixture still errors the file's 32 other users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test that asks for the shared scratch_config fixture reaches its own assertions under the repo's scratch-profile runner, with no setup error raised from the fixture
- [ ] #2 ADR-126 admission is unchanged: no production guard, participant registry or source binding is reset or bypassed for tests
- [ ] #3 A test that then fails on its own assertion is fixed, or listed in this task with a named owner
- [ ] #4 The five files that re-point TLDW_CONFIG_PATH by hand (17 failures listed in the description) are repaired the same way, or each is listed in this task with a named owner
- [ ] #5 A negative control is recorded: with the repair reverted, the fixture's users error at setup again
- [ ] #6 Repaired files that pass in a PR lane's own environment and order join that lane, so the next breakage of this kind turns a PR check red
<!-- AC:END -->
