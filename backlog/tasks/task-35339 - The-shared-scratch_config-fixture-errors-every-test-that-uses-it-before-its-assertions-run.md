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
- Tests/Backup_Recovery/config_test_support.py
- Tests/conftest.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
- backlog/decisions/126-complete-local-backup-and-recovery.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-33002.17 review, round 2 (2026-10-17). `scratch_config` in Tests/Internal_Prompts/conftest.py is the shared fixture that points the app at a throwaway config file. Tests/UI, Tests/Agents, Tests/RAG and Tests/Web_Scraping re-export it from their own conftests. Its `write()` sets `TLDW_CONFIG_PATH` to a new per-test file and then reloads the config. By then the config source is already bound to the process's first path, and ADR-126's admission refuses the change by design: `_register_raw_participant` in Backup_Recovery/raw_participants.py raises `RecoveryRequired('raw_source_selection_changed')` when an owner's selected path moves. So every test that asks for the fixture errors at setup and never reaches its assertions.

Measured at 85bba8bb41 (branch p9-w5-t14) under the repo's scratch-profile runner, xdist loadfile: 101 setup errors in 12 files, every one raised from the fixture's `write()`.
- Tests/UI: test_settings_image_gen_panel.py 32, test_internal_prompts_panel_editing.py 7, test_internal_prompts_panel.py 4, test_internal_prompts_search_survives_save.py 1. The 14 tests in these files that passed do not ask for the fixture.
- Elsewhere: Tests/Internal_Prompts test_resolver.py 16, test_document_generation_migration.py 10, test_authoring.py 7, test_summarization_migration.py 6, test_prompt_improvement_prompts.py 4; Tests/Agents/test_agents_internal_prompts.py 6; Tests/Web_Scraping/test_websearch_internal_prompts.py 5; Tests/RAG/test_reranker_internal_prompts.py 3.

The same run found 16 more failures with the same refusal in four files that re-point `TLDW_CONFIG_PATH` by hand (`monkeypatch.setenv`) rather than through the fixture: Tests/Evals/test_eval_orchestrator_db_path.py 8, Tests/Persona_Buddy/test_persona_buddy_config_projection.py 5, Tests/test_user_data_dir_isolation.py 2, Tests/Wizards/test_first_run_setup_wizard.py 1 (test_summary_footer_shows_the_effective_config_path).

One more failure in that run has neither shape: Tests/Agents/test_agents_internal_prompts.py::test_agent_lessons_suffix_follows_configured_protocol_and_precedes_environment. It takes only `tmp_path`, and neither its file nor Tests/Agents/conftest.py sets `TLDW_CONFIG_PATH`. It fails alone too (node mode, 1 failed). Its first config read (AgentService._build_model_request, then render_tool_protocol, the resolver's `_config_value`, get_cli_setting) is refused because the root conftest's per-test sandbox redirect (Tests/conftest.py, the `TLDW_CONFIG_PATH` setenv after `keep_bootstrap_profile`) moved the path after the config bound. That is the TASK-33370 / TASK-34000.50 per-test-redirect class, not a hand re-point.

No PR lane runs any of the failing tests, so nothing turns red. Only test.yml's Core Tests (pushes to main, dispatch) and nightly-deep (`pytest ./Tests/`) reach them. Two of the files already have node-level PR-lane entries. derived-artifacts.yml's admission-sensitive step (Tests/CI/test_ci_queue_pressure_contract.py's ADMISSION_TARGETS) runs 12 nodes of test_first_run_setup_wizard.py and two `bootstrap_profile` nodes of test_settings_image_gen_panel.py (test_backend_rows_paint_each_name_whole_before_one_status_column, test_a_long_probe_result_shortens_the_name_and_keeps_test_in_the_row). scripts/ui_pr_gate_census.txt runs test_settings_image_gen_panel.py::test_backend_row_controls_visible_and_clickable_under_real_css. AC #6 adds the repaired nodes beside those entries. The image-gen file's `bootstrap_profile` nodes stay in ADMISSION_TARGETS and never go into the census, so that file does not join the census whole.

TASK-33370 fixed this failure class for the DB, Personas and config suites at their shared seam; this fixture was not in its set. TASK-33002.17 converted one test of test_settings_image_gen_panel.py (test_backend_row_controls_visible_and_clickable_under_real_css) to `@private_profile_test`, with the real config writer storing its keys in the case's private profile. That is one working shape. The fixture still errors the file's 32 other users. The first candidate is the repo's helper for this exact move, `select_config_source` / `install_config_source` in Tests/Backup_Recovery/config_test_support.py. It selects the new path first, imports a fresh real config module, and rebinds only the test's explicit consumers. TASK-33370's notes and its lessons-testing-evidence.md entry (TASK-33370 and TASK-33373) name it as the sanctioned shape for a test that selects another config source, and Tests/UI/conftest.py already imports it. The precedent is `_select_scratch_config_source` in Tests/Chat/test_console_settings_defaults.py, which re-points `TLDW_CONFIG_PATH` at `tmp_path/config.toml` and "keeps the gate and satisfies it". The resolver reads config lazily (`from tldw_chatbook.config import get_cli_setting` inside `_config_value`), so it sees the fresh module. The helper's limit is recorded in the comment above test_first_run_setup_wizard.py's `pytestmark`: re-selecting per test left 20 wizard tests red, because consumers that import config at module level stay bound to the old module. Reach for the heavier `@private_profile_test` subprocess shape only where that limit bites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test that asks for the shared scratch_config fixture reaches its own assertions under the repo's scratch-profile runner, with no setup error raised from the fixture
- [ ] #2 ADR-126 admission is unchanged: no production guard or participant registry is reset or bypassed; only a test's own explicit consumers may be rebound to a freshly selected source (the TASK-33370 shape)
- [ ] #3 A test that then fails on its own assertion is fixed, or listed in this task with a named owner
- [ ] #4 The four files that re-point TLDW_CONFIG_PATH by hand (16 failures listed in the description) are repaired the same way, or each is listed in this task with a named owner
- [ ] #5 A negative control is recorded: with the repair reverted, the fixture's users error at setup again
- [ ] #6 Repaired files that pass in a PR lane's own environment and order join that lane, so the next breakage of this kind turns a PR check red
- [ ] #7 Tests/Agents/test_agents_internal_prompts.py::test_agent_lessons_suffix_follows_configured_protocol_and_precedes_environment, refused at its first config read under the root conftest's per-test redirect (the TASK-33370 class), reaches its own assertions under the scratch-profile runner, both alone and in its file
<!-- AC:END -->
