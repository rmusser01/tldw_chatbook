---
id: TASK-38756
title: "Prompt Management and Prompts DB tests red on dev (gateway signature, lazy exports, a self-contradicting privacy pin)"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- tests
- prompts
priority: medium
references:
- Tests/Prompt_Management/test_prompt_improvement_service.py
- Tests/Prompt_Management/test_prompt_batch_models.py
- Tests/Prompts_DB/test_prompts_db_batch_mutations.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Prompt_Management/__init__.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

A `-n 4` run of the three files through the scratch-profile wrapper gave 53 failed, 167 passed and 3 errors. Four of the failures are real assertions:
1. `test_prompt_improvement_service.py::test_task10_gateway_signature_and_application_cap_are_reused` (:292). It pins `ConsoleProviderGateway.complete_auxiliary`'s parameters to `("self", "request", "route")` (:295). Dev's method (`Chat/console_provider_gateway.py:5686`) also takes `rule_lease`.
2. `test_prompt_batch_models.py::test_prompt_management_lazy_public_exports_preserve_class_identities` (:117). It pins `Prompt_Management.__all__` to the four service names. Dev's `__all__` (`Prompt_Management/__init__.py:21`) also lists `PROMPT_DRAFT_SHELF_CAPACITY` and `PROMPT_DRAFT_SHELF_MAX_PAGE_SIZE`, first.
3. and 4. `test_prompts_db_batch_mutations.py::test_batch_success_diagnostics_are_aggregate_and_privacy_safe` (:1076) and `::test_batch_failure_is_bounded_and_privacy_safe` (:1117). Each forbids `str(second_id)` anywhere in the captured log. `second_id` is 2, and the same tests require `"operation=delete count=2"` in that log. The product's line ("Prompt batch mutation committed operation=delete count=2") carries no identifier, so the pin contradicts itself.

The other 49 failures and all 3 errors are in `test_prompt_improvement_service.py`, and each is `RecoveryRequired: raw_source_selection_changed`. It comes from a config read inside the code under test: `prompt_improvement_prompts.py:86`, then `Internal_Prompts/resolver.py:147`, `config.get_cli_setting`, and `Backup_Recovery/raw_participants.py:307`. That is the class TASK-38111 owns, but this file is not on its list. Run serially, the file gives the same result: 50 failed, 52 passed, 3 errors.

None of the three files runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four nodes passes and pins the contract dev intends today; the notes say for each whether the test or the code moved on purpose
- [ ] #2 The two privacy pins still fail if a prompt id, uuid, name, keyword or body reaches the log (shown with a temporary source edit), and no longer match inside an aggregate count
- [ ] #3 Run on its own through the scratch-profile wrapper, `test_prompt_improvement_service.py` fails only on its own assertions, or the notes trace its RecoveryRequired failures to TASK-38111's cause and record the file against that task
<!-- AC:END -->
