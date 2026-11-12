---
id: TASK-38756
title: "Prompt Management and Prompts DB tests red on dev (gateway signature, lazy exports, a self-contradicting privacy pin)"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 07:11
updated_date: '2026-11-05 21:10'
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
- [x] #1 Each of the four nodes passes and pins the contract dev intends today; the notes say for each whether the test or the code moved on purpose
- [x] #2 The two privacy pins still fail if a prompt id, uuid, name, keyword or body reaches the log (shown with a temporary source edit), and no longer match inside an aggregate count
- [x] #3 Run on its own through the scratch-profile wrapper, `test_prompt_improvement_service.py` fails only on its own assertions, or the notes trace its RecoveryRequired failures to TASK-38111's cause and record the file against that task
<!-- AC:END -->

## Implementation Plan

1. For each of the four nodes, find from `git log -S` whether the test or the code moved, and on purpose.
2. Re-pin each test to the contract dev intends; keep the two privacy pins able to fail, shown with temporary source edits that log each private field.
3. Trace `test_prompt_improvement_service.py`'s RecoveryRequired failures from their own tracebacks and record the file against TASK-38111 (which owns that error) instead of fixing it here.

## Implementation Notes

Base 29592ecc9b. The plan was written after the history reads, before any source or test edit was committed.

**Per node: what moved.**
1. `test_task10_gateway_signature_and_application_cap_are_reused`: the code moved on purpose. d24c4c04b2 (2026-10-12, native response rules) added a keyword-only `rule_lease=None` to `ConsoleProviderGateway.complete_auxiliary`; prompt improvement still calls `complete_auxiliary(request, route=None)`. The pin now lists `rule_lease` and requires it keyword-only with default `None`, so the improvement call shape stays valid.
2. `test_prompt_management_lazy_public_exports_preserve_class_identities`: the code moved on purpose. 9bd65942df (2026-09-26, "consolidate boot-resident limits") put `PROMPT_DRAFT_SHELF_CAPACITY` / `PROMPT_DRAFT_SHELF_MAX_PAGE_SIZE` on the package facade and in `__all__`. The pin now matches dev's six-name `__all__`; the class-identity asserts are unchanged.
3. and 4. The two privacy pins: neither the pin nor the product log moved; the test's id seeding silently stopped working. The tests seed ids 71,234,567 / 81,234,567 so an id cannot hide inside `count=2`. 7970fbcada (2026-11-01, mix-in v5->v6 migration) rebuilds `Prompts` and deliberately keeps its AUTOINCREMENT sequence, which leaves a `('Prompts', 0)` row in `sqlite_sequence` even in a fresh database; `_set_next_prompt_id` INSERTed a second row, which SQLite ignores, so the ids fell back to 1 and 2 and `str(second_id)` matched the count. The helper now updates the existing row (inserting only when absent), and each test asserts its seeded ids, so the seeding cannot decay silently again. The pins were not self-contradictory; the seed was stale.

**AC#2 (pins can still fail).** Temporary edits to `PromptsDatabase.soft_delete_prompts` logged one private field at a time (`logger.info('leak {}', ...)`, read through the raw connection so no DEBUG query line leaked the id first); both pins went red for each of id, uuid, name, keyword and body, then the source was restored. That run found a real hole: stored keywords are lower-cased (`_normalize_keyword`), so a leaked keyword never matched the upper-case `"PRIVATE KEYWORD"` pin (2 passed with the keyword leak). The forbidden check is now case-insensitive, and the failure pin also forbids a body (its first row now has a private body), so every one of the five kinds is red in both tests.

**AC#3.** On its own (`-n 4`, scratch wrapper) `test_prompt_improvement_service.py` gives 49 failed, 53 passed, 3 errors, and all 52 raise `RecoveryRequired: raw_source_selection_changed`; no own assertion fails. The 3 setup errors come from the file's `scratch_cli_config` fixture (redirect `TLDW_CONFIG_PATH` + `load_settings(force_reload=True)`); the 49 come from `prompt_improvement_prompts.py:86` -> `Internal_Prompts/resolver.py:147` -> `config.get_cli_setting`. With a temporary module-level `bootstrap_profile` marker the 49 pass and the 3 fixture errors remain. Recorded in TASK-38111's description (dated paragraph); not fixed here, since that task owns the error.

**Runs.** The three files at base: 53 failed, 167 passed, 3 errors. After: `test_prompt_batch_models.py` + `test_prompts_db_batch_mutations.py` + the gateway node: 119 passed; the improvement file as above.

**Files.** `Tests/Prompt_Management/test_prompt_improvement_service.py`, `Tests/Prompt_Management/test_prompt_batch_models.py`, `Tests/Prompts_DB/test_prompts_db_batch_mutations.py`, TASK-38111's task file. None of these test files runs on a PR lane.
