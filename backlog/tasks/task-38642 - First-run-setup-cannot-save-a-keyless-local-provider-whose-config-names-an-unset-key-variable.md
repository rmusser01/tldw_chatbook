---
id: TASK-38642
title: First-run setup cannot save a keyless local provider whose config names an unset key variable
status: Done
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 06:05
labels:
- follow-up
- bug
- wizards
- providers
priority: high
references:
- tldw_chatbook/UI/Wizards/first_run_setup_state.py
- tldw_chatbook/Chat/provider_readiness.py
- Tests/Wizards/test_first_run_setup_integration.py
dependencies:
- TASK-34100.6
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02).

**The regression.** b64f9c7cf2 ("preserve provider credentials and native routes", PR #3294, TASK-34100.6, merged 2026-11-01) changed `build_first_run_provider_commit` (first_run_setup_state.py:512-532).
- For credential source "none", it now asks `resolve_provider_credential` for an env-var name.
- It then puts that name into the draft's `credential_env_var` for every source except "environment".
- When the provider's settings name a key variable that is not exported, `resolve_provider_credential` returns `(None, None, env_var)` (provider_readiness.py:687).
- The shipped config does exactly that for llama.cpp: `api_key_env_var = "LLAMA_CPP_API_KEY" # If you set one on the server` (config.py:5241).
- `ProviderSetupDraft` refuses a "none" draft that carries an env var ("Credential setup is invalid.", provider_setup_persistence.py:300-303). The wizard's commit is therefore rejected with "First-run provider commit rejected (category=validation)".
- So a first run that sets up a keyless local server without exporting the variable cannot save. The live TUI was not checked.

**Evidence.**
- `Tests/Wizards/test_first_run_setup_integration.py` through the scratch-profile wrapper gave 6 failed, 15 passed and 17 errors, with `-n 4` and serially. The 17 errors are setup errors, added to TASK-38111. The 6 failures are `assert await container.commit_staged_provider_setup(...)` at :954, :1013 and :1110:
  - `test_saving_setup_exit_moves_an_untouched_console_chat_to_setup_choice` [chat, home, library, library_notes]
  - `test_start_chatting_lands_setup_choice_when_a_config_write_beats_the_handoff`
  - `test_a_listed_model_setup_saves_leaves_the_saved_model_list_alone`
- The last one is a PR-gated node ("PR Fast Lane (admission-sensitive 2)"). It failed on PR #3294's own run 41266290761 and on later PR runs 41267322915, 41268462903 and 41288804522. Most PR runs since 2026-11-01 skip that step after an earlier step fails, which hides it.
- A read-only probe that emulated a "none" draft carrying no env var gave 21 passed (only the 17 setup errors left).

**A fix already exists in open PRs.** Open PRs #3346 (prompt mix-ins) and #3342 (draft, recovery and required CI) both change the selector to `resolved_env_var if source == "environment" else None`. #3346 also adds `test_provider_commit_carries_environment_selector_only_for_environment_source`. Whichever lands first closes this task once the criteria below are checked on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A first-run commit with credential source "none" saves when the provider's config names an unset key variable, and the saved draft carries no env-var selector; "environment" and "stored" keep their behaviour
- [x] #2 The six tests named above pass on dev, and the admission-sensitive 2 node passes in a PR run
- [x] #3 A regression test fails at 88af9f9379 and passes with the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-11-03. PR #3346 (merged 2026-11-03, dev 18482c1cf7) carried the fix and the regression test. This task only records the checks on dev.
- **AC#1:** `build_first_run_provider_commit` now sets `credential_env_var` to the resolved variable only for source "environment" (first_run_setup_state.py:512-518). `test_provider_commit_carries_environment_selector_only_for_environment_source` pins all three sources. "environment" keeps its selector; "none" and "stored" carry none.
- **AC#2:** on dev 18482c1cf7, through the scratch-profile wrapper with `-n 4`, the six tests named in the description plus the three regression cases gave 9 passed. The admission-sensitive 2 node `test_a_listed_model_setup_saves_leaves_the_saved_model_list_alone` passed in #3346's PR run 41573532021, job 125717364994, in the step "Run admission-sensitive suites 2" (1437 passed).
- **AC#3:** under a temporary edit that restored the 88af9f9379 selector (`else resolved_env_var` for every non-draft source), the regression case `[none]` and `test_a_listed_model_setup_saves_leaves_the_saved_model_list_alone` failed (2 failed, 2 passed). With the edit reverted, they pass.
<!-- SECTION:NOTES:END -->
