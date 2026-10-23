---
id: TASK-36384
title: Put the quit-persistence and wizard listed-pick pins on a PR lane
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- ci
- tests
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/test_config_persistence_owner.py
- Tests/ProductionApp/test_config_profile_isolation.py
- Tests/Wizards/test_first_run_setup_wizard.py
- Tests/Wizards/test_first_run_setup_integration.py
dependencies:
- TASK-35578
- TASK-35835
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35835 (PR #3216) and TASK-35578 (PR #3214) add pins that no PR lane runs. They run only on main pushes and nightly, so a PR that breaks them merges green. The UI census carries only `Tests/UI` nodes, and these live elsewhere. They need a target in a `derived-artifacts.yml` lane, a workflow edit those PRs could not make while draft PR #3036 changes the same file.
- `Tests/test_config_persistence_owner.py`, the whole file: 9 tests after TASK-35835, about 8 s. They pin that the quit leaves a fresh template byte-identical, upgrades v1 ciphertext in place, refuses a value written under another password between its reads, and writes nothing when there is nothing to do.
- `Tests/ProductionApp/test_config_profile_isolation.py::test_real_app_quit_persists_only_the_effective_config`. Run as a lone file, it errors at setup with `RecoveryRequired`; beside a sibling file that imports the app, it passes.
- `Tests/Wizards/test_first_run_setup_wizard.py::test_model_step_says_a_listed_pick_is_not_added_to_the_saved_model_list` (from TASK-35578).
- `Tests/Wizards/test_first_run_setup_integration.py::test_a_listed_model_setup_saves_leaves_the_saved_model_list_alone` (from TASK-35578).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each named test runs in a PR-required lane, with a matching Tests/CI lane-contract pin
- [ ] #2 The real-quit test runs in that lane without the lone-file RecoveryRequired setup error
- [ ] #3 The lane's measured job time stays at least 3 minutes under its cap
<!-- AC:END -->
