---
id: TASK-42015
title: "Delete the uncalled ChatScreen first-send onboarding methods"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:38
labels:
- follow-up
- console
- cleanup
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_native_chat_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41904 (round 17). `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag` have no production callers; only `test_console_record_first_send_repairs_corrupt_config_value` calls the first. The live save is `_persist_first_send_when_quiescent`, whose `_record_successful_first_send` already guards a non-dict `console` or `onboarding`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both methods and the test that keeps them alive are removed, with the corrupt-value case covered on the live path if it is not already
<!-- AC:END -->

## Implementation Plan

1. Grep the whole repo for both method names (done: only chat_screen.py, the one test, and historical plan/QA docs that no test reads).
2. Delete `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag`, and `test_console_record_first_send_repairs_corrupt_config_value`.
3. The live path has no corrupt-value test: add one in `Tests/UI/test_console_runtime_ownership.py` (a lane target) that runs `ConsoleRuntime._record_successful_first_send` over a non-dict `console` and a non-dict `onboarding`, and checks the in-memory flag and the saved config file.
4. Show it red under a temporary source edit (skip the write on a corrupt value), then green; run covering tests, ruff, ratchets and preflight.

## Implementation Notes

- **Grep.** Across the whole repo the two names appear only in `chat_screen.py`, the one test, two backlog task files, and historical docs (`Docs/superpowers/plans/2026-07-03-console-setup-card-phase2.md`, `Docs/superpowers/qa/2026-10-05-console-pr2995-review-corrections/.../source-carry.json`, a ruling under `Docs/superpowers/qa/2026-11-01-...`, `Docs/QA/perf-structural-audit-2026-09-27/data/findings.json`). No test or script reads those docs; they record past states and were left alone.
- **Deleted** `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag` (`tldw_chatbook/UI/Screens/chat_screen.py`, -31 lines, -2 methods) and `test_console_record_first_send_repairs_corrupt_config_value` (`Tests/UI/test_console_native_chat_flow.py`). `save_setting_to_cli_config` and `work` stay imported for other callers (`ruff --select F` clean).
- **Live path.** Nothing tested `ConsoleRuntime._record_successful_first_send` with a corrupt value. New `test_first_send_repairs_a_corrupt_onboarding_value_and_still_saves[console|onboarding]` in `Tests/UI/test_console_runtime_ownership.py` (a whole-file target of the admission-sensitive lane, so no lane edit) calls it over `{"console": "corrupt"}` and `{"console": {"onboarding": "corrupt"}}`, lets the real `_persist_first_send_when_quiescent` run, and checks the in-memory flag and the saved config file.
- **Red/green.** Under a temporary edit that returns early on a non-dict value (the "silently skipping the write" the old test named), both nodes failed with `AssertionError: the first-send save was skipped`; reverted, both pass.
- **Runs.** `-k first_send` in the ownership file: 5 passed. Ownership file + `Tests/Chat/test_console_onboarding_state.py` at `-n 4` (load 28): 113 passed, 2 failed; the 2 (`test_missing_key_renders_card_with_provider_step_active`, `test_endpoint_problems_relabel_step_one`, "Finish provider setup" copy) fail identically with `chat_screen.py` at base, so they are pre-existing. Size ratchets (`test_screen_size_ratchet.py`, `test_library_modules_size_ratchet.py`, `test_module_size_ratchet.py`): 285 passed; the shrink is within the 200-line/10-method slack, so no budget edit.
- **Diagnostic inventory.** Preflight's production diagnostic inventory reported `chat_screen.py` 118 -> 117 calls; `--statements` named the one removed statement, the deleted method's `logger.warning("Failed to persist Console onboarding flag: {}", exc)`, so the pin was regenerated with `--write`.
- **Review rev1.** The new test now hands the runtime `copy.deepcopy(console_value)`: the repair edits the dict in place, so the `[onboarding]` parameter came out as `{"onboarding": {"first_send_completed": True}}` and an in-process rerun would have found the flag already set and skipped the save.
- **`ChatScreen._console_first_send_completed_cached` stays.** The deleted method was one of its two writers; the other is the memoizing read in `_console_first_send_completed` itself (plus `None` in `__init__`), so it is a live memo, not a dead field. Removing it is not behaviour-neutral: a live read would see the flag `ConsoleRuntime._record_successful_first_send` sets in `app.app_config` during the same visit, so a later empty tab on that screen would turn quiet at once. Today the screen keeps its first read until a later ChatScreen visit (as at base, since the deleted method was never called). That is a product change, proposed as a follow-up, not part of this cleanup.
- Files: `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_runtime_ownership.py`, `Docs/security/production-diagnostic-inventory.json`.
