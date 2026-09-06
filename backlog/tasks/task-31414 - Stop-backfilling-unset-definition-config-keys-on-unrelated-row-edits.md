---
id: TASK-31414
title: Stop backfilling unset definition config keys on unrelated row edits
status: Done
assignee: []
created_date: '2026-09-04 22:40'
updated_date: '2026-09-05 22:28'
labels:
  - scheduling
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saving ANY single-row edit on a local `recurring_question` definition backfills `config.generation_mode`, `config.scope` and `config.finding_policy` to concrete defaults, regardless of which row the user actually edited. Traced empirically through the shared preview pipeline — `preview_automation_definition` (`Scheduling/automation_preview.py:101`) -> `validate_recurring_question_config` (`Scheduling/automation_validation.py:145`) — which normalizes those three with defaults on every local save.

This is pre-existing behaviour, not a redesign regression: the create/edit modal never exercised it because it always sends explicit values for all three. Redesign PR-3 Task 4 made per-row in-pane editing the FIRST caller that can send a payload leaving them genuinely absent, which is what made it reachable.

It is runtime-identical today — the backfilled defaults are what the executor would resolve anyway — so this is a display-honesty defect, not a behaviour one. The cost is that `DefinitionDetail` deliberately renders an absent key as "Not set" rather than a plausible guess (its module docstring states the rule: honest meaning "we have no value", never a plausible guess), and the backfill silently converts "no value" into a stored concrete reading the user never chose. A later change to any default then also silently changes definitions the user never configured.

PR-3 Task 4's review adjudicated it out of that task's scope and deferred it (commit `8b6e7501d6`: "Finding 3 (config-normalization backfill on unrelated edits) stays deferred per the review's own adjudication -- not touched"). `Tests/UI/test_schedules_workbench.py::test_not_set_model_preserved_across_an_unrelated_edit` records the exclusion in its own docstring — it deliberately scopes its round-trip claim to `input`/`notification_policy` and NOT to the `config` trio.

The fix is a normalizer that distinguishes "validate what was supplied" from "fill in what a create needs", so the create path keeps its defaults and the edit path preserves absence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Editing one row of a local definition leaves config keys the payload did not carry absent in storage
- [x] #2 A definition created through the create modal still gets its full explicit config, unchanged from today
- [x] #3 The Details rows keep rendering Not set for a genuinely absent key after an unrelated edit, proven by reading the stored row and not only the painted value
- [x] #4 test_not_set_model_preserved_across_an_unrelated_edit extends its round-trip claim to the config trio, and its scoping docstring is replaced by the real assertion
- [x] #5 Validation strictness is unchanged: a supplied-but-invalid value is still rejected exactly as today
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read validate_recurring_question_config / automation_preview.py / scheduling_service.py save_definition, _merge_definition_payload, _definition_db_fields_from_preview to trace where scope/finding_policy/retention_policy/generation_mode get backfilled.
2. Add a mode='create'|'update' keyword to validate_recurring_question_config; gate the scope/finding_policy/retention_policy/generation_mode backfill on mode!='update' so an edit only normalizes a key the payload actually carried.
3. Thread mode through preview_automation_definition (already computes mode) into the validator call.
4. Fix _definition_db_fields_from_preview's dedicated finding_policy/retention_policy DB columns to omit the kwarg (not write {}) when the corresponding config sub-key is absent, so an update's SQL never touches a column the edit didn't carry -- otherwise the config-side fix would just move the same backfill bug into the dedicated columns as a {}-wipe.
5. Extend test_not_set_model_preserved_across_an_unrelated_edit (AC4) to assert scope/finding_policy stay absent in config, dedicated columns stay byte-identical, and the Sources row still paints Not set.
6. Add unit tests in test_automation_validation.py for create-mode backfill (unchanged), update-mode absence-preservation, and update-mode still rejecting a supplied-invalid value.
7. Add a pin test for _resolve_finding_policy(None) in test_automation_execution.py so execution-time resolution is proven unchanged.
8. Run the Scheduling + schedules_workbench suites; ruff/mypy on touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`validate_recurring_question_config` (automation_validation.py) gains a keyword-only `mode: str = "create"` param. `backfill = mode != "update"`: each of scope/finding_policy/retention_policy/generation_mode is only normalized-with-defaults when `backfill` or the key is already present in the incoming config -- a genuinely-absent key on an update stays absent in `normalized["config"]` instead of being invented; a present key (even an invalid one) is still validated/normalized identically in both modes, so strictness (AC5) never relaxes. `automation_preview.py`'s `preview_automation_definition` already computed `mode` for the mode-required/not-allowed checks; it now also threads that same value into the validator call (one-line change).

Root-cause follow-through: `scheduling_service.py`'s `_definition_db_fields_from_preview` writes finding_policy/retention_policy to TWO places -- inside the `config` JSON column and as their own dedicated DB columns (the executor reads the dedicated column, not config's copy). Its old unconditional `config.get("finding_policy") or {}` would have turned the validator's new omission into a DIFFERENT bug: writing an empty-dict WIPE over a real dedicated-column value on every unrelated edit. Fixed by only including the dedicated-column kwarg when `"finding_policy"/"retention_policy" in config` -- when absent, the kwarg is omitted entirely so `update_automation_definition`'s SQL `SET` clause never touches that column, leaving whatever was stored exactly as it was (not backfilled, not wiped).

Both ported-validator module docstrings got a short note flagging `mode="update"` as a deliberate LOCAL divergence from server byte-parity, so a future re-sync from the server module doesn't flatten it back to unconditional backfill.

Verified the full modal (create AND edit) is unaffected: it always sends config.scope/generation_mode/finding_policy explicitly (confirmed in automation_definition_form.py's _build_payload), so `in option_config` is always true there regardless of mode -- only the new per-row inline-edit path (task-4's _definition_edit_payload, which sends a payload touching only the one edited key) can genuinely omit a key, which is exactly the case this task targets.

Tests: extended test_not_set_model_preserved_across_an_unrelated_edit (Tests/UI/test_schedules_workbench.py) to assert scope/finding_policy stay out of stored config after an unrelated generation_mode edit, that the dedicated finding_policy/retention_policy columns are byte-identical before/after (proving no wipe), and that the Sources row still paints "Not set" (painted value, not just the DB read -- AC3). Added 4 new unit tests in test_automation_validation.py (create-mode backfill unchanged, update-mode leaves-absent, update-mode only-backfills-the-supplied-key, update-mode still-rejects-invalid-supplied-value) and one pin test in test_automation_execution.py for `_resolve_finding_policy(None)` resolving to the same balanced-findings default at execution time (unchanged, per the task's own runtime-identical framing).

Verification: Tests/Scheduling/ full suite (1072 passed), Tests/UI/test_schedules_workbench.py full file (158 passed), ruff/mypy diffed against baseline show zero new findings on the touched lines (pre-existing UP017/mypy issues elsewhere in scheduling_service.py are unrelated, confirmed via git show diff).

Files touched: tldw_chatbook/Scheduling/automation_validation.py, tldw_chatbook/Scheduling/automation_preview.py, tldw_chatbook/Scheduling/services/scheduling_service.py, Tests/Scheduling/test_automation_validation.py, Tests/Scheduling/test_automation_execution.py, Tests/UI/test_schedules_workbench.py.
<!-- SECTION:NOTES:END -->
