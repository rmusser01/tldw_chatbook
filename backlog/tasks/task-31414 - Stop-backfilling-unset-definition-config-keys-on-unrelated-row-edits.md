---
id: TASK-31414
title: Stop backfilling unset definition config keys on unrelated row edits
status: To Do
assignee: []
created_date: '2026-09-04 22:40'
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
- [ ] #1 Editing one row of a local definition leaves config keys the payload did not carry absent in storage
- [ ] #2 A definition created through the create modal still gets its full explicit config, unchanged from today
- [ ] #3 The Details rows keep rendering Not set for a genuinely absent key after an unrelated edit, proven by reading the stored row and not only the painted value
- [ ] #4 test_not_set_model_preserved_across_an_unrelated_edit extends its round-trip claim to the config trio, and its scoping docstring is replaced by the real assertion
- [ ] #5 Validation strictness is unchanged: a supplied-but-invalid value is still rejected exactly as today
<!-- AC:END -->
