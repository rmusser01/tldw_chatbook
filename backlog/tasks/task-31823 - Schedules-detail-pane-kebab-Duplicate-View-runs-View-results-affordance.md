---
id: TASK-31823
title: 'Schedules detail-pane kebab: Duplicate/View-runs/View-results affordance'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-06 05:56'
updated_date: '2026-09-06 19:22'
labels:
  - scheduling
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec §5's kebab menu (Duplicate / View runs / View results / Edit in full… / Delete-Archive) on the reminder/definition detail panes was deliberately deferred during the schedules redesign (task_detail.py lifecycle row comment: "no kebab -- plan ruling 1") and again scoped out of the 31712 form/detail-pane polish pass (controller ruling: re-scope rather than build speculative kebab UI in a polish pass). This task designs and delivers that affordance (or the specific subset still missing -- Duplicate and View-runs/View-results shortcuts do not exist from either detail pane today).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate, View runs, and View results are each reachable from a reminder or definition detail pane (kebab menu or equivalent per-action controls)
- [x] #2 The affordance follows the existing lifecycle-row disabled+reason idiom (UX-073) for any action that cannot apply to the current row
- [x] #3 Existing lifecycle actions (Edit/Acknowledge/Run now/Enable/Disable/Delete) are unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task ACs + redesign spec section 5 kebab list; grep task_detail.py's "no kebab -- plan ruling 1" comment and definition_detail.py's existing Last-run/Unread-results row activations (already post ViewDefinitionAuditRequested/ViewDefinitionResultsRequested) to see what already exists vs is genuinely missing.
2. Shape decision: a second compact button row (Duplicate / View runs / View results) under each pane's existing lifecycle row, not a popup kebab menu -- lower risk, no new modal/dismissal/focus-trap surface, matches the established "plain buttons over popup" idiom the lifecycle row already set.
3. events.py: add DuplicateTaskRequested/DuplicateDefinitionRequested messages.
4. task_detail.py: compose the secondary-actions row (Duplicate/View runs/View results); View results permanently disabled with a UX-073 reason (reminders have no results surface); View runs reuses the existing scroll-to-"Recent runs:" affordance; Duplicate posts the new message and is gated into the existing set_lifecycle_lock loop (same transfer lock as Edit/Delete).
5. definition_detail.py: compose the same three buttons; View runs/View results post the SAME existing ViewDefinitionAuditRequested/ViewDefinitionResultsRequested messages the Last-run/Unread-results rows already post (zero new workbench wiring for those two); Duplicate is gated on lock+family via a new _refresh_duplicate_button, mirroring Pause/Resume's own gate.
6. schedules_workbench.py: handle DuplicateTaskRequested via the existing create_reminder path (owner_id="local", name-disambiguated payload copying only authored fields); handle DuplicateDefinitionRequested via the existing save_definition create path (owner_id="local", a _duplicate_definition_payload helper mirroring AutomationDefinitionForm._build_payload's shape).
7. CSS: add .detail-secondary-action-button + the two secondary-actions row ids to css/features/_scheduling.tcss; rebuild the bundle and verify check_bundle_sync.py.
8. Tests: reachability + disabled-reason + duplicate-correctness (field copy, name disambiguation, no shared mutable state, local-ownership ruling) + navigation-target tests, split across test_schedules_transfer_actions.py (reminder pane), test_schedules_workbench.py (definition pane bare-harness + real-service integration), and test_schedules_automations_tab.py (View-runs navigation via the existing AutomationsMockService harness). Run existing lifecycle tests before/after as a pin.
9. Docs/User_Guide/schedules.md: document the new buttons in both detail-pane sections + add a Verified-against stamp.
10. Close out: tick ACs, Implementation Notes, mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shape: a second compact button row (Duplicate / View runs / View results) under each pane's existing lifecycle row -- not a popup kebab menu. Lower risk (no new modal/dismissal/focus-trap surface), matches the "plain buttons" idiom the lifecycle row itself already set, and is trivially testable via Button.Pressed. Justified in-code in both compose() docstrings.

Duplicate ownership ruling: a duplicate always lands on owner_id="local" regardless of the source row's own owner -- duplicating a server-owned reminder/definition is a plain new local draft, never an implicit transfer/push. Fresh id, no borrowed run/transfer/incident history, and (for definitions) lifecycle resets to "configured" (active) even when the source was paused -- all via the same defaults create_reminder/create_automation_definition already give every other fresh create (nothing bespoke needed).

Reuse:
- Reminder pane "View runs" reuses 31712 AC#5's existing scroll-to-"Recent runs:" affordance verbatim (`_request_view_runs`, also now the single implementation `on_detail_value_row_activated`'s history-link branch calls too).
- Reminder pane "View results" is permanently disabled (UX-073 reason: reminders have no `automation_results` surface -- that table is keyed by definition_id, a recurring_question concept).
- Definition pane "View runs"/"View results" post the SAME ViewDefinitionAuditRequested/ViewDefinitionResultsRequested messages the existing Last-run/Unread-results row activations already post -- zero new workbench-side handling for those two; the buttons are a second entry point onto an already-built navigation.
- Duplicate (both panes) goes through the existing create_reminder/save_definition create paths. New: DuplicateTaskRequested/DuplicateDefinitionRequested messages (events.py), two workbench handlers, and a _duplicate_definition_payload() helper (mirrors AutomationDefinitionForm._build_payload's shape, sourced from the stored row instead of form widgets).

Disabled+reason (UX-073): Duplicate joins Edit/Delete in TaskDetail.set_lifecycle_lock's existing gate (mid-transfer); DefinitionDetail gets a new _refresh_duplicate_button gated on the SAME combined reason (_lifecycle_lock_reason or _family_note) that already drives Pause/Resume and the #scheduling-automation-detail-why Static. View results (reminder pane) is unconditionally disabled with its own reason line in a new #scheduling-task-detail-secondary-why Static.

Lifecycle actions unaffected (AC#3): pinned by the existing test_schedules_transfer_actions.py / test_schedules_workbench.py lifecycle tests, all still green (extended, not replaced, to also assert Duplicate joins the same gate).

Terminology fix: my first draft of the reminder pane's "View results" reason used "Reminders don't produce..." -- test_schedules_terminology.py's AST sweep (task-23106: no bare "reminder" noun in user-facing copy on task_detail.py/schedules_workbench.py/reminder_form.py) correctly flagged it. Reworded to "Scheduled tasks don't produce automation results...".

CSS: new .detail-secondary-action-button class (plain class on each Button, never an ancestor-scoped bare-type rule) + #scheduling-task-detail-secondary-actions / #scheduling-automation-detail-secondary-actions row ids, in css/features/_scheduling.tcss; bundle rebuilt via css/build_css.py, check_bundle_sync.py green.

Modified/added files: tldw_chatbook/Scheduling/events.py, tldw_chatbook/UI/Screens/scheduling/task_detail.py, tldw_chatbook/UI/Screens/scheduling/definition_detail.py, tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py, tldw_chatbook/css/features/_scheduling.tcss (+ regenerated tldw_cli_modular.tcss / screen_feature_scheduling.tcss), Docs/User_Guide/schedules.md, Tests/UI/test_schedules_transfer_actions.py, Tests/UI/test_schedules_workbench.py, Tests/UI/test_schedules_automations_tab.py.

Verification: Tests/UI/test_schedules_workbench.py + test_schedules_automations_tab.py + test_schedules_transfer_actions.py + test_schedules_terminology.py + test_detail_value_row.py + test_schedules_results_tab.py all run foreground: 299 passed, 1 failed. The 1 failure (test_no_bare_reminder_noun_in_schedules_screen_copy, a PRE-EXISTING line-471 "Fixed: a reminder always notifies..." offender in task_detail.py) reproduces byte-for-byte identically against base b69bd85b7 in a throwaway worktree (removed after) -- confirmed pre-existing, out of this task's scope, not caused by this change.

--- Review round 1 (REQUEST-CHANGES, addressed as a new commit) ---

Finding 1 (MAJOR, fixed): _duplicate_definition_payload carried no lifecycle key, so a PAUSED/archived/disabled source's Duplicate always landed "configured" (active) via create_automation_definition's own create-path default -- silently making it eligible for the due-run selector (list_armable_automation_definitions gates strictly on lifecycle='configured'), i.e. an LLM pipeline could start spending on schedule with zero further user action. Fixed with a ruling: lifecycle now carries forward for an ACTIVE source (unchanged, no code needed -- the create path's own default already does this); a PAUSED/ARCHIVED/DISABLED/unrecognized source collapses to PAUSED via one follow-up SchedulingService.set_definition_lifecycle(new_id, "pause") call after the create succeeds (save_definition's create path structurally cannot carry lifecycle through -- _definition_db_fields_from_preview builds its fields dict purely from the preview's normalized_config, which has no lifecycle key at all -- so a post-create follow-up write is the only way to apply it, not a payload field). Archived/disabled were deliberately NOT preserved verbatim: a duplicate starting in either state would be invisible to a plain Resume click, a worse outcome than paused. Side effect: the pause follow-up is a real second DB write, so a paused-source copy's version is 2, not 1 (update_automation_definition's own bump_version=True default) -- asserted explicitly in the updated test. Revert-checked in a throwaway worktree at the pre-fix commit (5baeec50a): both the fixed existing test and the new dedicated paused-collapse test fail there, the active-source control test still passes (removed after).

Finding 2 (MINOR/docs, fixed): Duplicate always goes through save_definition's create path, so task-31414's normalize-on-create backfill concretizes a source's UNSET generation_mode/scope/finding_policy/retention_policy into concrete defaults (e.g. "Balanced findings") in the copy -- display/storage-only, runtime behavior is identical since the execution-time resolvers (automation_execution.py) already fall back to the exact same defaults for an absent key. Now stated explicitly here and in Docs/User_Guide/schedules.md's automation-pane paragraph + Verified-against stamp.

New tests: test_duplicate_button_collapses_a_paused_source_to_a_paused_copy_not_due_for_selection (paused source -> paused copy, NOT in list_armable_automation_definitions) and test_duplicate_button_keeps_an_active_source_active_and_due_for_selection (active source -> active copy, control, unchanged behavior) in Tests/UI/test_schedules_workbench.py. Existing test_duplicate_button_creates_a_disambiguated_local_copy_of_a_definition's lifecycle/version assertions corrected to match (was silently asserting the bug as correct behavior).
<!-- SECTION:NOTES:END -->
