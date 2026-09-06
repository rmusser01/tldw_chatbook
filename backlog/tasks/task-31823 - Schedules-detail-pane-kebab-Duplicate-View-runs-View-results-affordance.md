---
id: TASK-31823
title: 'Schedules detail-pane kebab: Duplicate/View-runs/View-results affordance'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-06 05:56'
updated_date: '2026-09-06 20:41'
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
CURRENT BEHAVIOR (as shipped after the final review wave):

Shape: a second compact button row under each pane's existing lifecycle row -- not a popup kebab menu. Reminder pane: Duplicate, View runs (2 buttons -- View results is deliberately NOT rendered here, see below). Definition pane: Duplicate, View runs, View results (3 buttons). Lower risk than a popup (no new modal/dismissal/focus-trap surface), matches the "plain buttons" idiom the lifecycle row itself already set, trivially testable via Button.Pressed.

Duplicate ownership ruling: always lands on owner_id="local" regardless of the source row's own owner -- duplicating a server-owned reminder/definition is a plain new local draft, never an implicit transfer/push. Fresh id, no borrowed run/transfer/incident history.

Duplicate lifecycle ruling (definitions): an ACTIVE ("configured") source duplicates active, unchanged (the create path's own default already does this, no extra code). A PAUSED/ARCHIVED/DISABLED/unrecognized source collapses to PAUSED via one follow-up SchedulingService.set_definition_lifecycle(new_id, "pause") call made AFTER the create succeeds and OUTSIDE the create's own try/except -- save_definition's create path structurally cannot carry lifecycle through in the payload itself (_definition_db_fields_from_preview builds its fields dict purely from the preview's normalized_config, which has no lifecycle key at all), so a post-create follow-up write is the only way to apply it. Archived/disabled are deliberately NOT preserved verbatim (a duplicate starting in either would be invisible to a plain Resume click, worse than paused). Side effect: the pause follow-up is a real second DB write, so a collapsed copy's version is 2, not 1.

Duplicate pause-failure path (definitions): the pause follow-up's own failure -- a non-"saved" outcome, or a raise -- is NOT a duplicate failure (a real local row already exists) and never reported as one. It gets exactly ONE honest warning toast ("Duplicated '<name>', but the copy could not be paused -- it is active...") and returns immediately -- never also the plain "Duplicated ... as a new local automation" success toast (which would contradict the warning), and never the create's own "Failed to duplicate" error (which would misreport an ACTIVE copy on disk as a failed create, the opposite of what happened).

Reminder pane has no View results button/reason at all (removed in the final review wave): a reminder has no automation-results target and never will (automation_results is keyed by definition_id, a recurring_question-only concept), so a permanently-disabled button plus its own permanent explanation Static was pure standing weight on the one pane whose outstanding problem is running out of vertical room at the 24-row floor -- not the *conditionally* unavailable case UX-073's disabled+reason idiom is for. AC#1 only requires the action reachable from a reminder OR definition pane; the definition pane's own View results (still gated on the transfer lock like Pause/Resume) satisfies it.

Reuse:
- Reminder pane "View runs" reuses 31712 AC#5's existing scroll-to-"Recent runs:" affordance verbatim (`_request_view_runs`, also now the single implementation `on_detail_value_row_activated`'s history-link branch calls too).
- Definition pane "View runs"/"View results" post the SAME ViewDefinitionAuditRequested/ViewDefinitionResultsRequested messages the existing Last-run/Unread-results row activations already post -- zero new workbench-side handling for those two; the buttons are a second entry point onto an already-built navigation.
- Duplicate (both panes) goes through the existing create_reminder/save_definition create paths. New: DuplicateTaskRequested/DuplicateDefinitionRequested messages (events.py), two workbench handlers, a _duplicate_definition_payload() helper (mirrors AutomationDefinitionForm._build_payload's shape, sourced from the stored row instead of form widgets), and the set_definition_lifecycle pause follow-up described above.

Disabled+reason (UX-073): Duplicate joins Edit/Delete in TaskDetail.set_lifecycle_lock's existing gate (mid-transfer); DefinitionDetail gets a new _refresh_duplicate_button gated on the SAME combined reason (_lifecycle_lock_reason or _family_note) that already drives Pause/Resume and the #scheduling-automation-detail-why Static.

Lifecycle actions unaffected (AC#3): pinned by the existing test_schedules_transfer_actions.py / test_schedules_workbench.py lifecycle tests, all still green (extended, not replaced, to also assert Duplicate joins the same gate).

Unset-config-key concretization (definitions, display-only): Duplicate's create path runs task-31414's normalize-on-create backfill, so a source's UNSET generation_mode/scope/finding_policy/retention_policy get written as concrete defaults (e.g. "Balanced findings") in the copy rather than staying "Not set". Runtime behavior is identical either way -- the execution-time resolvers (automation_execution.py) already fall back to the exact same defaults for an absent key.

CSS: .detail-secondary-action-button class (plain class on each Button, never an ancestor-scoped bare-type rule) + #scheduling-task-detail-secondary-actions / #scheduling-automation-detail-secondary-actions row ids, in css/features/_scheduling.tcss; bundle rebuilt via css/build_css.py, check_bundle_sync.py green.

Modified/added files: tldw_chatbook/Scheduling/events.py, tldw_chatbook/UI/Screens/scheduling/task_detail.py, tldw_chatbook/UI/Screens/scheduling/definition_detail.py, tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py, tldw_chatbook/css/features/_scheduling.tcss (+ regenerated tldw_cli_modular.tcss / screen_feature_scheduling.tcss), Docs/User_Guide/schedules.md, Tests/UI/test_schedules_transfer_actions.py, Tests/UI/test_schedules_workbench.py, Tests/UI/test_schedules_automations_tab.py.

Final verification (foreground): Tests/UI/test_schedules_workbench.py + test_schedules_transfer_actions.py + test_schedules_terminology.py + test_schedules_automations_tab.py + test_detail_value_row.py + test_schedules_responsive_floor.py all green except the one PRE-EXISTING red (test_no_bare_reminder_noun_in_schedules_screen_copy, line-471 "Fixed: a reminder always notifies..." offender in task_detail.py; test_the_docked_task_detail_pane_scrolls_to_reveal_history_past_the_fold, the pane's own standing scroll-past-fold problem) -- both reproduce byte-for-byte identically at base, confirmed pre-existing across three separate attribution rounds, out of this task's scope.

=== HISTORY (oldest first) ===

--- Initial implementation ---

First shipped commit 5baeec50a: the shape, ownership ruling, and reuse choices above, EXCEPT the reminder pane originally also rendered a permanently-disabled View results button (removed in the final review wave, see below), and Duplicate's definition-side lifecycle originally reset unconditionally to "configured" (fixed in round 1, see below).

Terminology fix (same commit): my first draft of the reminder pane's "View results" reason used "Reminders don't produce..." -- test_schedules_terminology.py's AST sweep (task-23106: no bare "reminder" noun in user-facing copy on task_detail.py/schedules_workbench.py/reminder_form.py) correctly flagged it. Reworded to "Scheduled tasks don't produce automation results...". (Moot now that the button/Static are removed entirely, but the AST-sweep discipline generalizes.)

--- Review round 1 (REQUEST-CHANGES, commit 624189b9d) ---

Finding 1 (MAJOR, fixed): _duplicate_definition_payload carried no lifecycle key, so a PAUSED/archived/disabled source's Duplicate always landed "configured" (active) via create_automation_definition's own create-path default -- silently making it eligible for the due-run selector (list_armable_automation_definitions gates strictly on lifecycle='configured'), i.e. an LLM pipeline could start spending on schedule with zero further user action. Fixed with the lifecycle-collapse ruling now stated at the top of these notes. Revert-checked in a throwaway worktree at the pre-fix commit (5baeec50a): both the fixed existing test and the new dedicated paused-collapse test fail there, the active-source control test still passes (removed after).

Finding 2 (MINOR/docs, fixed): the unset-config-key concretization behavior (now stated at the top) was undocumented; added here and in Docs/User_Guide/schedules.md.

New tests: test_duplicate_button_collapses_a_paused_source_to_a_paused_copy_not_due_for_selection (paused source -> paused copy, NOT in list_armable_automation_definitions) and test_duplicate_button_keeps_an_active_source_active_and_due_for_selection (active source -> active copy, control, unchanged behavior) in Tests/UI/test_schedules_workbench.py. Existing test_duplicate_button_creates_a_disambiguated_local_copy_of_a_definition's lifecycle/version assertions corrected to match (was silently asserting the bug as correct behavior).

--- Re-review (commit b3ad92e18) ---

Added a warning notify for a pause follow-up that returns non-"saved" -- but without suppressing the unconditional success toast below it (fixed in the final review wave, finding F1) and without any test (fixed in the same wave) and without handling a RAISING pause follow-up at all (also fixed in the same wave).

--- Final whole-branch review wave (F1/F2/F4, this commit) ---

F1 (MINOR, fixed): the pause follow-up's own try/except is now hoisted OUT of the create's try/except entirely (see the "Duplicate pause-failure path" ruling at the top) -- one outcome, one honest toast, never a contradictory pair, never misreported as "Failed to duplicate". New tests: test_duplicate_pause_followup_returning_non_saved_warns_without_a_success_toast and test_duplicate_pause_followup_raising_warns_and_never_reports_failed_to_duplicate (both stub only SchedulingService.set_definition_lifecycle on a real service; save_definition itself is real). Revert-checked individually against the pre-fix commit (b3ad92e18, throwaway worktree, removed after): both fail there with the exact predicted wrong toast ('error' instead of 'warning', or 2 notify calls instead of 1).

F2 (MINOR, design simplification, fixed): removed the reminder pane's permanently-disabled View results button and its #scheduling-task-detail-secondary-why reason Static entirely (see the ruling at the top) -- AC#1 only requires reachability from a reminder OR definition pane. test_view_results_is_permanently_disabled_with_a_reason replaced with test_view_results_is_not_rendered_on_the_reminder_pane (asserts absence via detail.query(...), not just .disabled). Definition pane untouched.

F4 (INFO, addressed): these notes restructured so current behavior opens the record instead of a now-superseded initial description, per this same review's own finding.
<!-- SECTION:NOTES:END -->
