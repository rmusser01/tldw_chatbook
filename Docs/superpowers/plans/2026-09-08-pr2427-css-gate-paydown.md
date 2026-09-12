# PR 2427 CSS gate paydown implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Track the steps below without weakening the qualification gates.

**Goal:** Clear the 281/274 selector and 811,541/804,000 startup-byte failures while preserving appearance and interactions.

**Architecture:** Re-key seven existing rules to dedicated classes/IDs on their existing subjects. Reuse `ScreenOwnedSplit` and the app's route-owned stylesheet loader for Watchlists-only rules; shared/mixed rules remain eager. No new loading framework or per-widget CSS sources.

**Tech stack:** Python 3.12, Textual 8, existing TCSS builder and pytest.

**Approved design:** User approved the proposals recorded in `backlog/docs/pr-2427-rebase-reconciliation.md` and the final approval question after `c27723b623`. This plan implements that design, not a visual redesign.

ADR required: no new ADR.
ADR path: `backlog/decisions/097-boot-budget-ratchets.md`.
Reason: direct application of existing cost deferral and existing route-owned CSS contracts. No cap exception.

## Task 1: Narrow the seven added broad selectors

Files: `tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py`, `new_task_choice_modal.py` in that directory, `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py`, `tldw_chatbook/UI/Library_Modules/skill_import_choice_modal.py`, `tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py`; generated `tldw_chatbook/css/widget_defaults_{self,scoped}.tcss`; tests in `Tests/UI/test_widget_css_consolidation.py` and a focused selector-parity test if the existing file lacks suitable coverage.

- [x] Record existing rule subjects and computed styles for representative mounted controls before edits. Cover normal/disabled/focus states and small-terminal geometry; preserve inheritance and cascade specificity. Keep IDs, nesting, labels, callbacks, and actions unchanged.
- [x] Run the unchanged selector census to reproduce 281 > 274: `.venv/bin/python -m pytest Tests/Performance/test_textual_css_fastpath.py --basetemp=<unique-temp-root> -q`.
- [x] Add a regression checking that the seven owner scopes contain no descendant bare-type subjects while their intended controls retain the baseline properties. Observe the structural assertion fail before editing production code.
- [x] Replace only these subjects, adding dedicated classes to exact existing controls when needed: AutomationDefinitionForm `.button-container Button` and `> VerticalScroll`; NewTaskChoiceModal `.new-task-choice-actions Button`; ProviderContinuationRecoveryCallout `Button`; SkillImportChoiceModal `#skill-import-choice-actions Button`; ToolProfilesPanel `.tool-profile-actions Button` and `Button`. Preserve ancestry where it carries cascade precedence; do not substitute a weaker unscoped selector.
- [x] Regenerate with `.venv/bin/python tldw_chatbook/css/build_css.py`; never hand-edit generated sheets.
- [x] Run the complete fastpath and widget consolidation files, the new parity file, and affected complete scheduling/skill-import/provider-recovery/tool-profile test files found by caller census. Compare mounted computed styles and painted/hit-tested controls against the captured baseline, not merely class presence. Fastpath:5 passed; parity:27 passed. Complete affected groups:289 passed and331 passed/2 pre-existing Scheduling failures. Consolidation:32 passed/1 pre-existing four-declaration failure. These remaining gates are explicitly tracked, not waived.
- [x] Obtain spec then correctness review; commit only this attributable selector change with its tests/evidence. Both reviews pass after adding the paired full-app-tier baseline controls; see reconciliation report.

## Task 2: Defer Watchlists-only stylesheet bytes

Files: `tldw_chatbook/css/build_css.py`, `tldw_chatbook/app.py`, generated `tldw_chatbook/css/tldw_cli_modular.tcss` and `screen_feature_watchlists.tcss`, `Tests/UI/test_css_build_integrity.py`, `Tests/Performance/boot_budget_snapshots/boot_css_bytes.json` after an under-cap measurement.

- [ ] Extend existing route-loading tests before implementation: absent at ordinary Console boot, present before Watchlists first paint, no duplicate parsing on repeat entry, and present for Watchlists as initial route. Add the new sheet to the union/reproduction checks. Observe missing split/route failures.
- [x] Reconfirm all moved selector tokens have compose consumers only in Watchlists. The conservative split found 11,203 bytes. A fresh exact-token audit safely adds six units (1,568 bytes): overview-card/overview-failed-runs/overview-first-run are composed only in `UI/Watchlists_Modules/overview_pane.py`; wl-workbench-body only in `watchlists_workbench.py`; wl-centre-status and wc-empty-actions only in `UI/Screens/watchlists_collections_screen.py`. No foreign-screen consumers or later-module collisions were found. Generic/mixed selectors remain eager. The expanded source partition moves12,771 bytes, with final boot cost still to measure.
- [ ] Add `ScreenOwnedSplit(module="features/_watchlists.tcss", sheets={"watchlists": "screen_feature_watchlists.tcss"}, prefixes={"watchlists": ("watchlists", "wl", "wc", "overview")}, pinned=frozenset())` according to that verified consumer audit.
- [ ] Add `TAB_WATCHLISTS_COLLECTIONS: ("screen_feature_watchlists.tcss",)` to the existing `_SCREEN_OWNED_ROUTE_CSS`. Do not add it to global boot CSS or the screen's `CSS_PATH`; harness styling tiers must remain intact.
- [ ] Regenerate all sheets with the existing builder. Run complete `test_css_build_integrity.py`, `test_widget_css_consolidation.py`, `Tests/Performance/test_boot_css_byte_budget.py`, and `test_boot_budget_ratchet_messages.py`. Actual startup total, not projection, must remain <=804,000; no raised constants or force snapshot refresh.
- [ ] Run complete Watchlists destination-shell, overview-loading, inspector, select-overlay and run-detail affected files. Verify real navigation/initial-route and 160x45 / 235x52 painted controls, with a compact supported terminal check where existing coverage requires it. Preserve all original behavioral assertions.
- [ ] Only after under-cap evidence, run `.venv/bin/python scripts/update_boot_budget_snapshots.py --only css`; verify the snapshot and generated artifacts again. Apply ADR-097 downward tightening only if its standard-slack condition actually holds.
- [ ] Obtain spec then correctness review. Commit this split separately from selector changes and controller moves.

## Task 2b: Consolidate four pre-existing modal defaults without cap exceptions

Files: the three source modules declaring LibraryCharacterRepairDialog,
RoleplayDraftNavigationDialog/RoleplayDraftRecoveryDialog, and
ConsoleAppearancePickerModal; existing generated widget-default sheets;
focused full-tier parity and affected complete modal tests.

ADR required: no new ADR.
ADR path: `backlog/decisions/097-boot-budget-ratchets.md`.
Reason: existing default-tier BUNDLED_CSS mechanism, unchanged UI behavior and
all numerical caps. This addresses the independently reproduced consolidation
guard failure under the user's request to address all PR issues.

- [ ] Read the exact owner tasks/source, reproduce the unchanged consolidation failure, and capture full-app-tier incumbent computed styles, geometry, focus/disabled paint and hit targets for all four dialogs.
- [ ] Add failing class-subject/consolidation and paired cascade controls before production changes. Give only existing Library Select/Horizontal/Button, Navigation Vertical/Button, and Appearance action Button subjects dedicated classes so the broad-type census does not grow.
- [ ] Convert the three effective default blocks to BUNDLED_CSS, preserving their default cascade tier through the established builder. Verify the Recovery alias scopes Navigation-named selectors under a non-subclass Recovery owner and is genuinely inert; remove only that redundant alias if paired normal/focus/disabled controls prove unchanged presentation.
- [ ] Rebuild, measure actual startup bytes and selector census, and require <=804,000 /274. Read-only simulation predicts +4,206 bytes; expanded Watchlists deferral provides1,568 bytes beyond the original plan, but projections are not evidence. No allowlist addition, cap raise, comment compression, or default-to-app tier migration.
- [ ] Run complete consolidation/fastpath/boot budget/parity files and actual affected modal files. Refresh the CSS snapshot only after real under-cap measurement. Obtain spec then correctness review and commit separately from the Watchlists split.

Task 2b independent plan review: ready. In addition to class-key comparisons,
Appearance's original comma-separated Clear/Cancel declaration must be compared
against the generated registration for both buttons, because consolidation
normalizes selector scoping. Recovery must be compared against its original
class-level alias, not a reconstructed approximation.

Task 2/2b review checkpoint: independent spec and correctness reviews pass.
Measured combined cost after the header-only wording correction is803,075/804,000
bytes and273/274 broad subjects; no cap was raised. Complete modal/navigation
47tests and CSS/consolidation51tests pass. Full-app fixture loading and real
backend filter seeding were reconciled without suppressing production reloads.
The complete Watchlists re-run remains pending with two independently reproduced
pre-split focus-contrast failures explicitly open for separate approval.
The shared generated header now describes app/owning-screen loading accurately.

## Task 3: Integration and PR qualification

- [ ] Run both complete CSS budget files together and all six `scripts/preflight.sh` checks using the isolated interpreter. Retain failure/warning evidence honestly.
- [ ] Review combined source/generated diffs and visual parity evidence. No production logs or diagnostic manifests should change except attributable generated inventory updates reviewed separately.
- [ ] Update TASK-31932 and reconciliation evidence, push checkpoints, inspect current Qodo review and GitHub Actions, and leave merge blocked until every remaining size gate and final-head check is satisfied.

Workspace: `.worktrees/pr2427-review-recovery` only. Use its `.venv/bin/python`; never mutate the original checkout or its environment. Each pytest run uses a fresh task-owned temporary root. The root agent owns shared reports, plan progress, final artifact regeneration, Git publication and merging; workers do not stage another worker's files.
