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

- [ ] Record existing rule subjects and computed styles for representative mounted controls before edits. Cover normal/disabled/focus states and small-terminal geometry; preserve inheritance and cascade specificity. Keep IDs, nesting, labels, callbacks, and actions unchanged.
- [ ] Run the unchanged selector census to reproduce 281 > 274: `.venv/bin/python -m pytest Tests/Performance/test_textual_css_fastpath.py --basetemp=<unique-temp-root> -q`.
- [ ] Add a regression checking that the seven owner scopes contain no descendant bare-type subjects while their intended controls retain the baseline properties. Observe the structural assertion fail before editing production code.
- [ ] Replace only these subjects, adding dedicated classes to exact existing controls when needed: AutomationDefinitionForm `.button-container Button` and `> VerticalScroll`; NewTaskChoiceModal `.new-task-choice-actions Button`; ProviderContinuationRecoveryCallout `Button`; SkillImportChoiceModal `#skill-import-choice-actions Button`; ToolProfilesPanel `.tool-profile-actions Button` and `Button`. Preserve ancestry where it carries cascade precedence; do not substitute a weaker unscoped selector.
- [ ] Regenerate with `.venv/bin/python tldw_chatbook/css/build_css.py`; never hand-edit generated sheets.
- [ ] Run the complete fastpath and widget consolidation files, the new parity file, and affected complete scheduling/skill-import/provider-recovery/tool-profile test files found by caller census. Compare mounted computed styles and painted/hit-tested controls against the captured baseline, not merely class presence.
- [ ] Obtain spec then correctness review; commit only this attributable selector change with its tests/evidence.

## Task 2: Defer Watchlists-only stylesheet bytes

Files: `tldw_chatbook/css/build_css.py`, `tldw_chatbook/app.py`, generated `tldw_chatbook/css/tldw_cli_modular.tcss` and `screen_feature_watchlists.tcss`, `Tests/UI/test_css_build_integrity.py`, `Tests/Performance/boot_budget_snapshots/boot_css_bytes.json` after an under-cap measurement.

- [ ] Extend existing route-loading tests before implementation: absent at ordinary Console boot, present before Watchlists first paint, no duplicate parsing on repeat entry, and present for Watchlists as initial route. Add the new sheet to the union/reproduction checks. Observe missing split/route failures.
- [ ] Reconfirm all moved selector tokens have compose consumers only in Watchlists. The prior in-memory conservative split found 33 selectors / 25 tokens / 11,203 bytes. Audit repo-relative consumers; keep any shared tokens pinned and mixed rules eager.
- [ ] Add `ScreenOwnedSplit(module="features/_watchlists.tcss", sheets={"watchlists": "screen_feature_watchlists.tcss"}, prefixes={"watchlists": ("watchlists",)}, pinned=frozenset())` if the consumer audit still supports no pins.
- [ ] Add `TAB_WATCHLISTS_COLLECTIONS: ("screen_feature_watchlists.tcss",)` to the existing `_SCREEN_OWNED_ROUTE_CSS`. Do not add it to global boot CSS or the screen's `CSS_PATH`; harness styling tiers must remain intact.
- [ ] Regenerate all sheets with the existing builder. Run complete `test_css_build_integrity.py`, `test_widget_css_consolidation.py`, `Tests/Performance/test_boot_css_byte_budget.py`, and `test_boot_budget_ratchet_messages.py`. Actual startup total, not projection, must remain <=804,000; no raised constants or force snapshot refresh.
- [ ] Run complete Watchlists destination-shell, overview-loading, inspector, select-overlay and run-detail affected files. Verify real navigation/initial-route and 160x45 / 235x52 painted controls, with a compact supported terminal check where existing coverage requires it. Preserve all original behavioral assertions.
- [ ] Only after under-cap evidence, run `.venv/bin/python scripts/update_boot_budget_snapshots.py --only css`; verify the snapshot and generated artifacts again. Apply ADR-097 downward tightening only if its standard-slack condition actually holds.
- [ ] Obtain spec then correctness review. Commit this split separately from selector changes and controller moves.

## Task 3: Integration and PR qualification

- [ ] Run both complete CSS budget files together and all six `scripts/preflight.sh` checks using the isolated interpreter. Retain failure/warning evidence honestly.
- [ ] Review combined source/generated diffs and visual parity evidence. No production logs or diagnostic manifests should change except attributable generated inventory updates reviewed separately.
- [ ] Update TASK-31932 and reconciliation evidence, push checkpoints, inspect current Qodo review and GitHub Actions, and leave merge blocked until every remaining size gate and final-head check is satisfied.

Workspace: `.worktrees/pr2427-review-recovery` only. Use its `.venv/bin/python`; never mutate the original checkout or its environment. Each pytest run uses a fresh task-owned temporary root. The root agent owns shared reports, plan progress, final artifact regeneration, Git publication and merging; workers do not stage another worker's files.
