---
id: TASK-1264
title: First-run setup wizard (hermes-modeled onboarding)
status: Done
assignee: []
created_date: '2026-07-28 22:16'
updated_date: '2026-07-29 19:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Guided, skippable, re-runnable setup wizard per Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New user with fresh config is offered the wizard once after startup
- [x] #2 Quick and Full tracks both complete and land in a working app
- [x] #3 Every step is skippable; Esc asks for confirmation and completed steps stay saved
- [x] #4 Wizard is re-runnable from Settings and the command palette with current values prefilled
- [x] #5 Secrets are masked everywhere and encryption offer uses the existing mechanism
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md across 12 tasks: a pure state module (first_run_setup_state.py: offer/resume gating, track/step-id derivation, commit builders, secret-presence and prefill readers, summary-row builder, the commit-sections allowlist oracle) plus a WizardScreen/WizardContainer subclass (FirstRunSetupWizard.py) built on the untouched BaseWizard framework. Every step commits its own section(s) on Next via SetupWizardContainer.commit_config, which enforces the allowlist oracle at runtime and mirrors writes into app_config. Auto-offer and the resume toast are wired through app.py's _push_initial_screen (both splash and no-splash boot paths); re-entry exists from Settings > Diagnostics > Run Setup Wizard and the command palette (Setup: Run setup wizard...), both pushing FirstRunSetupWizard(rerun=True).

Task 12 (this close-out) added:
- Tests/Wizards/test_first_run_setup_integration.py: 6 tests round-tripping wizard commits through a REAL temp TOML file (TLDW_CONFIG_PATH) via save_settings_to_cli_config/load_cli_config_and_ensure_existence -- provider+model commit, wizard-state flags gating should_offer_wizard/should_show_resume_toast, rerun prefill+secret-presence never leaking the raw key via repr(), an upgrader config (pre-existing key, no wizard state) never auto-offering, and summary rows matching persisted state. TestEncryptionAtRest exercises enable_config_encryption end-to-end (pycryptodome IS installed in this venv, so it runs unmarked, not @pytest.mark.optional -- confirmed via `python -c "import importlib.util as u; print(u.find_spec('Cryptodome'))"`).
- Tests/UI/test_first_run_wizard_live_contract.py: 6 app-level Pilot tests against the real TldwCli app (no interactive terminal is available in this environment; see backlog/docs/lessons-live-verification.md), covering every mechanically-checkable item of the brief's live checklist: (1) fresh config + splash disabled -> wizard auto-offers over the initial screen; (2) Esc -> confirm dialog -> "Finish later" -> dismissed, with the started flag proven to land in the REAL isolated config file, then a second TldwCli instance built against that same persisted state shows the resume toast and does NOT re-push; (3) full track, skip every step end-to-end -> wizard dismisses -> Home reachable -> shell nav still works; (4) FirstRunSetupWizard(rerun=True) pushed over a real Settings screen (via Settings > Diagnostics > Run Setup Wizard) -> finishing via "Done" (exit_route=None) pops cleanly back to Settings with no navigation side effect; (5) 80x24 terminal -- Back/Next/Cancel all visible, unclipped, cross-checked against Screen._compositor.render_strips() (not just pre-paint widget state, per the lessons doc); (6) rapid, unsettled Back/Next mashing across provider<->model does not crash or double-advance (the wizard cleanly reaches summary and completes with each active step visited exactly once).
- Tests/Wizards/test_first_run_setup_wizard.py::TestCommandPaletteReentry: closes a real gap found auditing AC #4 -- nothing anywhere exercised SetupWizardProvider (app.py's command-palette bridge) before this. Two tests: the action pushes FirstRunSetupWizard(rerun=True), and an unknown action_id is a no-op.
- Docs/User_Guide/First_Run_Setup.md: the user guide page, exactly as specified in the task-12 brief.

Live-verification evidence (per backlog/docs/lessons-live-verification.md -- evidence, not vibes): the brief's 7-item checklist is covered as follows.
1. Fresh config, splash ON: splash plays, Home renders, wizard appears on top -- NEEDS HUMAN SPOT-CHECK (splash rendering is genuinely visual; see repro command below).
2. Quick track end-to-end with a real/dummy key -> summary -> "Start chatting" lands in Chat: covered at the unit level (Tests/Wizards/test_first_run_setup_wizard.py::test_next_button_click_drives_quick_track_to_completion, ::test_summary_first_run_exit_buttons_set_expected_routes) plus app-level (test_first_run_wizard_live_contract.py::test_back_next_mashing_...completes the quick track and lands on TAB_CHAT).
3. Fresh config, splash OFF: wizard still auto-offers -- PASS, test_first_run_wizard_live_contract.py::test_fresh_config_splash_disabled_wizard_auto_offers.
4. Esc mid-wizard -> confirm -> Finish later -> relaunch -> resume toast, no re-push -- PASS, test_first_run_wizard_live_contract.py::test_escape_finish_later_dismisses_and_next_boot_resumes_via_toast (drives a REAL second TldwCli boot against the persisted config file).
5. Full track, skip every step -> app fully usable -- PASS, test_first_run_wizard_live_contract.py::test_full_track_skip_everything_leaves_app_usable.
6. Re-run from Settings > Diagnostics: prefilled values, key shown as "configured", exits return to Settings ("Done") -- the exit-to-Settings path is PASS at app level (test_rerun_over_settings_done_returns_to_settings); prefill/"configured" copy is covered at the unit level (test_first_run_setup_wizard.py's rerun-prefill tests) and the integration suite's secret-presence-never-leaks test.
7. 80x24 terminal: wizard renders without clipped navigation -- PASS, test_wizard_navigation_visible_at_80x24 (region bounds + Screen._compositor.render_strips() cross-check).

NEEDS HUMAN SPOT-CHECK (2 items -- genuinely require human eyes, no interactive terminal available in this environment):
- Checklist item 1 (splash-ENABLED boot: splash plays, Home renders, wizard appears on top).
- Overall visual look-and-feel / screen-reads-correctly-top-to-bottom (per lessons-live-verification.md's "Verify at the surface the user touches").
Repro command for both:
  TLDW_CONFIG_PATH=/tmp/wizard-live-test/config.toml /path/to/.venv/bin/python -m tldw_chatbook.app
(use this repo's own .venv, e.g. .venv/bin/python -m tldw_chatbook.app; TLDW_CONFIG_PATH must point at a scratch path that does not exist yet, so the app treats it as a genuinely fresh first run -- never point this at ~/.config/tldw_cli/config.toml.)

A mash-testing timing trap worth recording (see the lessons doc addition): pilot.click() resolves its target from a widget's own CACHED region, and that can go stale (confirmed directly: app.get_widget_at() at a button's own reported region center resolved to its PARENT, not the button, after this wizard's Summary step filled in async content) without pilot.click() raising -- it just returns False silently. Every state-changing interaction in the new live-contract file drives the widget directly (Button.press() / setting RadioButton.value) instead, which is what a click ultimately posts, sidestepping compositor-timing flakiness irrelevant to what those tests check.

Files modified/added: Tests/Wizards/test_first_run_setup_integration.py (new), Tests/UI/test_first_run_wizard_live_contract.py (new), Tests/Wizards/test_first_run_setup_wizard.py (TestCommandPaletteReentry added), Docs/User_Guide/First_Run_Setup.md (new), backlog/tasks/task-1264*.md (closed out), backlog/docs/lessons-live-verification.md (new entry).

Full affected suite green: Tests/Wizards/ + Tests/Widgets/test_splash_screen_config_read.py + Tests/UI/test_product_maturity_phase1_first_run.py + Tests/UI/test_first_run_wizard_live_contract.py + Tests/Chatbooks/ = 286 passed, 1 skipped (pre-existing, unrelated: Tests/Chatbooks/test_chatbook_performance.py needs --run-slow), 0 failed.
<!-- SECTION:NOTES:END -->
