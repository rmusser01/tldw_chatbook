---
id: TASK-31741
title: >-
  First-run wizard: Esc dead after provider discovery + contradictory
  key-required copy
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 23:14'
updated_date: '2026-09-06 01:16'
labels:
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Release UAT (dev tip 8e9d1128d4, loaded machine) reported Escape dead on the first-run wizard's Provider step after a failed OpenAI model discovery, while the footer Exit button worked; the same step showed 'You can continue anyway.' beside a hard 'API key required.' block, and Welcome promised every step is skippable. Make Escape's effect impossible to lose invisibly, and make the copy stop promising what the readiness gate refuses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Esc on the Provider step after a failed model discovery opens the Exit setup dialog (live tmux + pilot test)
- [x] #2 A second Esc arriving before the exit dialog has ever painted is absorbed; the dialog can never be dismissed unseen
- [x] #3 Discovery-failure status never says 'You can continue anyway.' while Next is hard-blocked on a missing API key; no-key providers keep the reassurance
- [x] #4 Welcome no longer promises 'every step can be skipped with Next' and names Esc as the universal out
- [x] #5 User guide First_Run_Setup.md matches the shipped copy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live via tmux (done, 4 variants) and harness (done) - Escape opens dialog everywhere; identify mechanism as unsettled-dialog toggle, not a dead key.
2. TDD Bug A: failing test - a second Escape arriving after the grace but BEFORE the dialog has ever painted must not dismiss it (simulate choked render by swallowing the settle callback). Plus a faithful end-to-end sequence pin (real Enter/Enter/Escape with failed discovery + focus-attached assert).
3. Fix: anchor _SettlingGuardedConfirmationDialog's settle clock to first PAINT (call_after_refresh) instead of on_mount; treat not-yet-painted as not settled (swallow Escape).
4. TDD Bug B: copy assertions - provider discovery-failure status must not say 'continue anyway' when Next is hard-blocked by a missing key; Welcome must not promise 'every step can be skipped'.
5. Fix copy: readiness-aware failure sentence at FirstRunSetupWizard.py:2485; soften Welcome subtitle (point at Esc as the universal out).
6. Targeted tests + preflight + live tmux verification; PR to dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LIVE-REPRO EVIDENCE (relfix worktree, scratch profile, tmux -L wizfix, 235x52, no API keys):
- Faithful sequence (Enter past Welcome; Enter on Provider -> OpenAI selected, discovery FAILED with 'Couldn't discover models for OpenAI. You can continue anyway.'; then Escape) run 4x: splash on, splash off, idle CPU, 14x 'yes' CPU load. In EVERY run the first Escape OPENED the 'Exit setup?' dialog. A second Escape >0.5s later silently DISMISSED it (dialog toggles).
- Harness (run_test, real key events, discovery worker really failing, ctrl+n refusal included): app.focused = ProviderChoiceList, attached and displayed; Escape opens the dialog. No orphaned focus on this path.
- Only surviving mechanism consistent with the live report ('nothing happens' twice, button later works): under heavy load the dialog PAINTS late; the settle clock starts at on_mount, so a second is-this-thing-on Escape >0.5s later dismisses the never-painted dialog invisibly. Escape's effect exists but is invisible and self-undoing.

SHIPPED (branch fix/first-run-wizard-esc-and-provider-copy -> PR to dev).

Bug A root cause — NOT the suspected orphaned focus. Faithful reproductions (4 live tmux walks: splash on/off, idle + 14-core load; plus a run_test harness walk with the discovery worker really failing and a ctrl+n refusal) all showed Escape WORKING: app.focused stayed the attached ProviderChoiceList and the first Escape opened the Exit dialog every time. The reproducible defect: Escape TOGGLES the dialog, and _SettlingGuardedConfirmationDialog anchored its 0.5s double-tap grace to on_mount wall-clock (FirstRunSetupWizard.py:10007 pre-change). On the UAT machine (full pytest sweep running) the dialog's PAINT lags seconds behind the push, so a second Escape >0.5s later dismissed a dialog nobody had ever seen — net observable "Escape does nothing, twice" while the mouse Exit button works. Also: the bottom hint line stays visible while the dialog is up, so footer-anchored captures can't tell the states apart.

Fix A: settle clock starts at first delivered frame (on_mount -> call_after_refresh(_mark_settled)); _opened_at None (never painted) absorbs Escape unconditionally. TASK-2314 battery unchanged and green.

Fix B (copy only): ProviderStep._discovery_failure_status renders "You can continue anyway." only when _current_provider_readiness().ready; blocked state says "Add an API key below to continue, or go Back." Welcome now: "most steps can be skipped with Next — Esc exits setup." User guide First_Run_Setup.md updated + stamp.

Paired test arms:
- test_escape_cannot_dismiss_an_exit_dialog_that_never_painted: FAILED pre-fix (1 failed, run recorded 2026-09-05), passes post-fix.
- Tests/Wizards/test_task_31741_wizard_copy.py (3 tests): all FAILED pre-fix, pass post-fix.
- test_provider_discovery_failure_keeps_escape_exit_alive: sequence pin (passes both sides for Escape; blocked-copy assert fails pre-Bug-B-fix).
- Tests/Wizards/ full: 815 passed post-fix.

Test repair (investigated, not papered over): test_focus_scrolls_offscreen_widget_into_view_when_step_overflows pressed Detect + fixed 0.2s pause, but the injected detect seam NEVER delivers rows in this harness (verified: calls=[] on base AND fix; local_state=failed) — its "genuine overflow" guard held only via transient mid-layout state, and my ~1.35ms readiness call in the discovery worker's completion path tipped it (failed 4/4 with fix, passed on base). Repaired by rendering the rows directly via _render_detection_results — deterministic settled overflow; 3/3 green.

Pre-existing, untouched: test_local_provider_probe_feedback_is_visible_and_adjacent fails identically on unmodified origin/dev (file-swap arm). Also observed twice, out of scope: with splash enabled the wizard occasionally mounts then self-dismisses to Home with zero input (setup_started gets persisted); intermittent, exists on origin/dev.

Live post-fix verification (tmux, fresh scratch profile, no keys): Welcome shows new copy; blocked Provider state shows "Add an API key below to continue, or go Back."; Escape opens Exit dialog; double-Escape within grace keeps it open; settled second Escape still cancels.

Full-file arms for Tests/UI/test_first_run_wizard_live_contract.py: two randomized full runs completed (87/94 and 86/94 passed) with every failure triaged — all either pass deterministically in targeted re-runs (load/order flakes; the machine was simultaneously running another session's 10-worker full-suite gate) or fail identically on unmodified origin/dev (file-swap arm: test_local_provider_probe_feedback_is_visible_and_adjacent). A third, deterministic full run was starved by that gate suite after 35+ minutes and was abandoned; targeted deterministic evidence stands.
<!-- SECTION:NOTES:END -->
