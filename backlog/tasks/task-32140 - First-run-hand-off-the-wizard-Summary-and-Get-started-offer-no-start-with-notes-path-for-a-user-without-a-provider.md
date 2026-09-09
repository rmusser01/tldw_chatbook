---
id: TASK-32140
title: >-
  First-run hand-off: the wizard Summary and Get started offer no
  start-with-notes path for a user without a provider
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 15:56'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - onboarding
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both assessors on a fresh profile: the Summary step's actions are provider setup, Explore Home and settings; the post-setup Console card lists provider, model and first message. A local-first user who came for notes is told the only thing they can do needs an API key. task-32072 (merged in #2531) added 'Add your first document'; a notes path is still missing. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Summary step offers 'Write your first note' and the Get started card offers 'Write a note in Library' (per task-7-brief.md's verbatim per-surface copy), both landing in Library ▸ Notes ▸ New note
- [x] #2 Covered by a wizard test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a Write your first note button beside Add your first document on the wizard Summary step, and a sentinel exit_route (EXIT_ROUTE_LIBRARY_NOTES) that app.py rewrites to TAB_LIBRARY + LIBRARY_NAV_CONTEXT_NOTES_CREATE (same destination as the new_note quick action).\n2. Add a needs-no-provider Write a note in Library action to ConsoleSetupModal alongside the detected-server action, wired through WorkbenchActionRequested to the same Library New note route in ChatScreen.\n3. Add the label copy to console_onboarding_state.py.\n4. Write failing tests in Tests/UI/test_library_notes_wave_onboarding.py, then implement to green.\n5. Update First_Run_Setup.md and console.md; live-verify on the fresh profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a 'Write your first note' button to the wizard Summary step beside 'Add your first document', wired to a new sentinel exit_route (EXIT_ROUTE_LIBRARY_NOTES in FirstRunSetupWizard.py) that app.py's _continue_first_run_wizard_result rewrites to TAB_LIBRARY + LIBRARY_NAV_CONTEXT_NOTES_CREATE -- the same destination the command-palette 'new_note' quick action already uses (Library Row CREATE_NOTE). Added a matching 'Write a note in Library' action to ConsoleSetupModal's Get started card, shown whenever the card is blocking (needs no provider, unlike the numbered steps), wired through a new WorkbenchActionRequested action id (CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION) into ChatScreen.on_console_workbench_action_requested, which posts the same NavigateToScreen. Copy constants (CONSOLE_SETUP_NOTES_ACTION_LABEL/_TOOLTIP) live in console_onboarding_state.py per its existing copy-constant pattern. Files: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py, tldw_chatbook/app.py, tldw_chatbook/Widgets/Console/console_setup_modal.py, tldw_chatbook/Chat/console_onboarding_state.py, tldw_chatbook/UI/Screens/chat_screen.py, Docs/User_Guide/First_Run_Setup.md, Docs/User_Guide/console.md, Tests/UI/test_library_notes_wave_onboarding.py (5 new tests, TDD RED->GREEN). Live-verified on the fresh onboarding profile: wizard Summary -> Write your first note -> lands on Library/Notes/New note; Console Get started card (no provider) -> Write a note in Library -> same destination. AC#1's original wording asked the Get started card to offer literally 'Write your first note'; the brief's task-7-brief.md gives distinct verbatim copy per surface -- 'Write your first note' on the wizard Summary, 'Write a note in Library' on the Console card -- and I implemented the brief's two strings (matches each surface's existing voice; both land on the identical New note route). Per review round 1: reconciled by amending AC#1's text to name both exact strings rather than leaving the original single-string wording checked against a differently-worded implementation.

Addendum (PR #2538 review, Qodo findings 3 and 4): the Summary's five exit actions (chat/library/notes/home/settings) no longer fit one non-wrapping Horizontal at either supported wizard size (80x24 or 120x40) -- verified with a live pilot probe that "Explore Home" and "Review settings" were pushed fully off-screen and absent from the compositor's visible widgets at both sizes. Split the row into two docked Horizontals ([chat, library] / [notes, home, settings]) via a new .setup-summary-actions-row CSS class; both rows now fit within 80 columns. Also added the Returns-documented docstring Qodo flagged on SetupSummaryStep.compose_step. test_summary_three_actions_visible_and_focused_on_full_track (already broken before this PR by task-32072's 4th button, confirmed against the pre-onboarding base commit in a throwaway worktree) was renamed to test_summary_five_actions_... and extended to check all five current buttons are on-screen and unclipped at both sizes. Files: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py, tldw_chatbook/css/features/_wizards.tcss (+ rebuilt tldw_cli_modular.tcss), Tests/UI/test_first_run_wizard_live_contract.py.
<!-- SECTION:NOTES:END -->
