---
id: TASK-33008
title: 'Phase 8: First run connects in place and always returns to Console'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-8
  - console
  - settings
  - ux
  - readiness
dependencies:
  - TASK-33001
  - TASK-33004
  - TASK-33005
  - TASK-33007
references:
  - 'tldw_chatbook/UI/Screens/chat_screen.py'
  - 'tldw_chatbook/UI/Navigation/pending_handoff_store.py'
  - 'tldw_chatbook/UI/Navigation/conversation_settings_navigation.py'
  - 'tldw_chatbook/Widgets/Console/console_setup_modal.py'
  - 'tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py'
  - 'tldw_chatbook/UI/Screens/settings_screen.py'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
  - 'Docs/User_Guide/First_Run_Setup.md'
  - 'PRODUCT.md'
  - 'backlog/decisions/012-provider-credential-settings-boundary.md'
  - 'backlog/decisions/033-application-session-state-ownership.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/097-boot-budget-ratchets.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 8 of the model-configuration redesign (judge-synthesis.md §4 P8; spec §8). It ships as one PR.

Why: verified finding C1's remaining gap is a recovery dead end. When a Console chat needs a key, _open_console_provider_recovery (chat_screen.py:20533-20565) posts NavigateToScreen(TAB_SETTINGS) with the category, provider, model and field, but gives no way back. The user adds the key and then has to find Console again by hand.

A working pattern already exists. The Chat settings modal's own credential detour has a revisioned return: ConversationSettingsReturnIntent is staged on HandoffChannel.CONVERSATION_SETTINGS_RETURN (chat_screen.py:3556-3600), and Settings shows a return continuation (settings_screen.py:16980-17004, handler :28656). ADR-033 (033-application-session-state-ownership.md:13-23) requires handoffs to go through PendingHandoffStore.

First run also has three smaller problems:
- 'Use detected <server>' silently adopts the server's first model (chat_screen.py:15397-15479).
- The Get started card draws a snow field behind itself (ConsoleSetupBackdrop, console_setup_modal.py:113). PRODUCT.md:29 rules out 'gratuitous ASCII decoration'.
- The wizard's model step mounts failures such as 'Authentication failed — this API key was rejected' as disabled radio options (FirstRunSetupWizard.py:3836-3906).

Owner decisions:
- D4 keeps credentials in Settings. ADR-012 (012:10, :31) says recovery must open the exact credential control, so the cloud path is a round trip with a return, not inline key entry.
- D2's non-generating key check (phase 5) is how the user confirms the key before returning.
- D1's pristine-chat convergence (phase 1) makes the returning first-run chat use the new defaults.

The judge's budgets: first run with a local server takes 2 keys; with a cloud key it takes about 11 keys and ends back in Console.

Constraints:
- chat_screen.py has 32 lines of headroom (25,331 against 25,363, Tests/Architecture/test_screen_size_ratchet.py:85), so new logic goes in UI/Console_Modules.
- FirstRunSetupWizard.py already measures 10,462 lines against its 10,404 row (test_module_size_ratchet.py:81). That is a red on dev today, and this phase must not make it worse.
- ConsoleSetupModal stays the documented non-dismissible gate (ADR-031, task-16211 refinement).
- First-run copy may only teach keys that the footer shows (ADR-031 rule 5).

Absorbs TASK-32572 (Tab does not cycle the Get started card's buttons). Its 100x30 capture is replaced by the program's full-screen targets.

Depends on:
- Phase 1: pristine-chat convergence (D1).
- Phase 4: the switcher, with its pick and filtered modes and NEEDS SETUP rows.
- Phase 5: the key check and the in-place guidance for out-of-app failures.
- Phase 7: the Providers & Models card that hosts the return action.

After this phase, task-1379 re-runs the Settings critique.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First run with a detected local server takes 2 key presses (Enter, Enter) from the Get started card to a ready composer, measured live at 211x44.
- [ ] #2 First run with a cloud key takes no more than 12 key actions from the Get started card to a ready composer in the same chat, counting a paste as one. The count includes the key check and the return, and the user never navigates back to Console by hand. Measured live at 211x44.
- [ ] #3 Every Console provider-recovery path that leaves for Settings offers a one-key return to the chat it came from. This closes C1's recovery dead end.
- [ ] #4 No Console surface accepts or displays an API key; keys are entered only in Settings ▸ Providers & Models (D4, ADR-012).
- [ ] #5 chat_screen.py stays within its line and method budget, and FirstRunSetupWizard.py does not grow.
- [ ] #6 No ADR-097 ratchet rises; boot CSS bytes go down with the backdrop's removal.
- [ ] #7 Keyboard-only live captures of both journeys at 211x44 and 235x52 are attached, taken with the real stylesheet and a scratch TLDW_CONFIG_PATH.
- [ ] #8 Docs/User_Guide pages updated, each with a Verified-against stamp: console.md (Get started), settings.md (return action) and First_Run_Setup.md (model-step status lines).
- [ ] #9 ./scripts/preflight.sh passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Capture fixes (after `a57bc97f87`, Phase 8 full-screen captures).** Five capture-fix waves and their rounds (70 commits to `1c1730f4b0`, about 680 production lines) fixed what the live captures showed; `qa/model-config-33008-captures/README.md` records each retake. By area:
- Console status row: the cost chip fits the room left by the chips before it, drops whole figures, keeps a focused chip in view and refits after a model switch or a layout change. No subtask's acceptance criteria cover this work; it is recorded here, not in a subtask.
- Footer: under the Get started card it names only the keys that act there (no F6), and under a dialog over the Console only Ctrl+P and Ctrl+Q (`CONSOLE_WORKBENCH_SHORTCUTS_COVERED`).
- Switch model: Enter's label follows the highlighted row and keeps its width, NOT RUNNING wraps and repacks on resize, MLX-LM it cannot check reads "not checked", Find counts in the singular, and no blank row after a probe shortens the list.
- Get started card: a long provider name is shortened so its readiness word stays on one line, and step 1 names the provider that word is about.
- Wizard (TASK-33008.5's area): a refusal of a check that sent no key asks for a key (key-required copy), the Authentication section drops "(optional)" and scrolls into view, the status box hugs its line (`#setup-model-choice.-status`), and the pinned notice keeps off the frame (`.setup-step-error` padding).
- Settings: a key check after the save redraws the return line ("the provider accepted the key"), and the unsaved-changes and Return without saving copy were reworded.

**Review fixes (2026-10-16).** The review of `1c1730f4b0`: the wizard's Test 401, Claude subscription and cleared-key cases and the status picker's Tab stop (see TASK-33008.5), the setup card's focus guard under a dialog, the return line's key-accepted copy for another provider, docs and docstrings, ratchet rows re-pinned to measured sizes, polling waits in the wizard's compose-crash tests, and the phase's new tests gated in the PR lanes.
<!-- SECTION:NOTES:END -->
