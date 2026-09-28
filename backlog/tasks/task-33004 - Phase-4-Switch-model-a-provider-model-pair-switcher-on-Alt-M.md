---
id: TASK-33004
title: 'Phase 4: Switch model, a provider·model pair switcher on Alt+M'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-4
  - console
  - ux
  - readiness
dependencies:
  - TASK-33001
  - TASK-33002
  - TASK-33003
references:
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/mockups-211x44.md'
  - 'tldw_chatbook/Widgets/Console/console_model_popover.py'
  - 'tldw_chatbook/UI/Console_Modules/model_switcher.py'
  - 'tldw_chatbook/UI/Screens/chat_screen.py'
  - 'tldw_chatbook/Chat/console_settings_apply.py'
  - 'tldw_chatbook/Chat/console_settings_defaults.py'
  - 'tldw_chatbook/Chat/console_chat_controller.py'
  - 'tldw_chatbook/Widgets/model_search_picker.py'
  - 'tldw_chatbook/Chat/console_command_grammar.py'
  - 'tldw_chatbook/UI/Console_Modules/left_rail.py'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/012-provider-credential-settings-boundary.md'
  - 'backlog/decisions/066-local-provider-thinking-controls.md'
  - 'backlog/decisions/146-console-custom-endpoint-registry.md'
  - 'backlog/decisions/020-automatic-model-catalog-refresh.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/097-boot-budget-ratchets.md'
  - 'backlog/decisions/147-agent-provider-routing.md'
  - 'Tests/Architecture/test_screen_size_ratchet.py'
  - 'Docs/User_Guide/console.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 4 of the model-configuration redesign (judge-synthesis.md section 2a, section 3 'Keep and reshape', and section 4 P4; spec §8). It ships as one PR.

The Alt+M quick popover (Widgets/Console/console_model_popover.py, 1,432 lines) is a 170x32 form, not a switcher (C6):
- Its 2x2 action grids put 1-row buttons in 3-row cells (:253-262).
- Temperature is clipped at the fold of a body about 20 rows tall.
- Focus lands on the scroll body, because on_mount sets none (:682-687).
- Max tokens is a read-only Static, because ADR-095:74 limits the quick mask.

Other problems in the same area:
- Choosing a provider rebases with no model (:1090-1101). That is how a provider switch inherited a model from another provider (C7(a)); phase 1 guards the resolver by provider.
- The shared ModelSearchPicker never marks or pre-highlights the committed model (Widgets/model_search_picker.py:643-644, :697-705, :866-875; C7(b)).
- There are no recents, so an A/B toggle takes 16-20 keys and Alex's switch-and-tune task takes 45 actions.
- /model drops its argument (Chat/console_command_grammar.py:110; UI/Screens/chat_screen.py:19211 and :19362-19383).
- Chat settings has no Console key of its own: Ctrl+O is unbound anywhere in the app, yet the spec's footer teaches it.

This phase replaces the popover's compose() with 'Switch model':
- provider·model pairs grouped as PREVIOUS / RECENT / READY PROVIDERS / NEEDS SETUP
- a ● CURRENT mark on the current pair
- the previous pair highlighted when the switcher opens
- a one-row value strip of exactly the quick mask: Temperature, Max tokens, Streaming

Owner decisions applied:
- D3: the ADR-095 amendment of 2026-09-26 (already drafted) adds max_tokens to the quick mask, and the quick surface's editable fields equal that mask, so Thinking stays in Chat settings (spec §8 deliberate change 3).
- D4: credentials stay in Settings. NEEDS SETUP rows route there, and Console gets no key entry.

Absorbed tasks:
- task-338: Streaming in the rail.
- task-32859: one provider-selection builder, which the switcher consumes.
- task-194: popover display names. It is popover-only; spec deliberate change 4 closes it here, and the chips are phase 2's separate fix.

Depends on:
- phase 1: the provider-guarded default model and the single supported-field function
- phase 2: shared field labels, Source words and display names
- phase 3: compact control tokens and the scoped CSS leaks

Constraints:
- The ConsoleModelPopover class stays, and so do the ids console-popover-apply, -temperature, -streaming, -save-model-default and -make-new-chat-default. 17 test files query popover ids.
- chat_screen.py has 32 lines of headroom (25,331 of 25,363, Tests/Architecture/test_screen_size_ratchet.py:85), so switcher logic lives under UI/Console_Modules/.
- The popover is imported on the boot path (task-32644), so ADR-097's _ui_ready census applies.
- Textual Input owns Ctrl+A/E/D/K/U/W/X/C/V while Find has focus, so the switcher's chords are Ctrl+N and Ctrl+O.

Out of scope:
- favourites, pins and Alt+1..9 (the synthesis waits for user testing)
- any readiness probing
- task-18922's one-turn override, which stays open
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Alt+M, the Provider/Model chips, the palette entry and /model [query] all open Switch model. At 211x44 and 235x52 it is 120 columns wide with auto height up to 80% of the screen, and focus starts in Find.
- [ ] #2 Alex's task (switch to a Sonnet model, set temperature 0.9 and max tokens 8192, return to the composer) takes 14 key presses. The A/B toggle takes 2, a model-only change takes 5, and a model change saved as the model default takes 16 or fewer.
- [ ] #3 No switcher path applies a provider without a model (closes C7(a) in the UI). The current pair is marked, and the previous pair is highlighted when the switcher opens (closes C7(b)). ModelSearchPicker also marks the committed model.
- [ ] #4 C6 is closed: no 3-row button cells remain, and at 211x44 Temperature and Max tokens are editable and visible without scrolling.
- [ ] #5 The quick default mask in code is temperature, max_tokens and streaming, matching the ADR-095 amendment of 2026-09-26 (D3).
- [ ] #6 Ctrl+O opens Chat settings from the Console, and the Console footer, F1 help and palette teach Alt+M, Ctrl+O and /model [query].
- [ ] #7 This PR closes task-338, task-32859 and task-194.
- [ ] #8 chat_screen.py ends the phase within its line and method budget (Tests/Architecture/test_screen_size_ratchet.py:85). If the phase shrank the file, the budget is lowered to the measured size.
- [ ] #9 No ADR-097 ratchet rises: the boot CSS byte budget (608,090, Tests/Performance/test_boot_css_byte_budget.py:117) and the _ui_ready census (1,031, Tests/Performance/test_ui_ready_module_census.py:150). Modules used only by the switcher are not imported on the boot path. If dev is already red, the branch's number equals dev's.
- [ ] #10 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added.
- [ ] #11 Every printed key hint matches a working binding (ADR-031 rule 4).
- [ ] #12 Live captures at 211x44 and 235x52, taken with a scratch TLDW_CONFIG_PATH profile, are attached to the PR.
- [ ] #13 Docs/User_Guide pages are updated: console.md covers Switch model, /model [query], Ctrl+O, the keys table and the rail's Change action, with a Verified-against stamp.
- [ ] #14 ./scripts/preflight.sh passes.
<!-- AC:END -->
