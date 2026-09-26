---
id: TASK-33006
title: 'Phase 6: Chat settings layout — core first, honest fields, pairs-only model change'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-6
  - console
  - ux
  - css
  - a11y
dependencies:
  - TASK-33001
  - TASK-33002
  - TASK-33003
  - TASK-33004
  - TASK-33005
references:
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'tldw_chatbook/Widgets/Console/console_settings_modal.py'
  - 'tldw_chatbook/Widgets/Console/console_provider_picker.py'
  - 'tldw_chatbook/css/features/_console_panels.tcss'
  - 'tldw_chatbook/css/core/_variables.tcss'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/161-component-pattern-library.md'
  - 'backlog/decisions/150-design-token-system-and-design-language.md'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 6 of the model-configuration redesign "Switchboard with field truth" (judge-synthesis.md §2(b) mock and §4 P6; spec §8). It ships as ONE PR. It is new work: task-32864 explicitly excludes this modal. It closes the UI half of C8(1) (the hidden-for-provider line). For chats that hold work, it also closes C1(b) through 'Use saved defaults', the D1 action the ADR-095 amendment of 2026-09-26 names. Untouched chats have already converged since phase 1.

Why. The Chat settings modal, today titled 'Conversation settings' (console_settings_modal.py:1648), has five problems:
- It opens connection-first. The Connection section holds its own ConsoleProviderPicker (:1719-1735) and ModelSearchPicker (:1794).
- Every tuning field is inside an 'Advanced generation' Collapsible (:1954-1960): Temperature, Top P, Min P, Top K, Response max tokens, Seed, Presence, Frequency and a Streaming toggle button (:1962-2034). The reasoning rows that follow each carry a 'Support not verified for this model.' static (GENERATION_CONTROL_UNKNOWN_COPY, :847).
- Anthropic shows Min P, Seed, Presence and Frequency. Its PROVIDER_PARAM_MAP entry lacks them (Chat_Functions.py:281-300), and project_chat_handler_kwargs drops them silently (:1431-1435).
- A provider can be chosen without a model, because the provider picker (console_provider_picker.py, 455 lines, used only here: :156, :1725) is separate from model search.
- A chat that holds work has no way to adopt newly saved defaults.

'Conversation settings' also appears in 47 strings across the code, many of them user-visible notices (for example chat_screen.py:3747-3835 and :15168, settings_screen.py:12231-12239 and :16976).

In the review, Alex needed Tab ×13 to reach Apply (report.md). The module sits at zero ratchet headroom: 7,807 lines against a 7,807 budget (Tests/Architecture/test_module_size_ratchet.py:68).

Target: judge mock (b), 150x22 over Console at 211x44. The layout is a Model row, Core fields, then Sampling / Connection / Request estimate / name as one-row disclosures, with a Source column and a help line per field, and 'Applies to this chat only'. It builds on earlier phases:
- P1: the single supported-fields function and D1 convergence.
- P2: the field table.
- P3: compact tokens, the Esc dirty guard and the fold-hint fix.
- P4: the switcher's pick-only mode and the Source-word resolver.
- P5: the shared readiness words.

Constraints:
- ADR-150/161: geometry lives in tokens in core/_variables.tcss only.
- ADR-031 rule 2: no Ctrl+C, V, X, S, D, Z, A, R or W.
- ADR-033: keep the commit models.
- ADR-095: Apply is conversation-owned, and explicit source-owned chats never rebase.
- ADR-097: ratchets never rise.
- Target sizes: 211x44 first, then 235x52. Smaller sizes are not redesigned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The phase ships as one PR containing every subtask below
- [ ] #2 At 211x44 and 235x52, Chat settings shows the whole Model view without scrolling: the model row, the core fields and the footer actions, core first, each field with a Source word and one help line
- [ ] #3 For Anthropic, one line names the fields it does not accept, instead of four editable dead fields (C8(1) UI closed)
- [ ] #4 The model is changed only by choosing a provider·model pair in the switcher's pick mode, and ConsoleProviderPicker and its module are deleted
- [ ] #5 A chat with any work can adopt newly saved defaults through 'Use saved defaults' and Apply, and no configuration is written by that path (C1(b) for chats with work, D1)
- [ ] #6 No user-visible copy names this modal 'Conversation settings'; it is 'Chat settings' everywhere
- [ ] #7 console_settings_modal.py shrinks, and its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in this PR. Boot CSS bytes and the _ui_ready census do not rise (ADR-097)
- [ ] #8 New geometry comes only from tokens in core/_variables.tcss. The dimension-literal and Python-style ratchets in Tests/UI/test_component_pattern_governance.py (:266, :291) stay at their floors (ADR-150/161)
- [ ] #9 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added, and every key the modal advertises works
- [ ] #10 Every existing test that pinned the old layout is rewritten on purpose and listed in the PR description, and the Context and memory view's tests pass unchanged
- [ ] #11 Live evidence: tmux captures at 211x44 and 235x52 from a scratch profile (TLDW_CONFIG_PATH), using the production stylesheet and real keypresses. They cover an Anthropic chat, a Not-ready chat and a chat using 'Use saved defaults'
- [ ] #12 Docs/User_Guide pages updated: console.md (Chat settings) and settings.md (the Console-modal reference near :373), with Verified-against stamps
- [ ] #13 ./scripts/preflight.sh passes
<!-- AC:END -->
