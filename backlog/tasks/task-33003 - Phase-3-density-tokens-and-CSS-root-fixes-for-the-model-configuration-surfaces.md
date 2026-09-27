---
id: TASK-33003
title: 'Phase 3: density tokens and CSS root fixes for the model-configuration surfaces'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-3
  - css
  - console
  - settings
  - a11y
  - ux
dependencies: []
references:
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'tldw_chatbook/Widgets/Console/console_settings_modal.py'
  - 'tldw_chatbook/css/features/_conversations.tcss'
  - 'tldw_chatbook/css/features/_evaluation_unified.tcss'
  - 'tldw_chatbook/css/components/_agentic_terminal.tcss'
  - 'tldw_chatbook/css/features/_console_panels.tcss'
  - 'tldw_chatbook/css/features/_settings.tcss'
  - 'tldw_chatbook/css/core/_variables.tcss'
  - 'tldw_chatbook/css/Themes/themes.py'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'Tests/UI/test_component_pattern_governance.py'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/150-design-token-system-and-design-language.md'
  - 'backlog/decisions/161-component-pattern-library.md'
  - 'backlog/decisions/097-boot-budget-ratchets.md'
  - 'DESIGN.md'
  - 'backlog/docs/design-language.md'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 of the model-configuration redesign ('Switchboard with field truth', qa/model-config-ux-review-2026-09-26/judge-synthesis.md section 4). It ships as one PR.

At full screen, the Chat settings modal spends its rows on chrome and blank space, and focus and boundaries are close to invisible (report.md priority issues 1 and 2). The verified root causes are CSS and one constant, not layout:
- C5(a): MODAL_CONTROL_HEIGHT = 3 (Widgets/Console/console_settings_modal.py:168) is applied through DEFAULT_CSS at :1056-1071 and through the app-tier rule css/components/_agentic_terminal.tcss:43-52. Label cells are 3 rows too (css/features/_console_panels.tcss:65-74).
- C5(b): three unscoped global rules make one collapsed section cost about 7 rows: css/features/_conversations.tcss:298-303 and :313-316, and css/features/_evaluation_unified.tcss:59-63.
- C5 width: controls are about 125 columns wide because of a 75% cap (_console_panels.tcss:77-81).
- C5(c): the fold hint never reads scroll_y (console_settings_modal.py:3850-3866).
- C8(2): Esc, a backdrop click and Cancel discard an edited draft without asking (:4041-4054). This contradicts ADR-031's task-16211 refinement.
- C8(4): the Settings category rail signals focus only with a 1.1-1.2:1 background swap, and a focused active row looks the same as an unfocused one (css/features/_settings.tcss:193-209).
- Report issue 2 also measured a Select highlight at 1.12:1 and a focused primary button that gets darker.

This phase absorbs task-25890 (the boundary note escapes the impact pane). The synthesis says this phase also covers task-32465 for these surfaces. At HEAD, though, none of these files renders a RadioSet or RadioButton: the Chat settings modal, the quick model popover, and settings_screen.py (checked with grep). There is nothing of task-32465 to cover here, so it stays open unchanged.

Constraints:
- Geometry comes only from tokens defined in css/core/_variables.tcss (ADR-150/161). The floor for raw literals is a hard zero (Tests/UI/test_component_pattern_governance.py:266-289).
- ADR-097 ratchets never rise.
- console_settings_modal.py has zero headroom: its row in Tests/Architecture/test_module_size_ratchet.py:68 is 7,807, which is its current length. Modal edits are paid for by moving its DEFAULT_CSS control rules to the app tier.
- No binding from ADR-031 rule 2 (Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+S, Ctrl+D, Ctrl+Z, Ctrl+A, Ctrl+R, Ctrl+W).

Not in this phase: the inverted select-row height in Settings (css/features/_settings.tcss:465-470), field order, and the quick model popover. Design for 211x44 first and 235x52 second.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, under the production stylesheet, every Input, Select and in-form Button in Chat settings is one row tall, and a collapsed section costs at most 2 rows (closes C5(a) and C5(b)).
- [ ] #2 At 211x44, Chat settings numeric fields are sized to their value, not about 125 columns wide.
- [ ] #3 The Chat settings fold hint is hidden once the body is scrolled to the bottom and shows whenever content remains below (closes C5(c)).
- [ ] #4 Esc, a backdrop click and Cancel never discard edited Chat settings without asking (closes C8(2)).
- [ ] #5 A focused Settings category row is visible at 3:1 or better, and a focused active row is distinguishable from an unfocused active row (closes C8(4)).
- [ ] #6 The grid-line and control-edge boundaries reach at least 3:1 against surface and panel on every shipped theme whose colours resolve.
- [ ] #7 On Chat settings and Settings ▸ Providers & Models, focused buttons never get darker and Select highlights are visible at 3:1 or better.
- [ ] #8 task-25890 is closed: the boundary note stays inside the impact pane, and its strict xfail is removed.
- [ ] #9 All new geometry uses tokens defined only in css/core/_variables.tcss. test_dimension_literal_ratchet and test_python_style_ratchet (Tests/UI/test_component_pattern_governance.py:266, :291) stay at zero, and the hex ratchet (Tests/UI/test_design_token_governance.py:170) does not rise.
- [ ] #10 No ADR-097 ratchet rises: the boot CSS byte budget (608,090, Tests/Performance/test_boot_css_byte_budget.py:117) and the _ui_ready census (1,031, Tests/Performance/test_ui_ready_module_census.py:150). If dev is already red, the branch's number equals dev's.
- [ ] #11 console_settings_modal.py ends the phase at no more than 7,807 lines. If it shrank, its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in the same PR.
- [ ] #12 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added.
- [ ] #13 Live captures at 211x44 and 235x52, taken with a scratch TLDW_CONFIG_PATH profile, are attached to the PR. They cover the Chat settings Model view (collapsed and expanded), the unsaved-edits prompt and Settings rail focus.
- [ ] #14 The Docs/User_Guide pages are updated: console.md (Chat settings density and the Esc prompt) and settings.md (rail focus), each with a Verified-against stamp.
- [ ] #15 ./scripts/preflight.sh passes, including CSS bundle sync.
<!-- AC:END -->
