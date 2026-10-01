# Plan: Phase 3: density tokens and CSS root fixes for the model-configuration surfaces (TASK-33003)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (the binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33003 - Phase-3-density-tokens-and-CSS-root-fixes-for-the-model-configuration-surfaces.md

## Phase goal (parent)

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
- console_settings_modal.py has zero headroom: its row in Tests/Architecture/test_module_size_ratchet.py (line 99 at phase start) is 7,802, which is its current length. Measured at the PR head: 7,764 lines, and the row (line 101) is lowered to 7,764. Modal edits are paid for by moving its DEFAULT_CSS control rules to the app tier.
- No binding from ADR-031 rule 2 (Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+S, Ctrl+D, Ctrl+Z, Ctrl+A, Ctrl+R, Ctrl+W).

Not in this phase: the inverted select-row height in Settings (css/features/_settings.tcss:465-470), field order, and the quick model popover. Design for 211x44 first and 235x52 second.

### Parent acceptance criteria

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
- [ ] #11 console_settings_modal.py ends the phase at no more than 7,802 lines. If it shrank, its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in the same PR.
- [ ] #12 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added.
- [ ] #13 Live captures at 211x44 and 235x52, taken with a scratch TLDW_CONFIG_PATH profile, are attached to the PR. They cover the Chat settings Model view (collapsed and expanded), the unsaved-edits prompt and Settings rail focus.
- [ ] #14 The Docs/User_Guide pages are updated: console.md (Chat settings density and the Esc prompt) and settings.md (rail focus), each with a Verified-against stamp.
- [ ] #15 ./scripts/preflight.sh passes, including CSS bundle sync.

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3. Start EVERY shell command with `cd <that path> &&`. Background shells reset cwd to the main checkout, which holds another session's uncommitted work: never touch it.
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv). Run pytest FROM the worktree cwd. Ad-hoc scripts need PYTHONPATH=<worktree>.
- NEVER write the real user profile (~/.config/tldw_cli, ~/.local/share/tldw_cli). Never bypass or disable test isolation (Tests/private_profile.py / private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH overrides), and never run a test body outside its fixtures. Any live app run uses TLDW_CONFIG_PATH=<scratch>/config.toml with a unique [general] users_name, and captures at FULL SCREEN: 211x44 (primary) and 235x52.
- ADR-126 storage gate: RecoveryRequired in a clean worktree is environmental. Prefer gate-free unit tests plus real-implementation tests at the seam, and compare failure-name sets against origin/dev.
- Size ratchets never rise (ADR-097). console_settings_modal.py has ZERO headroom: net lines must be <= 0. Check Tests/Architecture/test_module_size_ratchet.py and test_screen_size_ratchet.py before and after.
- Geometry (heights, widths, spacing) only through tokens in tldw_chatbook/css/core/_variables.tcss (ADR-150/161; raw dimension literals are banned in .tcss sheets). Rebuild the CSS bundle with the repo script; never hand-edit the bundle.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W. Footer hints must match working bindings.
- TDD: write the failing test first. For config and provider surfaces, add at least one real-implementation integration test (no kwargs fakes).
- Where a fix changes behaviour an existing test pins, rewrite that test on purpose and name it in the report.
- Commit per task: `fix(model-config): <summary> (TASK-33003.N)` or `feat(...)`, ending with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push. NEVER merge origin/dev into the branch (the owner requires rebases).
- Backlog hygiene per CLAUDE.md: in the subtask file, tick the ACs you satisfy, set status Done, and add `## Implementation Notes`. Update the Docs/User_Guide page CONTENT where behaviour changed. Do NOT add "Verified against" paragraphs (dev CLAUDE.md, TASK-33125): record what was verified, on which branch and date, in the task's Implementation Notes.
- Before you report DONE: run the covering tests plus `PYTHON=<venv> ./scripts/preflight.sh` (the exit code must be 0; do not pipe it through tail).

Phase 3 builds on Phase 2 (PR #2878: the field table, and console_settings_modal.py now at 7,802 lines with its ratchet row lowered to match, so net lines must stay <= 0 against 7,802). This worktree is cut from the Phase 2 branch; it will be rebased onto dev after #2878 merges. Design tokens: backlog/docs/design-language.md ($ds-control-height-compact exists) and css/core/_variables.tcss. Rebuild the CSS bundle with the repo's build script (never hand-edit it). Every visual change must be verified live at 211x44 AND 235x52 with capture-pane -e for contrast. Measure contrast ratios from the ANSI output across at least the default theme plus two others.

## Task 1: Scope the leaked global Collapsible and Select rules to their owning features (TASK-33003.1)

Task file: backlog/tasks/task-33003.1 - Scope-the-leaked-global-Collapsible-and-Select-rules-to-their-owning-features.md
Depends on: —

### Why

Three unscoped global rules shape every Collapsible and Select in the app, including those in the Chat settings modal:
- css/features/_conversations.tcss:298-303 forces every CollapsibleTitle to 3 rows. An .-active variant sits at :305-310.
- css/features/_conversations.tcss:313-316 makes every Select full width and puts a blank row beneath it.
- css/features/_evaluation_unified.tcss:59-63 gives every Collapsible a tall border and a bottom margin.

Add Textual's default padding and the section margin, and a collapsed section in Chat settings costs about 7 rows while every Select row gains a blank line (verified-claims C5, corrected points (a) and (b)).

The leak is already known. css/components/_agentic_terminal.tcss:1741-1744 overrides it for one canvas (task-2043) instead of fixing the source. Several MCP tests exist only to defend against the Select rule under the production bundle (Tests/UI/test_console_mcp_approval.py:1970, Tests/UI/test_mcp_tools_mode.py:825-897, Tests/UI/test_mcp_schema_form.py:581). Scoping these rules to the features that own them fixes the root cause for every screen at once. It is also the precondition for measuring one-row controls in Chat settings.

### Acceptance criteria

- [ ] #1 Outside the features that own these rules, three things hold, measured from rendered regions in Chat settings and in Settings ▸ Providers & Models at 211x44 under the production stylesheet: no CollapsibleTitle is forced to 3 rows, no Collapsible takes the tall border and bottom margin from _evaluation_unified.tcss, and no Select takes the bottom margin or full width from _conversations.tcss.
- [ ] #2 Before-and-after captures at 211x44 show the Conversations and Evaluation features rendering their collapsibles and selects exactly as before.
- [ ] #3 The task notes list every other destination that renders a Collapsible or Select, with a 211x44 capture of each one whose layout changed. None overlaps, clips or loses a control.
- [ ] #4 The production-bundle tests that guard against these leaks stay green unchanged: test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css (Tests/UI/test_console_mcp_approval.py:1970) and the Select-width cases in Tests/UI/test_mcp_tools_mode.py and Tests/UI/test_mcp_schema_form.py.
- [ ] #5 No comment still describes a leak that no longer exists. Overrides that only undid these rules are removed, and comments such as the task-2043 note at css/components/_agentic_terminal.tcss:1741-1744 are removed or corrected.

### References

- tldw_chatbook/css/features/_conversations.tcss
- tldw_chatbook/css/features/_evaluation_unified.tcss
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/components/_widgets.tcss
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_mcp_tools_mode.py
- Tests/UI/test_mcp_schema_form.py

## Task 2: Render Chat settings controls one row tall from the compact control token (TASK-33003.2)

Task file: backlog/tasks/task-33003.2 - Render-Chat-settings-controls-one-row-tall-from-the-compact-control-token.md
Depends on: TASK-33003.1

### Why

Chat settings draws every Input, every Select row and every in-form Button three rows tall (C5(a)):
- MODAL_CONTROL_HEIGHT = 3 (Widgets/Console/console_settings_modal.py:168) feeds the DEFAULT_CSS rules for rows, labels and controls (:1056-1071).
- The view tabs are fixed at 3 rows (:1073-1075), and the width-tier code switches heights between 1 and that constant (:3255-3292).
- The app-tier sheet repeats the Input height (css/components/_agentic_terminal.tcss:43-52).
- Labels carry min-height 3 (css/features/_console_panels.tcss:65-74). The report also found Context-tab Select labels sitting one row below their controls.
- Buttons are borderless app-wide (css/components/_buttons.tcss:12-15), so each shows one label line above two blank rows.

The fix needs nothing new:
- A compact control-height token already exists ($ds-control-height-compact, css/core/_variables.tcss:177).
- DESIGN.md:248 already defines a dense one-row form convention for Settings.
- The three-row height was never an ADR decision.

The modal has zero line headroom (its row in Tests/Architecture/test_module_size_ratchet.py: 7,802 at phase start, 7,764 at the PR head). Moving these DEFAULT_CSS rules to the app tier pays for the change and removes widget-tier rules that lose to app-tier CSS anyway (backlog/docs/lessons-textual.md:978).

Known traps:
- Removing a border turns on the global focus outline over one-row content (backlog/docs/lessons-testing-evidence.md:7220).
- A one-row field can paint nothing while tests that read .value still pass.

A test pins today's value: test_console_settings_modal_sizing_uses_named_constants (Tests/UI/test_console_session_settings.py:4555-4560).

### Acceptance criteria

- [ ] #1 At 211x44 and 235x52, under the production stylesheet, every Input, Select and non-footer Button in both Chat settings views renders exactly one row tall. This includes the view-switch tabs and the recovery actions.
- [ ] #2 No label cell or one-line summary (readiness or scope) makes its row taller than its tallest control or its text.
- [ ] #3 In the Context and memory view, each Select sits on the same row as its label.
- [ ] #4 A focused one-row field shows its value unobscured, with no global focus outline painted over it, and differs visibly from an unfocused field. This is verified by a painted-text probe driven with real key presses, not by reading .value.
- [ ] #5 Control heights come from the existing compact control-height token. The modal's DEFAULT_CSS no longer carries the control-height rules now at console_settings_modal.py:1056-1075 and :1098-1101; they live in app-tier sheets.
- [ ] #6 test_console_settings_modal_sizing_uses_named_constants (Tests/UI/test_console_session_settings.py:4555) is deliberately rewritten to pin the rendered one-row contract, replacing its MODAL_CONTROL_HEIGHT == 3 assertion.
- [ ] #7 Controls outside Chat settings keep their current heights, including the endpoint template modal's own MODAL_CONTROL_HEIGHT (Widgets/Console/console_endpoint_template_modal.py:89).
- [ ] #8 At 211x44, the task notes record how many editable Model-view controls are visible without scrolling, before and after, and the number at least doubles.
- [ ] #9 DESIGN.md's dense-form control convention (DESIGN.md:248) and backlog/docs/design-language.md section 2.3 name Chat settings as a dense form that uses the compact control height.

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/features/_console_panels.tcss
- tldw_chatbook/css/core/_variables.tcss
- Tests/UI/test_console_session_settings.py
- Tests/Architecture/test_module_size_ratchet.py
- DESIGN.md
- backlog/docs/design-language.md
- backlog/docs/lessons-textual.md
- backlog/docs/lessons-testing-evidence.md

## Task 3: Size Chat settings fields to their value type with width tokens (TASK-33003.3)

Task file: backlog/tasks/task-33003.3 - Size-Chat-settings-fields-to-their-value-type-with-width-tokens.md
Depends on: TASK-33003.2

### Why

At 211 columns, the short numeric inputs in Chat settings are about 125 columns wide. Controls fill up to 75% of a modal that is itself 85% of the viewport (css/features/_console_panels.tcss:77-81 and :200-203; C5, corrected statement). A 4-character value in a 125-column field wastes the width that help and source text need, and it makes the eye travel. ADR-161 spec 3.10 says a value with real semantic weight graduates to a named token, and raw dimension literals are banned outside css/core/_variables.tcss (Tests/UI/test_component_pattern_governance.py:266-289).

### Acceptance criteria

- [ ] #1 At 211x44 and 235x52, numeric fields in Chat settings are at most 12 columns wide, edge included. This covers Temperature, Top P, Min P, Top K, Max tokens, Seed, the penalties, the budgets and the context limits.
- [ ] #2 Each enum Select is no wider than its longest option plus the arrow.
- [ ] #3 No free-text field (model id, URL, env var name) is wider than 64 columns.
- [ ] #4 Field widths are the same at 211 and 235 columns: they are sized by value type, not as a percentage of the viewport.
- [ ] #5 Every width comes from a named token defined only in css/core/_variables.tcss. test_dimension_literal_ratchet and test_python_style_ratchet stay at zero.
- [ ] #6 A 211x44 live capture of the Model view is attached to the task notes.

### References

- tldw_chatbook/css/features/_console_panels.tcss
- tldw_chatbook/css/core/_variables.tcss
- Tests/UI/test_component_pattern_governance.py
- backlog/decisions/161-component-pattern-library.md

## Task 4: Make the Chat settings fold hint follow the scroll position (TASK-33003.4)

Task file: backlog/tasks/task-33003.4 - Make-the-Chat-settings-fold-hint-follow-the-scroll-position.md
Depends on: —

### Why

The '▼ more — scroll for the rest' hint only checks whether the body overflows (_sync_fold_hint, Widgets/Console/console_settings_modal.py:3850-3866). None of its callers (:2709, :2716, :3216, :3361, :3574, :3672, :3677, :3892) runs on scroll, so the hint still promises more content after the user has reached the bottom (C5(c)). The sibling ConsoleBoundedSection already watches scroll_y (Widgets/Console/console_bounded_section.py:234). No test or ADR pins the hint at the bottom; existing tests pin only that it shows while the body overflows.

### Acceptance criteria

- [ ] #1 When the content overflows, the fold hint is visible while content remains below the viewport and hidden once the body is scrolled to the bottom. Only scrolling drives this, proven by a pilot test that uses real scroll or key presses.
- [ ] #2 Scrolling back up from the bottom shows the hint again, with no other trigger.
- [ ] #3 The recovery-summary variant of the hint follows the same rule.
- [ ] #4 The existing overflow pins stay green unchanged: Tests/UI/test_console_resize_reflow.py (:846-856) and Tests/UI/test_console_context_controls.py (:980-985).

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Widgets/Console/console_bounded_section.py
- Tests/UI/test_console_resize_reflow.py
- Tests/UI/test_console_context_controls.py

## Task 5: Ask before Esc, backdrop or Cancel discards edited Chat settings (TASK-33003.5)

Task file: backlog/tasks/task-33003.5 - Ask-before-Esc-backdrop-or-Cancel-discards-edited-Chat-settings.md
Depends on: —

### Why

Three gestures reach _request_settings_close (:4041-4054) through request_safe_cancel: Esc (bound at Widgets/Console/console_settings_modal.py:1140), a backdrop click, and the Cancel button (:4250-4253). That method checks only for a pending memory reset or a running compaction, then dismisses. An edited draft is therefore thrown away without a word (C8(2)).

ADR-031's task-16211 refinement sets three rules:
- A generic gesture never discards by itself.
- The modal owns its dirty guards.
- Any displayed Escape hint must describe the action that is actually active.

No existing task covers this: task-32864 lists this modal as out of scope (:27).

The existing close guard (#console-settings-close-guard, :2611-2631), with its reset and compaction modes, is the seam to extend. The only tests that pin this path use unedited drafts: test_console_settings_modal_cancel_discards_draft (Tests/UI/test_console_session_settings.py:5254) and test_console_settings_modal_escape_dismisses_none (:5302).

### Acceptance criteria

- [ ] #1 If either Chat settings view holds any unapplied change, including edits carried in from the quick model surface, then Esc, a backdrop click and the Cancel button each open an unsaved-edits prompt instead of closing. Nothing is discarded until the user chooses.
- [ ] #2 The prompt names the edited fields and offers Apply to this chat (Enter), Discard (d) and Keep editing (Esc). Keep editing returns focus to the control that had it.
- [ ] #3 Apply from the prompt commits through the same Apply-to-this-chat path as the modal's Apply button and writes no configuration. If the draft is invalid, the modal stays open with its validation summary.
- [ ] #4 With no edits, Esc, a backdrop click and Cancel close immediately, as today. test_console_settings_modal_cancel_discards_draft (Tests/UI/test_console_session_settings.py:5254) and test_console_settings_modal_escape_dismisses_none (:5302) stay green unchanged.
- [ ] #5 The existing memory-reset and compaction close guards keep their precedence, copy and buttons.
- [ ] #6 While edits are unsaved, the visible Esc hint says that closing will ask first; otherwise it says Esc simply closes (ADR-031 task-16211).
- [ ] #7 Tests/UI/test_console_modal_dismissal.py stays green.
- [ ] #8 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added.
- [ ] #9 A 211x44 live capture of the prompt is attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Widgets/modal_dismissal.py
- Tests/UI/test_console_session_settings.py
- Tests/UI/test_console_modal_dismissal.py
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md

## Task 6: Raise boundary, focus and highlight contrast to at least 3:1 (TASK-33003.6)

Task file: backlog/tasks/task-33003.6 - Raise-boundary-focus-and-highlight-contrast-to-at-least-3-1.md
Depends on: —

### Why

Boundaries and focus are close to invisible. Field borders measure about 1.05:1 and the modal frame about 1.01:1 (report.md issue 2). $ds-grid-line is $surface-lighten-1 and $ds-control-edge is $surface-lighten-2 (css/core/_variables.tcss:11, :43). The agentic_terminal theme also sets ds-grid-line explicitly, to #26364D on a #0D1626 panel (css/Themes/themes.py:1561-1564).

On the Settings category rail, focus only swaps $ds-surface-panel for $ds-surface-raised and adds bold (css/features/_settings.tcss:198-203). That measures 1.107:1 in agentic_terminal and 1.22:1 in textual-dark, where the focused fill is darker than the unfocused one. A focused active row looks exactly like an unfocused active row (:193-197 versus :205-209), and two tests pin that (C8(4)).

The same report issue measured the Select highlight at 1.12:1 and found that a focused primary button gets darker, so focus reads as disabled.

WCAG 1.4.11 asks for 3:1 on focus indicators and component boundaries. ensure_readable_text_hues (css/Themes/themes.py:83) is the precedent: it pins generated text hues per theme on both the shipped-theme and user-theme load paths, and Tests/UI/test_theme_contrast.py gates it.

### Acceptance criteria

- [ ] #1 On every shipped theme whose colours resolve, the resolved grid-line and control-edge colours measure at least 3:1 against both surface and panel. ANSI palettes are excepted, as in ensure_readable_text_hues. The floor also applies to themes that set ds-grid-line explicitly.
- [ ] #2 A test in Tests/UI/test_theme_contrast.py fails if any shipped theme falls below that floor, or any user-saved theme loaded through load_user_themes.
- [ ] #3 A focused Settings category row differs from the same row unfocused, for both active and inactive rows, by at least 3:1 non-text contrast or by a non-colour cue (a glyph or an edge).
- [ ] #4 On Chat settings and Settings ▸ Providers & Models, a focused button is never lower in contrast against its surface than the same button unfocused, measured in agentic_terminal and one light theme.
- [ ] #5 The highlighted option in a Select overlay or OptionList on those surfaces differs from unhighlighted options by at least 3:1, or carries a ▶ glyph.
- [ ] #6 Two tests are deliberately rewritten wherever they pin identical styling for the active and active-focused states: test_settings_active_category_focus_style_keeps_label_readable (Tests/UI/test_settings_configuration_hub.py:4953) and test_settings_category_active_states_use_selected_contract (Tests/UI/test_non_obscuring_focus_contract.py:1139). They keep their intent: readable labels and no dominant geometry.
- [ ] #7 Contrast is measured in a running terminal at 211x44, in agentic_terminal and one light theme, and the values are recorded in the task notes. This covers the Chat settings frame and field edges, the Settings rail focus cue, focused buttons and Select highlights.

### References

- tldw_chatbook/css/core/_variables.tcss
- tldw_chatbook/css/Themes/themes.py
- tldw_chatbook/css/features/_settings.tcss
- tldw_chatbook/css/components/_buttons.tcss
- Tests/UI/test_theme_contrast.py
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/test_non_obscuring_focus_contract.py
- DESIGN.md

## Task 7: Keep the Settings boundary note inside the impact pane (TASK-33003.7)

Task file: backlog/tasks/task-33003.7 - Keep-the-Settings-boundary-note-inside-the-impact-pane.md
Depends on: —

### Why

This subtask absorbs task-25890. Under the production stylesheet, #settings-boundary-note (UI/Screens/settings_screen.py:22406) renders below #settings-impact-pane, whose overflow is hidden (css/features/_settings.tcss:61-68). The workbench-geometry contract caught this only once it was pointed at production CSS. Its settings parameter is a strict xfail that cites the task (Tests/UI/test_destination_visual_parity_correction.py:2385-2395). The inspector is where the Settings screen explains scope and ownership, so content that falls out of the pane is lost guidance. The fix belongs in this CSS root-fix phase.

### Acceptance criteria

- [ ] #1 Under the full production stylesheet (bundle plus split sheets), #settings-boundary-note stays inside #settings-impact-pane at 211x44, 235x52 and the contract's 140x42.
- [ ] #2 The same change removes the strict xfail on the settings parameter of test_runtime_and_settings_default_states_preserve_workbench_geometry (Tests/UI/test_destination_visual_parity_correction.py:2385-2395).
- [ ] #3 At 211x44 and 235x52, no inspector row is clipped or unreachable.
- [ ] #4 A live capture confirms the fix, not only the harness, which has masked this defect once already.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/css/features/_settings.tcss
- Tests/UI/test_destination_visual_parity_correction.py
- backlog/tasks/task-25890 - Settings-boundary-note-escapes-the-impact-pane-under-production-CSS.md

## Task 8: Chat settings choice rows paint their Select, not an empty error edge (TASK-33003.8)

Task file: backlog/tasks/task-33003.8 - Chat-settings-choice-rows-paint-their-Select-not-an-empty-error-edge.md
Depends on: —

### Why

In Conversation settings for a llama.cpp chat, the Reasoning effort row shows no control at 211x44 or 235x52. The only thing painted is a lone red '█' on the row above the label (qa/model-config-33002-captures/conversation-settings-reasoning-fields-*.txt). origin/dev 89dd84943a shows the same thing, with the old label 'Reasoning', so the defect predates Phase 2. It was found in the Phase 2 capture triage.

Cause: each provider-choice row (Widgets/Console/console_settings_modal.py:2027-2048 at 867762f3ef, and the Reasoning summary, Verbosity and Thinking rows that follow, whose validation Statics are at :2068, :2090 and :2112) yields a validation Static with class console-settings-error. That Static stays displayed while it is empty, because _sync_generation_choice_validation (:5006) only updates its text. The class sets no width (css/features/_console_panels.tcss:205, and DEFAULT_CSS at :1039 adds border-left: thick $error). Textual gives a widget with no width the whole row, so the Select (1fr, .console-settings-control) resolves to zero width. What remains visible is the empty Static's thick left edge, painted as '█'. Any provider that shows these rows is affected, so reasoning and thinking levels cannot be chosen in Chat settings.

### Acceptance criteria

- [ ] #1 At 211x44 and 235x52, under the production stylesheet, every displayed provider-choice row (Reasoning effort, Reasoning summary, Verbosity, Thinking) paints its Select, showing its value or its blank prompt. This is proven by a painted-text probe driven with real key presses, not by reading .value.
- [ ] #2 An empty choice-validation line paints nothing and takes no row width, and a restored obsolete value still shows its recovery copy beside the control.
- [ ] #3 The probe covers a llama.cpp chat and a provider that shows all four choice rows.
- [ ] #4 A 211x44 live capture from a scratch TLDW_CONFIG_PATH profile is attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/css/features/_console_panels.tcss
- qa/model-config-33002-captures/conversation-settings-reasoning-fields-211x44.txt

## Task 9: Keep a focused Settings field's whole guide inside the inspector view (TASK-33003.9)

Task file: backlog/tasks/task-33003.9 - Keep-a-focused-Settings-fields-whole-guide-inside-the-inspector-view.md
Depends on: —

### Why

In the Phase 2 capture qa/model-config-33002-captures/settings-console-behavior-fallbacks-temperature-focused-211x44.txt, the Console Behavior Temperature fallback has focus. Its focused-field guide shows Focused setting, Purpose and Saved as above the inspector fold, but the Validation line sits below it, behind '▼ more — scroll the inspector'. The capture triage found this.

The inspector fold is not new, and neither is the focus-driven scroll that should bring the guide into view (the first-row map at UI/Screens/settings_screen.py:1861 and the scroll pass near :24535). Both exist on origin/dev 89dd84943a. With the same inspector scroll position, dev's Console Behavior guide sits in the same slot and is also below the fold. Since the TASK-33002 final review, focusing a generation fallback replaces the guide rows in place with the field-table rows. The scroll pass's own docstring names that case as the one that leaves stale scroll bounds.

A live re-drive at 211x44 could not reproduce the capture: click focus and Shift+Tab then Tab focus both showed the whole guide, with Validation at row 27 of 44. The focus route that produced the capture is still unknown. The '/' field-search jump is a candidate.

### Acceptance criteria

- [ ] #1 The task notes record the focus route that leaves the guide's tail below the fold, with a reproduction.
- [ ] #2 At 211x44 and 235x52, focusing any Console Behavior generation fallback leaves every row of its focused-field guide, from Focused setting to Validation, inside the inspector's visible region. This holds for click, Tab and '/' field search.
- [ ] #3 The same holds for the Providers & Models model-default fields.
- [ ] #4 A pilot test under the production stylesheet drives real focus changes and asserts that the guide's rows lie inside the inspector body's visible region.

### References

- tldw_chatbook/UI/Screens/settings_screen.py
- qa/model-config-33002-captures/settings-console-behavior-fallbacks-temperature-focused-211x44.txt
