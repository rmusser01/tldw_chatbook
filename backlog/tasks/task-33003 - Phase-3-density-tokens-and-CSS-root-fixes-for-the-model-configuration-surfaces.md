---
id: TASK-33003
title: >-
  Phase 3: density tokens and CSS root fixes for the model-configuration
  surfaces
status: Done
assignee: []
created_date: '2026-09-26 11:47'
updated_date: '2026-09-30 16:30'
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
  - qa/model-config-ux-review-2026-09-26/judge-synthesis.md
  - qa/model-config-ux-review-2026-09-26/verified-claims.md
  - qa/model-config-ux-review-2026-09-26/report.md
  - tldw_chatbook/Widgets/Console/console_settings_modal.py
  - tldw_chatbook/css/features/_conversations.tcss
  - tldw_chatbook/css/features/_evaluation_unified.tcss
  - tldw_chatbook/css/components/_agentic_terminal.tcss
  - tldw_chatbook/css/features/_console_panels.tcss
  - tldw_chatbook/css/features/_settings.tcss
  - tldw_chatbook/css/core/_variables.tcss
  - tldw_chatbook/css/Themes/themes.py
  - Tests/Architecture/test_module_size_ratchet.py
  - Tests/UI/test_component_pattern_governance.py
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
  - backlog/decisions/150-design-token-system-and-design-language.md
  - backlog/decisions/161-component-pattern-library.md
  - backlog/decisions/097-boot-budget-ratchets.md
  - DESIGN.md
  - backlog/docs/design-language.md
  - Docs/User_Guide/console.md
  - Docs/User_Guide/settings.md
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
- [x] #1 At 211x44 and 235x52, under the production stylesheet, every Input, Select and in-form Button in Chat settings is one row tall, and a collapsed section costs at most 2 rows (closes C5(a) and C5(b)). (Met: TASK-33003.2's one-row test at 211x44 and 235x52; TASK-33003.1's collapsed-section test, now at both sizes (final fix wave, M1).)
- [x] #2 At 211x44, Chat settings numeric fields are sized to their value, not about 125 columns wide. (Met: TASK-33003.3, 12 columns, the same at 211 and 235.)
- [x] #3 The Chat settings fold hint is hidden once the body is scrolled to the bottom and shows whenever content remains below (closes C5(c)). (Met: TASK-33003.4.)
- [x] #4 Esc, a backdrop click and Cancel never discard edited Chat settings without asking (closes C8(2)). (Met: TASK-33003.5. The final fix wave made the Esc hint name a pending reset or compaction (I4). The Streaming-Inherit credential crash predates this phase; TASK-33003.10 fixed it on this branch.)
- [x] #5 A focused Settings category row is visible at 3:1 or better, and a focused active row is distinguishable from an unfocused active row (closes C8(4)). (Met: TASK-33003.6. At the final head, the focus edge measures 3.83:1 on the active row and 5.12:1 on an inactive row, at 211x44 and 235x52.)
- [x] #6 The grid-line and control-edge boundaries reach at least 3:1 against surface and panel on every shipped theme whose colours resolve. (Met: TASK-33003.6's gate over every shipped, built-in and user theme.)
- [x] #7 On Chat settings and Settings ▸ Providers & Models, focused buttons never get darker and Select highlights are visible at 3:1 or better. (Met: TASK-33003.6.)
- [x] #8 task-25890 is closed: the boundary note stays inside the impact pane, and its strict xfail is removed. (Met: TASK-33003.7. task-25890 is Done, and no xfail is left.)
- [x] #9 All new geometry uses tokens defined only in css/core/_variables.tcss. test_dimension_literal_ratchet and test_python_style_ratchet (Tests/UI/test_component_pattern_governance.py:266, :291) stay at zero, and the hex ratchet (Tests/UI/test_design_token_governance.py:170) does not rise. (Tokens only, and this phase adds no offender. Both ratchets are red at origin/dev 75c06af39a with the same offenders, so the "stay at zero" clause is delegated to TASK-33003.22. The hex ratchet passes.)
- [x] #10 No ADR-097 ratchet rises: the boot CSS byte budget (608,090, Tests/Performance/test_boot_css_byte_budget.py:117) and the _ui_ready census (1,031, Tests/Performance/test_ui_ready_module_census.py:150). If dev is already red, the branch's number equals dev's. (Met: both ratchets pass at the final head. The _ui_ready ceiling is 1,033, not the stale 1,031 (ruling 2).)
- [x] #11 console_settings_modal.py ends the phase at no more than 7,807 lines. If it shrank, its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in the same PR. (Met: 7,764 lines, down from 7,802 at phase start (ruling 1: the 7,807 here was stale), and the row is lowered to 7,764.)
- [x] #12 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added. (Met: the diff adds no Ctrl binding. The prompt's `d` is handled in a prompt-scoped on_key.)
- [x] #13 Live captures at 211x44 and 235x52, taken with a scratch TLDW_CONFIG_PATH profile, are attached to the PR. They cover the Chat settings Model view (collapsed and expanded), the unsaved-edits prompt and Settings rail focus. (Captured at the rebased final head: qa/model-config-p3-2026-09-28/final/ (live_final.sh, scratch TLDW_CONFIG_PATH profile). Attaching them is delegated to the PR description, which links that folder.)
- [x] #14 The Docs/User_Guide pages are updated: console.md (Chat settings density and the Esc prompt) and settings.md (rail focus), each with a Verified-against stamp. (Met, without stamps: worktree CLAUDE.md forbids "Verified against" paragraphs (ruling 3). What was verified, where and when, is recorded in these notes.)
- [x] #15 ./scripts/preflight.sh passes, including CSS bundle sync. (Met: exit 0 at the final head.)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 3 ships as one PR. Chat settings becomes a dense form: one-row controls, fields sized by value type, a fold hint driven by scrolling, and a prompt before any close gesture discards edits. The leaked global Collapsible and Select rules are gone, and boundary, focus and highlight contrast reach 3:1 on the model-configuration surfaces. Nine subtasks ran serially, 1 → 2 → 8 → 3 → 4 → 5 → 6 → 7 → 9 (ruling 4), each with its own review rounds, followed by a whole-branch final review and one final fix wave. The SDD ledger and rulings are in .superpowers/sdd/plan-2026-09-28-model-config-p3/ (gitignored); every ruling this file relies on is restated here.

**Commits.** The branch was rebased onto origin/dev 75c06af39a on 2026-09-30. The 20 Phase 2 commits (#2878) were dropped because they were patch-identical to what merged.
- Plan: ef07dfc220
- .1, scope the leaked Collapsible/Select rules: 31c65ba131, a4749d0d85
- .2, one-row controls: 45e82ab962, 241032728d
- .3, value-type widths: 7a395deecd, 370aded2b6
- .4, fold hint follows scroll: 22f25866e8, 526d1e7e63
- .5, ask before discarding: 26107762ea, 3b343f1f78, bb10c47098, 5e14a03f82
- .6, contrast of at least 3:1: 9233e73858, da59d60822, 81fb61ea58, 7d570f1156
- .7, boundary note inside the pane: 8b97127eb9, 6f6721d6e9
- .8, choice rows paint their Select: d4728b73f3, 40c5813667
- .9, the guide stays in view: 7fda3b3ed1, 204f60dba8, 50817ab238
- Final fix wave: 1e6ffced5b (code, tests, docs, captures), then the close-out commit that carries these notes.
- The rebase had two conflicts, resolved as the final review's I1 described. In tldw_chatbook/app.py, TASK-33011 had decomposed TldwCli's bases; ThemeVariableDefaultsMixin now sits before App[None], and app.py stays at 5,712 lines, which equals its ratchet row. In backlog/docs/lessons-testing-evidence.md, both sides appended at EOF, and both entries are kept. `git range-diff` shows 22 patch-identical commits and 2 changed, the two conflict commits.

**Behaviour changes (for the PR description).**
- Chat settings:
  - Every Input, Select, in-form Button, view tab and recovery action is one row tall.
  - A collapsed section costs 2 rows; it used to cost about 7.
  - Inputs and Selects carry a one-column left edge instead of a tall border. A focused field shows a thick edge and a fill, not an outline.
  - Numeric fields are 12 columns wide. An enum Select is as wide as its longest option plus 5. Caps: names 32, model ids 48, URLs 64. Widths are the same at 211 and 235 columns.
  - The fold hint shows while content remains below and hides at the bottom. Only scrolling drives it.
  - Esc, a backdrop click and Cancel ask before discarding edits. The prompt names the edited fields and offers Apply to this chat (Enter), Discard (d) and Keep editing (Esc).
  - The footer hint reads "Esc close", "Esc close (asks: N unsaved)", "Esc close (asks: memory reset)" or "Esc close (asks: compaction running)".
  - The Reasoning effort, Reasoning summary, Verbosity and Thinking rows paint their Select. Before, each was a zero-width Select behind an error edge and could not be used.
- App-wide:
  - Every Collapsible title is one row.
  - A focused title shows a fill plus bold underline, with no bottom border.
  - Bare Selects no longer get the leaked blank row and full width. The owning Conversations and Evaluation roots are retired, so their rules were deleted.
- Contrast:
  - Grid lines and control edges resolve through a guard-generated $tldw-boundary, pinned at 3:1 or better on every shipped, Textual built-in and user theme.
  - Focused buttons paint $tldw-focus-fill and are never darker than at rest.
  - The highlighted option in Chat settings and in Providers & Models is an inverted bar.
  - Settings rail focus is a thick edge at 3:1 or better, and a focused active row differs from an unfocused one.
- Settings:
  - The Scope Inspector's boundary note stays inside the pane (task-25890 is closed and its strict xfail removed).
  - Inspector rows lose the blank row between them.
  - A 36-column inspector floor applies from 134 columns.
  - A focused field's whole guide stays in the inspector view after a resize.

**Tests rewritten on purpose.**
- .1:
  - test_non_obscuring_focus_contract::test_shared_collapsible_header_focus_is_underlined_and_non_heavy
  - test_schedules_workbench::test_task_detail_frequency_group_no_longer_wastes_padding_rows (8 → 6 rows)
- .2:
  - test_console_settings_modal_sizing_uses_named_constants, now test_console_settings_modal_controls_render_one_row_tall[211x44, 235x52]
  - test_console_settings_modal_select_uses_compact_focus_outline, now ..._select_uses_dense_form_focus_edge
  - ..._focused_inputs_keep_value_row_visible
  - test_focus_accessibility::test_console_settings_input_focus_does_not_outline_single_row_value
- .5:
  - test_console_endpoint_discovery::test_create_endpoint_with_live_controller_rebase_settles (answers Discard)
  - test_console_context_window_modal::test_full_modal_refreshes_capacity_and_ignores_late_previous_model[dismiss]
- .6:
  - test_settings_active_category_focus_style_keeps_label_readable
  - test_settings_category_active_states_use_selected_contract
  - test_console_settings_modal_select_overlay_is_readable, which pinned dead SelectOverlay Option rules
  - test_global_button_focus_uses_two_non_obscuring_cues
  - test_feature_buttons_inherit_shared_button_focus_contract_without_duplicate_rules, whose Button:focus fill is now $tldw-focus-fill
- .7: test_picker_and_editor_stack_together_at_one_threshold (180/181 → 181/182)
- .8: test_console_settings_modal_saves_replaced_temperature_input and test_console_settings_modal_replaces_focused_sampling_input. They now press Escape before clicking; the race they hid is TASK-33003.12.
- Final fix wave: test_console_settings_modal_select_overlay_is_readable again. Its bundle leg now reads only the bundle; before, it re-checked the source.

**Final fix wave (2026-09-30), from the whole-branch review.**
- C1, test-suite regression. The app stylesheets reference $tldw-boundary and $tldw-focus-fill. Only TldwCli's ThemeVariableDefaultsMixin and themes.py's import-time guard over Textual's built-in themes supply them. Any harness App that loaded the sheets before something imported themes.py raised UnresolvedVariableError.
  - Reproduced: 96 failed in the 10 named files, run one per process.
  - Fixed at the one place every in-process test passes through: Tests/conftest.py imports the module eagerly, as production does. The test_mcp_recovery_compact subprocess Host inherits the mixin.
  - After the fix: 64/64, 16/16, 14/14, 34/34, 4/4, 38/38, 3/3, 20/20, 6/6, and 7/8 for test_watchlists_operation_card. Its one red, test_task_state_persists_only_valid_canonical_receipt_identity, is red at origin/dev too.
- I1: rebased, as above.
- I2: console_settings_modal.py's row is lowered from 7,790 to its measured 7,764.
- I4: the Esc hint names the guard Esc opens first, in the close path's precedence: reset, then compaction, then unsaved. _sync_completion_actions now re-reads it when compaction starts and finishes (+1 modal line). test_esc_hint_names_the_reset_and_compaction_guards_esc_opens was red first.
- I3, I5, I6: ticks qualified inline (.1 AC#4, .2 AC#8, .3 AC#5), riders filed (below), and this ledger reconciled.
- Minors fixed:
  - M1 and 2.4: the collapsed-section and recovery-action tests also run at 235x52.
  - 1.6: the focused-title check asserts that focus landed.
  - 5.3: test_prompt_labels_match_the_labels_the_modal_renders, with a negative control (a renamed label fails it).
  - 6.2: the overlay contract's bundle leg reads only the bundle.
  - 3.2: the container-scoped width rule names its ceiling.
  - M3: console.md says Apply shows dimmed; it no longer says the prompt "offers only Discard and Keep editing".
  - M4 and 2.6: design-language §2.3.
  - 8.5: the lessons-textual citation points at committed evidence.
  - M5: task-file corrections in .1, .3 and .7.
  - M6 and 7.1: the _settings.tcss comment and .7's notes no longer cite spec §6.
  - 7.5: .7 records its contrast exemption.
  - .5's notes no longer contradict themselves about the hint.
- M2: parent AC#13 re-captured at the rebased head, below.

**Riders filed (subtasks of this task, To Do by design).**
- .10 (high): Configure credential crashes when Streaming is Inherit.
- .11: the view-tab ' · Selected' crop.
- .12: a click during a picker list's collapse lands on the wrong control, plus the console-settings-control-support class-coverage red.
- .13 (owner decision): custom endpoints and their registry family's choice rows.
- .14: '/' cannot reach Console Behavior's fallbacks.
- .15: Library pane frames measure 1.28:1.
- .16: the theme sanitiser keeps 'auto NN%' on background-used names.
- .17: the SelectionList check glyph shows under the cursor on an unchecked row.
- .18: live keystroke loss after clearing a number field.
- .19: the five drift reds in test_console_modal_dismissal.py.
- .20: the approval batch-row guard is not gated, and its harness needs the splash turned off.
- .21: the credential round-trip [model] test is timing-dependent.
- .22: the inherited governance-ratchet offenders.
- The Alt+M popover's unsaved prompt is not a new rider: TASK-33004.5 AC#14 already owns it.

**Delegations and open owner calls.**
- AC#9, "stay at zero": the inherited offenders go to TASK-33003.22. Phase 3 adds none.
- AC#13, "attached to the PR": the PR description links qa/model-config-p3-2026-09-28/final/.
- Owner calls still open:
  - the reading of .2 AC#8 (expanded Model view 3 → 8; the collapsed view holds only 3 editable controls);
  - ruling 16 (the ▶ glyph moves to the Phase 4 switcher; TASK-33004.4 AC#19 owns it);
  - Task 7's inspector floor (a 133/134-column edge) was accepted by the controller on 2026-09-30;
  - TASK-33003.13.

**Pre-existing reds (identical at origin/dev 75c06af39a; not this branch's).**
- Governance ratchets: test_dimension_literal_ratchet (_agentic_terminal 1, _settings_splash_theme 2, _workflows 11) and test_python_style_ratchet (library_skill_work_pane.py:170). Owned by TASK-33003.22.
- Module-size ratchet: 9 other rows red. The console_settings_modal row passes.
- test_css_class_coverage_contract (TASK-33003.12).
- test_console_modal_dismissal.py: 5 drift reds (TASK-33003.19) and [escape] (TASK-33006.4).
- Tests/Docs/test_console_library_controls_docs.py: 3 reds.
- test_watchlists_operation_card: 1.
- test_credential_round_trip_keeps_the_restored_edit_unsaved[model]: failed 5 of 6 serial runs at the pre-rebase tip and at the rebased tip before this wave (TASK-33003.21).
- ADR-126 RecoveryRequired in plain local runs of full-app tests is environmental. Those tests ran with the repo's bootstrap_profile marker on a scratch TLDW_TEST_CONFIG_ROOT.

**Verification (branch feat/model-config-p3-density, 2026-09-30, rebased onto origin/dev 75c06af39a).**
- Covering run of 36 targets, `-n 5`, with the repo's bootstrap_profile marker and a scratch TLDW_TEST_CONFIG_ROOT. The targets are the 29 test files that reference the modal, the unsaved module or `_open_console_settings`, plus the theme-contrast, governance, token, boot-CSS, _ui_ready, module-size, focus-contract and Tests/Docs gates. The two trees ran concurrently.
  - Pre-wave tree (a git archive of 50817ab238): 158 failed, 2774 passed.
  - Final tree: 152 failed, 2784 passed.
  - All 152 final reds are red on the pre-wave tree too, so there are 0 final-only reds. The 6 pre-wave-only reds are timeouts in console_internals_decomposition and native_chat_flow, and they pass on the final tree.
  - The shared reds are this worktree's known environment and dev set: `#console-left-rail` NoMatches and ADR-126 RecoveryRequired in the native-chat-flow, hub and internals files, plus the inherited ratchet, dismissal, class-coverage and Docs reds listed above.
- C1: the 10 files, one pytest process each, give 206 passed and 1 failed at the final head. The one red is dev's watchlists node. Before the fix: 96 failed.
- The unsaved-guard file passes in the covering run (31 tests). The new I4 and label tests were red first. The label test was also checked with a negative control.
- Ratchets:
  - Boot CSS budget and _ui_ready census pass. The bundle is unchanged by the wave, because the sheets strip comments.
  - The module-size row for console_settings_modal.py passes at 7,764 = 7,764. The other 9 rows are red at origin/dev too.
  - The hex ratchet passes.
- ruff: the changed files have the same counts as before the wave.
- CSS bundle and split sheets reproduce (check_bundle_sync).
- `PYTHON=<venv> ./scripts/preflight.sh`: exit 0 at the final head.
- Live check at the rebased head: qa/model-config-p3-2026-09-28/final/live_final.sh at 211x44 and 235x52. It ran with a scratch HOME, XDG and TLDW_CONFIG_PATH, users_name t33003final_verify, a null keyring and the splash off.
  - Captures: Model view collapsed and expanded, an edited Temperature, the unsaved prompt ("1 unsaved edit to this chat: Temperature.", with Apply to this chat, Discard and Keep editing), and the rail at rest, active-focused and inactive-focused.
  - The hint read "Esc close (asks: 1 unsaved)" after the edit, and `d` closed the modal.
  - Rail focus edge, measured with final/measure_rail.py: 3.83:1 on the active row and 5.12:1 on the inactive row, at both sizes.
  - The reset and compaction hint states need an active memory and a provider, so they are proven by the pilot test only.
- Real profile: 0 files in ~/.config/tldw_cli or ~/.local/share/tldw_cli are newer than the wave's start marker. No test or live run pointed at it.

**Post-close fix (2026-09-30): number fields read as sized inside Advanced generation (AC#2).**
- Defect: with Advanced generation expanded, the disclosure body and the fields both painted (30,30,30) in textual-dark. The 12-column fields therefore read as full-row fields behind a one-column edge. The shipped captures showed it, measured at field/body 1.00:1.
- Cause: not the `ConsoleSettingsModal Collapsible` rule (_console_panels.tcss:51). That rule does paint $ds-surface-panel, but only under the title row. The body is Textual's `Contents` child, which the global `Collapsible > Contents { background: $surface }` (components/_widgets.tcss:95) painted with the fields' own fill.
- Fix: `ConsoleSettingsModal Collapsible > Contents { background: $ds-surface-panel; }`. The body now paints the modal panel the Context view's fields sit on. Fields keep $ds-surface-raised and the $ds-control-edge edge. The rule also covers the Conversation identity and Request estimate disclosures. It adds 85 bytes to the boot CSS, which passes the budget: headroom goes from 164 to 79 bytes against 608,090.
- Test: test_console_settings_disclosure_fields_read_as_sized, over textual-dark (the default), agentic_terminal and textual-light. For each shown field it asserts three things: the cell past the field's right end differs from the field's fill; the edge or the fill reaches 3:1 against that cell; and that cell equals the surface beside a Context view field. It was red on all three themes before the fix and is green after.
- Live check with `capture-pane -e`: a scratch HOME/XDG/TLDW_CONFIG_PATH profile, users_name p3fix_verify_*, tmux socket p3fix. The Temperature and Max tokens rows measured as follows:
  - textual-dark, at 211x44 and 235x52: field (30,30,30) on body (36,47,56). Fill 1.22:1, edge 3.05:1.
  - agentic_terminal, at 211x44: fill 1.04:1, edge 3.33:1.
  - textual-light, at 211x44: fill 1.08:1, edge 3.12:1.
  - These are the Context view's numbers in each theme. The 3:1 comes from the edge. None of the existing surface tokens gives a 3:1 fill against the panel.
  - The app log has 0 unhandled_exception or app_stopping lines.
- Captures: qa/model-config-33003-captures/chat-settings-model-expanded-{211x44,235x52} were re-taken. Only the colours changed; the text is identical.
- Lesson: lessons-textual.md, "A Collapsible's background paints only its title row".
- Covering run: 36 files, `-n 6`, plain local run (no bootstrap_profile marker). The branch had 814 failed and 2175 passed. origin/dev, over the 34 of those files it has, had 815 failed and 1654 passed.
  - 2 reds are on the branch only. Both are test_credential_round_trip_keeps_the_restored_edit_unsaved, in a file dev does not have. Both raise ADR-126 RecoveryRequired, and they fail the same way at the pre-fix head c8a015d370.
  - The other reds are shared with dev: the environmental RecoveryRequired and the inherited reds listed above.
  - The module-size ratchet has 9 red rows, the same set as dev. console_settings_modal.py is unchanged at 7,764 lines.
- `PYTHON=<venv> ./scripts/preflight.sh` exits 0.

**Triage (2026-09-30): a view switch keeps the other view's scroll. Pre-existing, filed as TASK-33006.7.**
- Symptom: at 211x44, scroll the Model view down (Advanced generation expanded), then choose Context and memory. The Context view opens at Conversation budget, and Model capacity is hidden above it. The branch reproduces this live.
- origin/dev 75c06af39a reproduces it too, live, from a scratch `git worktree add --detach` with a scratch profile. The worktree was removed afterwards.
- Cause: the views share #console-settings-body, and `_show_context_view` focuses the Budget strategy Select without resetting the scroll.
- Not fixed here, by instruction.
**Rider fix (2026-09-30): TASK-33003.10, Configure credential no longer exits the app.**
- Streaming at Inherit is now a valid snapshot value (`None`). It round-trips, so the reopened modal shows Inherit, not Off.
- Two more crashes in the same round trip are fixed:
  - The return intent's focus allowlist was missing 13 snapshot focus targets, the provider picker among them. A drift test now covers every snapshot focus target.
  - A blank context Select raised `InvalidSelectValueError` in the reopened modal.
- Configure credential now answers a refused snapshot with a notice and keeps the draft.
- console_settings_modal.py is still 7,764 lines.
- The repro test fails against origin/dev 75c06af39a. It passes here, as does the live repro at 211x44. The covering run shows 0 branch-only reds against the pre-fix head. `preflight.sh` exits 0. Details are in TASK-33003.10.
<!-- SECTION:NOTES:END -->
