---
id: TASK-31429
title: Console context rail - theme-aware color grammar for active, value, and state
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 22:30'
updated_date: '2026-09-04 23:55'
labels:
  - console
  - ui
  - theme
dependencies: []
priority: medium
---

## Renumbering provenance

Filed 2026-09-04 22:30 as **TASK-31420** (the id was free across every ref at
filing). PR #2383 landed a different `task-31420` ("Register the ask_user
restraint description in the internal-prompts registry", created 19:28) on
dev the same evening as a merge commit, which a `git log --all
--diff-filter=A` sweep without `-m` cannot see. Per the older-arrival-keeps-id
rule (TASK-19601) this task moved to TASK-31429; every inbound reference
(code comments, tests, docs, the PR) moved with it. See
`backlog/docs/lessons-backlog-hygiene.md` (2026-09-04 entry).

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console left rail ("Console context") is monochrome: labels, values, the active workspace/conversation, and agent/run state all render in the same white/grey, so users cannot tell state at a glance, cannot see which conversation is current versus which row merely has keyboard focus, and cannot separate fixed labels from live values.

Give the rail a small semantic color grammar that rides the existing theme infrastructure: two new `$ds-*` tokens that reference Textual's generated polarity-aware variables (so every built-in, Orb, and user-saved theme recolors the rail with no theme-editor changes), plus the existing `$ds-status-*` tokens for state. Approved design (brainstorm 2026-09-04): primary hue = what you are in; accent hue = a value; status hues = state; grey = labels and help copy. Section headers (shared with the Library/Home rails) and the Inspector rail are out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active workspace value, the active-conversation line, and the selected conversation row render in the theme-primary text hue (`$ds-active-fg` -> `$text-primary`) at rest; keyboard focus keeps its existing distinct treatment and wins when a selected row is focused; empty placeholders stay muted
- [x] #2 Label/value pairs in the rail (Workspaces, Model, Details) render labels muted and non-bold and values in the theme-accent text hue (`$ds-value-fg` -> `$text-accent`)
- [x] #3 The Agent status line carries a state class derived from the run status (running/done/stuck/error/cancelled) and is colored with the matching `$ds-status-*` token; idle and unavailable stay uncolored
- [x] #4 Conversation rows carry a class per ConsoleRunMarker (running, needs-approval, finished-ok, finished-failed, subagent-unseen) and are colored with the matching `$ds-status-*` token; a selected row overrides the marker color
- [x] #5 The Model section not-ready line uses `$ds-status-error-readable` instead of the decorative error hue
- [x] #6 Both new tokens are defined in `_variables.tcss` as `$`-references (no hex literals) and the all-themes AA contrast gate covers `$text-primary` and `$text-accent` over the rail surfaces; any theme that fails gets a per-theme `variables` override
- [x] #7 Live verification in the real app on textual-dark, textual-light, and one Orb theme shows the grammar (selected row, values, a running agent) and the regenerated CSS bundle is committed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The Console context rail now carries a four-meaning colour grammar driven entirely by the active theme: primary hue = what you are in, accent hue = a value, `$ds-status-*` = state, muted = labels/help. Two new tokens (`$ds-active-fg: $text-primary`, `$ds-value-fg: $text-accent`) in `_variables.tcss`, ~12 rail rules in `_agentic_terminal.tcss`, two small class-toggle seams in Python, and a readability pin in themes.py.

**Approach.**
- CSS: `.console-workspace-status-value` / `.console-model-section-value` -> `$ds-value-fg`; labels drop `bold`; `#console-active-workspace-value`, `#console-workspace-selected-conversation.…-active` (id+class, because the placeholder rule on the same Static is an id rule) and `.console-workspace-conversation-row-selected` -> `$ds-active-fg`; selection no longer borrows the focus tint (fg + bold only), so focus and selection read as two signals. Five `.console-workspace-conversation-row-<ConsoleRunMarker.value>` rules ordered BEFORE `-selected` so selection wins at equal specificity; five `.console-agent-section-status-<status>` rules; `#console-model-section-recovery` -> `$ds-status-error-readable`.
- Python: `console_agent_status_state()` / `apply_console_agent_status_state()` in `UI/Console_Modules/agent.py` parse the status word out of the existing "Agent: <status> …" / "Sub-agent · <status>" line (no payload-shape change; the timer-path census pin on the `.update()` receiver is preserved by a second `query_one`). `console_workspace_context.py` adds the marker class in `_conversation_button` via a glyph-keyed map derived from `CONSOLE_RUN_MARKER_GLYPHS`, and the `-active` class on the selected-conversation line only when a summary exists.
- **Deviation from the plan (documented):** Textual 8.2.8 derives `text-primary`/`text-accent` as a 66% tint of the contrast text toward the hue; 20 of the 70 shipped themes still fail AA on their own surfaces (measured). Instead of 40 hand-edited `variables` dict entries, `themes.py` gained `ensure_readable_text_hues()`, applied to every shipped theme at import and to every user-saved theme in `create_theme_from_dict`: where a tint fails 4.5:1 on `surface` or `panel` it blends further toward the text pole until it clears. These are GENERATED names no tcss defines, so the entry is honoured (the mechanism note atop themes.py). Side effect worth knowing: `$ds-chat-user-accent` (also `$text-primary`) gets the same, slightly more readable value on those 20 themes.

**Evidence.**
- TDD: 57 new tests in `Tests/UI/test_console_rail_color_grammar.py` (pure helpers, class toggles on unmounted widgets, mounted class application through `_sync_console_agent_section` and the tray's recompose, source + generated-sheet declaration contracts, a source-order pin for selected-over-marker, and a mounted rule-match probe asserting the three tokens RESOLVE to the running theme's `primary` / `text-accent` / `text-primary`). `test_theme_contrast.py` now gates `text-primary`/`text-accent` on all 70 themes plus a user-saved pastel theme through `load_user_themes`. All were watched red before the implementation.
- Live (real app, tmux, scratch profile, llama-server at :9099): `capture-pane -e` decoded per run; textual-dark painted active line + selected row `#57a5e2` (= text-primary), values `#ffc473` (= text-accent), labels `#a7abaf`, "Agent: running · step 3" `#0178d4` (= primary); textual-light `#002d4f` / `#a86d1c` / `#004578`; apricot `#7c4621` / `#345425` / `#bc6b32` — each equal to the theme's computed generated value. The launch rewrote no tracked generated file.
- Baseline failures confirmed on a pristine detached `origin/dev` worktree, NOT introduced here: `test_css_class_coverage_contract` (identical 157-line failure list both trees), the three `test_timer_path_static_update_inventory` tests, both `test_console_agent_controller` bridge tests, and `test_console_parallel_runs::test_navigation_guard_survives_stay_then_renavigate_then_leave_by_coordinates`.

**Files.** `tldw_chatbook/css/core/_variables.tcss`, `css/components/_agentic_terminal.tcss`, generated `css/tldw_cli_modular.tcss` + `css/screen_agentic_{console,library,settings}.tcss`, `css/Themes/themes.py`, `UI/Console_Modules/agent.py`, `UI/Screens/chat_screen.py`, `Widgets/Console/console_workspace_context.py`, `Tests/UI/test_console_rail_color_grammar.py` (new), `Tests/UI/test_theme_contrast.py`, `Docs/User_Guide/console.md`, `backlog/docs/lessons-live-verification.md`.

**Out of scope (by design):** section headers (shared `DestinationRailSectionHeader` with Library/Home), Inspector-rail parity (can adopt the same two tokens), the collapsed Details "not configured" rows.

**Review follow-up (Qodo on PR #2393, 4 findings, all addressed):** Google-style `Args:`/`Returns:` added to the two new agent.py helpers; the rail row focus rule corrected from the never-matching descendant form `.console-workspace-conversation-row Button:focus` to `.console-workspace-conversation-row:focus` (focus was in fact already painted by `ConsoleTerminalWorkspace Button:focus`, proven by the new mounted probe before the change; the fix makes the row-level contract rule real and `test_console_keyboard_trust.py` pins it); and `ConsoleLeftRail.compose` now seeds the status colour class from its `agent_status_line`, so a recomposed rail is coloured even when the screen-side payload memo skips the sync (`test_fresh_rail_compose_applies_the_agent_status_class`).
<!-- SECTION:NOTES:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
Reason: adds two design tokens and a handful of state classes inside the existing `$ds-*` token layer and the Console rail's existing CSS; no component boundaries or contracts change.

1. TDD, tests first: (a) extend `Tests/UI/test_theme_contrast.py` so the resolved-token AA gate covers `text-primary` and `text-accent` on every theme and the no-literal check covers the two new tokens; (b) new `Tests/UI/test_console_rail_color_grammar.py` pinning the pure helpers (agent status line -> state class, run-marker glyph -> row class), the class toggles on unmounted widgets, and the CSS declarations in source and in the generated console sheet.
2. `_variables.tcss`: add `$ds-active-fg: $text-primary;` and `$ds-value-fg: $text-accent;` with a mechanism comment.
3. `_agentic_terminal.tcss`: recolor `.console-workspace-status-value`, `.console-model-section-value`, `#console-active-workspace-value`, `.console-workspace-selected-conversation-active`, `.console-workspace-conversation-row-selected`; drop bold from `.console-workspace-status-label`; `#console-model-section-recovery` -> readable error; add agent-status and row-marker state rules.
4. Python: `agent.py` pure helper + apply function called from `ChatScreen._sync_console_agent_section`; `console_workspace_context.py` adds the marker class in `_conversation_button` and the `-active` class on the selected-conversation line when a summary exists.
5. Rebuild the CSS bundle (`build_css.py`), fix any theme the AA gate flags via `Theme.variables` overrides (generated names only), run the rail/theme test files, then live-verify per AC #7 and update the Console user guide stamp.
<!-- SECTION:PLAN:END -->
