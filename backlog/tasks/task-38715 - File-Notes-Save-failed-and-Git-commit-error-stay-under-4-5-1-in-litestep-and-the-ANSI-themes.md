---
id: TASK-38715
title: "File Notes Save failed and Git commit error stay under 4.5:1 in litestep and the ANSI themes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 07:23
labels:
- follow-up
- accessibility
- theme
- library
priority: low
references:
- tldw_chatbook/css/components/_agentic_terminal.tcss
- tldw_chatbook/css/Themes/themes.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36528
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36528's measurements). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36528 changed the shared rule for the File Notes "Save failed" status and the Git commit error (`.file-notes-git-commit-error, #file-notes-save-status.-error`, `css/components/_agentic_terminal.tcss:454-461`). It now paints `$text` on `$error 14%`. That pair passed 4.5:1 in textual-dark (12.98:1), textual-light (10.90:1) and high_contrast_yellow_black (14.58:1), and in 88 of the 91 shipped themes. Three still fall short:
- **litestep** (`css/Themes/themes.py:2167`): 3.54:1.
- **ansi-dark** and **ansi-light**: 1.92:1. These Textual built-ins use ANSI palette entries such as `color(1)`, which the terminal maps to real colours, so the computed ratio is nominal.

Both are error states that ask the user to act, and the text is the only signal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In litestep, "Save failed" and the Git commit error paint at 4.5:1 or better
- [x] #2 In ansi-dark and ansi-light the pair reaches 4.5:1, or the task notes record why a ratio computed from ANSI palette entries does not describe what the user sees
- [x] #3 No theme that passes 4.5:1 today falls below it, and test_high_stakes_file_notes_states_are_legible_in_shipped_themes still passes
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: One app-tier colour pair and its pins; no architectural change.

1. Re-measure the pair in all 91 themes the CSS-true workspace harness registers (TASK-36528's probe shape: paint "Save failed" and a Git commit form error, read the compositor style).
2. Find why litestep falls short and pick a recipe that lifts it without dropping any other theme under 4.5:1.
3. Pin litestep in `test_high_stakes_file_notes_states_are_legible_in_shipped_themes` (red first) and the new tokens in `test_css_build_integrity.py`; rebuild the bundle.
4. Record what the ANSI ratio means (AC#2).

## Implementation Notes

**Cause.** The shared rule painted `$text` on `$error 14%`. The tint has alpha, so it blends over whatever the host paints, here the theme's `$background`. litestep's `$background` is its mid-tone slate "desktop" (#7188a3; its text normally sits on the light-gray `$surface`/`$panel`). The blend came out #7b7b90, where `$text`'s auto ink picks white (#ededf0): 3.54:1. On that tone only a pure black ink passes (about 5.1:1; pure white is about 4.1, and 87% black about 4.2), and a fixed black ink fails on every dark theme, so the background had to change, not the ink.

**Fix.** The rule now paints the same 14% error tint on an opaque base: `background: $panel; background-tint: $error 14%`. The pair no longer depends on the host's fill, so "Save failed" and the Git commit error paint identically wherever they sit (the probe measured both: equal in every theme). `$surface` was also measured: litestep 10.18, but the weakest other theme fell to 4.87 (paradise_virtua), against 5.18 for `$panel`.

**Measurements (CSS-true harness, 120x40, all 91 themes).** Base b8608e4cdb: litestep 3.54, ansi-dark/ansi-light 1.92, the other 88 at 7.22 or better. After: litestep 9.49; textual-dark 10.00, textual-light 9.62, high_contrast_yellow_black 12.36; every non-ANSI theme at 5.18 or better (lowest catppuccin-frappe 5.18, atom-one-dark 5.59, catppuccin-macchiato 5.75). Margins shrink on some dark themes because `$panel` is lighter than `$background` (largest drops catppuccin-mocha 10.82 -> 6.08, tokyo-night 10.83 -> 6.23), but none goes under 4.5:1 (AC#3).

**ANSI themes (AC#2).** ansi-dark and ansi-light define `$error`, `$panel` and `$text` as ANSI palette entries. Textual cannot blend an alpha tint into a palette entry, so the 14% tint is dropped. Before, the pair painted the terminal's default foreground on `color(1)` (ANSI red); now it paints the default foreground on the default background, the terminal's own body-text pair, in bold. The computed ratios are not what the user sees. The test helper resolves each colour through Rich's default terminal theme with `get_truecolor()`, whose default argument treats a DEFAULT background as the default *foreground* (black), so default-on-default computes as 1.00:1. The old 1.92:1 was Rich's stock #800000 for `color(1)` against black. The terminal's own palette decides both pairs. Default-on-default is that palette's ordinary text pair, legible whatever the palette. Default-on-red depends on how the palette maps `color(1)`, and is poor on light palettes (black on dark red). One visible trade-off: in the two ANSI themes the error chip loses its red fill. The bold "Save failed" copy still marks the state.

**Red/green.** With the CSS reverted to base, both cases of the legibility test failed `litestep: 'Save failed' paints at 3.54:1, below 4.5:1`, and the integrity pin failed `assert '$error 14%' == '$panel'`. With the fix: both legibility cases, `test_css_build_integrity.py` and `test_component_pattern_governance.py` 51 passed (`-n 4`, load ~26). `test_design_token_governance.py` and `test_library_file_notes_git.py`: 227 passed, 2 failed, neither caused by this change. `test_prepare_session_fixed_regions_remain_visible_at_40_by_20` also fails 2 of 2 at base CSS (TASK-38693 owns it), and `test_hidden_unexpected_action_failure_refreshes_on_reopen` passed on rerun (a whole-file flake TASK-38693 lists).

**Not covered.** The measurements use the CSS-true harness, where the workspace sits on the screen's `$background`. Because the base is now opaque, the production host's fill no longer changes the pair.

**Files.** `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated with `build_css.py`), `Tests/UI/test_library_file_notes_workspace.py`, `Tests/UI/test_css_build_integrity.py`.

