---
id: TASK-31264
title: Themed light modes expose hard-coded slate colors in stock widget states
status: Done
assignee: []
created_date: '2026-09-04 13:47'
updated_date: '2026-09-04 14:38'
labels:
  - ui
  - themes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under the new light themes from PR #2374 (observed live under apricot at 2516735cfd), some widget states paint a slate blue-gray that no theme variable controls: the model-catalog consent dialog's non-primary 'Don't check' button renders bg #4f6379 / fg #e8eaed, and the Library rail's selected row (Skills) shows a similar slate highlight. Traced far enough to rule out the obvious suspects: the dialog's own DEFAULT_CSS uses $panel/$background correctly, no app .tcss contains these hexes, and they do not derive from the active theme's palette via Textual's ColorSystem — so the origin is likely Textual stock component styles or a widget-tier default the app never overrides (root cause unconfirmed; note the widget-tier-CSS-loses-to-app-tier lesson). Cosmetic on dark themes, clearly foreign on light ones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source of the slate paint is identified and recorded in this task
- [x] #2 Non-primary dialog buttons and rail selected rows follow the active theme's palette under a light theme (verified live under apricot)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce minimally: stock Button under apricot with no app CSS (paints correctly -> app-tier source)\n2. Reproduce with bundle CSS + rule-match dump -> Button:focus { background: $ds-focus-bg }\n3. Trace $ds-focus-bg -> _variables.tcss literal #51677e; prove tcss $var definitions shadow Theme.variables (minimal probe)\n4. Replace the three dark-tuned literals with Textual's generated polarity-aware variables; measure across all 70 themes\n5. Pin with tests, rebuild generated CSS, update literal-pinned tests, live-verify under apricot
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: a $name: value definition in a tcss source shadows the theme's variables dict for that source (proven with a minimal Stylesheet probe: file tokens append after app tokens and last-token-wins). _variables.tcss froze three dark-tuned literals for EVERY theme: $ds-focus-bg #51677e (the slate focus paint on buttons and rail rows - both reported symptoms), $ds-status-error-readable #ff8fa3, $ds-text-placeholder/$ds-text-disabled-readable #8a8a8a. This also means PR #2374's light-theme dict additions for those tokens were inert at runtime. Fix: point the tokens at Textual's generated polarity-aware variables ($block-cursor-blurred-background, $text-error, $text-muted) - agentic_terminal resolves $text-error to #FF929C, within a hair of the old hand-tuned value, so dark themes keep their look. litestep_dark's error nudged #e05a48->#e26655 so generated text-error clears 4.5:1. Files: css/core/_variables.tcss, Widgets/Library/library_note_import_canvas.py (local fallback), css/Themes/themes.py (litestep_dark), regenerated bundle + screen_agentic_* sheets, tests: test_theme_contrast.py (2 new pins), literal-assert updates in test_console_workspace_action_menu.py + test_library_file_notes_workspace.py. Verified live under apricot: focused buttons paint #e4c7ae warm tint, selected rail row #e2c2a7; fix also healed test_disabled_file_notes_actions_keep_legibility_at_40_columns. Remaining theme-dict inertness (dict ds-* entries are documentation, not overrides) recorded in lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
