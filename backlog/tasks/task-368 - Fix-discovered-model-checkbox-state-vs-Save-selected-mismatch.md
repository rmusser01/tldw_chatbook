---
id: TASK-368
title: Fix discovered-model checkbox state vs Save selected mismatch
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-23 08:10'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After discovery, the single result renders as '▐X▌ gemma-4-26B-...gguf' — visually checked. Clicking 'Save selected' immediately returned 'Select discovered models to save.' Clicking the model row itself changed NOTHING visually (cell attrs identical, still ▐X▌; the ▐ ▌ cap glyphs even render fg==bg, i.e., invisible), yet after that click 'Save selected' succeeded ('Saved 1 discovered model(s) to Llama_cpp.'). The visual state and the selection model are disconnected; I only recovered by trial and error.

**Repro:** Discover models against http://127.0.0.1:9099/v1 -> result row shows ▐X▌ -> click Save selected -> 'Select discovered models to save.' -> click the model row (no visual change) -> click Save selected -> saves.

**Verifier note:** Consistent with code: the SelectionList options are built unselected on first discovery (settings_screen.py:4923-4942, selected only if in the initially-empty _model_discovery_selected_model_ids), so the row the reviewer saw as '▐X▌…' was in fact unselected — the theme leaves Textual ToggleButton component classes (toggle--button) unstyled/mis-contrasted so the X reads as checked in both states and toggling produces no visible change (reviewer measured identical cell attrs; caps render fg==bg). Save-selected honesty path confirmed at 5114-5127. Brand-new surface (2026-07-11 discovery commit), no prior art in ledger or backlog. Downgraded P1→P2: the warning toast ('Select discovered models before saving.') guides recovery within one click, but the control's state display genuinely lies — worth a prompt fix.

**Source:** Console UX expert review 2026-07-20 (finding j1-discovered-model-selection-mismatch; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-28-model-list-clean.png`, `j1-29-model-row-after-click.png`, `j1-25-save-selected.png`, `j1-30-save-selected-retry.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Checked visual == selected state
- [x] #2 Toggling must produce a visible change
- [x] #3 If nothing is selected, the row should render visibly unchecked
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (theming, not state): the discovered-model `SelectionList` selection
state was already tracked correctly — the review's "▐X▌ in both states" was a
CSS defect. `css/components/_lists.tcss` had a row-cursor rule that ALSO targeted
the toggle component classes (`selection-list--button-selected` /
`-selected-highlighted` / `-highlighted`), painting selected and highlighted
identically (`$surface`/`$text`). That erased the checked/unchecked distinction
Textual's default provides, and because Textual renders the toggle caps (`▐▌`)
in the button's BACKGROUND colour over the row background, `$surface`≈row made
the caps `fg==bg` (invisible).

Fix: removed the toggle classes from the row-cursor rule (they are the checkbox
glyph, not the row highlight — the row highlight is `option-list--option-
highlighted`, untouched) and styled the toggle explicitly — unselected renders
an empty box (`$ds-grid-line` fg==bg, no inner `X`), selected renders a bright
framed `▐X▌` (`$ds-focus-fg` on `$ds-focus-bg`, bold). So checked==selected,
toggling shows a clear empty-box→filled-box change, and an unselected row reads
visibly unchecked.

Verified visually by rendering a themed `SelectionList` against the REAL built
CSS bundle and exporting a Textual SVG screenshot (served-app-equivalent): the
selected row shows a bright framed X, unselected rows show an empty box.
Regression-locked by `Tests/UI/test_settings_discovered_model_toggle.py`, which
loads the bundle + `agentic_terminal` theme (the lightweight pilot harness does
not load the bundle, so this is the only way to catch a component-style
regression) and asserts the selected toggle style differs from the unselected in
both fg and bg. Source `_lists.tcss` edited and the bundle regenerated with
`build_css` (reproducible, minimal diff).
<!-- SECTION:NOTES:END -->
