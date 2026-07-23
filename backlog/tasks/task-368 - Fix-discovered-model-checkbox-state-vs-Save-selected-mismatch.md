---
id: TASK-368
title: Fix discovered-model checkbox state vs Save selected mismatch
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
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
- [ ] #1 Checked visual == selected state
- [ ] #2 Toggling must produce a visible change
- [ ] #3 If nothing is selected, the row should render visibly unchecked
<!-- AC:END -->
