---
id: TASK-541
title: RAG settings screen v2 UX upgrades (from sr design review)
status: Done
assignee: []
created_date: '2026-07-24 03:30'
updated_date: '2026-07-25 06:40'
labels:
  - rag
  - settings
  - ux
  - followup
dependencies:
  - task-503
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
V2 items from the senior UX/HCI design review of the SP3 RAG settings screen (task-503, PR #829). The review's 9 quick wins (clone-flow guidance, decoupling caption, backfill nudge, terminology unification, ⚠ legend, provenance sub-line, Delete danger styling, RAG test action, inspector fit) shipped in the SP3 PR; these are the structural/deeper upgrades deferred as v2. Review context is in the SP3 PR discussion and Docs/superpowers/qa/rag-settings-sp3-2026-07/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Split "manage profiles" from "edit config" structurally: a distinct profile picker/list region and an editor explicitly titled with the profile being edited (removes the dropdown/editor decoupling ambiguity that the v1 caption papers over; consider preview-on-select via a Select.Changed handler).
- [x] #2 Pre-commit re-index confirmation: when an index-determining (⚠) field changed AND the current index is built, Save confirms with the real blast radius (e.g. "This empties the current index (N vectors). Search returns nothing until you Backfill. Save anyway?").
- [x] #3 Context-sensitive Scope Inspector: guidance follows the expanded group / focused field (reranking guidance when in Reranking, etc.) instead of one static block.
- [x] #4 Replace state-labeled toggle buttons ("Enabled") with checkboxes or "X: On/Off" + action labels for citations and reranking; hide or dim+annotate Reranker model / Rerank results while reranking is disabled.
- [x] #5 First-run starter panel: instead of a wall of disabled fields, a brief "Search already works on Hybrid Basic. Clone to tune, or Backfill to enable semantic results" orientation with direct actions.
- [x] #6 Keyboard accelerators for the profile workflow (Set active / Clone / Backfill) honoring the keyboard-first posture; document them in the footer or category help.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered as 6 sequential SDD tasks on this branch (T1 checkboxes+dimmed rerank AC4, T2 re-index confirm+debounce AC2, T3 context-sensitive inspector AC3, T4 manage/edit split+preview-on-select AC1, T5 first-run starter panel AC5, T6 keyboard accelerators+captures AC6).

T6 (this closing task): added `a`/`c`/`b` BINDINGS (Set active/Clone/Backfill) to SettingsScreen, guarded to the LIBRARY_RAG category and to the same text-entry-focus check s/r/t use. Factored `_trigger_library_rag_profile_set_active` out of the button handler (mirroring Task 5's clone/backfill triggers) so the key and the button share one code path exactly. Footer discoverability: `SETTINGS_SHORTCUTS` is now category-aware (`_register_footer_shortcuts` appends a new `LIBRARY_RAG_SHORTCUTS` tuple only while LIBRARY_RAG is active; re-registered on every category switch, not just on_mount) rather than using Textual's native check_action/Footer machinery, which this screen doesn't use (it has its own AppFooterStatus + register_footer_shortcuts seam).

Tests (Tests/UI/test_settings_rag_profile_region.py, 102 passed): dispatch-when-RAG-active for each action, silent no-op for a non-RAG category, silent no-op via the text-entry-focus guard, a real end-to-end pilot test proving typed 'a'/'c'/'b' into the (restrict-digits) top-k Input never fire the actions (Input consumes printable keys before they reach screen BINDINGS), and a footer-hint test asserting the three new hints appear only while LIBRARY_RAG is active. Gates: Tests/RAG/ 537 passed/8 skipped; Tests/UI/ -k settings 715 passed/8 pre-existing-unrelated failures (same baseline set documented in Task 5's report: nav-overflow-hint overlap, theme-editor-open timeout, 6 chat_api_key tests).

QA captures: extended (derived copy, not a mutation of) the SP3 capture rig into Docs/superpowers/qa/rag-settings-v2-2026-07/ (capture_rag_settings_v2.py + svg_to_png.py), producing 5 new SVG+PNG pairs for the screen gate: first-run starter panel, preview-on-select (banner+title+disabled fields), the pre-commit re-index confirm modal, checkbox toggles + dimmed rerank fields, and the context-sensitive inspector following a focused rerank field. Each state needed scrolling the relevant widget into view first (the editor card composes well below the fold in the bare-harness viewport) -- driven via the pilot, all verified visually. One pre-existing, environment-only rendering gap noted (not fixed, not a regression): Checkbox glyphs/labels render as an empty box in this specific bare-CSS-bundle capture harness (also present in the already-accepted SP3 05 capture for the equivalent old Button) -- confirmed via direct widget inspection that the real value/label/disabled state is correct; it's a cairosvg/harness-layout quirk, not a product defect (the real DestinationHarness-based pytest suite renders and asserts Checkboxes correctly).
<!-- SECTION:NOTES:END -->
