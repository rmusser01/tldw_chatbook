---
id: TASK-2089
title: 'Roleplay: surface the existing accelerators (F-038)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 10:58'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F6 pane cycle, Ctrl+1-4 mode jumps, Space dictionary toggle appear in no footer/chip/tooltip; footer shows 3 of ~10 bindings. Evidence: personas_screen.py:425-460. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mode chips/tooltips and footer advertise the working shortcuts,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests first: footer context renders f6 pane / ctrl+1-4 mode / [ ] mode / renamed 'ctrl+enter draft' (was attach), space toggle only in dictionaries mode; mode chip tooltips carry their Ctrl+N hint. 2. personas_screen._shortcut_context: add the always-on accelerators, per-mode space toggle, rename attach->draft. 3. Mode chip tooltips gain ' (Ctrl+N)'. 4. Update footer/chip tests; run workbench + footer-hint suites + ruff. ADR required: no - shortcut disclosure copy; no binding behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Footer shortcut context now advertises the working accelerators per state: f6 pane, ctrl+1-4 mode, [ ] mode always; space toggle only in dictionaries mode (the pane's space binding only acts on dictionary rows); ctrl+enter hint renamed 'attach' -> 'draft' to match the F-032 Send to Console draft CTA. Mode chip tooltips now carry their Ctrl+N jump key (mirrors MODE_CHIP_ORDER bindings). Files: tldw_chatbook/UI/Screens/personas_screen.py (_shortcut_context, mode chip compose); tests in test_personas_workbench.py (footer context content/gating, chip tooltips, import-refresh draft label). Verified: targeted 7 + gate 306 passed (full workbench + footer hints); ruff clean. ADR: not required (shortcut disclosure copy; no binding behavior changed).
<!-- SECTION:NOTES:END -->
