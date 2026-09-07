---
id: TASK-2234
title: 'Roleplay: rename the preview toggle to state the payoff (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:00'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The preview conversation (safest way to learn what a character is) hides behind a subdued 'Preview conversation' toggle whose purpose lives only in a tooltip. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Toggle label states the payoff (e.g. 'Try a test chat (nothing saved)')
- [x] #2 Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no - label copy change on the preview toggle; no behavior, boundary, or schema change. ADR path: N/A.
1. Rename the preview toggle label to state the payoff: 'Try a test chat (nothing saved)' (with the existing ▸/▾ glyph swap and ids unchanged); keep the elaborating tooltip.
2. Update label assertions in Tests/UI/test_personas_preview.py and Tests/UI/test_personas_library_toolbar_layout.py; sweep for other 'Preview conversation' copy references.
3. Run preview + toolbar-layout + workbench tests + ruff; commit code + task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The preview toggle now states the payoff: 'Try a test chat (nothing saved)' (PREVIEW_TOGGLE_LABEL constant; the ▸/▾ glyph still tracks expand state, ids and tooltip unchanged). ADR: none (label copy only, stated in the plan).

Copy sweep: the living user guide (Docs/User_Guide/roleplay-chat-dictionaries.md and .../characters-and-personas.md) now references the renamed toggle, and its Console-handoff + export sections were brought up to the shipped task-2232/2233 vocabulary (the CTA pair 'Chat now' / 'Send to Console draft', the show-only-when-assigned voice-profile checkbox) - those sections were stale since F-032/task-523. Historical plans/specs/QA reports and the transient 'Preview conversation staged in Console.' notification (names the staged payload, not the toggle) were left as-is.

Tests (TDD red first): new test_toggle_label_states_the_payoff pins the exact collapsed/expanded labels; the toolbar-layout toggle assertions strengthened to the full labels. test_personas_preview.py + test_personas_library_toolbar_layout.py + test_personas_preview_restore.py: 49 passed; test_personas_workbench.py: 306 passed; ruff clean.

Files: tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py, Tests/UI/test_personas_preview.py, Tests/UI/test_personas_library_toolbar_layout.py, Docs/User_Guide/roleplay-chat-dictionaries.md, Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md.
<!-- SECTION:NOTES:END -->
