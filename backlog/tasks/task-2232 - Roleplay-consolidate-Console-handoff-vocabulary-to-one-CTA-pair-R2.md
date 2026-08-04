---
id: TASK-2232
title: 'Roleplay: consolidate Console handoff vocabulary to one CTA pair (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 18:39'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four names for one action family: 'Chat now', 'Send to Console draft', 'Continue this chat in Console' (preview), footer 'ctrl+enter draft'. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One primary + one secondary label pair used identically in inspector, preview, footer, and tooltips
- [x] #2 Readiness/blocked copy uses the same vocabulary
- [x] #3 Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no - copy/vocabulary consolidation on existing buttons and hints; no behavior, boundary, or schema change. ADR path: N/A.
1. Fix intent per surface from code semantics: inspector 'Chat now' (intent=start_chat, immediate provider reply) = PRIMARY; inspector attach, preview open-console, and conversation continue-console (all _stage_handoff drafts with a suggested prompt, no auto-send) = SECONDARY.
2. Rename the preview button and the conversation-row button to 'Send to Console draft' (preview also takes the console-action-secondary emphasis of the pair); rename the ctrl+enter binding description and the footer ShortcutAction label to 'Send to Console draft'.
3. Replace the both-gates-blocked fallback copy 'Chat blocked: X' with pair vocabulary ('Chat now and Send to Console draft blocked: X') in the readiness line and disabled tooltips.
4. Update tests to the new labels (TDD red first), then run persona tests + ruff; commit code + task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One CTA pair everywhere: primary 'Chat now', secondary 'Send to Console draft'. ADR: none (copy consolidation only, stated in the plan).

Intent decision (from code semantics, not surface copy): the preview's open-console button and the saved-conversation 'Continue in Console' button BOTH call _stage_handoff with a suggested prompt and no auto-send - the same draft mechanism as the inspector's attach - so they take the SECONDARY label, not 'Chat now' (which is reserved for intent=start_chat, the only path that needs an immediate provider reply). The preview button also moved from console-action-subdued to console-action-secondary so the pair's emphasis matches.

Changes: preview button relabeled 'Send to Console draft' (id unchanged); conversation-row button relabeled (id unchanged); ctrl+enter binding description and footer ShortcutAction label renamed 'Send to Console draft' (footer now reads 'ctrl+enter Send to Console draft'); the both-gates-blocked fallback 'Chat blocked: X' became 'Chat now and Send to Console draft blocked: X' in the inspector readiness line and disabled-button tooltips; the conversation controller's stale 'continuing in Console' notification/comment copy aligned. Kept: 'Chat now blocked: X' (provider gate, already pair vocabulary), 'Ready to chat in Console.', selection/unsaved/kind guidance lines, and the preview button's explanatory tooltip.

Tests: expectations updated first (TDD red: 5 failures), then source. test_personas_preview.py + test_personas_inspector_pane.py: 70 passed; test_personas_workbench.py: 306 passed; ruff clean on all touched files.

Files: tldw_chatbook/Widgets/Persona_Widgets/{personas_preview_pane,personas_inspector_pane}.py, tldw_chatbook/UI/Screens/personas_screen.py, tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py, Tests/UI/{test_personas_preview,test_personas_inspector_pane,test_personas_workbench}.py.
<!-- SECTION:NOTES:END -->
