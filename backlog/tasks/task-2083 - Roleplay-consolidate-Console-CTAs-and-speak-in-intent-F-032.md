---
id: TASK-2083
title: 'Roleplay: consolidate Console CTAs and speak in intent (F-032)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 08:49'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Attach to Console / Start Chat / Open in Console are three near-synonymous CTAs; gating copy says 'Console blocked: select an item'. Evidence: personas_inspector_pane.py:346-364. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One primary and at most one secondary Console action with intent names,Readiness copy uses intent language,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update failing tests first: new labels (Chat now primary / Send to Console draft secondary / Continue this chat in Console), intent readiness copy. 2. PersonasInspectorPane: reorder+rename the two Console buttons (ids/handlers/gating unchanged - per-intent gating from task-523 preserved), rewrite readiness branches in intent language (guidance no-selection line from task-2082 stays), rename provider-block tooltip/notify copy. 3. PersonasPreviewPane: rename Open in Console button. 4. Sweep stale comments/docstrings; run persona suites + UAT + parity + ruff. ADR required: no - label/copy changes only; gating logic, ids, and message flow unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Consolidated the three near-synonymous Console CTAs: inspector now has one primary 'Chat now' (console-action-primary, id personas-start-chat, intent=start_chat) and one secondary 'Send to Console draft' (id personas-attach-to-console, intent=attach - the old Attach semantics); preview button renamed 'Continue this chat in Console'. Ids, handlers, messages, and the task-523 per-intent gating are untouched - verified by probe: Send to Console draft stays enabled while Chat now disables on an unready provider. Readiness copy now speaks intent: 'Pick a character or persona to start chatting.' / 'Save or discard your edits to chat in Console.' / 'Console chat is for characters and personas.' / 'Chat now blocked: <reason>' / 'Ready to chat in Console.' Files: tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py, personas_preview_pane.py, tldw_chatbook/UI/Screens/personas_screen.py (notify + comments); tests updated in Tests/UI/test_personas_{inspector_pane,preview,workbench}.py, test_uat_first_time_character_chat.py, test_destination_visual_parity_correction.py (plus comment sweeps in test_console_session_settings.py). Verified: final gate 541 passed/0 failed across 14 roleplay-affected files (workbench, state, inspector, preview, library panes, toolbar layout, attach, world-books, UAT, parity, phase1, footer hints, root-state); console_session_settings 134 passed separately; ruff clean. ADR: not required (labels/copy only; no logic, schema, or boundary change).
<!-- SECTION:NOTES:END -->
