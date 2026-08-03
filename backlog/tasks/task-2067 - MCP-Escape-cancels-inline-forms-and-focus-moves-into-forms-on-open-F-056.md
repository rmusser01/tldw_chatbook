---
id: TASK-2067
title: 'MCP: Escape cancels inline forms and focus moves into forms on open (F-056)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 18:19'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No inline form (profile, import, mutations, delete-confirm, test-tool) binds Escape; focus is never moved into a form on open, so keyboard users must Tab to reach inputs. Evidence: mcp_profile_form.py:126-127 et al. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape closes/cancels each inline MCP form,Opening a form moves focus to its first input,Tests cover escape and initial focus
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI keyboard-interaction fix). Steps: 1. Survey each inline form's cancel/close path (profile form Cancel, import panel close, mutations panel close, delete-confirm disarm, test-tool panel Close) and its first input. 2. RED tests per form: Escape triggers the same path as its Cancel/Close control; opening the form focuses its first input. 3. Implement minimal BINDINGS (escape) on each form widget + focus() on open (on_mount of the form or the host's show path). 4. Run MCP form/workbench/inspector tests + ruff.
<!-- SECTION:PLAN:END -->
