---
id: TASK-2064
title: 'MCP: make inspector Advanced… a reversible toggle (F-053)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 17:30'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing Advanced… persists advanced_visible=True forever (recomposed on every visit) with no hide path, revealing jargon content. Unlabeled one-way door. Evidence: mcp_inspector.py:752-774,831-839. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Advanced content can be hidden again from the same control,State does not trap future visits without an explicit user choice,Tests cover show/hide round trip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI interaction fix; same config keys, no new contract). Behavior choice: persisted REVERSIBLE toggle -- the reveal button stays mounted and flips to 'Hide advanced'; hiding persists advanced_visible=False. Whatever the user last explicitly chose is what future visits show, so neither direction is a one-way door (session-only rejected: it would churn ~30 tests that mount the pane via the advanced_visible fixture and would discard a legitimate 'always show' preference once a hide path exists). Steps: 1. RED tests: show->hide->show round trip with persisted sequence; mount-visible renders toggle as 'Hide advanced'; update existing reveal tests (button relabelled, not removed). 2. mcp_inspector.py compose(): always yield the toggle button (label from flag); collapsible additionally when visible. 3. on_button_pressed dispatches _toggle_advanced (same disable-first + exclusive worker pattern). 4. _reveal_advanced(): relabel/re-enable button instead of removing it. 5. New _hide_advanced(): persist False, remove collapsible, relabel/re-enable. 6. Run MCP inspector/servers/workbench/parity tests + ruff.
<!-- SECTION:PLAN:END -->
