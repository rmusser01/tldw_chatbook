---
id: TASK-2064
title: 'MCP: make inspector Advanced… a reversible toggle (F-053)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 17:43'
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
- [x] #1 Advanced content can be hidden again from the same control
- [x] #2 State does not trap future visits without an explicit user choice
- [x] #3 Tests cover show/hide round trip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI interaction fix; same config keys, no new contract). Behavior choice: persisted REVERSIBLE toggle -- the reveal button stays mounted and flips to 'Hide advanced'; hiding persists advanced_visible=False. Whatever the user last explicitly chose is what future visits show, so neither direction is a one-way door (session-only rejected: it would churn ~30 tests that mount the pane via the advanced_visible fixture and would discard a legitimate 'always show' preference once a hide path exists). Steps: 1. RED tests: show->hide->show round trip with persisted sequence; mount-visible renders toggle as 'Hide advanced'; update existing reveal tests (button relabelled, not removed). 2. mcp_inspector.py compose(): always yield the toggle button (label from flag); collapsible additionally when visible. 3. on_button_pressed dispatches _toggle_advanced (same disable-first + exclusive worker pattern). 4. _reveal_advanced(): relabel/re-enable button instead of removing it. 5. New _hide_advanced(): persist False, remove collapsible, relabel/re-enable. 6. Run MCP inspector/servers/workbench/parity tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: persisted REVERSIBLE toggle. The reveal Button (#mcp-inspector-advanced-reveal) is now always composed (label/tooltip from _advanced_visible); on_button_pressed dispatches _toggle_advanced via the existing disable-first + exclusive-worker pattern; _reveal_advanced relabels/re-enables the button ('Hide advanced') instead of removing it; new _hide_advanced mirrors it (persists advanced_visible=False, removes the collapsible, flips label back). Persistence semantics: whatever the user last explicitly chose is what future mounts compose -- no trap in either direction. Session-only was considered and rejected (documented in plan + commit): with a real hide path the one-way door is gone, and session-only would silently discard a legitimate always-show preference and churn ~30 tests mounting the pane via the advanced_visible fixture. Files: tldw_chatbook/UI/MCP_Modules/mcp_inspector.py; Tests/UI/test_mcp_inspector.py (new round-trip + mount-hide-toggle tests; 5 existing reveal tests updated; stale skips-reveal test removed). Test gotcha: Textual 8.2.7 Button._on_click swallows clicks while the previous press's '-active' class is present (0.3s), so the round-trip test waits 0.4s between toggle clicks. Verification: 105 passed test_mcp_inspector.py; 53 passed (servers_mode + phase6 + 2 MCP geometry parity tests); 195 passed test_mcp_workbench.py; ruff clean. ADR: not required (UI interaction fix, same config keys). Not done: advanced_open (collapse/expand) persistence untouched; commit c4f4ac2 (see git log for exact hash).
<!-- SECTION:NOTES:END -->
