# MCP Screen UX Correction Staged Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the MCP screen correction pass so the rendered Textual UI matches the approved terminal-native direction, fills the usable viewport, exposes clear columns, explains blocked actions, and passes actual screenshot approval before PR completion.

**Architecture:** Keep the existing `MCPScreen` route and `UnifiedMCPPanel` service contract intact. Improve the screen through mounted regressions, minimal Textual layout changes, and CDP-observed screenshot QA rather than broad rewrites. Treat screenshot approval as a hard gate for completion.

**Tech Stack:** Python 3.11+, Textual, pytest, textual-web, Chrome DevTools Protocol, existing `tldw_chatbook` UI modules.

---

## Current Evidence And Problem List

Source evidence:

- Actual screenshots showed earlier MCP layouts rendering in only part of the terminal, with unclear panel boundaries and missing top navigation context.
- Later screenshots improved viewport use and introduced panel framing, but the screen still needs a final guarded pass for spacing, copy, and visible controls.
- CDP capture is required for all future UI approval. Rendered SVGs, ASCII-only layouts, or code-level snapshots are not acceptable approval evidence.

Identified issues to address:

- The MCP workbench must fill the available screen area at real viewport sizes instead of collapsing into a partial-width or partial-height layout.
- The top app navigation and Textual status bar must remain visible unless an explicitly approved focused mode is active.
- The three-column layout must be visually obvious and structured for later draggable resizing.
- The left source/scope column must keep `Source`, `Section`, `Server`, `Scope`, and `Scope Entity` visible without pushing the primary `Section` affordance out of view.
- Empty and blocked action states must explain exactly what the user can do next, especially that `Inventory` is the path to runnable tools.
- Disabled controls must not look like broken controls. They need nearby recovery text and accurate tooltips.
- Long diagnostic copy must not crowd out primary controls or wrap into unusable fragments at common terminal widths.
- Screenshot QA must reject black, stale, clipped, or forced-viewport captures.

Suggested improvements to implement:

- Use clear terminal-native panel borders, pane titles, and subtle resize handles.
- Use concise recovery copy in the readiness pane instead of verbose diagnostic prose.
- Keep overview content scannable with counts and next action before any raw payload detail.
- Add focused regressions for durable selectors and visible state instead of brittle full-screen string matching.
- Update QA evidence only after a valid real screenshot is captured and approved.

## File Structure

### Modify

- `tldw_chatbook/UI/Screens/mcp_screen.py`
  - Keep the route-level MCP purpose concise and user-facing.
- `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
  - Own workbench layout, pane borders, source/scope controls, action readiness copy, action control display, and resize-handle affordances.
- `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
  - Own rendered overview/inventory content, scannable summary copy, and next-step guidance.
- `tldw_chatbook/UI/Navigation/main_navigation.py`
  - Preserve explicit overflow hint copy in the top navigation.
- `Tests/UI/test_unified_mcp_panel.py`
  - Add/maintain regressions for action readiness, overview next steps, inventory hierarchy, and blocked states.
- `Tests/UI/test_destination_shells.py`
  - Add/maintain mounted screen regressions for MCP workbench columns and screen copy.
- `Tests/UI/test_shell_product_model_visibility.py`
  - Add/maintain navigation overflow hint regression.
- `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md`
  - Update only with valid CDP screenshot evidence and final approval state.
- `backlog/tasks/task-11.4 - Phase-4.4-MCP-source-scope-and-action-readiness.md`
  - Update implementation notes and QA evidence after approval.

### Create If Needed

- `Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png`
  - Final approved screenshot candidate captured from the actual running app through CDP.
- `/private/tmp/cdp-capture-mcp-real-viewport.js`
  - Temporary capture helper. Do not commit this file.

## Stage 1: Lock The Regressions

**Purpose:** Prevent another visual-only pass from regressing core usability.

- [ ] Add or verify a mounted MCP test that asserts `#mcp-workbench`, `#mcp-server-tree-pane`, `#mcp-detail-pane`, and `#mcp-readiness-pane` exist and are displayed.
- [ ] Add or verify a mounted MCP test that asserts the visible labels include `Source`, `Section`, `Server`, `Scope`, and `Scope Entity`.
- [ ] Add or verify a mounted MCP test that asserts `Section` appears before lower-priority controls when rendered in the source/scope pane.
- [ ] Add or verify a mounted MCP test that asserts empty overview action readiness says `Run Action disabled` and points to `Section: Inventory`.
- [ ] Add or verify a mounted MCP test that asserts policy-blocked actions explain that runtime policy blocks actions instead of silently disabling the run button.
- [ ] Add or verify a navigation regression that the top navigation exposes `More: Ctrl+P`.
- [ ] Run:

```bash
python -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns Tests/UI/test_shell_product_model_visibility.py::test_navigation_exposes_explicit_overflow_hint --tb=short
```

Expected result: all focused MCP/navigation tests pass.

## Stage 2: Finish Layout And Copy Polish

**Purpose:** Make the screen readable and actionable at real terminal sizes.

- [ ] Keep `MCPScreen` purpose copy short: MCP is for servers, scoped tools, permissions, and audit readiness.
- [ ] Keep the MCP workbench as three columns: source/scope, server detail, readiness/actions.
- [ ] Keep column dividers visible but not visually louder than pane borders.
- [ ] Keep the source/scope pane narrower than detail, but wide enough for selected values to remain legible.
- [ ] Keep `Section` near the top of the source/scope pane so the Inventory path is discoverable.
- [ ] Keep disabled payload editing hidden or clearly marked when no runnable action exists.
- [ ] Keep readiness copy concise:

```text
Blocked
Run Action disabled
No actions available for local overview.
Select Section: Inventory to inspect runnable MCP tools.
```

- [ ] Keep overview detail scannable:

```text
Unified MCP Overview

Summary
Source: local
Server: local
Scope: personal
Section: overview
Tools: 0
Resources: 0
Prompts: 0
External server profiles: 0
Governance rules: 0

Next: select Inventory to inspect tools and actions.
```

- [ ] Avoid adding new runtime behavior in this stage. This is a usability and presentation pass over existing service data.

## Stage 3: Validate In The Running App

**Purpose:** Prove the actual rendered UI is usable, not just test-passable.

- [ ] Start textual-web with an isolated HOME/XDG directory so local machine config does not affect screenshots.
- [ ] Open the MCP route in the browser through textual-web.
- [ ] Capture with CDP from the actual browser viewport. Do not use forced CDP viewport emulation.
- [ ] Reject the screenshot if any of these are true:
  - The capture is black.
  - The top navigation is missing.
  - The bottom Textual status bar is missing.
  - The MCP workbench fills only a small fraction of the window.
  - The three columns are not visually distinct.
  - `Source` or `Section` are hidden.
  - The readiness pane does not explain the blocked action state.
- [ ] Save the valid screenshot under `Docs/superpowers/qa/product-maturity/phase-4/`.
- [ ] Present the actual screenshot to the user for approval before calling the screen done.

Recommended capture helper behavior:

- Open a fresh Chrome/CDP target or explicitly select the current textual-web target.
- Wait for a non-empty terminal canvas and stable layout dimensions.
- Capture the real terminal viewport.
- Perform a simple non-black image check before accepting the file as evidence.

## Stage 4: Update QA And Backlog Evidence

**Purpose:** Make completion auditable by future agents.

- [ ] Update `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md` with:
  - focused pytest commands and results,
  - `git diff --check` result,
  - textual-web/CDP capture method,
  - real viewport size,
  - final screenshot path,
  - user approval status.
- [ ] Mark superseded screenshots as superseded in the QA doc. Do not delete files unless explicitly approved.
- [ ] Update `backlog/tasks/task-11.4 - Phase-4.4-MCP-source-scope-and-action-readiness.md` with final implementation notes and QA evidence.
- [ ] Check only acceptance criteria that are actually proven by tests plus screenshot approval.

## Stage 5: Final Verification

**Purpose:** Package the PR safely after visual approval.

- [ ] Run focused MCP verification:

```bash
python -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py::test_mcp_destination_embeds_unified_mcp_management_panel Tests/UI/test_destination_shells.py::test_mcp_destination_labels_server_first_workbench_columns Tests/UI/test_shell_product_model_visibility.py::test_navigation_exposes_explicit_overflow_hint --tb=short
```

- [ ] Run the phase replay subset only if it is known to be stable on the current branch. If failures are unrelated stale-copy/timing failures, document them instead of hiding them.
- [ ] Run:

```bash
git diff --check
```

- [ ] Review the diff for accidental generated CSS edits or broad unrelated changes.
- [ ] Commit only the MCP correction files, QA doc, plan doc, and backlog task updates.
- [ ] Open or update the PR against `dev`.

## Completion Gate

This work is not complete until all of these are true:

- Focused MCP tests pass.
- `git diff --check` passes.
- A CDP screenshot from the actual rendered app is captured and shown to the user.
- The user approves the screenshot.
- QA evidence references the approved screenshot.
- Backlog task notes are updated with implementation and verification evidence.

Do not mark the screen approved based on ASCII diagrams, DOM snapshots, SVG mockups, or screenshots captured from forced viewport emulation.
