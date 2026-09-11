---
id: TASK-32290
title: 'Docs: approvals and MCP permissions pages match shipped behaviour'
status: Done
assignee: []
created_date: '2026-09-10 19:17'
labels:
  - docs
  - approvals
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
agent-runs-and-tools.md lists four decisions while the card has five, says Always allow is MCP-only while local workspace tools also get it, quotes a path-warning string that differs from the code, and its approval-card SVG shows the pre-TASK-1846 single-line layout. mcp.md is a stub with no explanation of Inherit, Allow, Ask, Off, Space cycling, the kill switch, or that an explicit tool-level Allow bypasses the risk floor. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both pages describe the five decisions, their scopes and where to undo them.
- [x] #2 The path-warning text matches the code and the approval-card SVG is regenerated from the current layout.
- [x] #3 mcp.md documents the Permissions matrix states, Space cycling, the kill switch and the risk-floor rule for explicit tool-level Allow.
- [x] #4 Both pages carry an updated 'Verified against' stamp.
<!-- AC:END -->

## Implementation Plan

1. Read the shipped copy in the lane-B card/Console and lane-C MCP-hub worktrees and quote it verbatim.
2. Rewrite the Approvals section of `console/agent-runs-and-tools.md` around lane A's already-rewritten activity-line paragraph.
3. Start `mcp.md` from lane C's version and add the missing Permissions-mode material.
4. Regenerate the approval-card SVG from the real card; stamp both pages.

## Implementation Notes

Documented the approval card and the MCP Permissions matrix as they actually ship, quoting every locked string from source rather than paraphrasing. `Docs/User_Guide/console/agent-runs-and-tools.md`: the Approvals section now lists five decisions (`_DECISION_OPTIONS`) with the scope line each one paints (`DECISION_SCOPE_COPY`, verbatim), says which rows narrow the set (MCP all five; local workspace tools drop `Always · these args`; built-ins get Once/This session/Deny because **Always** is the only decision that writes to disk; raw shell is Run once / All shell · session / Deny and defaults to Deny), quotes `_PATH_PRECHECK_SUFFIX` exactly ("path outside allowed folders; will fail even if approved"), adds the visible risk/definition reason lines and the `needs decision · ` prefix, the ticking `Auto-denies in M:SS` countdown (`format_approval_deadline`), a new "Reaching a card from the keyboard" subsection (Alt+A, the footer/F1 entries, the ◆-tab route) and an "After you decide" subsection (`denied by you` / `blocked (Off)` / `blocked (kill switch)` plus the **Sent to the model** disclosure). The old "Always allow (MCP tools only)" claim is corrected — it covers MCP *and* local workspace tools. `Docs/User_Guide/mcp.md` (started from lane C's copy so its 32280/32281/32284/32291 sections are preserved) gains "Permissions mode — Allow, Ask, Off": the four states, tool → server → global precedence, the `•` override marker, Space cycling with `_LEGEND_TEXT` quoted verbatim, the gate breadcrumb, the risk floor and the rule that an *explicit* tool-level Allow is never floored (`permission_store.resolve`), and the kill switch's real label/hint; the leftover "checkboxes" description of the Tool gates rows is fixed (they are buttons). `First_Run_Setup.md` now points the Tools row at MCP ▸ Servers ▸ Tool gates instead of "there is no Tools category" and describes the step's copy; `settings.md` stops quoting a raw-shell label that no longer exists. The SVG was regenerated from lane B's card with `scripts/regen_approval_card_svg.py` (new, ~60 lines: mounts `ChatApprovalCard` with `APP_STYLESHEETS`, one MCP row, `save_screenshot`) — the old image was a whole-Console shot showing the pre-TASK-1846 single-line layout and the retired "Approve once" select label. Trade-off: the new image is card-only, so it loses surrounding Console context but is reproducible whenever the card changes. Verification: `Tests/MCP/test_mcp_documentation_contract.py` + `Tests/Docs/test_console_library_controls_docs.py` give 41 failed / 55 passed both before and after, identical failure *name* sets (pre-existing `README.md` and `Docs/Design/MCP.md` contract drift, neither file touched here); `./scripts/preflight.sh` all green. Stamps on all four pages name the lane heads read: `fix/approval-wave-b-card` @ e7409210cc and `fix/approval-wave-c-hub` @ a999fcf6e6.
