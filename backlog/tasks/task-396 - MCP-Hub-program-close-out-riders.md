---
id: TASK-396
title: MCP Hub program close-out riders
status: To Do
assignee: []
created_date: '2026-07-21 01:44'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual non-blocking items from the Phase 6 final review and merge round, filed at program close (PRs #624/#632/#639/#654/#675/#722 all merged): (1) _refresh_server_discovery failure path — when select_server_target rejects a finding-carried target, _selected_server_key still flips and old-target data can cache under the new key (fail-soft, self-heals on next selection); (2) DataTable cursor_foreground_priority defaults to 'css' so semantic cell colors vanish on the cursor row — one-liner per table, also affects personas tables; (3) destination-shells harness reads real user config for mcp.hub_state keys — add the monkeypatch for parity with the visual-parity file; (4) spec §16.6 'density/shortcut pass' disposition: shortcuts shipped incrementally in Phases 1-4, density inherited from the shared ds-inspector contract — recording here that no dedicated pass ran; (5) formatting churn accepted on 5 MCP files during the final 246-commit rebase (dev's formatter pass vs our pre-format branch) — a repo-wide format run would reconcile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each item fixed or explicitly declined with reasons recorded here
<!-- AC:END -->
