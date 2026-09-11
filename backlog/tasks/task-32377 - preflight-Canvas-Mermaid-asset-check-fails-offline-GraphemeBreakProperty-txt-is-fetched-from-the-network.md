---
id: TASK-32377
title: preflight Canvas Mermaid asset check fails offline (GraphemeBreakProperty.txt is fetched from the network)
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - tooling
  - ci
dependencies: []
priority: low
---

## Description

`./scripts/preflight.sh` reports "Canvas Mermaid assets do not reproduce: declared input is unavailable: GraphemeBreakProperty.txt" whenever `scripts/vendor_canvas_mermaid.py` cannot fetch that input from unicode.org; the failure is identical on untouched branches, so it masks real drift and forces every reader to re-diagnose it. Seen across all four approval-card wave worktrees on 2026-09-11.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] The Canvas Mermaid check either vendors its Unicode inputs in-repo or reports "skipped: input unavailable" as a distinct non-failing status when offline
- [ ] preflight's summary distinguishes an unavailable input from a reproduction mismatch
