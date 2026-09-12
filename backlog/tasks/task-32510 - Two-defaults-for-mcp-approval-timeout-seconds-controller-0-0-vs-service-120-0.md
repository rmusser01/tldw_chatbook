---
id: TASK-32510
title: Two defaults for [mcp] approval_timeout_seconds: controller 0.0 vs service 120.0
status: To Do
assignee: []
created_date: '2026-09-12 00:10'
labels:
  - mcp
  - config
dependencies: []
priority: low
---

## Description

The Console approval card resolves `[mcp] approval_timeout_seconds` through `ConsoleChatController._resolve_mcp_approval_timeout_seconds` with `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0` (wait indefinitely; the `config.py` template documents 0 as the default), while `UnifiedMCPControlPlaneService.approval_timeout_seconds` and `live_server_request_wiring.py` fall back to 120.0 for the same key. Nothing injects the service value into the controller today, so the card waits indefinitely and external MCP-client waits auto-deny after 120 s — one key, two documented defaults. Surfaced by a Qodo finding on #2600 that read the service default as the card's.

Source: Qodo review rounds on the approval-card fix-wave PRs #2586/#2594/#2597/#2600 (2026-09-11); recorded in the lane ledgers, not fixed in the wave.

## Acceptance Criteria

- [ ] One documented default for the key, or two clearly named keys/paths, decided and recorded in the config template comment
- [ ] The service, the controller, and the live-server wiring agree with that decision
- [ ] The user guide states the resulting behaviour for the approval card
