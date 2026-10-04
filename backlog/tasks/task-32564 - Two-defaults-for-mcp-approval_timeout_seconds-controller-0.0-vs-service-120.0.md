---
id: TASK-32564
title: >-
  Two defaults for [mcp] approval_timeout_seconds: controller 0.0 vs service
  120.0
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-12 00:10'
updated_date: '2026-10-04 14:42'
labels:
  - mcp
  - config
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console approval card resolves `[mcp] approval_timeout_seconds` through `ConsoleChatController._resolve_mcp_approval_timeout_seconds` with `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0` (wait indefinitely; the `config.py` template documents 0 as the default), while `UnifiedMCPControlPlaneService.approval_timeout_seconds` and `live_server_request_wiring.py` fall back to 120.0 for the same key. Nothing injects the service value into the controller today, so the card waits indefinitely and external MCP-client waits auto-deny after 120 s — one key, two documented defaults. Surfaced by a Qodo finding on #2600 that read the service default as the card's.

Source: Qodo review rounds on the approval-card fix-wave PRs #2586/#2594/#2597/#2600 (2026-09-11); recorded in the lane ledgers, not fixed in the wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One documented default for the key, or two clearly named keys/paths, decided and recorded in the config template comment
- [x] #2 The service, the controller, and the live-server wiring agree with that decision
- [x] #3 The user guide states the resulting behaviour for the approval card
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/067-indefinite-human-approval-waits.md
Reason: direct implementation of the accepted default-0 and <=0-no-deadline policy across existing MCP consumers; no new configuration or runtime boundary.

1. Add regression coverage for missing/unparsable config, nonpositive elicitation waits, positive timeout, denial and cancellation.
2. Align the service and live elicitation defaults with the Console's 0-second default; use no deadline for nonpositive values while preserving terminal cleanup and positive ceilings.
3. Update the config template, user guide and ADR consequence to describe the one policy.
4. Run focused MCP/Console and static checks, review the diff, and require passing final-head PR gates before integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned service and live MCP confirmation fallback with the Console's existing default0 per ADR067. Live confirmations arm a deadline only for positive ceilings; nonpositive/default/unparsable values remain pending until a decision or cancellation. Positive expiry and finally-based terminal cleanup remain intact. Config template, MCP/Console guides and the stale ADR consequence now document the one policy.

Tests cover all three consumers' default/invalid/nonpositive/positive config, live approval/denial/cancellation after a virtual1000-second wait, terminal late-approval refusal, and existing positive expiry/factory behavior. Nine pre-existing real-config nodes use the supported bootstrap_profile marker. Added policy, bridge and live-wiring tests to PR Fast Lane. Reviewed bridge default/fallback expectations now pin0.0. RED:18 expected failures. GREEN:98 combined focused checks; preflight and zero-added-diagnostic lint comparison pass. Existing ADR067 applies; no new ADR. PR gate and final review qualification pending.
<!-- SECTION:NOTES:END -->
