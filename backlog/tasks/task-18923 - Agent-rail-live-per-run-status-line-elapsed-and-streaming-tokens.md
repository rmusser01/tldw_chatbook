---
id: TASK-18923
title: 'Agent rail: live per-run status line (elapsed + streaming tokens)'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-19 09:55'
updated_date: '2026-09-12 21:59'
labels:
  - console
  - agents
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's live token-flow spinner idea (2026-08-19 hermes-release review). While a run streams, the Console Agent rail shows only "running · step N" and token totals arrive post-hoc on the cost chip. Add a live status line during streaming: elapsed time plus the tokens received so far for the in-flight reply, updating at most ~1/s (reuse the once-a-second survivor tick timer pattern, task-15664) and tearing down when idle. Extend the same treatment to live children in the fleet panel. Figures must be honest: provider-reported usage where available, else an explicitly-labeled local count — never a fabricated dollar cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 During streaming the Agent section shows a live line with elapsed time and the current turn's token count, updating at most once per second and stopping when nothing is live (idle CPU unaffected)
- [x] #2 Token figures use provider-reported usage or are explicitly labeled approximate/local; no cost estimate is invented for the live figure
- [x] #3 Live children in the fleet panel show the same live elapsed/token treatment where the child's usage is observable
- [x] #4 The tick reuses/stops per the survivor-tick discipline: self-stopping when nothing is live, no per-chunk repaint cost
- [x] #5 Tests pin the render, the 1/s cadence bound, idle teardown, and honest-labeling of non-provider counts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/156-live-per-run-stream-usage-attribution.md. Reason: an additive AgentService-to-Console run-attribution and lifecycle contract. Execute Docs/superpowers/plans/2026-09-12-live-per-run-usage.md using Docs/superpowers/specs/2026-09-12-live-per-run-usage-design.md: exact run/call attribution and bounded scalar bridge, then existing-cadence UI rendering and documentation, with targeted tests and independent review. Final accounting and existing timers remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and independently reviewed the full current-call usage path in ADR-156 (backlog/decisions/156-live-per-run-stream-usage-attribution.md). AgentService scopes exact run identity; the shared adapter and bridge keep bounded scalar state, accept only explicit valid provider output fields, exclude synthesized fallback copy, and otherwise label cumulative UTF-8 estimates as local. Call/terminal cleanup and primary/child attribution are covered. Final accounting did not change.

Console primary and fleet rows show elapsed time and provider/local output labels through the existing primary poll and survivor timer. Zero/unavailable counts are omitted; terminal budget labels remain distinct. Existing wrapping keeps task and count painted at wide/narrow widths. User guide updated. Source commits: 16c10faca7, 2f39215549, ca9418b33a, c1b2bf984b, 3580bc6ae0. Backend review and two fix rounds, plus UI review and typing/doc clarification, approved.

Verification: backend focused 16-case provenance/lifecycle gate and later 10-case terminal extraction gate passed (overlapping, not summed); UI four-file run passed 88 in 117.22s. The FD warning led to exact fixture-owned database/profile-lock teardown; after cleanup the 15 new usage selectors and entire 12-case fleet panel module passed, both with regular descriptors 2→2. Residual sockets+2/pipe+1 and inherited RequestsDependencyWarning remain disclosed. Two painted widths independently inspected; scoped Ruff has no added diagnostics, edited-hunk formatting and whitespace pass. No full-suite/live-provider claim. Initial backend broad behavioral coverage was sensitivity rather than preimplementation RED; report preserves this deviation. Unchanged per-chunk telemetry extraction containment remains for final branch review, without reopening this task's accepted UI contract.

Evidence and reports: .superpowers/sdd/2026-09-12-live-per-run-usage/task-1-report.md, task-2-report.md and scratch/task-2-review-round1.md. Implementation plan: Docs/superpowers/plans/2026-09-12-live-per-run-usage.md. Modified service, gateway, adapter/bridge, Console agent renderer, focused service/Chat/UI tests and guide. Root preserves evidence; parent program and unresolved worktree recovery remain open.
<!-- SECTION:NOTES:END -->
