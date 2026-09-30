---
id: TASK-18920
title: 'Console approval cards: deny with an optional reason'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-19 09:55'
updated_date: '2026-09-30 19:15'
labels:
  - console
  - agents
  - approvals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `/deny <reason>` interaction (identified in the 2026-08-19 hermes-release review). Today the Console approval card's Deny decision is silent — the model receives a bare refusal and often retries the same denied call, burning turns. Add an optional free-text reason to the Deny path (row decision select, the single-call fast Deny button, and Deny all): the reason is delivered to the model as part of the denied tool result, clearly labeled as user-authored, so the agent course-corrects instead of retrying. Purely additive — an empty reason must behave exactly as today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Deny decision path (per-row select, fast Deny button, Deny all) accepts an optional bounded free-text reason; an empty reason produces today's denied-result behavior byte-for-byte
- [x] #2 A provided reason reaches the model inside the denied tool result with an explicit user-authored label (e.g. "Denial reason (from user):") that cannot be confused with tool output or an approval
- [x] #3 Reason text is length-bounded with an honest truncation note, sanitized as untrusted input, and never written anywhere the denied result itself does not already go (no new logs/exports)
- [x] #4 Sub-agent approval cards keep per-card scoping: a deny-with-reason resolves only its own card, exactly as Deny does today
- [x] #5 UI tests cover reason entry on each deny path, empty-reason equivalence, bounded input, and the model-visible denied-result content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/067-indefinite-human-approval-waits.md; backlog/decisions/195-console-live-tool-call-presentation.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: transient denial metadata uses the existing scoped approval and refused-result paths; no storage, permission, or provider boundary is widened.

1. Keep decision strings intact and carry normalized, bounded per-row denial reasons with the existing ephemeral answer provenance.
2. Add one optional reason field to each approval row; Select + Submit, fast Deny, and Deny all + Submit share that field. Preserve unchanged-round drafts and clear reasons for replacement rounds.
3. Append explicitly user-authored reasons only to actual denied tool results, through both review hooks and direct MCP approval; keep permission stamps and audit records body-free.
4. Pin mounted decision paths, normalization/truncation, round ownership, same-name calls, and model-visible refusals with focused tests; update the Console guide and run affected UI/runtime checks.
5. CI qualification: supply the new reason fields and real reason collector to the existing mixed-submit ownership test double, assert its input is locked before publishing, and verify both submission orders plus mounted approval paths on latest dev.
6. Preserve ADR-097 after rebase: remove declaration whitespace from existing approval rules, rebuild the stylesheet, and verify compact disclosure geometry and the unchanged CSS-byte limit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per-row optional denial reasons in the existing compact approval card. All Deny paths share a bounded disclosure field; unchanged-round drafts survive sync and replacement rounds clear them. Approval verdict strings remain unchanged, and quoted user-authored reasons reach only the exact explicit denied result through controller review and direct local/MCP/virtual owners. Batch permission stamps and audit metadata remain body-free. Empty reasons preserve existing refusal behavior.

Existing ADR-067 and ADR-195 apply; no new ADR was required. Changed approval provenance, input validation, controller/provider gates, approval widget/CSS and Console guide. Consolidated identical approval declarations to keep the startup CSS ratchet unchanged. Focused mounted, model-flow, direct-owner, same-name and simultaneous-round checks pass; native 80x24 and 100x40 journeys pass. Verification and baseline limitations: Docs/superpowers/qa/2026-09-29-console-tool-followups.md.

CI qualification on latest dev: repaired the existing mixed-submit namespace fixture with reason input state and the real collector, retaining lock-before-publish and deny-only reason assertions. Both CI failures reproduce before repair; the affected approval/denial group passes 42 cases. Streaming/startup checks pass 34 cases; the CSS case exposed a 65-byte overrun after rebase, resolved by removing 71 whitespace bytes from existing approval rules and rebuilding. Final UI/token/CSS qualification passes 39 checks (608,084/608,090 B; module census 1031/1033). Existing ADR-097 applies; no budget was raised. QA and the ownership-harness testing lesson are updated.
<!-- SECTION:NOTES:END -->
