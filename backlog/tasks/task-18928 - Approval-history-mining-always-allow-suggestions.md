---
id: TASK-18928
title: 'Approval history mining: "Always allow" suggestions surface'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - agents
  - approvals
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `approvals suggest` (2026-08-19 hermes-release review). Chatbook users repeatedly pick "Approve once" for the same safe MCP tools. Mine the user's own approval history into proposed "Always allow" entries, surfaced for explicit review in Settings (MCP permissions area): the user accepts or dismisses each proposal; accepting writes a real Always-allow entry through the existing permission store. Nothing is ever auto-allowed — the feature only drafts proposals, and runtime approval behavior does not change until the user accepts one. Proposals are definition-fingerprint aware, consistent with the existing "(definition changed)" badge semantics, and high-risk tool classes are excluded from suggestion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Settings surface lists suggestions mined from approval history (tool identity + definition fingerprint aware; a tool whose definition changed is not proposed against the stale fingerprint)
- [ ] #2 Accepting a suggestion writes an explicit Always-allow entry through the existing permission-store API — the same path the MCP screen uses today
- [ ] #3 Dismissing a suggestion is remembered (not re-proposed) and dismissals are reviewable/clearable from the same surface
- [ ] #4 Tools in the existing high-risk classes are never suggested; the exclusion set is shared with the approval card's high-risk badge so the two cannot drift
- [ ] #5 Tests prove no runtime approval behavior changes merely because suggestions exist (until one is accepted), plus mining, fingerprint, and dismissal-persistence coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: extends the existing MCP permission-store surfaces; proposal-only, no auto-apply, no new trust boundary (ADR-032's permission boundary governs; link it from implementation notes).

1. Source approval decisions (where Approve once/session outcomes are recorded today; add minimal durable history if none exists — schema decision documented in-task)
2. Mining: frequency + fingerprint filter + high-risk exclusion
3. Settings surface with accept/dismiss, dismissal persistence
4. No-runtime-change tests, mining tests, docs (mcp.md / settings guide)
<!-- SECTION:PLAN:END -->
