---
id: TASK-32825
title: Keep MCP server actions bound to their displayed target
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 00:33'
updated_date: '2026-09-19 00:57'
labels:
  - mcp
  - ui
  - ownership
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Queued server actions must never connect, disconnect, edit or delete a different profile when selection or readiness changes. Accepted deletion must retain the exact confirmed target while the toolbar updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retired or hidden server toolbar controls cannot dispatch or arm actions for a replacement detail view.
- [x] #2 An accepted delete confirmation retains its original server identity across asynchronous toolbar teardown and subsequent selection changes.
- [x] #3 Current toolbar actions, safe Keep and Escape behavior, and keyboard-visible confirmation remain usable at compact and wide sizes.
- [x] #4 Deterministic regression, targeted neighboring tests and bounded native evidence qualify the repair and document remaining server lifecycle scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: backlog/decisions/161-component-pattern-library.md; existing MCP lifecycle/service contracts remain unchanged.
Reason: correct UI action identity and confirmation lifetime inside existing handlers; no new storage, runtime authority or service boundary.
1. Pin delayed Connect/Edit/Refresh/Disconnect/Delete/Confirm delivery after detail replacement, including the pre-removal await gap and accepted confirm during a held toolbar rebuild.
2. Associate live toolbar widgets with their displayed target, retire that association synchronously before replacement, and capture an accepted confirmation target before awaiting presentation work. Preserve normal safe confirmation behavior.
3. Run the focused server canvas and relevant Workbench lifecycle regressions serially with private profiles. Check generated artifacts and scoped static diagnostics.
4. Inspect real-app keyboard confirmation/selection journeys in dark/light compact/wide private profiles, with source/lifecycle receipts. This bounded action review does not claim external connection or server execution qualification.
5. Independent review, update the MCP ledger and save a separate bounded draft PR. PR2711 remains untouched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retire current toolbar target associations before replacement awaits, reject detached or effectively hidden controls, and capture accepted deletion identity before presentation work. Keep focus uses the same live-target guard. Compact toolbar uses a two-column token-backed grid with existing action-class selectors; wide geometry and the 274 CSS candidate ceiling are preserved. 109 distinct targeted cases pass, including 17 new regressions and final CSS checks. Four final real-TTY theme/size journeys and twelve inspected captures verify safe Keep/Escape, ignored retired Connect and exact real-store Alpha deletion while Beta remains selected. Clean exit, released lock, ten healthy private databases, unchanged defaults and eight matching source hashes pass. No introduced Ruff findings; new files and changed production ranges formatted. All seven preflight guards pass, with CSS rechecked after selector rekey. Independent review found a hidden-mode gap, repaired and reproduced by three regressions; final review has no blockers. Initial compact clipping and selector-ratchet failures are retained. Existing ADR-161 applies; no new ADR. Evidence: Docs/superpowers/qa/2026-09-18-mcp-server-actions/README.md. Separate draft PR save pending; PR2711 inspector CI repair remains separate. External connection/execution and wider destination reviews remain open.
<!-- SECTION:NOTES:END -->
