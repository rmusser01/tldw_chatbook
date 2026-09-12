---
id: TASK-31674
title: Reconcile Console delegate wiring evidence and final screen ratchets
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:11'
updated_date: '2026-09-05 18:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep architecture evidence tied to the actual bounded command controller callbacks after the approved extractions, and retain the final measured size reductions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delegate binding evidence follows the explicit command-controller route and rejects broken wiring.
- [x] #2 Final screen size ceilings are lowered to measured source without widening other guards.
- [x] #3 Affected architecture suites pass with non-vacuity tests and static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve existing direct and decorator binding checks; follow only the explicitly extracted command dispatch route through its constructor port and wiring lambda. 2. Add negative regression variants for disconnected screen, controller, constructor and wiring ownership. 3. Lower final measured Console and Library screen ceilings after extraction and run affected architecture/static checks. ADR required: no. ADR path: N/A. Reason: test-only reconciliation of approved bounded controller ownership; no new runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled only the explicit commands route: screen delegation, controller callback use and constructor assignment, and screen-owned constructor wiring lambda. Added one positive and five disconnected-owner/dispatch/constructor/callback negative cases; existing direct and decorator paths stay unchanged. Lowered final measured ChatScreen16966/563 to16873/559 and Library41325/1301 to41324/1301. Complete wave6 and size-ratchet suites46passed12.58s (/private/tmp/tldw-review-console-architecture-final-20260905.xml). Full-file Ruff, changed-range format, diff whitespace and self-review pass. No new ADR or runtime behavior.
<!-- SECTION:NOTES:END -->
