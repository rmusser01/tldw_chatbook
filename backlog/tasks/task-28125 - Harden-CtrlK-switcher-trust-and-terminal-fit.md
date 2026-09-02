---
id: TASK-28125
title: Harden Ctrl+K switcher trust and terminal fit
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 04:42'
updated_date: '2026-09-02 05:04'
labels:
  - console
  - ux
  - trust
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
  - backlog/tasks/task-21351 - Add-activity-views-to-CtrlK-session-switcher.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the current Ctrl+K switcher's wrong-target risks and clipped result navigation before the approved activity-switcher redesign lands, while preserving the existing activation and persistence boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Submitting a just-edited query activates the first result for that exact query, never a stale pre-debounce result.
- [x] #2 F2 renames only the explicitly highlighted open session; a saved or unavailable target remains open and explains why rename is unavailable.
- [x] #3 At supported terminal sizes, results scroll with the keyboard candidate, modal chrome stays reachable, and the complete modal fits the viewport.
- [x] #4 Current, candidate, open-session fleet state, queued prompts, and non-openable rows are textually distinguishable without relying on color alone.
- [x] #5 Empty-query Enter activates the most-recently-used other open tab when one exists, while explicit navigation and nonblank search activate the highlighted result.
- [x] #6 Focused regression, compositor, scoped Ruff/format, and diff checks pass; the broader TASK-21351 activity/receipt architecture remains untouched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md and backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md (existing, unchanged)
Reason: this is a pre-redesign trust repair using existing store/UI boundaries; TASK-21351 remains the owner of durable activity receipts, Active/History projections, and acknowledgement policy.

1. Add born-red store/state tests for MRU-other identity, fleet-state projection, semantic filter tokens, and explicit openability.
2. Add born-red modal tests for immediate-query Enter, strict F2 targeting, synthetic candidate navigation, scrolling, empty-state guidance, and 60x18 geometry.
3. Implement minimal MRU tracking in ConsoleChatStore and thread the preferred tab into the existing Ctrl+K modal constructor.
4. Replace positional DOM focus with one explicit highlighted index, current-query synchronous submit, strict rename eligibility, and a scroll-owning results body with fixed modal chrome.
5. Add grouped open-agent/saved-chat presentation, textual CURRENT/candidate/fleet/queue/unavailable labels, and documented semantic filters without changing activation destinations.
6. Run focused tests, production-stylesheet compositor geometry, scoped Ruff/format, diff checks, and an Impeccable polish pass; record evidence and close only this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact-query submission, explicit candidate targeting, safe F2 behavior, keyboard-owned scrolling, grouped operational state labels and filters, contextual onboarding, and process-local MRU-other navigation with a restored-session fallback. The store, pure switcher projection, modal, and real `ChatScreen` activation seam were updated; focused store/state, modal, controller, and production-path tests were added or extended. The implementation deliberately leaves TASK-21351's durable Active/History receipt architecture unchanged because TASK-20937 remains open.

Verification: 39 focused switcher tests passed, including production-stylesheet 60x18 compositor geometry. Scoped Ruff, format, and diff checks passed, and the Impeccable detector returned no findings. The repository-wide CSS ratchet still reports five unrelated `DEFAULT_CSS` offenders already present at `HEAD`; none is in this diff. ADR check: ADR-031 and ADR-085 remain accurate and unchanged. The section-contiguity regression produced a reusable testing lesson in `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
