---
id: TASK-33620
title: >-
  Console unification: one run-state and readiness truth rendered by every
  status surface
status: To Do
assignee: []
created_date: '2026-09-30 03:03'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: the Console states run state, send readiness and staged context in about nine places: the header pill, the status-strip run chip, the Inspect authority line, the Inspect Provider row, the rail edge badge, the tab marker, the rail row glyph, the jump pill and the composer reason strip. Each one computes that state on its own, and some do it by substring-matching display copy ('provider'+'blocked' becomes 'setup'). active_run is routed through the 'Provider setup needed: {blocker}' template. Queue, vision and turn-commit failures travel in the setup_blocked_reason slot. As a result a healthy run reads 'Run: Blocked', a blocked send reads 'Ready', an emergency stop leaves no trace, attention in other tabs disappears from the tab being viewed, and staged sources, titles and streaming state go stale differently on each surface.

What unifying means here: one per-session presentation of run state (ConsoleRunPresentation). It holds a closed state set (Ready, Sending, Running, Waiting for you, Paused - decision needed, Failed, Stopped, Held) plus a reason and a next action. It is derived once from the controller's run status, preparation pause kind, kind-aware pending rounds (approval or question), emergency stop and the last terminal outcome. Every status surface renders from it. Send blockers become typed codes instead of free-text reasons. Cross-tab counts come from the same projection ('0 here, 1 in other tabs'). Staged context, chat titles and turn phase follow the same rule: one owner, many renderers. This umbrella is the parent of the theme's P0/P1 fixes (G2-02, G3-05, G4-09, GAP2-16, G2-03, G1-03, G2-08, GAP5-05, G2-05, G1-10). Those fixes should land on the shared projection rather than patch individual surfaces.

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'One truth per concept') and qa/console-ux-review-2026-09-29/findings.md / findings.json (verified statements, repros, evidence).

Related P2/P3 in the ledger: G1-04, G1-12, G1-24, G1-47, GAP1-03, G3-04, GAP1-02, GAP2-03, G1-19, G1-21, G2-18, GAP5-02, GAP5-24.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Console status surface (header pill, status-strip run chip, Inspect authority and Provider rows, rail edge badge, tab marker, rail row glyph, jump pill, composer reason strip) renders run state from one per-session source, and a test fails if any of them computes run state independently
- [ ] #2 A healthy active run is never reported as blocked or as needing provider setup on any surface, and a blocked, stopped or failed send is never reported as Ready on any surface
- [ ] #3 Send blockers are carried as typed codes rather than free-text reasons, and a lint fails the build when Console code branches on substrings of user-facing copy
- [ ] #4 Pending approvals and questions in other tabs are visible from the tab being viewed, with counts taken from the same projection that drives the per-tab markers
- [ ] #5 Staged sources, chat title and turn phase each have one owner, so after send, rename, tab switch and workspace switch every surface that shows them agrees (verified by a test that compares all rendering surfaces)
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
