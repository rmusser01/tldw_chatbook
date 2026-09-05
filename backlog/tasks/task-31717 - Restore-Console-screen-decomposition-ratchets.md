---
id: TASK-31717
title: Restore Console screen decomposition ratchets
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:34'
updated_date: '2026-09-05 16:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move cohesive Console responsibilities into the established controller and region boundaries so the current screen satisfies its existing size contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All Console Architecture size checks pass without raising existing ceilings.
- [x] #2 Moved behaviors retain late-bound dependencies and existing Textual ownership.
- [ ] #3 Focused Console tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: Mechanical ownership extraction implements DESIGN.md section 7 and the approved screen decomposition design, preserving existing runtime and persistence boundaries.
1. Move settings durability into a named controller, preserve app lifetime admission and test the settings/default flows.
2. Move settings navigation, provider selection, row menus, and cohesive projection clusters sequentially with explicit late-bound dependencies.
3. Remove obsolete screen forwarding methods after updating their production callers; preserve Textual event and lifecycle edges.
4. Run focused Console behavior and Architecture checks, inspect final counts, and record exact evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Extracted seven cohesive owners through the existing wiring graph: settings durability/navigation, provider selection, commands, context/cost projections, submission/stash lifecycle, and row actions/Markdown export. Textual hooks, DOM ownership and timer edges remain on the screen; the app-owned runtime and persistence lifetime are unchanged. Existing ADR-033 and DESIGN.md §7 apply; no new architectural decision was introduced.

The screen fell from 21,882 lines / 677 direct AST methods to 16,873 / 559, below the unchanged 16,966 / 563 ceilings. The method delta is 64 net ownership moves plus 54 exact redundant getter/setter methods replaced by the existing writable controller-state descriptor. No mixins, generic screen proxies, raised caps or code compression were used. Constructor/lifecycle reads preserve the wired contract; bare fixtures were explicitly migrated.

Final focused evidence: 48 row-action/persistence/export tests, 54 parallel-run/draft-snapshot tests, 141 command/raw/question tests, 246 state-owner tests, 81 context/cost tests (two independently reproduced baseline failures excluded), and 42 size/worker/bare-shell guards passed in their respective runs. Full Ruff checks on the changed production ownership graph and affected row/worker tests passed, as did the ten-file formatter check. Earlier stage evidence and ownership details are recorded in `backlog/docs/console-decomposition-repair-31717.md`.

The worker source guard now follows the existing conditional summary alias and both summary targets, with explicit local-table handling and fail-loud unknown aliases. Star-row and rewind fixtures were corrected at their real input/scheduler boundaries after reproducing queued-refresh races. The coordinating task owns baseline style/cost-cache repairs, shared inventory reconciliation and the full suite. AC #3 and Done remain pending that integrated evidence; this record does not claim a clean full sweep.
