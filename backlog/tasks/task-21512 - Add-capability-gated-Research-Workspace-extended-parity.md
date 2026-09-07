---
id: TASK-21512
title: Add capability-gated Research Workspace extended parity
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - parity
  - ux
dependencies:
  - TASK-21507
  - TASK-21508
  - TASK-21509
  - TASK-21510
  - TASK-21511
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-extended-parity.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the audited Research Workspace control namespace with honest capability-gated More outputs, work products, workspace/help controls, search, diagnostics, and owner links while keeping unsupported and planned features non-deceptive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `More outputs…` presents Learn, Analyze, and Present groups without duplicate commands and adds Mind Map, Timeline, Data Table, Slides, and Audio Summary only where a real generator and canonical owner exist.
- [ ] #2 Executive Brief, Literature Matrix, Corpus Gap Finder, Evidence-Bound Hypotheses, and Research Proposal Pack become actions only when their adapter has a real generator and owner; otherwise each displays authority, owner, reason, and recovery.
- [ ] #3 Output options are progressively disclosed by type and unsupported controls never execute a substitute action, switch authority, or claim persistence owned by another system.
- [ ] #4 Workspace and Help menus, current-authority global workspace search, templates, collections, import/export/BibTeX, recent/pinned/archived state, banner/split preferences, status, onboarding, and keyboard help follow the approved core/context/owner-link/capability/planned classification.
- [ ] #5 Create agent task, ACP history, sandbox diagnostics, MCP/ACP/provider/runtime remediation, Console handoff, Settings workspace management, Library, Artifacts, and Study actions navigate to their real owners instead of recreating management UI.
- [ ] #6 Research Dossier, Competitive Market Memo, and Technical Project Spec are visibly Planned and are not focusable or dispatchable actions.
- [ ] #7 Unknown or stale capabilities fail closed, refresh by capability revision, preserve the selected authority, and expose readable recovery; footer hints advertise only currently implemented actions.
- [ ] #8 Wide/medium/narrow/short layouts keep the complete enabled control set keyboard reachable and screen-reader named without hiding authority, processing route, blocked state, or recovery.
- [ ] #9 Targeted capability-matrix, owner-link, menu/search, output-owner, inert-planned-control, accessibility, geometry, and live supported-server-output tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 and the approved design already classify the audited controls and define their owners. This phase activates only existing real contracts and must raise a new ADR if it introduces a new generator, canonical owner, or server contract.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-extended-parity.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
