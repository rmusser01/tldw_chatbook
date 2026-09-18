---
id: TASK-32646
title: Review Library Skills browsing and editor journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 17:53'
updated_date: '2026-09-15 18:52'
labels:
  - library
  - skills
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through Skills selection, editor disclosure, save and dirty-draft exits with readable keyboard focus and existing trust boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills rows and Overview/Edit actions are keyboard reachable and readable at wide and compact sizes in both themes.
- [x] #2 Basic/Advanced disclosure and invocation controls preserve unsaved field contents and exact stored allowlists.
- [x] #3 Successful save, Back, Cancel and Discard settle with a usable focus destination while stale or dirty exits preserve the intended data.
- [x] #4 Targeted regression and governance tests plus native isolated-profile evidence qualify the reviewed journeys and stored skill contents.
- [x] #5 Editor name-collision warnings cover the current built-in tool and Console command names.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/076-library-lifecycle-progressive-disclosure.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/009-local-skill-trust-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Review and repair existing editor presentation and keyboard lifecycle without changing storage, trust, execution or service ownership.

1. Exercise real local Skills through production-CSS list, Overview, Edit, Basic/Advanced and invocation controls at 170x48 and 80x24 in both themes.
2. Reproduce focus, data or feedback defects in save, Back, Cancel and Discard journeys before implementing targeted repairs. Preserve exact untouched allowlists and all trust gates.
3. Run touched Skills reader/canvas/state/service and token/bundle/wiring checks only. Verify actual native browsing/edit/save/reopen/discard with a fresh private profile, read-only persistence and normal exit.
4. Update Skills guide, workflow audit, QA evidence and task notes; self-review and commit locally. Import and trust-approval workflows remain subsequent review work. No full suite or dev integration.

Allocation: fresh reachable paths and 49 live worktrees max 32645; no candidate content reference across 262 refs. CLI allocation verified and corrected if below swept candidate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Skills browsing and editor journeys. Save returns focus to the available lifecycle action; Back/Discard reveal the list, and Cancel uses the canonical Browse Skills route. Deferred row focus waits for loading and the canvas rebuild, preventing outgoing rows from cancelling the handoff. Edits made during save remain dirty with committed version/trust metadata for the next explicit save. Exact untouched allowlists are preserved; collision warnings include twenty newer runtime/Console names.

Validation: 277 distinct targeted checks pass (273 Skills/domain/governance, two shared-focus, two controller-size). Forced Back/Cancel barriers and a held-write second save cover timing defects. Native private run-008 passes 170x48 dark and 80x24 light; six captures inspected, both SKILL.md files and exact allowlists verified read-only, no cancelled draft, ten SQLite integrity checks, zero messages, normal Ctrl+Q exit 0 and owned shell closed. Independent final review has no remaining actionable findings. No new Ruff diagnostics; new files pass lint/format; diff whitespace clean.

Known baseline exception: Library screen size check remains red at 35,203 lines against 33,204; HEAD had 35,204. Budget unchanged. Controller is 3,137 against 3,142. Nine stale canvas assertions were updated to current layout/token/footer contracts. No full repository suite, provider execution, import/trust approval or dev integration.

Changed the Skills controller, canvas, state warning set and Library screen; added keyboard journeys; updated existing canvas tests, Skills guide, workflow audit, QA evidence and testing lesson. Evidence: Docs/superpowers/qa/2026-09-15-skills-editor/README.md.

ADR required: no. Existing ADR-009, ADR-076, ADR-086, ADR-150 and ADR-161 govern the unchanged storage, trust and ownership boundaries. Next component: Skills import and trust journeys.
<!-- SECTION:NOTES:END -->
