---
id: TASK-19025
title: Simplify Library Skill editing
status: Done
assignee: []
created_date: '2026-08-21 08:20'
updated_date: '2026-08-21 09:11'
labels:
  - library
  - ux
  - skills
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give first-time users a concise Skill editor while preserving invocation semantics, exact allowlist content, trust safety, and efficient lifecycle actions for returning users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible Skills default to a concise Basic view without changing stored SKILL.md content or invocation semantics.
- [x] #2 Advanced remains available while actionable safety states expand trust details without overwriting the remembered preference.
- [x] #3 User and agent invocation remain independent, including an explained neither/reference-only state.
- [x] #4 The Advanced tool picker preserves untouched ordered, duplicate, and unknown allowlist names and changes them only after explicit selection edits.
- [x] #5 New, clean, dirty, conflict, delete, and mutation states expose only lifecycle-valid actions with guarded recovery.
- [x] #6 Mode disclosure preserves draft content, native focus, undo, and scroll across supported terminal sizes.
- [x] #7 Only touched-component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`.
This task implements the accepted Skill disclosure contract without changing
trust, runtime approval, SKILL.md storage, or service ownership.

1. Add pure mode, invocation, trust-expansion, and exact allowlist helpers over
   the existing immutable `SkillEditorState`.
2. Keep Basic and Advanced mounted over one draft and switch with targeted
   display updates that preserve focus, undo, and scroll.
3. Replace comma-separated tool editing with one bounded searchable native
   SelectionList while retaining unknown, duplicate, and ordered names until
   an explicit picker change.
4. Distill trust and lifecycle actions without moving trust/script ownership
   or weakening dirty, conflict, delete, and mutation guards.
5. Prove production geometry and safety with touched/direct-owner tests,
   update the ASCII-only guide, review, and close through Backlog CLI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a profile-remembered Basic/Advanced Skill editor over one mounted
  immutable draft. Basic contains the primary writing and independent
  invocation choices; Advanced adds run context, supporting information, and
  a bounded native tool checklist without introducing another editor model.
- Preserved exact ordered, duplicate, and unknown tool names until an explicit
  checklist edit. Filtering, catalog changes, disclosure, and no-edit saves do
  not rewrite imported allowlists; the checklist remains a restriction, never
  a permission grant.
- Made safety and lifecycle presentation truthful: actionable trust expands in
  either mode; healthy trust is compact; new/clean/dirty/conflict/delete/busy
  states expose only valid controls. Lifecycle updates patch mounted widgets,
  Ctrl+S follows the same availability predicate, and first save points to
  trust review.
- Production-CSS tests cover Basic and Advanced at 100x30 and 170x48,
  including the 60-item bounded keyboard tool picker, contained trust/actions,
  focus restoration, undo, scroll, dirty/conflict recovery, and user-focus
  precedence. The ASCII-only Skills guide documents both views and lifecycle.
- Focused evidence: 163 passed in
  `Tests/Library/test_library_skills_state.py` and
  `Tests/UI/test_library_skills_canvas.py`; 10 passed in
  `Tests/UI/test_css_build_integrity.py`. Ruff passed all five changed Python
  owners/tests; the CSS build and bundle parity check passed; `git diff
  --check` passed. Whole-file Ruff formatting was claimed only for the newly
  conforming pure-state owner; the large canvas/test owners retain pre-existing
  formatter drift and were not bulk-reformatted.
- Required inverses each failed the named contract and were immediately
  restored: sort/deduplicate captured allowlists; rewrite unknown/duplicate
  names during filter/catalog presentation; recompose on mode switch; hide an
  actionable trust state in Basic; expose Delete on a new draft. The restored
  focused owner gate is the 163-pass result above.
- ADR required: no. ADR-076 already owns this progressive-disclosure contract;
  trust/runtime approval, SKILL.md storage, and service ownership did not
  change. The Impeccable detector found only two unrelated pre-existing color
  advisories outside this task's CSS hunk.
- Per user direction, repository-wide pytest and a live/tmux profile were not
  run; only modified/touched Skill components and their direct CSS owner are
  claimed.
<!-- SECTION:NOTES:END -->
