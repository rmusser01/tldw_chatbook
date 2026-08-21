---
id: TASK-19025
title: Simplify Library Skill editing
status: In Progress
assignee: []
created_date: '2026-08-21 08:20'
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
- [ ] #1 Eligible Skills default to a concise Basic view without changing stored SKILL.md content or invocation semantics.
- [ ] #2 Advanced remains available while actionable safety states expand trust details without overwriting the remembered preference.
- [ ] #3 User and agent invocation remain independent, including an explained neither/reference-only state.
- [ ] #4 The Advanced tool picker preserves untouched ordered, duplicate, and unknown allowlist names and changes them only after explicit selection edits.
- [ ] #5 New, clean, dirty, conflict, delete, and mutation states expose only lifecycle-valid actions with guarded recovery.
- [ ] #6 Mode disclosure preserves draft content, native focus, undo, and scroll across supported terminal sizes.
- [ ] #7 Only touched-component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

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
