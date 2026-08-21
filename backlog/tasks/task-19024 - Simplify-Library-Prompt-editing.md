---
id: TASK-19024
title: Simplify Library Prompt editing
status: In Progress
assignee: []
created_date: '2026-08-21 07:09'
labels:
  - library
  - ux
  - prompts
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give first-time users a concise Prompt editor while preserving exact structured Prompt data, safety states, and efficient lifecycle actions for returning users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible Prompts default to a concise Basic view without changing their stored block representation.
- [ ] #2 Advanced remains available, while incompatible or safety-sensitive Prompts force an explained Advanced view without overwriting the remembered preference.
- [ ] #3 Basic edits preserve block identities, ordering, metadata, version history, and ordinary save/conflict behavior.
- [ ] #4 New, clean, dirty, conflict, and mutation states expose only lifecycle-valid actions with guarded recovery.
- [ ] #5 Mode and action disclosure preserve draft content, native focus, undo, and scroll across supported terminal sizes.
- [ ] #6 Only touched-component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A. This task implements the Prompt-specific disclosure and
lifecycle composition already accepted by ADR-076; it does not change Prompt
storage, service, safety, or versioning ownership.

1. Add pure Basic eligibility and preference coercion over the existing
   `PromptEditorState` and immutable block working copy.
2. Keep Basic and Advanced regions mounted over one draft and switch them with
   targeted display updates so draft content, focus, undo, and scroll survive.
3. Persist one Prompt-only profile preference while deriving temporary forced
   Advanced presentation for incompatibility, conversion, conflict, or unsafe
   update states.
4. Replace overlapping global actions with a lifecycle-valid action strip and
   an inline More actions disclosure that routes to existing handlers.
5. Prove exact representation round-trip, safety overrides, mutation/error
   behavior, and production geometry with touched/direct-owner tests only.
6. Update Prompt documentation, record inverses/static evidence, review, and
   close through Backlog CLI.
