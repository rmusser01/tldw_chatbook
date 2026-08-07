---
id: TASK-2852
title: Library evidence handoff can land in a locked Console with no receipt
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - search-rag
  - console-handoff
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-04, observed at dev `6ffa56516`, fresh profile with no provider
configured).

Search/RAG → select evidence → "Use in Console" navigated to Console's locked onboarding
("Get started / Composer unlocks after setup") with zero receipt of the selection — no staged-
evidence chip, no toast, no trace. The flagship Library→Console handoff silently ate the user's
selection and stranded them on a setup screen that never mentions it.

The staged-evidence strip DOES exist on a configured Console (shipped in PR #1320); this is the
unconfigured edge: the handoff is neither gated nor warned when Console cannot accept work.
Re-verify at current dev before implementing (RAG-truth PR #1385 merged since observation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When Console is locked (setup incomplete/no provider), "Use in Console" either warns before navigating (naming that evidence is saved and what unlocks it) or is disabled with that reason at the button
- [ ] #2 If navigation proceeds, the locked Console surface shows a visible receipt that Library evidence is staged and will be usable after setup
- [ ] #3 The configured-Console path is unchanged (staged-evidence strip still appears; regression-covered)
- [ ] #4 Live TUI verification on a fresh profile confirms the chosen behavior
<!-- AC:END -->
