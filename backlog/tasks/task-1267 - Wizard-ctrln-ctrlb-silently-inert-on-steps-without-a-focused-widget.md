---
id: TASK-1267
title: Wizard ctrl+n/ctrl+b silently inert on steps without a focused widget
status: To Do
assignee: []
created_date: '2026-07-29 22:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the first-run wizard final-review fix wave (see Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md). Textual key-binding resolution walks ancestors of the focused widget; on steps like Provider (RadioSet with no default-pressed button) nothing is focused after on_show, so the container's ctrl+n/ctrl+b bindings never fire. Pre-existing focus-management gap, orthogonal to the crash fixed in the wave; needs a focus strategy in each step's on_show (or a screen-level binding), without modifying BaseWizard.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ctrl+n advances from every wizard step without requiring a prior click
- [ ] #2 ctrl+b goes back from every step past Welcome
- [ ] #3 Pilot regression test covers at least Provider and Model steps
<!-- AC:END -->
