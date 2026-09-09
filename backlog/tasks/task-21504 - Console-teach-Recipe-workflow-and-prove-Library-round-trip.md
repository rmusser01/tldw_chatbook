---
id: TASK-21504
title: 'Console: teach Recipe workflow and prove Library round trip'
status: To Do
assignee: []
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 04:46'
labels:
  - console
  - prompts
  - recipes
  - ux
  - uat
dependencies: []
references:
  - .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the reusable Recipe path understandable to a first-time user and close the loop between building a Recipe in Console and finding, reopening, editing, and using it as a first-class item in Library > Prompts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The improvement chooser describes the Recipe mode by outcome as `Build a reusable prompt`, while preserving an explicit Recipe identity for saved Library artifacts.
- [ ] #2 The Recipe chooser explains Outcome-first, blank, and saved Recipe starting points in one concise sentence each before selection.
- [ ] #3 The first-use Outcome-first editor initially emphasizes Goal, Context and evidence, Constraints, and Output; optional Role, Personality, Collaboration style, Success criteria, and Stop rules remain editable and discoverable through progressive disclosure.
- [ ] #4 Saving a Recipe shows a confirmation naming `Library > Prompts` and provides an `Open Library` action focused on the newly saved Recipe.
- [ ] #5 A real UI round trip saves a new Recipe, finds it in Library with a Recipe label, reopens it losslessly, edits and fills it, reviews the generated Prompt, and applies the intended lanes to Console.
- [ ] #6 Recipe block order, Markdown/XML syntax, mapping hints, optional blocks, source identity, optimistic version behavior, and Prompt-versus-Recipe execution guards remain intact across that round trip.
- [ ] #7 The current real-app QA capture run completes against current dev and emits assertions and rendered evidence for the save-to-Library round trip and post-Apply session state without relying on removed Console helpers.
- [ ] #8 The guided and experienced-user Recipe flows remain keyboard-complete and usable at 140x40, 100x30, and 80x24 without hiding required content or actions.
<!-- AC:END -->

