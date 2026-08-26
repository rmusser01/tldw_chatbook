---
id: TASK-21504
title: 'Console: teach Recipe workflow and prove Library round trip'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 07:49'
labels:
  - console
  - prompts
  - recipes
  - ux
  - uat
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the reusable Recipe path understandable to a first-time user and close the loop between building a Recipe in Console and finding, reopening, editing, and using it as a first-class item in Library > Prompts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The improvement chooser describes the Recipe mode by outcome as `Build a reusable prompt`, while preserving an explicit Recipe identity for saved Library artifacts.
- [x] #2 The Recipe chooser explains Outcome-first, blank, and saved Recipe starting points in one concise sentence each before selection.
- [x] #3 The first-use Outcome-first editor initially emphasizes Goal, Context and evidence, Constraints, and Output; optional Role, Personality, Collaboration style, Success criteria, and Stop rules remain editable and discoverable through progressive disclosure.
- [x] #4 Saving a Recipe shows a confirmation naming `Library > Prompts` and provides an `Open Library` action focused on the newly saved Recipe.
- [x] #5 A real UI round trip saves a new Recipe, finds it in Library with a Recipe label, reopens it losslessly, edits and fills it, reviews the generated Prompt, and applies the intended lanes to Console.
- [x] #6 Recipe block order, Markdown/XML syntax, mapping hints, optional blocks, source identity, optimistic version behavior, and Prompt-versus-Recipe execution guards remain intact across that round trip.
- [x] #7 The current real-app QA capture run completes against current dev and emits assertions and rendered evidence for the save-to-Library round trip and post-Apply session state without relying on removed Console helpers.
- [x] #8 The guided and experienced-user Recipe flows remain keyboard-complete and usable at 140x40, 100x30, and 80x24 without hiding required content or actions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rename the chooser path around the reusable-prompt outcome and add concise starting-point guidance.\n2. Add progressive disclosure so Outcome-first opens with Goal, Context and evidence, Constraints, and Output emphasized while optional blocks remain editable and keyboard discoverable.\n3. Add a saved-Recipe confirmation with an Open Library action that selects the newly saved Recipe.\n4. Extend UI tests and the real-app QA harness through save, Library lookup, lossless reopen/edit/fill/review, Apply, and post-Apply state at all three responsive sizes.\n5. Run targeted tests, current-dev QA, rendered visual inspection, Ruff, compilation, and diff checks.\n\nADR required: no\nADR path: N/A; ADR-040 remains applicable.\nReason: this is onboarding, navigation, and verification within the existing versioned Recipe artifact and safe Apply boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed the chooser path around the reusable-prompt outcome, added concise guidance for all three Recipe starting points, and made Outcome-first progressively disclose five optional blocks after the four essentials. Recipe saves now show an in-modal Library > Prompts confirmation and deep-link through the existing Library navigation contract using the normalized local row identity.

The real-app QA flow exposed and fixed two integration gaps: direct-entry Recipe saves now load source capabilities lazily, and the Library handoff prefers `local_id` over composite normalized identities. The final UI round trip saves, deep-links, edits to optimistic version 2 with explicit starter content, reopens, fills, reviews, and applies User only while preserving the live System prompt. Targeted tests passed (`153 passed`), the full current-dev QA capture stage regenerated 43 visually inspected SVGs plus its observation manifest and exited 0, and focused static checks passed.

ADR required: no. ADR-040 remains the governing artifact/version and safe-Apply decision; no storage, runtime, provider, or cross-module boundary changed.
<!-- SECTION:NOTES:END -->
