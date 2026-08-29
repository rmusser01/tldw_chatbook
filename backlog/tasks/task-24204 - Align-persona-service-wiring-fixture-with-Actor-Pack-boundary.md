---
id: TASK-24204
title: Align persona service wiring fixture with Actor Pack boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 14:08'
updated_date: '2026-08-29 14:09'
labels:
  - tests
  - characters
  - actor-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the app-level Character and Persona service wiring characterization representative now that the same wiring method constructs type-checked Actor Pack services.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The wiring fixture supplies a CharactersRAGDB-compatible test double
- [x] #2 The app wiring characterization reaches and validates the CharacterPersonaScopeService assertions
- [x] #3 The complete containing test module and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this is test-fixture maintenance for an existing app wiring and Actor Pack type boundary; no service contract or ownership changes. Replace the obsolete bare database object with the narrow existing Mock(spec=CharactersRAGDB) test double, run the exact full-suite failure and complete containing module, then Ruff/format/compile/diff checks before closing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the obsolete bare-object database in the app Character/Persona wiring characterization with Mock(spec=CharactersRAGDB). The fixture now satisfies the existing Actor Pack activation type boundary while retaining an isolated, non-persistent wiring test; production code is unchanged. ADR required: no; ADR path: N/A. Verification: exact full-suite failure 1 passed; complete containing module 54 passed; Ruff check passed; Ruff format check passed; compileall passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
