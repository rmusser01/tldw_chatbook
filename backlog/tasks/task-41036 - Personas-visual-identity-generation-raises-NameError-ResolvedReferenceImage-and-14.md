---
id: TASK-41036
title: "Personas visual-identity generation raises NameError ResolvedReferenceImage, and 14 expression-generate tests are red on dev"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- bug
- personas
- tests
priority: medium
references:
- tldw_chatbook/UI/Screens/personas_screen.py
- Tests/UI/test_personas_expression_generate.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`PersonasScreen._visual_identity_reference` (`UI/Screens/personas_screen.py:12003`) builds a `ResolvedReferenceImage`, but the module imports that class only under `TYPE_CHECKING` (`:422`). So the call raises NameError at runtime, and any visual-identity generation that resolves a pack reference fails, including "generate all", restoring a missing canonical asset, and the provider-failure path.

`Tests/UI/test_personas_expression_generate.py` fails 14 tests on dev ecad404bd4 (2026-11-15) (`-n 4`, bootstrap profile):
- 5 fail on this NameError;
- 2 fail on fakes that lack a newer `submitted_...` keyword argument;
- 7 fail on cancellation, draining or call-ceiling assertions in the same generation paths.

TASK-36948.3's notes also recorded the NameError as pre-existing. The file runs on no PR lane, so CI does not see it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generating a visual identity from a pack reference works in the app; a test that drives the real `_visual_identity_reference` was red before the fix, and the task notes name the commit that broke it
- [ ] #2 Every node of Tests/UI/test_personas_expression_generate.py passes on dev at `-n 4`; each red node first gets a named cause
- [ ] #3 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
