---
id: TASK-31815
title: >-
  Cross-wave Library modal-inventory repair: the skills-era discovery blocker
  plus three stale delegator rows
status: To Do
assignee: []
created_date: '2026-09-06 04:42'
labels:
  - library
  - decomposition
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_modal_dismissal.py maintains a hand-declared (file, class, presenter, modal-type) inventory and rediscovers it by AST-parsing only the files named in _SUPPORTED_OWNER_SCOPES, then asserts the two sets match in BOTH directions. The file has been 1-red at every Library decomposition wave tip since the skills series: an unresolved modal constructor for LibraryScreen._present_library_skills_import_choice_if_needed aborts discovery before the comparison ever runs, so the guard currently proves nothing about ANY subsystem's rows -- it is a blocked guard, not a failing assertion. Three further row clusters are equally stale for the same underlying reason (the named owner is now a one-line delegator, or was pruned outright, while the real presenter moved into a Library_Modules controller): the ingest row keyed on handle_library_ingest_browse, and the two skill-trust passphrase presenter rows. Wave-6 task 3 repointed the four prompts rows and proved them by construction against the new _OwnerScope, but could not prove the FILE green end-to-end because of the blocker; that same blocker is why the recipe's documented-reds list carries a standing 1-red entry for this file. This is cross-wave repair work that belongs to no single subsystem series, which is why five consecutive waves each deferred it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/UI/test_library_modal_dismissal.py runs to completion with zero failures at HEAD -- no declared-but-undiscovered row and no discovered-but-undeclared edge, in both directions of its own bidirectional assertion
- [ ] #2 The skills-era discovery blocker is resolved: _discover_library_modal_edges completes without raising an unresolved-modal-constructor error for _present_library_skills_import_choice_if_needed, and the row names whichever owner actually constructs the modal today
- [ ] #3 The stale ingest row keyed on handle_library_ingest_browse names its real present-day owner and is rediscovered by the file's own AST walk (adding the owning module to _SUPPORTED_OWNER_SCOPES first, if it is not already listed -- without that, a repointed edge is never discovered and the assertion fails the other way)
- [ ] #4 The two skill-trust passphrase presenter rows (_request_library_skill_trust_passphrase and _request_library_skill_trust_bootstrap_passphrase) name their real present-day owners and are likewise rediscovered
- [ ] #5 Each repointed row is verified against the modal's CONCRETE type, not just its presenter name, so a row that resolves to the wrong modal class is caught
- [ ] #6 backlog/docs/library-decomposition-recipe.md section 7's documented-pre-existing-failures list has the test_library_modal_dismissal.py entry removed, with the commit that fixed it named
<!-- AC:END -->
