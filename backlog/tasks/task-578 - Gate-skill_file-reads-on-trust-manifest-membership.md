---
id: TASK-578
title: Gate skill_file reads on trust-manifest membership
status: To Do
assignee: []
created_date: '2026-07-25 14:35'
labels:
  - skills
  - security
  - trust
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Script execution now requires a bundle path to be present in the skill's trusted manifest, but `skill_file` reads still gate only on `validate_supporting_file_path`.

The trust scanner deliberately prunes junk paths (`node_modules/`, `.git/`, `__pycache__/`, `*.tmp`, `*.part`, `*~`, `*.pyc`, `.DS_Store`) from fingerprinting, so a file on such a path is never fingerprinted and never shown in the human trust review — yet an agent can still read its contents. A shipped bundle can therefore carry agent-readable instructions the reviewer never saw, and a script running under a standing grant can write more of them without perturbing the skill's digest.

This predates the script-execution work and is not a regression, and it cannot escalate into execution (only manifest-fingerprinted files can run). But it is the last place where "passes the path validator" still substitutes for "is trust material", so the two seams should agree.

The open question this task must settle: some real bundles legitimately read their own vendored data from pruned paths. Tightening reads may be a breaking change for them, so the fix needs an explicit decision rather than a silent tightening.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reading a bundle path that is not fingerprinted in the skill's trusted manifest is refused
- [ ] #2 The refusal is indistinguishable from a genuinely missing file, so it cannot be used to probe what a bundle contains
- [ ] #3 Reading a normal fingerprinted supporting file still works, including nested paths
- [ ] #4 An explicit decision is recorded for bundles that legitimately read their own vendored/pruned data — either a documented allowance or a documented breaking change
- [ ] #5 The residual-risk wording in Docs/Features/Skills-Script-Execution.md is updated to match the resulting behaviour
<!-- AC:END -->
