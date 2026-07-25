---
id: TASK-578
title: Gate skill_file reads on trust-manifest membership
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 14:35'
updated_date: '2026-07-25 16:33'
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
- [x] #1 Reading a bundle path that is not fingerprinted in the skill's trusted manifest is refused
- [x] #2 The refusal is indistinguishable from a genuinely missing file, so it cannot be used to probe what a bundle contains
- [x] #3 Reading a normal fingerprinted supporting file still works, including nested paths
- [x] #4 An explicit decision is recorded for bundles that legitimately read their own vendored/pruned data — either a documented allowance or a documented breaking change
- [x] #5 The residual-risk wording in Docs/Features/Skills-Script-Execution.md is updated to match the resulting behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the real exposure: probe which pruned paths validate_supporting_file_path already rejects, so the change is scoped by evidence not assumption
2. Record the AC#4 decision (tighten vs allow) with that evidence in the task notes
3. TDD: failing tests that a pruned-but-present path (node_modules/**, *.tmp) is refused by read_skill_file with the SAME error kind as a missing file, and that normal fingerprinted files incl. nested paths and SKILL.md still read
4. Generalise the existing script-side trust-material helper so both seams share one gate rather than duplicating it
5. Wire the gate into read_skill_file after containment, before any stat, preserving the oracle-safe error
6. Update Docs/Features/Skills-Script-Execution.md residual-risk wording
7. Run Tests/Skills + Tests/Agents + Tests/Chat
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the read seam agree with the execution seam: skill_file requires a file to appear in the skill's trusted manifest, WITH a user-directed exemption for vendored dependency data.

Scoped by evidence first. Probing validate_supporting_file_path showed most of the pruned set was ALREADY unreachable — the validator rejects any path segment not starting with an alphanumeric, covering .git/, .github/, .hg/, .svn/, __pycache__/ and ~-suffixed names. The genuinely reachable-but-pruned surface was node_modules/**, Thumbs.db, and *.tmp/*.part/*.swp/*.pyc/*.pyo; of those, all but node_modules text and *.tmp/*.part text are binary or editor artifacts reads already refused.

DECISION (AC#4), user-directed: grant an exemption for bundles reading their own vendored data. Reads under a vendored dependency tree (VENDORED_READ_EXEMPT_DIRS = {node_modules}) are allowed without manifest membership, because requiring node_modules to be fingerprinted would defeat the pruning that keeps such bundles trustable at all. Deliberately narrow: transient editor/build artifacts (*.tmp/*.part/*.swp/*.pyc) are NOT vendored data and stay refused, and the exemption is READ-ONLY — execution still demands manifest membership with no exemption, so vendored code never becomes runnable.

Because an exempted file is by definition one no human saw at trust review, the read is labelled: the result carries trust_reviewed=False and the content is prefixed with a banner telling the agent to treat it as untrusted input rather than instructions. The banner rides in content because that is the only channel reaching the model. Residual documented plainly: a bundle can still place agent-readable text under node_modules/ that trust review never shows.

Implementation: generalised the script-side helper (_script_path_is_trust_material -> _path_is_trust_material) so both seams share ONE gate, wired into read_skill_file after containment and before any stat (a pure manifest lookup, so a refused file is indistinguishable from a missing one, AC#2), plus _is_vendored_read for the exemption. Result shape gained trust_reviewed (one existing exact-shape assertion updated accordingly). Also removed a dead 'import stat' in export_skill that ruff flagged on every PR touching this file.

Tests: Tests/Skills/test_skill_file_trust_material.py (12 tests) — RED first, refusal tests failing DID NOT RAISE against the old code. Covers: fingerprinted/nested/body reads still work; non-vendor pruned paths refused; refusal indistinguishable from missing; vendored reads allowed; vendored reads labelled + bannered; ordinary reads unbannered; and the exemption explicitly NOT extending to describe/run. Full run: Skills+Agents 610, Chat 2174/69 skipped, ruff clean.

Files: tldw_chatbook/Skills_Interop/local_skills_service.py, Docs/Features/Skills-Script-Execution.md, Tests/Skills/test_skill_file_trust_material.py, Tests/Skills/test_read_skill_file.py
<!-- SECTION:NOTES:END -->
