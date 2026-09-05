---
id: TASK-31654
title: Isolate Terminal qualification loader module aliases
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:06'
updated_date: '2026-09-05 17:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Terminal qualification tests independent of unrelated common modules already imported by other suites, and restore import state after each script load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qualification probes import their sibling common module even when sys.modules already contains an unrelated common module.
- [x] #2 Loading a probe restores the exact prior common module, qualified alias, and sys.path without leaking aliases.
- [x] #3 Targeted loader regressions and the dependency qualification file pass with scoped static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce probe imports with a foreign common sentinel and verify the current loader leaks sibling imports when no common module was present.
2. Scope qualification-directory precedence and common/qualified aliases with pytest MonkeyPatch.context, restoring exact preexisting import state on exit.
3. Run the loader regressions, complete dependency qualification file, and scoped lint/format checks; review and commit only task-owned files.
ADR required: no
ADR path: N/A
Reason: test-only import isolation; qualification scripts and product runtime boundaries remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced manual path/qualified-module cleanup with a scoped pytest MonkeyPatch.context. Probe loads explicitly bind a freshly loaded sibling common module only during execution, then restore exact preexisting common and task-qualified aliases and sys.path. Product code and standalone qualification scripts are unchanged.

Eight parameterized regressions cover all four sibling-importing scripts with foreign common sentinels and absent common aliases; they reproduced RED before the fix and passed afterward. They assert the loaded exception comes from the intended qualification module, exact sentinel restoration, no new qualified aliases, and unchanged sys.path. The complete dependency qualification file passed 208 tests in the isolated installed Python 3.12 environment. Full-file Ruff lint/format and git diff --check passed. Self-review completed. Recorded the observed import-cache collision and leak in lessons-testing-evidence.md.

ADR required: no; test-only import isolation preserves existing script/runtime contracts. Changed files: Terminal dependency qualification tests, this task, and testing-evidence lessons.
<!-- SECTION:NOTES:END -->
