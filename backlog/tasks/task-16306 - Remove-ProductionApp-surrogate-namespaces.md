---
id: TASK-16306
title: Remove ProductionApp surrogate namespaces
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:06'
updated_date: '2026-08-14 09:08'
labels:
  - testing
  - architecture
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ProductionApp evidence rooted in real application owners and native module/configuration seams instead of generic SimpleNamespace substitutes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ProductionApp tests contain no SimpleNamespace or MagicMock surrogate calls
- [x] #2 Video retention coverage uses an explicit mutable fixture contract
- [x] #3 Transformers coverage patches the real constants module and the optional picker uses a real module object
- [x] #4 The ProductionApp surrogate architecture gate and affected focused tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the failing surrogate-pattern architecture node as RED evidence.
2. Replace only the four generic namespace calls with a small typed retention fixture, direct constant patching, and ModuleType.
3. Run the architecture node, affected focused tests, and static checks.

ADR required: no
ADR path: N/A
Reason: This is test-evidence cleanup with no production boundary or behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed every generic namespace surrogate from ProductionApp evidence without weakening the architecture rule. Video retention now uses an explicit mutable dataclass, the Transformers cache path patches the real Hugging Face constants module, and the optional picker import is represented by a real ModuleType. Verification: the surrogate architecture node plus five affected focused tests passed (6 total); Ruff lint/format, py_compile, and git diff --check passed.
<!-- SECTION:NOTES:END -->
