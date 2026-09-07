---
id: TASK-520
title: Generate traversal paths without Hypothesis over-filtering
status: Done
assignee: []
created_date: '2026-07-24 18:39'
updated_date: '2026-07-24 18:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the path-validation traversal property deterministic and health-check clean by generating paths that escape the base directory by construction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The traversal property generates at least one leading parent-directory component without assume-based filtering
- [x] #2 Every generated path resolves outside the temporary base directory
- [x] #3 validate_path rejects every generated traversal with the outside-directory error
- [x] #4 No production path-validation behavior changes
- [x] #5 Repeated focused runs and the full path-validation property file pass
- [x] #6 Task documentation records the intermittent health check, ADR decision, verification, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the full-batch Hypothesis filter_too_much failure and confirm the exact test can also pass intermittently.
2. Replace the filtered generic component list with a strategy that constructs one or more leading `..` components plus a generated safe leaf.
3. Assert each generated path resolves outside the base and is rejected by validate_path.
4. Run repeated focused tests, the full property file, Ruff format/check, and diff check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This improves a property-test generator without changing security policy, production validation, dependencies, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the traversal property's assume-filtered generic component list with constructive generation: one to five leading parent-directory components and a bounded alphanumeric leaf. Every example now proves the resolved path is outside the base with `Path.relative_to` and requires `validate_path` to raise the outside-directory error. Removed the conditional assertion and broad exception swallowing; production path-validation code is unchanged.

Failure evidence: the full batch raised Hypothesis `FailedHealthCheck: filter_too_much` after only 9 successful inputs and 50 filtered inputs. A pre-change isolated rerun passed, confirming the failure was intermittent generator debt rather than a deterministic validator failure.

ADR required: no. ADR path: N/A. This is a test-strategy correction with no security-policy, dependency, or architecture change.

Verification: five consecutive focused runs passed; the full `Tests/Utils/test_path_validation_properties.py` file passed 11 tests; Ruff check passed; Ruff format check reported the file already formatted; `git diff --check` passed.
<!-- SECTION:NOTES:END -->
