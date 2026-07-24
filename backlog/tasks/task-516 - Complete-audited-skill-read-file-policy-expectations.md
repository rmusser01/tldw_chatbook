---
id: TASK-516
title: Complete audited skill read-file policy expectations
status: Done
assignee: []
created_date: '2026-07-24 18:30'
updated_date: '2026-07-24 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the full audited runtime-policy registry guard aligned with the existing local and server skill read-file actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The server_skills action-id expectation includes skills.read_file launch actions for local and server sources
- [x] #2 The server_skills expected action set exactly matches the current branch registry
- [x] #3 Runtime policy registry production definitions remain unchanged
- [x] #4 The focused core registry guard passes
- [x] #5 Task documentation records the merge-base failure, registry origin, upstream corroboration, ADR decision, and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the server_skills mismatch exposed after fixing the earlier watchlist oracle and verify the same drift exists on merge base.
2. Confirm skills.read_file is an intentional current-branch registry resource and that upstream independently added the same expectation.
3. Add only the current branch local/server read-file action IDs to the test oracle.
4. Run the focused core registry guard plus static checks; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This updates a stale test oracle for an existing policy resource and changes no enforcement, ownership, capability, dependency, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the exact server_skills test-oracle update by adding only `skills.read_file.launch.local` and `skills.read_file.launch.server`. Before the change, those were the complete actual-minus-expected delta and expected-minus-actual was empty; the focused core guard now matches the current registry. Production runtime-policy definitions were not changed.

Root cause and corroboration: the supplied merge-base comparison and current branch exposed the same stale expectation for the `skills.read_file` resource introduced by `bef1e945c7`. Upstream `d0d51759c` independently includes both read-file expectations. Its `skills.install_remote` expectations were deliberately excluded because that resource is absent from the current branch registry.

ADR required: no. ADR path: N/A. This is an exact test-oracle correction for an existing policy resource and changes no enforcement or architecture.

Verification: exact audited-registry guard 1 passed; full `Tests/RuntimePolicy` 248 passed with one existing Requests dependency warning. Ruff check passed, Ruff format check reported the test already formatted, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
