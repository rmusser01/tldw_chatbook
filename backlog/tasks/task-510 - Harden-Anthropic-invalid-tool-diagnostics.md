---
id: TASK-510
title: Harden Anthropic invalid-tool diagnostics
status: Done
assignee: []
created_date: '2026-07-24 17:45'
updated_date: '2026-07-24 17:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure malformed Anthropic tool entries are dropped without rendering caller-controlled values, leaking secrets, or invoking unsafe representations, and close the associated review-task documentation gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Malformed non-dict tools are dropped without invoking their `repr`.
- [x] #2 Malformed dictionary tools are dropped without logging key values or secrets.
- [x] #3 Diagnostics identify Anthropic and a bounded metadata-only rejection reason.
- [x] #4 Existing native-tool conversion and caching behavior remains unchanged.
- [x] #5 TASK-506's implementation plan contains the mandatory ADR decision block.
- [x] #6 Focused tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression coverage proving malformed tool diagnostics neither expose secret values nor invoke a raising `__repr__`.
2. Replace caller-controlled value rendering with fixed Anthropic-specific metadata-only reasons.
3. Add the missing mandatory ADR decision block to TASK-506.
4. Run focused Anthropic tests, Ruff format/check, and `git diff --check`; complete only after review.

ADR required: no
ADR path: N/A
Reason: This hardens diagnostics inside the existing Anthropic provider boundary and repairs task documentation without changing storage, service contracts, dependencies, or long-lived architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced caller-controlled repr previews in Anthropic invalid-tool warnings with fixed provider-specific rejection reasons, so malformed inputs are dropped without rendering secrets, traversing large values, or invoking custom repr implementations. Added RED/GREEN coverage using a raising-repr object and a secret-bearing dictionary while preserving existing native/OpenAI conversion, caching, and immutability behavior. Added the missing ADR decision block to TASK-506. Verification: focused RED failed at repr; GREEN exact test passed; full Anthropic native and mocked API coverage passed (34 tests); Ruff check passed; Ruff format check passed; git diff --check passed; independent review approved. ADR required: no. ADR path: N/A. Reason: diagnostic hardening within the existing Anthropic boundary and documentation repair only.
<!-- SECTION:NOTES:END -->
