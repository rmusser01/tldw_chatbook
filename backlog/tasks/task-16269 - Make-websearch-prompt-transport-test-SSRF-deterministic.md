---
id: TASK-16269
title: Make websearch prompt transport test SSRF-deterministic
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:43'
updated_date: '2026-08-14 20:45'
labels:
  - testing
  - web-search
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the result-summarization transport test deterministic while preserving the production SSRF guard in the exercised pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The summarization transport test reaches the scrape branch without DNS dependence.
- [x] #2 The real SSRF guard remains exercised and no network transport occurs.
- [x] #3 The original 25-file checkpoint and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the failing test and confirm the production SSRF refusal is correct.
2. Replace the DNS-dependent fixture URL with a deterministic public literal while retaining the real guard and fake scraper.
3. Run focused mutation evidence, the original checkpoint, and static checks.

ADR required: no
ADR path: N/A
Reason: test-fixture correction for an existing security boundary; no production behavior or architecture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the summarization transport test's DNS-dependent `example.com` input with an explicit globally routable address. The production SSRF policy still runs, while the existing fake scraper keeps the test offline.
- Preserved RED evidence: both the original 25-file checkpoint and the isolated node fell back to snippet content because the guard refused the environment-dependent hostname. Reverting the literal reproduces that failure.
- Verified the affected node (1 passed) and the original checkpoint (330 passed, 3 live-search skips). Ruff lint and `git diff --check` pass. Ruff format remains unchanged from its pre-existing red state on `HEAD`; no unrelated formatting churn was included.
- ADR required: no. This is a deterministic test-fixture correction for an existing security contract.
<!-- SECTION:NOTES:END -->
