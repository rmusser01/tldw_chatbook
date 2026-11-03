---
id: TASK-38613
title: App-core triage batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- cleanup
- utils
priority: low
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decision and cleanup items from the audit: scrypt N is 2^14 below current guidance, every settings save strips all config.toml comments, and the state package is mostly test-only with a plaintext password field. Each needs a keep-or-change decision. See bug-sweep-2026-11-02-dev.md app core findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision recorded on scrypt cost or raised toward guidance
- [x] #2 Comment-preserving config write evaluated or non-preservation documented
- [x] #3 state package reduced to live members
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Safe arm: scrypt decision and comment-stripping tradeoff recorded in code with follow-up paths; state package reduced to UIState with the plaintext password field removed. Commit 14e57c46e2.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
