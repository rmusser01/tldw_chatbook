---
id: TASK-38613
title: App-core triage batch from the 2026-11-02 audit
status: To Do
created_date: 2026-11-02 14:30
labels:
- cleanup
- utils
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decision and cleanup items from the audit: scrypt N is 2^14 below current guidance, every settings save strips all config.toml comments, and the state package is mostly test-only with a plaintext password field. Each needs a keep-or-change decision. See bug-sweep-2026-11-02-dev.md app core findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded on scrypt cost or raised toward guidance
- [ ] #2 Comment-preserving config write evaluated or non-preservation documented
- [ ] #3 state package reduced to live members
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
