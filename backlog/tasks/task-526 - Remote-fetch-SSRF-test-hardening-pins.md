---
id: TASK-526
title: >-
  Remote fetch: SSRF test-hardening pins (mapped-IPv6, full auth-host matrix, redirect shapes)
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-3 review verified these behaviors correct at runtime but left them unpinned by regression tests: IPv4-mapped IPv6 rejection (::ffff:169.254.169.254 - the classic SSRF bypass), auth-header scoping across all four GITHUB_AUTH_HOSTS (only 2/4 covered), and dedicated tests for userinfo-bearing and scheme-relative redirect Locations (currently only transitively covered via validate_url).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test pins rejection of a host resolving to an IPv4-mapped IPv6 private/link-local address.
- [ ] #2 The auth-scope test asserts header presence on all four GITHUB_AUTH_HOSTS and absence off-family.
- [ ] #3 Dedicated tests pin rejection of userinfo redirect targets and correct join/re-validation of scheme-relative Locations.
<!-- AC:END -->
