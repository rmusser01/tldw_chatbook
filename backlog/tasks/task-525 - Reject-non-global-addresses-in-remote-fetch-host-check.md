---
id: TASK-525
title: >-
  Remote fetch: reject non-global addresses (CGNAT/shared space) in host allow-check
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
skill_remote_fetch._assert_host_allowed rejects private/loopback/link-local/reserved/multicast/unspecified, but RFC 6598 shared address space (100.64.0.0/10, e.g. Tailscale-adjacent CGNAT ranges) passes all six predicates, so a DNS name resolving there is fetched. Adding a not-is_global check closes the gap in one line (final-review finding M2, deferred).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A host resolving to 100.64.0.1 (and other non-global special ranges) is rejected with the standard unreachable-host RemoteSkillError.
- [ ] #2 Public addresses (incl. the existing test fixtures) still pass; existing SSRF tests stay green.
- [ ] #3 A regression test pins the CGNAT rejection.
<!-- AC:END -->
