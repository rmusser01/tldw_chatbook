---
id: TASK-610
title: >-
  Remote fetch: reject non-global addresses (CGNAT/shared space) in host allow-check
status: Done
assignee: [rmusser01]
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
- [x] #1 A host resolving to 100.64.0.1 (and other non-global special ranges) is rejected with the standard unreachable-host RemoteSkillError.
- [x] #2 Public addresses (incl. the existing test fixtures) still pass; existing SSRF tests stay green.
- [x] #3 A regression test pins the CGNAT rejection.
<!-- AC:END -->

## Implementation Plan

1. Verify on the target Python (3.12) that `ipaddress.ip_address("100.64.0.1")` passes all six named predicates but `is_global` is False (premise check, empirical).
2. Write the red regression test in `Tests/Skills/test_skill_remote_fetch.py`: hostname resolving to 100.64.0.1 (CGNAT), 100.100.100.200 (Alibaba metadata, same /10), and the 100.64.0.1 IP literal must each raise `RemoteSkillError("That host is not reachable from here.")`.
3. Add `or not addr.is_global` to `_assert_host_allowed`'s rejection chain in `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`.
4. Run the targeted SSRF test files (`Tests/Skills/test_skill_remote_fetch.py`) — new test green, existing SSRF fixtures (public 93.184.216.34) stay green.

ADR required: no
Reason: One-line tightening of an existing security predicate in place; no new interface, storage, or policy surface. (A later task, task-609, consolidates this predicate with Utils/egress and owns any ADR there.)

## Implementation Notes

- **Approach**: Added `or not addr.is_global` to `_assert_host_allowed`'s rejection chain in `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`. Verified empirically on Python 3.12.11 first: `ipaddress.ip_address("100.64.0.1")` has `is_private=False` (and all six named predicates False) while `is_global=False` — the premise holds, and `not is_global` is the only floor that rejects RFC 6598 shared space. Same classification basis `Utils/egress._classify_ip` already uses (`"public" if ip.is_global else "private"`), so the two layers now agree on this range ahead of task-609's consolidation.
- **Evidence (red -> green)**:
  - Red: `.venv/bin/python -m pytest Tests/Skills/test_skill_remote_fetch.py -k cgnat -x -q` -> `1 failed` (`test_cgnat_shared_space_rejected` fetched the CGNAT host instead of rejecting).
  - Green: explicit node-id run of all 9 fetch/SSRF tests (happy path, private/mixed rejection, both new CGNAT tests, redirect revalidation, GitHub-family auth scoping, stream cap, both deadline tests) -> `9 passed`.
  - No regressions: file-wide run with the change is `36 passed, 15 failed` vs baseline HEAD (changes reverted via `git checkout HEAD -- <paths>`; no stash, per lessons) `34 passed, 15 failed` — the identical 15 are pre-existing failures with the documented TASK-32628 `RecoveryRequired("raw_source_selection_changed")` config-admission signature (every one constructs `GitHubAPIClient`, whose `__init__` reads real config getters under the per-test sandbox); they do not touch the fetch/SSRF predicate and are red at HEAD independent of this change.
- **Tests added**: `test_cgnat_shared_space_rejected` (hostname resolving to 100.64.0.1, to 100.100.100.200 Alibaba metadata inside the same /10, and the 100.64.0.1 IP literal — all must raise "That host is not reachable from here.") and `test_mixed_public_and_cgnat_rejected` (public + CGNAT mixed resolution rejects).
- **Files changed**: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`, `Tests/Skills/test_skill_remote_fetch.py`, this task file.
- **ADR required**: no (see plan; task-609 owns any ADR for the shared predicate).
