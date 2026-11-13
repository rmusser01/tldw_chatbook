---
id: TASK-40721
title: Prepare isolated self-hosted Linux CI capacity and reversible PR routing
status: Done
created_date: 2026-11-13 17:30
assignee:
- '@codex'
priority: high
updated_date: 2026-11-13 18:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce multi-hour GitHub runner waits using the owner's Linux server while preserving required test coverage and keeping CI isolated from existing server workloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runner capacity is sized from observed host resources and runs in disposable VMs without host mounts or access to private networks.
- [x] #2 PR test and control jobs can route to separate local pools with a hosted fallback; required check names and test selections remain unchanged.
- [x] #3 A representative existing Linux shard completes on the server with recorded timing, and focused workflow tests pass.
- [x] #4 Provisioning, runner renewal credentials, rollback, and any remaining activation prerequisite are documented; no unverified fleet is enabled as the required CI path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: New CI execution and credential/isolation boundary; preserve ADR-103's required gate and test selection.

1. Record measured queue delay, host resources, isolated worker design and rollback.
2. Prepare opt-in PR/control routing with hosted defaults and fork fallback; preserve required names, selectors and dependency graph. Test routing and existing CI contracts.
3. Prepare pinned disposable QEMU controller with four test workers and one control worker, App credentials outside guests and blocked private-network egress.
4. Build Ubuntu runner image and run a representative existing shard with timing and resource evidence.
5. Document activation prerequisites and rollback; enable routing only after runner renewal and execution work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Prepared reversible PR/control runner routing while preserving test selections, required names, job graph, fork fallback and bulk hosted execution. ADR-359 records the credential/isolation boundary. Eight 2-vCPU/6-GiB test VMs plus one control VM run under a 64-GiB/12-logical-CPU cgroup and bounded 100-GiB storage. The repository-only GitHub App key stays root-owned outside guests. The pinned controller plus checked-in tested patch reclaims deregistered/offline guests and bounds busy execution; systemd KillMode=mixed preserves draining.

Evidence: 586 focused CI cases passed plus the harness-dependent storage lease case separately; independent routing review passed. Actual guest public HTTPS/private-SSH boundary checks passed. Existing UI shard 36/36 at dev 1455ceeb: 116 passed, 1 existing Darwin-only skip, 714.60s, 619888 KiB maximum test-process RSS. GitHub qualification 43478237685 passed normal renewal; 43480488554 passed after intentionally disabling guest poweroff, with host reclamation logged and a new control job 23 seconds later. Race-enabled pool/config/controller/GitHub/QEMU tests, Go vet, systemd validation, and backlog/ADR checks passed. Deployed patched binary SHA256 is documented in scripts/ci/self_hosted/README.md.

Service enabled at boot; nine runners online; CI_CONTROL_RUNNER and CI_TEST_RUNNER set only after qualification. Workflow integration on dev and first complete PR-gate latency remain pending PR merge/checks; no full-suite success is claimed. Representative load supports eight slots but sustained concurrency remains to be measured. No full application suite run. Added the actual shutdown-qualification incident to lessons-testing-evidence.md. TASK-40722 separately fixes two pre-existing current-dev plugin backup census failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
