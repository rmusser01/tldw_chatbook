---
id: TASK-35312
title: Give the private-path refusal Windows-appropriate repair guidance
status: In Progress
assignee:
- Robert
created_date: 2026-10-18 00:00
labels:
- windows
- private-paths
- ux
priority: medium
dependencies: []
updated_date: 2026-10-31 17:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console UAT 2026-10-17 (`qa/console-uat-2026-10-17/`): on a Windows machine whose `%TEMP%`
genuinely grants other principals write access (verified via the guard's real NTFS ACL
projection), the refusal banner is correct but its repair guidance is POSIX-only: it suggests
`chmod g-w,o-w` and `sudo chown` (which cannot change Windows ACLs) and `$VAR=... tldw-cli`
shell syntax a Windows user cannot paste.

Scope decision: the guard's refusal semantics (ADR-029 charter, ADR-126 Windows ACL
implementation) are deliberately unchanged — only the guidance copy branches per platform.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:AC:BEGIN -->
- [x] On Windows, the shared-writable guidance explains the Windows ACL cause and offers an `icacls` inspection command instead of `chmod`
- [x] On Windows, the ALTERNATIVE block uses PowerShell env syntax
- [x] When the offender is under `%TEMP%`, the guidance says so and recommends keeping Chatbook's config/data under the user profile
- [x] POSIX output is byte-identical to today (existing tests green)
- [x] Both platform branches unit-tested via a monkeypatchable platform seam
<!-- SECTION:AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED tests for the Windows branches of `private_path_repair_hint`,
   `_shared_writable_guidance`, `_alternative_lines` (platform seam).
2. Implement the Windows copy in `Utils/startup_errors.py`.
3. POSIX regression: existing startup-error tests stay green.
PR3173 repair (2026-10-31): verify Windows and POSIX startup guidance, literal shell quoting and current CLI error boundary. Correct platform-dependent test assumptions without relaxing private-path enforcement. ADR required: no. ADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: copy-only fixes under existing native ACL policy.
<!-- SECTION:PLAN:END -->
## Implementation Notes

- Guidance branches by platform while private-path admission remains unchanged.
  Windows explains real ACL write grants, identifies shared TEMP, offers literal
  PowerShell `icacls` inspection and Properties > Security guidance, and uses
  PowerShell environment syntax. There is no speculative ACL repair command.
- Fixed the original double-quoted path interpolation bug: single-quoted
  PowerShell literals escape apostrophes and retain `$()`, backticks and spaces.
  The hostile-path regression was RED before this fix.
- POSIX copy remains covered explicitly. The emitter/CLI tests now assert their
  actual cross-platform contract. The startup subprocess uses an explicit private
  test config path; on Windows only its disposable empty child receives a real
  shared ACL, because chmod does not create the refusal under NTFS.
- All 22 startup/CLI tests pass on Windows, including actual fail-closed startup;
  together with the shutdown regressions the final direct run is 29 passed.
  No profile ACL, path-validation policy or real user HOME was changed by the runner.
- Files: startup_errors.py, its unit/integration tests and the existing test lane.
  ADR-029/126 govern the unchanged private boundary; no new ADR required. Status
  remains In Progress pending the coordinating agent's independent review.
