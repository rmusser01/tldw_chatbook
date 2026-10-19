---
id: TASK-35312
title: >-
  Give the private-path refusal Windows-appropriate repair guidance
status: Done
assignee:
  - Robert
created_date: '2026-10-18 00:00'
labels:
  - windows
  - private-paths
  - ux
priority: medium
dependencies: []
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: platform-branched guidance only; the refusal semantics stay fail-closed exactly per
  ADR-029/ADR-126 (the guard reads real NTFS ACLs; a %TEMP% that grants other principals write is
  correctly refused).
- `_is_windows()` seam (reads the `platform_files` facade, monkeypatchable) gates every branch:
  shared-writable guidance explains the ACL cause and offers `icacls "<path>"` inspection (plus
  the keep-under-user-profile recommendation, and a dedicated "this is your TEMP directory" note
  via `_path_under_windows_temp`); ownership guidance drops `sudo chown` for Properties > Security;
  generic guidance swaps `namei` for `icacls`; the ALTERNATIVE block uses PowerShell env syntax;
  `private_path_repair_hint` returns None on Windows (no safe mechanical one-liner — blind
  `icacls /remove` could break inheritance).
- POSIX output is byte-identical (gated branches only).
- Tests: 5 new Windows-branch tests in `Tests/Utils/test_startup_errors.py` (RED first) plus a
  platform-agnostic POSIX-identity test; two existing tests that implicitly assumed the POSIX
  branch are now pinned to it, and the double-emit subprocess test accepts either platform's
  repair line. Remaining Windows failures in these two files are pre-existing
  `Path("/home/...")`-vs-backslash mismatches (verified at baseline), unchanged by this task.
- Modified files: `Utils/startup_errors.py`, `Tests/Utils/test_startup_errors.py`,
  `Tests/test_cli_startup_private_path.py`.
- ADR check: no new ADR — copy-only change inside the existing private-data boundary; ADR-029 and
  ADR-126 linked here as the governing decisions.
