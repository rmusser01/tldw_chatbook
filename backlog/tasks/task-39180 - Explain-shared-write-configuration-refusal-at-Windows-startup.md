---
id: TASK-39180
title: Explain shared-write configuration refusal at Windows startup
status: Done
assignee: []
created_date: 2026-11-11 11:03
labels:
- uat
- startup
- windows
dependencies: []
priority: low
updated_date: 2026-11-11 20:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A configuration path under a shared-write ancestor is correctly refused on Windows. The startup preflight turns the specific reason into `configuration_unavailable` and directs users toward Backup & Restore without explaining the path problem.

### Reproduction

1. On Windows at local `origin/dev` `6bd97b112f`, create a new disposable config directory beneath an existing shared-write ancestor. In the observed case, Chatbook's native permission inspection reported the checkout as `0o766`.
2. Point `TLDW_CONFIG_PATH` at `config.toml` in that disposable directory and run `python -m tldw_chatbook.app`.
3. Observe **Recovery required: configuration_unavailable** with **Open Backup & Restore** and **Exit**.
4. Re-run against a fresh profile under a private ancestor. The setup wizard opens.

### Evidence

- [Startup reproduction and underlying diagnostic](../../qa/uat-core-workflows-2026-11-11/windows-startup-recovery.txt)

A direct preflight diagnostic reported `PrivatePathError unsafe_parent: shared_writable_parent`. `startup_preflight()` in `Backup_Recovery/launcher.py` maps that exception to the generic reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Windows startup still refuses configuration paths with a shared-write ancestor.
- [x] #2 The startup screen explains that folder permissions caused the refusal and provides an actionable next step such as choosing a private config path.
- [x] #3 The guidance applies when starting a fresh profile and does not imply that a backup is needed to resolve the permission error.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: ../decisions/029-local-private-data-boundary.md
Reason: Surface the existing private-path refusal without changing admission.

Test refusal propagation and the rendered guidance, preserve the typed shared-write reason in startup preflight, and show the private-path remedy on the first screen.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Startup preflight preserves the shared-write parent refusal and the startup host shows private-config guidance with Exit. The native path guard still rejects the original profile. [Native 80×24 rerun](../../qa/uat-core-workflows-2026-11-11/windows-startup-fixed.txt) shows the full remedy. The typed-refusal regression and four existing unlock-host cases passed within the 9-test focused run. [ADR-029](../decisions/029-local-private-data-boundary.md) still governs the storage boundary.
<!-- SECTION:NOTES:END -->
