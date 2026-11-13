---
id: TASK-40723
title: Keep self-hosted Python imports working in sanitized child processes
status: In Progress
created_date: 2026-11-13 18:37
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first full local CI gate fails because setup-python installs under the runner work directory and sanitized child processes lose access to the installed dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A setup-python interpreter imports installed pytest after LD_LIBRARY_PATH is removed and HOME changes.
- [ ] #2 Fresh disposable runners use the canonical tool-cache location while private-profile isolation remains unchanged.
- [ ] #3 The failing project-chat UAT is rerun and the corrected full-gate timing is tracked separately from the failed attempt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Repair the guest image toolchain within the accepted runner boundary.

1. Reproduce the sanitized-child import failure against the same official Python archive.
2. Configure the guest runner tool cache at /opt/hostedtoolcache and add an actual sanitized-child qualification assertion.
3. Publish a verified trusted base image, recycle old runners after jobs finish/cancel, rerun the failed UAT and normal PR gate. Preserve first-attempt timings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced the exact Python 3.12.16 archive failure in a trusted bake VM, then verified pytest imports with LD_LIBRARY_PATH removed and HOME changed after moving the installation to /opt/hostedtoolcache. The runner .env selects that canonical cache and the directory is runner-owned. Added a reusable guest customization script and sanitized-child checks for pinned 3.12.11 and current 3.12. No application fixture changes. Independent review passed; the temporary promotion helper was invoked after blocking systemctl stop completed and its reuse guard was tightened to require explicit inactive state. Fresh fleet is online. Live GitHub qualification and the failing UAT rerun remain pending, so this task stays In Progress.

Follow-up gate 43485256104 exposed an incomplete preloaded interpreter: setup-python cache initialization could not find pip. Removed the manually seeded Python 3.12.16 tree from a new trusted base; retain only canonical cache configuration and let the official action perform the full installation. Qualification now enables cache: pip to match real lanes. The task remains In Progress pending live cache qualification and UAT.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
