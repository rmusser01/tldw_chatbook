---
id: TASK-37234
title: Trace persistent project tool exclusions after disclosure recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28 18:29'
updated_date: '2026-10-28 18:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Fedora reporter still cannot inspect a new project after locally integrating PR #3284. The prior change proved schema recovery but did not establish why that run excluded the requested tools. Determine the actual build and refusal path, reproduce matching inputs and correct verified defects without widening tool authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tested commit and current refusal category are distinguished from the still-open GitHub PR state.
- [ ] #2 Tool recovery and command controls run on Linux Python 3.14, matching the interpreter family recorded in the original report.
- [ ] #3 Configuration-capture default mismatches are reproduced at their real consumer before a fix is attributed to the report.
- [ ] #4 Any implementation preserves explicit disabled tools, persona restrictions, folder authority and approval gates, with focused regression evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no for investigation and qualification under existing ADR-104 disclosure, ADR-032 local permissions and ADR-069 folder authority; a new boundary decision would be recorded before implementation. 1. Obtain the reporter tested commit and current exact refusal text while tracing composition. 2. Run the existing real recovery/command controls on Linux Python 3.14 as well as 3.12. 3. Reproduce capture-default versus provider/UI mismatches through actual configuration and catalog consumers. 4. Repair only evidenced causes, verify narrow controls and document what remains unconfirmed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reporter says local tools are on and current error is Tool schema not loaded, narrowing the path to registered/allowed but undisclosed tools. GitHub PR remains open; reporter-provided code inspection identifies local PR integration but the reporter git line is still pending. Supplied app logs show Python 3.14.8, two provider entries and four opaque agent run step records, not find/load arguments/results. Supplied a validated read-only SQLite step extraction. Real planner probe with the unknown reported llama_cpp model, native/fence mode and verified/unverified 32k windows retained find_tools/load_tools in all four cases. No planner or permission change is inferred. Linux Python 3.14.9 run 40746120405: 141 passed, one POSIX descendant control failed before PID publication within its 600 ms lifetime. Reworked only that controlled test to observe descendant/leader readiness before public cancellation, retain cleanup and budget assertions and independently prove the OS group absent; added an 800 ms slow-start control. Original incident cause remains unconfirmed pending actual discovery trace.
<!-- SECTION:NOTES:END -->
