---
id: TASK-34402
title: Investigate Console pauses on native Windows Linux and macOS
status: Done
created_date: 2026-10-04 19:17
assignee:
- '@codex'
updated_date: 2026-10-04 19:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify the local causes of slow Console interaction and pre-provider send delays, and distinguish shared behavior from platform-specific amplification using native evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retained native Windows Linux and macOS measurements identify the source revision and preserve storage safety checks.
- [x] #2 Three durable captured sends complete with an immediate provider recorder on each platform, or platform failures are documented with evidence.
- [x] #3 Report maps measured blocking work to source callers and distinguishes verified platform behavior from inference.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Observe guarded storage and native file calls without changing their behavior, using private profiles and an immediate provider recorder. 2. Run a bounded mounted Console probe on native Windows, Linux and macOS. 3. Correlate stage timings, event-loop delays and call counts with source paths and existing performance checks. 4. Retain results and document root causes and practical limits. ADR required: no. ADR path: N/A (existing ADR-126 applies). Reason: diagnostic measurements preserve runtime and security boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a bounded native full-Console probe with real guarded file-backed storage and three immediate captured replies, plus an unchanged-height control and per-helper caller attribution. Native Linux and macOS CI and local Windows completed three trace calls with three links and no dispatch checkpoints. Windows Server 2022 refused a workspace WAL with projected owner UID 0; exact default-owner mechanism remains unproven and is documented separately. Root causes: repeated synchronous refresh reads, serialized guarded admission with Windows full derivation, repeated POSIX connection-validation helpers, and redundant stylesheet work. Report and compact native evidence manifest are in qa/console-pause-investigation-2026-10-04; lessons-testing-evidence records the native counter blind spot. Native runs 37228072972 and 37228455116 retain raw evidence. Ruff lint/format and diff checks passed; read-only peer review found no Important issue, and measurement limits were tightened. ADR required: no; existing ADR-126 applies. No production performance behavior changed and no full suite was run. Fixes PR 3017 remains at 16e5f027 against dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
