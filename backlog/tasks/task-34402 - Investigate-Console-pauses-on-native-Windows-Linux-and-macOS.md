---
id: TASK-34402
title: Investigate Console pauses on native Windows Linux and macOS
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 19:17'
updated_date: '2026-10-06 00:23'
labels: []
dependencies: []
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

Preserve the original Fleet journey and 300-second limits while diagnosing the remaining pre-arm/retirement failure. Before another App run, qualify actual Windows launcher plus CPython descendant containment using the existing Notes process-tree controller; the three serialized real process-only controls now pass with all captured PIDs physically absent. Add only opt-in bounded original-body/stage checkpoints, with all checkpoint construction failing open on OSError so body exceptions retain priority. ADR required: no; diagnostic-only observation and reuse of the existing containment contract do not change application ownership or permissions.

Config publication causal measurement: serialize two fresh actual Windows cases (configured and default root) that seed and physically retire one original AgentRunsDB, then observe exactly one unchanged Library lifecycle writer with original local monitoring. Preserve raw-operation retirement, source pins, file/runtime generation effects, exact actor and hooks; report directory companion/decline facts and Native attempts without summing overlapping spans or changing any authority/cache. ADR required: no; test-only observation under existing ADR126. Production optimization requires a separately proven cause and registered boundary amendment.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a bounded native full-Console probe with real guarded file-backed storage and three immediate captured replies, plus an unchanged-height control and per-helper caller attribution. Native Linux and macOS CI and local Windows completed three trace calls with three links and no dispatch checkpoints. Windows Server 2022 refused a workspace WAL with projected owner UID 0; exact default-owner mechanism remains unproven and is documented separately. Root causes: repeated synchronous refresh reads, serialized guarded admission with Windows full derivation, repeated POSIX connection-validation helpers, and redundant stylesheet work. Report and compact native manifest are in qa/console-pause-investigation-2026-10-04; lessons-testing-evidence records the native counter blind spot. Native runs 37228072972 and 37228455116 retain raw evidence. Ruff lint/format and diff checks passed; peer review found no Critical or Important issue, and attribution and measurement limits were corrected. ADR required: no; existing ADR-126 applies. No production performance behavior changed and no full suite was run. Timings are pinned to original fix commit 16e5f027. Concurrent PR updates advanced PR 3017 to b825d974 with a newer dev merge; the only production difference is console_trace_service.py, while all examined UI, admission, config, native-file, SQLite and CSS sources remain unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->
