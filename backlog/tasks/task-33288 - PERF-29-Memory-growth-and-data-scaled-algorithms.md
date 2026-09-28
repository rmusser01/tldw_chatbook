---
id: TASK-33288
title: 'PERF-29: Memory growth and data-scaled algorithms'
status: To Do
created_date: 2026-09-28 18:04
labels:
- performance
- memory
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unbounded growth and repeated work:
- the realtime mic tap keeps every frame (about 173 MB/h)
- the dictionary version-history sidecar grows without bound
- opening a conversation loads every image BLOB of every branch
- the normalized trace re-base64s every history image per provider call (55 MiB peak)
- speech models are cached per TranscriptionService instance, so every Mic press reloads the model

Quadratic or cubic algorithms:
- the meeting sink rewrites JSONL per segment (O(n^2), 10.5 s CPU for 1,500 segments)
- the Console model picker is O(M^2) per keystroke (90 ms at 2,000 models)
- chunking offset synthesis is O(N*S)
- the fallback vector store uses O(N) list lookups Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-29; every issue with file:line is listed under PERF-29 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed buffer/history has a bound or streaming replacement
- [ ] #2 Speech models are cached process-wide per model/config
- [ ] #3 Meeting sink, model picker and chunking offset synthesis scale linearly (benchmarks recorded)
- [ ] #4 Unverified appendix items in this group are re-verified before being fixed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
