---
id: TASK-33289
title: 'PERF-30: Cold-feature and hygiene sweep (P2/P3 across TTS, Audio, Evals, Chunking,
  interop)'
status: To Do
created_date: 2026-09-28 18:04
labels:
- performance
- hygiene
- perf-audit-2026-09
priority: low
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The remaining P2/P3 findings in cold features: TTS, Audio, STT, Evals, Chunking, Backup_Recovery and the interop packages. Examples:
- the legacy TTS request imports all seven backends on the loop (torch via higgs)
- AudioService.convert_audio does sync decode/encode inside async def
- the model-catalog refresh fsyncs on the loop every launch
- the Environment tier spawns about 10 git processes every 10 s (TASK-31628)
- web-search relevance runs serially with deliberate sleeps

24 of these findings were never adversarially verified. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-30; every issue with file:line is listed under PERF-30 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All unverified items in this group are re-verified (confirmed or closed as refuted)
- [ ] #2 Confirmed P1 items are fixed; P2/P3 items are fixed or deferred with a recorded reason
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
