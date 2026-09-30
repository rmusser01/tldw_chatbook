---
id: TASK-33271
title: 'PERF-12: Console send path - one off-loop turn snapshot'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33265
- TASK-33266
labels:
- performance
- console
- chat
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first send spends 6.5-11.8 s before the provider call. Two per-send turn-snapshot builders run sync I/O on the loop:
- trust status and fingerprint for every installed skill (40-90 ms per skill)
- the MCP catalog and permissions, resolved twice (about 210 ms)
- the workspace registry, re-read three times
- world books and dictionaries with N+1 reads
- a cold RAG ConfigProfileManager built even when RAG is unused (794 ms)

World-info matching runs one regex per key over the scan text (219 ms at 1,000 entries). The credential sanitizer re-scans the transcript 5-7 times per send. Terminal persistence and trace reservation run write transactions on the loop. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-12; every issue with file:line is listed under PERF-12 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single turn-snapshot builder computes per-send configuration off the event loop
- [ ] #2 Skill trust/fingerprint and the MCP catalog are resolved at most once per send and cached across sends until their inputs change
- [ ] #3 Sending with RAG unused does not construct the RAG profile manager
- [ ] #4 World-info matching at 1,000 entries costs under 20 ms per send
- [ ] #5 Enter-to-provider-dispatch on the send probe is under 1 s with 5 installed skills
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
