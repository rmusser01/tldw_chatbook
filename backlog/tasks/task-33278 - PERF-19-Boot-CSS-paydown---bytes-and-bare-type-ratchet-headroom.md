---
id: TASK-33278
title: 'PERF-19: Boot CSS paydown - bytes and bare-type ratchet headroom'
status: To Do
created_date: 2026-09-28 18:03
labels:
- performance
- css
- startup
- perf-audit-2026-09
priority: low
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Boot-parsed CSS is at 607,640 of 608,090 bytes (450 B headroom) and the bare-type-subject rule ratchet at 273-274 of 274. About 58 KB of non-boot screen CSS (research, settings-theme, lab) still rides the boot bundle. The boot-time CSS staleness check costs 80-134 ms per source-tree boot against a documented ~0.3 ms (TASK-18910). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-19; every issue with file:line is listed under PERF-19 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Research, settings-theme and lab CSS load with their screens instead of at boot
- [ ] #2 Boot CSS bytes and bare-type-rule counts have at least 5% headroom
- [ ] #3 The CSS staleness check costs under 5 ms per boot
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
