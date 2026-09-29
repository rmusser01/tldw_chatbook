---
id: TASK-33276
title: 'PERF-17: Boot import diet and pre-import ratchet paydown'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33260
labels:
- performance
- startup
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Boot imports carry avoidable weight:
- the TTS/STTS stack (43 modules, about 97 ms) via @on Message classes
- tldw_profile_core for one integer (about 25 ms)
- requests via 7 boot modules
- the citation pydantic cluster (45-60 ms)
- AA hue pinning for all 91-94 themes at import (20-35 ms)
- the MCP gateway via MCP/server.py
- HF datasets via the Evals orchestrator
- all 87 splash effects
- image-protocol warm-up on every boot

app.py holds 404 of 470 module-scope names that are only used inside functions (TASK-33011). The screen pre-import payload ratchet is red at 554/500 modules. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-17; every issue with file:line is listed under PERF-17 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed import is deferred to first use, with the import-weight ratchet lowered to the new measured value
- [ ] #2 The screen pre-import payload is back under its budget
- [ ] #3 Only the active theme is contrast-pinned at startup
- [ ] #4 import tldw_chatbook.app loads measurably fewer modules (recorded before/after)
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
