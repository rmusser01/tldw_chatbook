---
id: TASK-35612
title: Investigate diagnostic exit and reduce Console presentation stalls
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 19:06'
updated_date: '2026-10-19 19:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the interrupted Windows diagnostic and remove measured presentation work that delays Console input while retaining native storage safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diagnostic exit is investigated with recorded exit codes and bounded reproduction evidence; unresolved causality is stated explicitly
- [x] #2 Hidden Home completion does not repaint; the next visit presents completed data
- [x] #3 Console inspector and checked display refresh avoid redundant work or coordinator waits without weakening freshness or storage checks
- [x] #4 Targeted regressions and serial original/current Windows profiles verify behavior and report measured results
- [x] #5 PR 3166 and evidence documentation are updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: routine diagnostic and presentation fixes within ADR-126/230/231 boundaries. 1. Reproduce the diagnostic timeout failure separately and record native exit codes. 2. Add regressions for hidden Home completions, inspector derivation, and checked-display lock contention. 3. Apply minimal fixes, preserving storage checks and fresh post-await presentation. 4. Run targeted tests and serial paired profiles. 5. Review, document limitations, and update PR3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Defer hidden Home completion until a successful visible cache warm; share inspector derivation for one synchronous build (9 settings loads to 1); defer checked-display coordinator contention with native fallback retirement before retry. All storage/source/pause/action checks retained. Diagnostic C timer replaced by Python-thread synchronous dumps; six standalone runs and five native app runs exit 0, including 49 accelerated dumps. Original diagnostic exit remains unreproduced/unclassified. ABBA median input 901->748 ms and 828->821 ms; worst returns improve 14.4%/9.7%. 46 distinct targeted cases pass, final new-regression/rail batch 15 passed. Repaired one baseline-confirmed bootstrap fixture marker. No new lint diagnostics; existing formatter/lint debt recorded. Independent reviews resolved native fallback exception edge. ADR required: no, existing ADR-126/230/231 boundaries preserved. Evidence: Docs/Development/evidence/windows-performance-2026-10-18/STALLS_RESULTS.md; plan: Docs/superpowers/plans/2026-10-19-console-presentation-stalls.md. PR update pending publication.

Published implementation commit dfd5a4be5e to PR3166 and updated its description with both timing pairs and unresolved diagnostic causality. Final focused batch: 15/15 passed; 46 distinct targeted cases pass across the recorded runs. Removed all 16 exact temporary fixture/profile directories created for this follow-up. PR: https://github.com/rmusser01/tldw_chatbook/pull/3166.
<!-- SECTION:NOTES:END -->
