---
id: TASK-35594
title: Prepare checked Console config before FULL presentation
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 15:55'
updated_date: '2026-10-19 16:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expired Console presentation projections enter native config admission synchronously during FULL reconciliation, delaying Windows interaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FULL presentation prepares the original checked config off-loop while preserving direct fallback and live action checks
- [x] #2 Source, session, visit, maintenance and cancellation changes retain authoritative FULL replay and finite read ownership
- [x] #3 Targeted native controls and repeatable before/after Windows profiles document benefits and limitations
- [x] #4 Related native lifetime and presentation fixtures exercise their assertions with current screen fields; no storage guards are bypassed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-126 checked-display and ADR-230 finite ownership apply. ADR-231 remains proposed; this does not implement general narrow polling or remove FULL effects. 1. Pin native fallback cause with original return-site observations and baseline. 2. Add regression controls for expired FULL presentation and retained direct fallback. 3. Await existing eligible config projection before synchronous presentation, recheck visit/source/maintenance and preserve whole-FULL replay on refusal. 4. Run targeted controls, independent review, paired profiling and update PR3166.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FULL now awaits the existing stock checked config projection before rail/control presentation, revalidates the attach visit and maintenance state, and preserves whole-FULL replay and original direct/custom fallback. Native storage checks, live action authority and physical read retirement are unchanged (existing ADR-126/230; no new ADR, ADR-231 remains Proposed). Added seven original-native regression controls and repaired six baseline-proven stale fixture cases; 107 targeted checks passed across focused runs. New helper/tests/scripts pass Ruff; chat_screen retains 22 unchanged lint findings. Paired Windows profiles remove native config main-thread samples 69 to 0 and reduce median worst heartbeat overshoot 436 to 243 ms (44.1%); overall median input arrival worsens 838 to 905 ms, so no general resume speedup is claimed. Earlier input timeouts reproduce on original code in character/native-admission preparation; both failed and successful measurements are documented in FULL_RESULTS.md. Independent review found no authority/fallback changes. Updated Console architecture, profiling tools, evidence and PR3166. Scope expanded only to related stale fixtures so existing native assertions execute.
<!-- SECTION:NOTES:END -->
