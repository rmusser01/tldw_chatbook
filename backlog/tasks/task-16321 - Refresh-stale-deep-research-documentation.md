---
id: TASK-16321
title: Refresh stale deep-research documentation
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:13'
updated_date: '2026-08-15 05:48'
labels:
  - research
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs/Development/research-sessions-runs-tranche-2.md references deleted files (DB/Research_DB.py and UI/Screens/research_screen.py) and its deferred-items list predates the SSE work and the web_deep_search tool. The parity matrix and gap ledger also reference the retired research screen. This misleads contributors about what exists and what remains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tranche-2 doc reflects current reality: only existing files are referenced and the deferred list is accurate
- [x] #2 Parity matrix and gap ledger research rows are updated where stale
- [x] #3 Task-255 deferred decision (orphan Research_Window) status is stated
- [x] #4 Docs-only change with no code modifications
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update tranche-2 doc: correct the stale landed-scope claims (research screen was removed by task-255) and the deferred list (SSE landed via the 2026-04-23 plan), annotate the historical verification commands whose target files are deleted, and add a 2026-08 status section covering the web_deep_search tool, citation verification (task-16331), and the engine task-16322 plus the task-255 deferred UI decision
2. Fix the capability-matrix Research Search row stale file reference to the deleted UI/Screens/search_screen.py
3. Leave dated plan documents untouched (historical records)
ADR required: no - docs only
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `Docs/Development/research-sessions-runs-tranche-2.md`: struck through the two no-longer-true landed-scope claims (dedicated Research screen, first-class navigation destination — both removed by task-255), moved SSE streaming from Deferred to Landed (the 2026-04-23 plan completed it), annotated the historical verification commands whose targets (`DB/Research_DB.py`, `UI/Screens/research_screen.py`) are deleted with pointers to their replacements, and added a dated 2026-08 Update section recording: the task-255 deferred UI decision (explicitly deferred to task-16322), the `web_deep_search` tool (task-1356), citation verification (task-16331), and the wiring dedupe (task-16332).
- `Docs/Parity/2026-04-21-capability-matrix.md`: removed the deleted `tldw_chatbook/UI/Screens/search_screen.py` from the Research Search / Provider Surfaces row's chatbook file list (the row's own status text already says UX adoption is pending, so no replacement reference was needed).
- Gap-ledger research rows were reviewed and left unchanged: their current-state/gap text (service seams present, UX adoption and full server session semantics pending) is accurate and references only existing files.
- Dated plan documents (`Docs/superpowers/plans/2026-04-23-research-live-events.md` etc.) deliberately untouched — they are historical records of what was planned at the time.
- Docs-only change; no code or tests modified.
<!-- SECTION:NOTES:END -->
