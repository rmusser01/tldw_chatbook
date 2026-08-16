---
id: TASK-16328
title: Expose the academic lane as a Research window toggle
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 16:43'
updated_date: '2026-08-15 17:16'
labels:
  - research
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The engine accepts an optional paper_search_fn (task-16326) but nothing user-facing sets it: the academic lane is dead code without a caller. Add a persisted toggle to the Research window that, when enabled, launches local runs with a paper search function covering arXiv and Semantic Scholar with per-provider degradation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Research window has an academic toggle persisted through save_state and restore_state and defaulted from config,Local runs launched with the toggle on construct the engine with a paper search function while off keeps today's web-only behavior,A shared default paper search function queries both arXiv and Semantic Scholar with DOI dedup and degrades per provider (one provider failing never blocks the other),Partial provider failure is surfaced without failing the run,Tests cover the toggle state the engine wiring and the dual-provider merge with mocked HTTP
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD academic_providers.search_papers: query arXiv and Semantic Scholar together, DOI-dedup across providers, degrade per provider (one failing never blocks the other; total failure raises for the engine lane to warn)
2. TDD window toggle: Checkbox in the Research toolbar defaulting from config (SearchSettings.research_academic_lane, default false), persisted via save_state/restore_state, and _start_local_engine passes paper_search_fn=search_papers only when enabled
3. Tests plus lint plus task close
ADR required: no - UI wiring over existing seams
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `academic_providers.search_papers(query)`: queries arXiv and Semantic Scholar concurrently (each off the event loop via `to_thread`), DOI-dedups across providers (first provider wins the DOI), and degrades per provider — one failing logs a warning and contributes nothing; only total failure raises so the engine lane records a warning, never a run failure. The S2 key resolves through the existing config/env resolver.
- `ResearchWindow` gained a `Checkbox("Academic (arXiv + S2)")` in the toolbar (`#research-academic-toggle`), an `academic_enabled` state persisted through `save_state`/`restore_state`, and a config default via `_academic_lane_default()` (`[SearchSettings] research_academic_lane`, default OFF — the lane costs network calls, so opt-in; config read failures also default OFF). `_start_local_engine` constructs the engine with `paper_search_fn=search_papers` only when the toggle is on; off keeps the web-only behavior exactly.
- Verified TDD: 3 dual-provider tests (mocked httpx: both-providers dedup, single-provider degradation, total failure raise) + 4 window tests (default off + state persistence, restore, config default via monkeypatched seam, engine wiring on/off) — all written first and watched failing; `Tests/UI/test_research_screen.py + Tests/Research/` = 111 passed; ruff clean.
<!-- SECTION:NOTES:END -->
