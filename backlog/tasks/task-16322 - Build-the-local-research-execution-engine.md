---
id: TASK-16322
title: Build the local research execution engine
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:14'
updated_date: '2026-08-15 06:07'
labels:
  - research
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local research runs are pure DB bookkeeping: launch_run writes rows with phase local_planning and nothing ever plans, searches, or synthesizes. The only real execution pipeline (web_deep_search / generate_and_search + analyze_and_aggregate) is disconnected from the run lifecycle. Port tldw_server dev's Research/ phase machine (drafting_plan then collecting then synthesizing then packaging) in miniature: drive the existing deep-search pipeline for a launched run, stream progress into research_run_events, store report and structured artifacts via save_artifact, and reach completed or failed status through the existing LocalResearchService transitions. This also resolves task-255's deferred decision by making Research_Window reachable again now that something local actually executes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A launched local run executes end to end (plan then collect then synthesize then package) and reaches completed or failed via existing LocalResearchService methods
- [x] #2 Phase transitions and per-step progress are recorded as research_run_events consumable by existing stream endpoints
- [x] #3 Final report and structured artifacts (report markdown plus sources plus verification summary) are stored via save_artifact and retrievable via get_bundle
- [x] #4 The existing deep-search pipeline is reused rather than forked
- [x] #5 Pause and cancel are honored between phases with clean terminal states
- [x] #6 Research_Window and ResearchController are reachable from navigation resolving the task-255 deferred decision
- [x] #7 ADR is created before implementation covering the local engine contract and its relationship to the server run contract
- [x] #8 Tests cover the state machine including failure and cancellation paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write ADR-068 (local research execution engine contract + relationship to server run contract) BEFORE implementation
2. TDD LocalResearchService.update_run_progress: phase and progress updates with events on both storage paths
3. TDD Research_Interop/local_research_engine.py: async UI-agnostic engine driving planning then collecting then synthesizing then packaging by REUSING generate_and_search and analyze_and_aggregate via an injectable runner; events and artifacts (report_v1.md, sources.json, verification_summary.json, plan.json) through LocalResearchService; pause and cancel honored between phases via control_state polling; failures go through fail_run
4. TDD UI/Screens/research_screen.py hosting ResearchWindow and register the research route in screen_registry and route_inventory (resolving the task-255 deferred decision)
5. Wire engine start: launching a local run from ResearchWindow runs the engine in a Textual worker
6. Full research and UI navigation test runs plus lint; close task with notes
ADR required: yes - backlog/decisions/066-local-research-execution-engine.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **ADR-068** (`backlog/decisions/066-local-research-execution-engine.md`) records the contract: a UI-agnostic engine reusing the pipeline via injectable seams, LocalResearchService as single writer, pause/cancel honored between phases, artifact names mirroring the server, and the task-255 reversal. Alternatives (embedding the server engine, extending the tool, deleting the window, forking the pipeline) are recorded with rejection reasons.
- **Engine** (`Research_Interop/local_research_engine.py`): `LocalResearchEngine.execute_run(run_id)` drives planning (10%) → collecting (45%, `search_fn` = `generate_and_search`) → synthesizing (75%, `analyze_fn` = `analyze_and_aggregate`) → packaging (95%) → `complete_run` (100%), with progress anchors matching the server's phase map. Pipeline exceptions route through `fail_run` with partial artifacts preserved (plan/collection survive, no report). Pause between phases leaves the run non-terminal with an `engine_paused` event (restart semantics on resume — phase-level resume is out of scope per the ADR); cancel resolves through `cancel_run` exactly once. Draft runs normalize to running first (`engine_started` event). Artifacts: `plan.json`, `collection_summary.json`, `sources.json`, `verification_summary.json` (carries task-16331's `citation_verification` when the synthesis branch produced one), `report_v1.md` (answer + rendered Sources), `bundle.json`.
- **Service seam**: `LocalResearchService.update_run_progress` added — engine-facing non-terminal transition (phase/progress/message + optional status/control for draft start), version-bumped, event recorded with data; external-DB path mirrors `update_run_state` like the other transitions. Terminal/control transitions stay on their dedicated methods.
- **Navigation**: `UI/Screens/research_screen.py` restored verbatim from history (pre-task-255, commit 7a6129009) — `BaseAppScreen` wrapper hosting `ResearchWindow` with save/restore state. screen_registry registers the real `ScreenRoute` again (replacing the library alias); the workbench owner stays `"library"` (consistent with real screens `study`/`writing` folding under the Library shell destination for nav highlighting), and the route-inventory comment documents that.
- **Window wiring**: `ResearchWindow.create_run` starts the engine after a LOCAL create, and `resume_selected_run` restarts it for non-terminal local runs — via `_start_local_engine`, which runs the engine in a Textual worker (group `research-engine-{run_id}`, exclusive) when mounted, skips gracefully without the app's `local_research_service`, and never blocks the create/resume flow. Headless (unmounted) calls report and do nothing, so no accidental pipeline spend in tests.
- **Verified (TDD, tests watched failing first)**: engine state machine 7 tests (happy path artifacts, draft normalization, pipeline failure, pause, cancel-once, terminal rejection) + service 3 progress tests + 5 window wiring tests; navigation pin updated (research resolves to ResearchScreen). Suites: Tests/Research/ + test_research_screen.py + test_workbench_route_inventory.py = 74 passed; test_screen_navigation.py = 127 passed; command palette / shell destinations / destination headers / master shell navigation = 71 passed individually. A combined single-process run of the navigation modules shows 3-4 failures BOTH with and without this change (verified on stashed baseline) — pre-existing cross-module test isolation flakiness on dev, not introduced here. Ruff clean on all changed files.
- **Known scope (per ADR)**: `autonomy_mode` recorded but not enforced locally (v1 runs autonomously; checkpoint review UI deferred); budgets/limits not yet enforced (task-16323); resume restarts from the top rather than resuming mid-phase.
- Files: new `local_research_engine.py`, `UI/Screens/research_screen.py`, `ADR-068`, `Tests/Research/test_local_research_engine.py`; modified `local_research_service.py`, `screen_registry.py`, `route_inventory.py`, `Research_Window.py`, `test_local_research_service.py`, `test_research_screen.py`, `test_screen_navigation.py`.
<!-- SECTION:NOTES:END -->
