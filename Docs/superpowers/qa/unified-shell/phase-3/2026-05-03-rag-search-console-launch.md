# Phase 3.8 RAG Search Console Launch

Task: `TASK-3.8`
Branch: `codex/unified-shell-phase3-rag-console-launch`

## Goal

Make Search/RAG results a real Console live-work source instead of only a Chat context handoff. A user who finds a relevant retrieved chunk can keep power-user flow by sending it directly into Console, while blocked server-backed RAG launches still explain how to recover before staging context.

## Implementation Summary

- Added a distinct `Use in Console` action beside existing `Use in Chat` on Search/RAG result cards.
- Reused existing Search/RAG handoff payload construction so Console launch metadata preserves result title, identity, source, score, runtime backend, display summary, and suggested prompt.
- Reused existing RAG runtime policy checks before staging server-backed RAG launches.
- Updated Console live-work source readiness to mark `RAG` as connected.

## Verification

- Red result: focused tests failed because `#use-in-console-0`, RAG connected readiness copy, and this evidence file did not exist.
- Focused green command: `.venv/bin/python -m pytest Tests/UI/test_ux_audit_smoke.py::test_valid_rag_search_console_launch_replays_from_mounted_window_to_app_seam Tests/UI/test_ux_audit_smoke.py::test_contract_blocked_rag_search_console_launch_explains_recovery_without_staging Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_source_readiness_marks_connected_sources_and_future_sources_unavailable Tests/UI/test_console_live_work_handoffs.py::test_console_renders_source_readiness_summary_without_pending_launch Tests/UI/test_console_live_work_handoffs.py::test_phase3_rag_console_launch_tracking_evidence_links_task_and_roadmap -q`
- Focused green result: `5 passed, 1 warning in 8.26s`.
- Broader focused command: `.venv/bin/python -m pytest Tests/UI/test_ux_audit_smoke.py Tests/UI/test_console_live_work_handoffs.py -q --tb=short`
- Broader focused result: `55 passed, 1 warning in 19.50s`.

Warning boundary: the remaining warning is the existing `requests` dependency version warning and is not caused by the Search/RAG Console launch path.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: Search/RAG result list with a retrieved result.
- Visual check: result cards keep `Use in Chat` and add a separate `Use in Console` action, preserving the existing fast handoff path.
- Functional result: selecting `Use in Console` stages `open_console_for_live_work` with a typed RAG launch payload.
- Recovery result: when runtime policy blocks server RAG under a local active source, no Console context is staged and the user sees source-authority recovery copy.

## Residual Risk

- Live embedding/vector search quality and provider availability are outside this shell slice.
- Console still does not execute a full RAG agent loop by itself; this slice stages selected retrieved evidence into the Console live-work surface.
