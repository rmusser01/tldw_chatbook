# Gate 1.5 Console Internals Decomposition

Date: 2026-05-07
Task: TASK-10.6
Closeout Task: TASK-10.6.5
Status: verified

## Scope

Gate 1.5 verifies that Console is no longer a shell wrapped around the full legacy `ChatWindowEnhanced` presentation. The verified scope covers Console-native display state, provider/model controls, staged context, transcript/session surface, composer actions, run inspector, RAG/source visibility, approvals, tool readiness, Chatbook artifact actions, and recovery states.

This evidence closes the Console internals decomposition gate only. It does not claim completion for Gate 1.6 Library-native Search/RAG, Workspaces, Collections, broader agent configuration, MCP/ACP runtime redesign, or live provider integration depth.

## Walkthrough

Console mounted in the Textual harness with the native workbench regions:

- `#console-control-bar` exposes provider, model, assistant/persona, RAG, staged source, tool, and approval labels outside the transcript.
- `#console-session-surface` hosts the native transcript/session area without `#chat-window`, `#chat-enhanced-sidebar`, `#toggle-chat-left-sidebar`, or `#chat-main-content`.
- `#console-native-composer` exposes Send, Stop, Attach, and Save Chatbook actions through the existing chat compatibility seams.
- `#console-run-inspector` shows live-work provenance, provider readiness, tool readiness, RAG/source state, artifact availability, approval state, and disabled reasons.

Repeated core-loop fixtures covered a default Console launch, staged Library Search/RAG handoff, blocked provider state, missing RAG/source state, pending approval state, and Chatbook artifact handoff state. Existing chat regressions covered handoffs, shell bar behavior, tab/session state, direct `ChatWindowEnhanced` compatibility, live-work launches, approvals/resume, and Chatbook artifact contracts.

## Functional Result

Console now presents as one native agentic workbench rather than a nested legacy Chat screen. Provider/model state and staged source authority are visible before send. The staged context tray preserves handoff provenance and recovery copy. The native composer keeps send/stop/attach/save controls reachable. The run inspector makes blocked provider and missing RAG/source states explicit and gives disabled reasons for unavailable approval, tool-call, and Chatbook actions.

Existing chat behavior remains covered through compatibility tests instead of relying on the old embedded chrome. The direct legacy `ChatWindowEnhanced` widget remains available for legacy/direct use, but the Console route no longer mounts the full legacy screen.

## Verification

Red evidence-tracking regression:

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_gate15_console_internals_evidence_is_tracked --tb=short`
- Result before closeout docs/task updates: failed with `FileNotFoundError` for `2026-05-07-gate-1-5-console-internals-decomposition.md`, proving the tracking test guarded the missing evidence.

Gate 1.5 focused implementation evidence from completed child tasks:

- `TASK-10.6.1`: pure Console display-state contracts and red mounted legacy-chrome guardrails.
- `TASK-10.6.2`: Console-native control bar and staged-context tray.
- `TASK-10.6.3`: native transcript/session surface and native composer action row.
- `TASK-10.6.4`: native run inspector for approvals, tools, RAG/source state, and Chatbook artifact actions.

Final closeout verification:

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_chat_tab_container.py Tests/UI/test_chat_window_enhanced.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_screen_state.py --tb=short`
- Result: `170 passed, 8 warnings in 70.64s`.
- `git diff --check`
- Result: passed with no output.

## Defects

- Gate 1.5 began with the known Gate 1 residual risk: Console mounted the full legacy `ChatWindowEnhanced` chrome inside the workbench. The child tasks replaced that route composition with Console-owned widgets while preserving compatibility seams.
- PR review feedback during the run-inspector slice identified blocked RAG/source and pending-approval state gaps. Those were addressed by centralizing inspector constants, sharing provider/model source precedence, reusing the non-negative integer seam helper, and consuming pending Artifacts Chatbook targets before fallback.

No P0 or P1 defect remains in the verified Gate 1.5 scope.

## Residual Risk

- Gate 1.6 still owns Library-native Search/RAG with source selection, query input, retrieval status, evidence lists, citations, snippets, and Console handoff evidence.
- Phase 3 still owns Workspaces, Library-owned Collections, deeper Import/Export/Search/RAG study flows, full server job history, and direct generated deck selection.
- Later agent phases still own MCP/ACP runtime depth, controlled agent configuration, schedules/workflows execution depth, and live integration parity.
- `ChatWindowEnhanced` remains in the codebase as a legacy/direct compatibility widget. Gate 1.5 only verifies that the Console route no longer presents it as the embedded product-center screen.

## Exit Decision

Pass for Gate 1.5. Console now has native internals for the product-center workbench and retains focused compatibility coverage for existing chat behavior. Continue Phase 3 with Gate 1.6 Library-native Search/RAG and the remaining Knowledge/Study workflow depth under TASK-10.
