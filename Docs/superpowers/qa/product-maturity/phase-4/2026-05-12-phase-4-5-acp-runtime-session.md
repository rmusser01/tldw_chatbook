# Phase 4.5 ACP Runtime Session Contract

Date: 2026-05-14
Branch: `codex/phase4-5-acp-runtime-session`
Backlog task: TASK-11.5
Screen: ACP

## Scope

Make ACP runtime/session state explicit without pretending the full ACP runtime exists:

- ACP owns runtime setup and session readiness copy.
- A configured runtime can be shown separately from a live session payload.
- Console follow remains disabled until a real ACP session payload is available.
- A real ACP session payload stages a live-work handoff into Console using the shared Console live-work contract.
- Settings remains limited to global defaults and is not presented as the ACP runtime owner.

## Automated Evidence

- Command: `python -m pytest -q Tests/UI/test_destination_shells.py::test_acp_missing_runtime_explains_acp_owned_setup_recovery Tests/UI/test_destination_shells.py::test_acp_configured_runtime_without_session_disables_console_follow Tests/UI/test_destination_shells.py::test_acp_session_payload_enables_console_follow_live_work_handoff --tb=short`
- Result: `3 passed, 1 warning in 32.14s`

## Screenshot Evidence

- Screenshot: `Docs/superpowers/qa/product-maturity/phase-4/acp-runtime-session-2026-05-14.png`
- User approval: approved
- Notes:
  - The screenshot should be captured from the actual running app or textual-web/CDP surface, not from SVG export or a code layout.
  - The automated fixture proves configured runtime and session-payload states because a real local ACP runtime fixture is not yet available in the app.

## QA Walkthrough Notes

- Missing runtime state remains honestly blocked with `Runtime not configured`, ACP-owned next action copy, disabled launch, and disabled Console follow.
- Configured runtime without a session shows `Runtime configured`, runtime label/version, `Session: none`, and disabled Console follow with the next action to start or resume a session in ACP.
- Configured runtime with a session payload enables `Follow ACP Session in Console` and calls `open_console_for_live_work` with `source="ACP"` and `target_id="local:acp_session:<id>"`.

## Residual Risks

- Full ACP runtime launch is not implemented in this slice; `Launch ACP Agent` remains disabled unless a future ACP runtime contract supplies launch behavior.
- Console can stage ACP live-work payloads, but no ACP server/session process is started by this PR.
- Future ACP runtime launch work should add its own screenshot approval once launch controls become actionable.
