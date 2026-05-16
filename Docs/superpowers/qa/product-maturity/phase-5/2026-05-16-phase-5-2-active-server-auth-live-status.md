# Phase 5.2 Active Server Auth Live Status QA

Date: 2026-05-16
Branch: `codex/phase5-2-server-auth-live-status`
Backlog: `TASK-12.2`

## Scope

Phase 5.2 verifies that the running app exposes local/server runtime source, active server, auth, and reachability state without changing source authority or requiring credentials for local mode.

## Implementation Evidence

- Home dashboard input now carries runtime source, server label, configured state, reachability, and auth state from `RuntimeSourceState`.
- The Home status summary renders source-honest labels such as `Mode: Local`, `Server: Not configured (local mode)`, `Server: Missing active server`, `Server: Auth required`, `Server: Auth expired`, `Server: Unreachable`, and `Server: Ready`.
- Home groups runtime, server sync, agent readiness, and active work into a dedicated `System Status` pane so the dashboard reads as a command center instead of three unrelated counters.
- Home empty-state action copy now uses direct keyboard affordances (`Enter: Open Library`, `Ctrl+P: Search commands`) instead of requiring users to infer what the selected action does.
- Home dashboard panes use clearly delineated bordered columns with a narrower attention queue and a wider status pane, preserving the later resizable-pane direction.
- Home adapter wiring passes the app-owned runtime policy into the dashboard adapter. No credential storage or server write fallback was added.

## State Matrix

| State | Expected Home status | Verification |
| --- | --- | --- |
| Local mode, no configured active server | `Mode: Local`; `Server: Not configured (local mode)` | Mounted regression and screenshot evidence |
| Server mode, missing active server | `Mode: Server`; `Server: Missing active server` | Mounted regression |
| Server mode, auth required | `Mode: Server`; `Server: Auth required` | Mounted regression |
| Server mode, session invalid | `Mode: Server`; `Server: Auth expired` | Mounted regression |
| Server mode, unreachable | `Mode: Server`; `Server: Unreachable` | Mounted regression |
| Server mode, reachable and authenticated | `Mode: Server`; `Server: Ready` | Mounted regression |

## Visual Evidence

- Actual rendered screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/home/phase-5-2-home-runtime-status-2026-05-16.png`
- Actual rendered polish screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/home/phase-5-2-home-system-status-polish-2026-05-16.png`
- Screenshot capture method: `textual-web` on `127.0.0.1:8827` with `PYTHONPATH` pinned to this worktree and an isolated temporary HOME/XDG profile.
- Screenshot file verification: `PNG image data, 2050 x 1240, 8-bit/color RGB, non-interlaced` for the runtime-status capture; `PNG image data, 2050 x 1240, 8-bit/color RGB, non-interlaced` for the polish capture.
- Visual approval: approved by user on 2026-05-16 after actual rendered screenshot review.

## Verification

```bash
python -m pytest -q Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UX_Interop/test_server_connection_contracts.py Tests/RuntimePolicy/test_server_context_provider.py --tb=short
```

Result: `104 passed, 8 warnings in 56.54s`.

Latest focused result after the UX/HCI polish pass: `105 passed, 8 warnings in 60.63s`.

```bash
git diff --check
```

Result: passed.

## Findings

- No P0/P1 implementation findings remain from focused regression coverage.
- Visual approval is complete for the Home Phase 5.2 polish screenshot.
