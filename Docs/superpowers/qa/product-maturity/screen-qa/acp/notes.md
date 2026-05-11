# ACP Screenshot QA Notes

Date: 2026-05-10
Branch: `codex/screen-qa-acp`
Backlog task: TASK-14.10
Commit:
Screen: ACP
Viewport: 2050x1240
Launch method: textual-web with isolated HOME/XDG profile and `default_tab = "acp"`
Screenshot method: Playwright Chromium PNG capture from textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/acp/baseline-2026-05-10-acp-rerun.png`
- Defects: ACP rendered as a weak three-pane row without full-height column treatment, explicit column headings, strong visual separators, or ACP-owned runtime setup copy. The first blank capture was rejected as invalid evidence and not committed.

## Interaction Smoke

- Goal: Verify the blocked ACP runtime state gives an honest recovery path instead of pretending agent launch or Console follow is available.
- Steps: Launched ACP as the default destination, inspected the visible runtime blocked state, and verified launch/follow controls remain disabled with target-specific reasons.
- Result: ACP shows runtime missing state, no active sessions, unavailable launch/follow controls, and ACP-owned recovery/setup copy.

## Fixes

- Summary: Reworked ACP into the approved compact three-column destination layout with explicit Agents/Sessions, Runtime Setup, and Compatibility/Actions columns. Runtime setup ownership now stays in ACP instead of routing users to Settings.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/acp/final-2026-05-10-acp-columns.png`
- User approval: approved in chat after actual rendered screenshot review

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_acp_runtime_blocked_state_uses_setup_and_compatibility_columns --tb=short`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_default_states_preserve_workbench_geometry Tests/UI/test_destination_visual_parity_correction.py::test_acp_runtime_blocked_state_uses_setup_and_compatibility_columns Tests/UI/test_destination_shells.py --tb=short`
- Results: CSS build passed; focused ACP regression passed; broader focused destination shell set passed with `84 passed`.

## Residual Risks

- ACP runtime/session execution remains future work; this pass validates the honest blocked runtime destination shell only.
