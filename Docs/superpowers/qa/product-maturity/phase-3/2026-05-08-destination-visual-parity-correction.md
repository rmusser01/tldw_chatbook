# Destination Visual Parity Correction QA

Date: 2026-05-08
Status: verified
Branch under test: `codex/destination-visual-parity-correction`
Implementation commits under test: `6d96082a` through `f4014068`, plus Task 8 closeout changes in this PR
Design spec: `Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md`
Implementation plan: `Docs/superpowers/plans/2026-05-08-destination-visual-parity-correction-implementation.md`

## Scope

This pass verifies visual parity between the approved destination ASCII contracts and the mounted Textual destination shells. It checks that each top-level destination exposes a persistent shell identity, compact global navigation, primary workbench geometry, object/list pane, detail pane, inspector/action pane, and at least one primary or recovery action without falling back to a single vertical explanation stack.

The tested terminal sizes are:

- `140x42`
- `100x32`

The verified destinations are:

- Home
- Console
- Library
- Artifacts
- Personas
- Watchlists
- Schedules
- Workflows
- MCP
- ACP
- Skills
- Settings

## Evidence Artifacts

Per-destination geometry dumps are saved under:

`Docs/superpowers/qa/product-maturity/phase-3/visual-parity/`

Each dump records the global navigation, destination identity, mode/filter strip when present, workbench region, object/list pane, detail pane, inspector/action pane, primary actions, and default state markers.

Representative compact evidence:

- `visual-parity/home-100x32.txt` records `#home-dashboard-grid` at `(x=0, y=9, w=100, h=18)` with attention, active work, and inspector columns visible.
- `visual-parity/chat-100x32.txt` records `#console-workspace-grid` at `(x=0, y=10, w=100, h=22)` and keeps `#console-send-message`, `#console-attach-context`, and `#console-save-chatbook` inside the viewport.
- `visual-parity/library-100x32.txt` records `#library-contract-grid` with source browser, source detail, inspector actions, and recoverable source error state visible.
- `visual-parity/mcp-100x32.txt` records the compact MCP server tree, detail, readiness panes, and disabled run action without overflowing.
- `visual-parity/settings-100x32.txt` records settings category, detail, impact panes, boundary note, and Appearance action inside the viewport.

## State Coverage

The mounted tests and text dumps verify default empty, blocked, loading, unavailable, or recovery states render inside the destination workbench geometry:

- Console: `#console-run-inspector-state`
- Library: `#library-source-empty`, `#library-source-loading`, or `#library-source-error`
- Artifacts: `#artifacts-console-unavailable` and loading state coverage
- Personas: `#personas-empty-state`, `#personas-loading-state`, or `#personas-service-error`
- Watchlists: `#wc-empty-state`, `#wc-loading-state`, or `#wc-service-error`
- Schedules: `#schedules-empty-state`, `#schedules-loading-state`, and `#schedules-console-unavailable`
- Workflows: `#workflows-loading-state` and `#workflows-console-unavailable`
- MCP: `#unified-mcp-content` and `#unified-mcp-status`
- ACP: `#acp-empty-state` and `#acp-console-unavailable`
- Settings: `#settings-boundary-note`

## Regression Found And Fixed

The first compact geometry evidence exposed a Console horizontal overflow: the composer actions could render beyond the right edge at `100x32` even though vertical visibility passed. The compact regression helper now checks horizontal bounds when a viewport width is provided, and the Console composer controls use bounded widths from source TCSS and the widget layer.

## Verification

Focused visual parity suite:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py --tb=short
```

Result: `58 passed, 1 warning`.

Focused destination compatibility suite:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_destination_visual_parity_correction.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_console_live_work_handoffs.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_screen_navigation.py \
  --tb=short
```

Result: `232 passed, 1 warning`.

CSS build:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Result: exited `0`. The pre-existing missing-module warning for `features/_evaluation_v2.tcss` remains unchanged.

Diff hygiene:

```bash
git diff --check
```

Result: passed.

Harness warnings observed during geometry capture:

- `RequestsDependencyWarning` from the local Python environment.
- Expected test-harness app initialization logs for prompts and local database assignment.
- Sandbox-denied attempts to write `/Users/macbook-dev/.config/tldw_cli/ui_state.toml` while capturing Console geometry.

These warnings did not fail the mounted verification and do not change the visual parity result.

## Residual Risks

- This is a visual shell parity correction, not a backend feature-depth pass.
- ACP runtime setup remains a destination-owned setup-needed state, not a fully implemented runtime.
- MCP compact mode is a visual workbench adapter around existing MCP behavior.
- Schedules and Workflows remain layout/recovery shells until sync engine and server-parity execution depth are implemented.
- Workspaces, deeper Import/Export, server sync, collection item membership, citations/snippets carry-through into Chat/artifacts/exported Chatbooks, and deeper Study/Search/RAG flows remain later Phase 3 or later-phase work.
