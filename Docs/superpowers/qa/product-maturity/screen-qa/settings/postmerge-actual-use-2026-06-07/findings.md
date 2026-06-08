# Settings post-merge actual-use sweep - 2026-06-07

## Scope

Actual Settings screen QA was performed through Textual-web/CDP against current `origin/dev` after PR 485 merged. This pass focused on Settings as the application configuration hub, including category selection, field editing, dropdowns, focused input visibility, validation feedback, Save/Revert/Test controls, read-only domain defaults, and top-level navigation behavior.

## Environment

- App served through `tldw-serve` at `http://127.0.0.1:8933`.
- Browser interaction and screenshots were captured through CDP/Textual-web.
- Tests were run with the project virtual environment active.

## Verified Behavior

- Settings renders as a full-width three-column configuration hub with grouped navigation, detail pane, and category-specific inspector.
- Providers & Models selection works: the provider dropdown opens, entries are readable, selecting a provider updates detail state, Save/Revert availability, model, endpoint, credential, and readiness copy.
- Focused text entry remains visible in provider/model fields while typing.
- Appearance dropdowns and text inputs remain visible while focused; invalid font size is blocked with inline status and inspector feedback.
- Console Behavior settings expose streaming, sampling, paste-collapse, background effect, and frame-rate controls; invalid frame rate is blocked with visible validation feedback.
- Library & RAG defaults expose search/citation/snippet settings; invalid snippet length is blocked with visible validation feedback.
- Storage, Privacy & Security, Diagnostics, and Advanced Config action paths execute visibly with status feedback instead of silent/no-op controls.
- Advanced Config validation and backup preview are usable and show state transitions without an observed UI freeze.
- Domain defaults for Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP, and ACP render explicit ownership/read-only guidance instead of implying unsupported edits.

## Evidence Map

- `01-settings-overview.png` - Settings hub layout.
- `02-providers-models.png` through `06-provider-reverted.png` - provider dropdown, selection, focused model input, and revert.
- `07-appearance.png` through `09-appearance-invalid-font.png` - appearance controls and validation.
- `10-console-behavior.png` through `12-console-behavior-invalid-save-blocked.png` - console behavior controls and blocked save.
- `13-library-rag.png` through `17-library-rag-key-revert.png` - Library/RAG defaults, dropdowns, validation, and revert.
- `18-storage.png` through `30-advanced-load-backup-result.png` - utility action paths and raw TOML checks.
- `31-domain-library-rag.png` through `39-domain-acp-defaults.png` - domain defaults coverage.
- `40-top-nav-console-click.png` through `44-top-nav-settings-playwright-text-click.png` - top-level navigation discrepancy probes.

## Corrective Follow-up

### Top navigation visible-border activation

From Settings, a CDP/CUA click on Console activated Console. From Console, repeated CDP/CUA clicks and a Playwright text click on Settings visually focused/underlined Settings but did not activate Settings in the captured browser session.

Follow-up investigation found the normal `Button.Pressed` path works, but the visible bottom row of a navigation tab hit-tests to `MainNavigationBar` rather than the owning `NavigationButton`. That means a user can click a visible tab border/focus area and see focus styling without navigation.

Fix implemented locally: `MainNavigationBar` now routes parent-row clicks that fall within a navigation button's visible region back to that button's navigation action. Regression coverage clicks the visible bottom row of Settings from a cached Console screen.

CDP screenshot re-verification is still pending because starting a fresh patched Textual-web server was blocked by the current approval gate, and the existing `127.0.0.1:8935` browser target was blocked by browser security policy. Automated behavior verification is green.

## Verification

- `python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short`
- Result: 187 passed, 1 warning.
- `python -m pytest -q Tests/UI/test_product_maturity_phase1_navigation_smoke.py::test_top_level_navigation_activates_visible_tab_border_from_cached_console_screen --tb=short`
- Result: 1 passed, 1 warning.
- `python -m pytest -q Tests/UI/test_screen_navigation.py::test_main_navigation_exposes_all_routed_primary_screens Tests/UI/test_screen_navigation.py::test_main_navigation_copy_and_order Tests/UI/test_screen_navigation.py::test_main_navigation_buttons_explain_compact_labels Tests/UI/test_screen_navigation.py::test_main_navigation_route_ids_match_shell_destinations Tests/UI/test_screen_navigation.py::test_screen_navigation_routes_reach_real_app_handler Tests/UI/test_product_maturity_phase1_navigation_smoke.py --tb=short`
- Result: 10 passed, 1 warning.
