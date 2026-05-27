# Settings Configuration Hub QA Notes

Date: 2026-05-25
Branch: `codex/settings-config-next-slice`
Screen: Settings
Spec: `Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md`
Evidence method: actual textual-web/CDP screenshots driven with Playwright browser automation.

## Approved Screenshots

- Overview: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/01-overview.png`
- Providers & Models: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/02-providers-models.png`
- Appearance: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/03-appearance.png`
- Storage: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/04-storage.png`
- Privacy & Security: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/05-privacy-security.png`
- Console Behavior: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/06-console-behavior.png`
- Diagnostics: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/07-diagnostics.png`
- Advanced Config: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/08-advanced-config.png`
- Focused Settings button final fix: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/11-focused-settings-button-no-heavy-outline.png`
- Category search polish: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/13-category-search-polish.png`
- Diagnostics shortcut validation: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/16-diagnostics-shortcut-cdp.png`
- Diagnostics worker shortcut validation: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/17-diagnostics-worker-cdp.png`

User approval: approved after actual rendered screenshot review.

## Rejected Or Superseded Evidence

- `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/09-focused-button-cdp.png`
- `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/10-clicked-save-button-cdp.png`

These captures showed that one-line Settings buttons inherited the global heavy focus outline, obscuring selected button labels. The final approved capture replaces that state with readable raised/underlined focus styling.

## Workflow Coverage

- Category switching was exercised through keyboard navigation in textual-web.
- Overview and each first-slice category rendered with the three-column Settings workbench.
- Advanced Config rendered guarded raw TOML controls with validation-first save policy.
- Focused Settings buttons no longer render as empty heavy-outline rectangles.
- Category search filters Settings sections, reports match count, distinguishes primary and secondary matches, and opens the top-ranked category on Enter.
- The active search field remains visually identifiable while filtering, and the wider Settings section column keeps filtered category labels readable.
- Scope Inspector preserves the boundary that runtime MCP, ACP, and tool control stay in their own destinations.

## Automated Verification

Commands were run with the project virtual environment active:

```bash
python tldw_chatbook/css/build_css.py
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Result: `58 passed`, one existing requests dependency warning, and `git diff --check` clean.

## Residual Risks

- Settings is now a first-slice configuration hub, not a complete replacement for every legacy config surface.
- MCP, ACP, Skills, Personas, Schedules, and Workflows ownership is intentionally documented as out-of-Settings runtime control.
- Additional persisted controls should be added category-by-category with the same validation, save/revert, and screenshot approval workflow.

## 2026-05-26 Provider Endpoint Follow-Up

Branch: `codex/settings-next-after-merge`

Approved screenshot:

- Provider endpoint control: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/14-provider-endpoint.png`

Coverage added:

- Providers & Models now exposes the provider endpoint/base URL alongside provider, model, streaming, and temperature controls.
- Endpoint saves to the provider-specific `api_settings.<provider>.<endpoint-key>` path while preserving existing endpoint-key conventions.
- Invalid endpoint values are blocked before save unless they are empty, and visible feedback is shown in the provider detail pane.
- Actual textual-web/CDP verification confirmed the endpoint row is visible and readable in the default Settings workbench layout.

Focused verification:

```bash
python tldw_chatbook/css/build_css.py
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Result: `61 passed`, one existing requests dependency warning, and `git diff --check` clean.

## 2026-05-26 Guided Settings Actions Follow-Up

Branch: `codex/settings-next-polish`

Approved screenshot:

- Guided Settings actions: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/15-guided-actions-overview-final.png`

Coverage added:

- Read-only or routed Settings categories now disable the global Save and Revert affordances instead of implying that a category-local save exists.
- Providers & Models and Console Behavior keep Save and Revert disabled until a draft change exists, then enable both controls from the same dirty-state signal.
- Scope Inspector guidance now explains the next valid action for the selected category before the boundary and impact copy.

Focused verification:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config Tests/UI/test_destination_shells.py::test_legacy_tools_settings_route_opens_mcp_not_global_settings --tb=short
git diff --check
```

Result: `66 passed` for the Settings configuration hub suite, `5 passed` for the focused destination shell suite, one existing requests dependency warning in each pytest run, and `git diff --check` clean.

## 2026-05-27 Diagnostics Shortcut Follow-Up

Branch: `codex/settings-next-ux-polish`

Approved screenshot:

- Diagnostics shortcut validation: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/16-diagnostics-shortcut-cdp.png`

Coverage added:

- The Settings `t` shortcut now runs Diagnostics validation and reload when Diagnostics is selected.
- The visible Diagnostics buttons and keyboard shortcut share the same validation/reload paths.
- The resulting status rows show both `Config validation: valid` and `Config reload: loaded` in the actual rendered Settings screen.

Focused verification:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_diagnostics_test_shortcut_runs_validate_and_reload --tb=short
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Result: `1 passed` for the red/green Diagnostics shortcut regression, `67 passed` for the full Settings configuration hub suite, one existing requests dependency warning, and `git diff --check` clean.

## 2026-05-27 Diagnostics Shortcut Review Fix

Branch: `codex/settings-next-ux-polish`

Approved screenshot:

- Diagnostics worker shortcut validation: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/screenshots/17-diagnostics-worker-cdp.png`

Coverage added:

- The Diagnostics `t` shortcut now dispatches validation/reload through an exclusive Textual thread worker instead of running config IO on the UI thread.
- The combined shortcut path validates the config file once, reloads only after a valid result, and applies config mutation/status rows back on the UI thread.
- Invalid config validation skips reload and reports the same validation failure in the reload status row.

Focused verification:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_diagnostics_combined_helper_validates_once Tests/UI/test_settings_configuration_hub.py::test_settings_diagnostics_combined_helper_skips_reload_when_invalid Tests/UI/test_settings_configuration_hub.py::test_settings_diagnostics_test_shortcut_runs_validate_and_reload --tb=short
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Result: `3 passed` for the focused review-fix regressions, `69 passed` for the full Settings configuration hub suite, one existing requests dependency warning, and `git diff --check` clean.
