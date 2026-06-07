# Settings Screenshot QA Notes

Date: 2026-05-11
Branch: `codex/screen-qa-settings`
Backlog task: TASK-14.12
Commit: PR #306 merge `6baee11e59039340c10e2027567f62922929e968`
Screen: Settings
Viewport: 2050x1240 textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8832` with isolated HOME/XDG config and `default_tab = "settings"`
Screenshot method: Playwright Chromium screenshot of textual-web
Fallback reason: none

## Baseline Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/settings/baseline-2026-05-11-settings.png`
- Defects: Settings rendered as an under-structured work area with no clear column ownership, oversized empty space, weak boundaries, and a visually ambiguous large-paste toggle.

## Interaction Smoke

- Goal: Verify Settings exposes a real app-level preference action and does not route runtime-specific MCP/ACP controls into global Settings.
- Steps: Opened Settings from the running app, inspected the global/console behavior areas, verified the large-paste Console setting is visible and toggleable, and verified the Appearance action routes to the customization surface.
- Result: Smoke path passed in focused mounted tests and the final screenshot shows the toggle as a readable button rather than an unreadable checkbox glyph.

## Fixes

- Summary: Converted Settings to the approved compact shell with a narrow settings-section column and wider preference-detail and scope-inspector columns. Added explicit column dividers, concise header/mode copy, a readable Console large-paste toggle, and a regression that verifies the left column remains narrower than the other two.

## Final Screenshot

- Path: `Docs/superpowers/qa/product-maturity/screen-qa/settings/final-2026-05-11-settings.png`
- User approval: approved in Codex thread after actual screenshot review

## Verification

- Commands:
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config --tb=short`
- Results: CSS build completed; focused Settings tests passed (`4 passed`, one existing requests dependency warning).

## Residual Risks

- Settings remains a destination shell for global preferences. Deeper category navigation and additional persisted preference controls remain later product-depth work, outside this screenshot QA pass.

## Providers And Models Functional QA

Date: 2026-05-30
Branch: `codex/settings-provider-model-profiles`
Backlog task: TASK-73.2
Screen: Settings Providers & Models plus Console inherited defaults
Launch method: `tldw-serve --host 127.0.0.1 --port 8898` with isolated HOME/XDG config and `default_tab = "settings"`
Screenshot method: Playwright Chromium screenshot of textual-web through CDP
Fallback reason: none

### Screenshots

- Settings selected provider+model profile: `Docs/superpowers/qa/product-maturity/screen-qa/settings/2026-05-30-settings-provider-model-profile.png`
- Console inherited profile defaults: `Docs/superpowers/qa/product-maturity/screen-qa/settings/2026-05-30-console-provider-model-profile-defaults.png`

### Verification Scope

- Confirmed Providers & Models exposes provider, model, endpoint, credential env var, and selected model default profile fields.
- Confirmed API key values are represented by env var name only; no token value appears in the screenshots.
- Confirmed Console session summary uses the same selected model profile defaults from config: OpenAI `gpt-4.1`, streaming off, and profile sampling values.
- User approval: approved in Codex thread after actual screenshot review.

## Server, Sync, Workspace, And Handoff Defaults QA

Date: 2026-05-31
Branch: `codex/settings-server-sync-defaults`
Backlog task: TASK-73.5
Screen: Settings Overview
Viewport: 2050x1240 textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8901` with isolated HOME/XDG config and `default_tab = "settings"`
Screenshot method: Playwright Chromium screenshot of textual-web through CDP
Fallback reason: none

### Screenshot

- Settings server/sync/workspace/handoff defaults: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-server-sync-defaults-2026-05-31-large.png`

### Verification Scope

- Confirmed Settings Overview renders the new server/sync/workspace/handoff section as status/default rows, not executable runtime controls.
- Confirmed the screen names the local/server authority as read-only in Settings.
- Confirmed sync dry-run/recovery copy, Library workspace visibility copy, and ACP handoff readiness appear in the same destination-native Settings shell.
- User approval: approved in Codex thread after actual textual-web screenshot review.

## OpenAI-Compatible Model Discovery QA

Date: 2026-06-04
Branch: `codex/openai-compatible-model-discovery-prd`
Backlog task: TASK-78
Screen: Settings Providers & Models
Viewport: 2050x1240 textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8894` with isolated HOME/XDG config and `default_tab = "settings"`
Screenshot method: Python Playwright Chromium screenshot of textual-web through CDP
Fallback reason: npm Playwright CLI wrapper could not be used because restricted network prevented fetching `@playwright/cli`; project venv already had Python Playwright installed.

### Screenshots

- Idle discovery controls: `Docs/superpowers/qa/product-maturity/screen-qa/settings/model-discovery-providers-models-cdp-2026-06-04.png`
- Safe recovery state: `Docs/superpowers/qa/product-maturity/screen-qa/settings/model-discovery-providers-models-recovery-cdp-2026-06-04.png`

### Verification Scope

- Confirmed Providers & Models renders the discover, save selected, clear, status, warning, and discovered-model list controls inside the destination-native Settings shell.
- Confirmed discovery recovery copy remains visible after a safe failed endpoint request and does not expose authorization headers or secret values.
- Confirmed Save selected and Clear remain disabled until discovered runtime models exist.
- User approval: passed by follow-up instruction to continue after actual screenshot review.

### Verification

- `python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_controls_render_for_eligible_provider Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_saves_selected_runtime_models Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_shows_ambiguous_provider_recovery --tb=short`
- Result: `3 passed, 1 warning`.
- `python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short`
- Result: `161 passed, 1 warning`.

## Advanced Config Action Reachability QA

Date: 2026-06-06
Branch: `codex/settings-category-actions-qa`
Backlog task: TASK-77
Screen: Settings Advanced Config
Viewport: textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8123` with isolated HOME/config.
Screenshot method: Playwright browser screenshot of textual-web through CDP.

### Screenshot

- Advanced Config action reachability and narrowed inspector: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-advanced-actions-narrow-inspector-2026-06-06.png`

### Verification Scope

- Confirmed Advanced Config renders `Validate Raw TOML`, `Load Backup`, and `Save Raw TOML` above the raw editor so safety controls remain reachable before a large TOML body.
- Confirmed Settings workbench column weights now favor the center detail pane and reduce the inspector width: sections `3fr`, detail `6fr`, inspector `2fr`.
- User approval: continued after actual textual-web screenshot review.

### Verification

- `python -m pytest -q Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_keeps_safety_actions_before_raw_editor Tests/UI/test_settings_configuration_hub.py::test_settings_advanced_config_shows_raw_editor_and_safety_actions --tb=short`
- Result: `3 passed, 1 warning`.

## Settings Actual-Use Corrective QA Closeout

Date: 2026-06-06
Branch: `codex/settings-actual-use-closeout`
Backlog task: TASK-77
Screen: Settings
Viewport: 2050x1240 textual-web browser capture
Launch method: `tldw-serve --host 127.0.0.1 --port 8910` with isolated HOME/XDG config, `PYTHONPATH` pinned to this worktree, `default_tab = "settings"`, and splash disabled.
Screenshot method: Playwright browser automation against textual-web through CDP.
Fallback reason: initial browser navigation waited on page lifecycle past timeout, but the server returned the page and assets; the existing open page was then driven directly through Playwright after disabling textual-web overlay pointer events.

### Screenshots

- Baseline overview: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-baseline-2026-06-06.png`
- Console Behavior saved edit state: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-console-max-tokens-saved-2026-06-06.png`
- Provider dropdown open: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-provider-dropdown-open-actual-2026-06-06.png`
- Provider selection applied: `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-provider-selection-mistral-2026-06-06.png`

### Verification Scope

- Confirmed actual rendered category navigation works through mouse selection and keyboard slash-search plus Enter.
- Confirmed Providers & Models renders provider, model, endpoint, credential, and selected model-default controls without clipped leading characters.
- Confirmed the provider dropdown opens without clipping and selecting `Mistral` updates the selected provider, marks the category unsaved, enables Save/Revert, and avoids blank-provider persistence.
- Confirmed Console Behavior numeric entry remains visible while typed, footer shortcuts do not leak while the input owns focus, and moving focus then pressing `s` saves with visible no-unsaved and saved-status feedback.
- Confirmed the narrowed inspector provides category-specific guidance for both Providers & Models and Console Behavior.
- User approval: approved in the Codex thread after actual screenshot review.

### Verification

- `python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract --tb=short`
- Result: `177 passed, 1 warning`.
- `file Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-baseline-2026-06-06.png Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-console-max-tokens-saved-2026-06-06.png Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-provider-dropdown-open-actual-2026-06-06.png Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-closeout-provider-selection-mistral-2026-06-06.png`
- Result: all four files are `PNG image data, 2050 x 1240, 8-bit/color RGB, non-interlaced`.
