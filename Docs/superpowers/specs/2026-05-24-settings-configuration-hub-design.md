# Settings Configuration Hub Design

Date: 2026-05-24
Status: Draft for review
Target branch: `dev`

## Purpose

Redesign the Settings destination so it becomes the main configuration point for the entire application.

Settings should let users inspect, edit, validate, save, and recover global app preferences from one coherent terminal workbench. It must not absorb operational controls that belong in destination-specific screens such as MCP, ACP, Skills, Personas, Schedules, or Workflows.

## Current Code Findings

The current top-level Settings screen is implemented in `tldw_chatbook/UI/Screens/settings_screen.py`.

Observed state:

- The screen is a mostly static three-column destination shell.
- The only persisted setting currently exposed is `console.collapse_large_pastes`.
- The category list is not interactive.
- `Sync Safety` is hardcoded as the active section.
- The detail pane mixes sync-safety copy, Console behavior, and placeholder global-settings text.
- Tests in `Tests/UI/test_destination_shells.py` mostly lock the shell layout and the one large-paste toggle.

Existing reusable pieces:

- `save_setting_to_cli_config()` and `load_cli_config_and_ensure_existence()` in `tldw_chatbook/config.py` provide config persistence.
- `Tools_Settings_Window.py` contains many older real settings forms: raw TOML, general config, API providers, database paths, RAG, chat, character, notes, TTS, embeddings, encryption, backup, vacuum, and restore.
- `EnhancedSettingsSidebar` and older settings sidebars contain chat-specific controls but are not suitable as a global Settings foundation.

Routing constraints:

- `settings` routes to `SettingsScreen`.
- Legacy `tools_settings` intentionally resolves to MCP, not global Settings.
- That MCP alias must remain intact.

## Product Role

Settings is the global configuration hub for:

- Providers and models.
- Overview/status.
- Appearance.
- Storage.
- Privacy and security.
- Console behavior.
- Diagnostics.
- Advanced/raw config access.

Settings may define global defaults for other modules, but destination-specific operations remain owned by their destination.

Examples:

- MCP server management stays in MCP.
- ACP runtime/session management stays in ACP.
- Skills import/validation/attachment stays in Skills.
- Persona library/editing stays in Personas.
- Schedule execution controls stay in Schedules.
- Workflow authoring/execution stays in Workflows.

Non-goals:

- Do not rebuild MCP, ACP, Skills, Personas, Schedules, or Workflows inside Settings.
- Do not make raw TOML editing the default path for normal users.
- Do not expose destructive database maintenance as a primary Settings action.
- Do not show full secret values in any guided Settings surface.
- Do not make Settings the place to run agents, execute tools, attach context, or manage live work.

## UX Principles

Follow Nielsen Norman usability principles with emphasis on:

- Visibility of system status: show unsaved changes, validation state, config source, and apply/restart impact.
- Match between system and real world: use product language such as provider, model, storage, privacy, diagnostics, not internal implementation labels.
- User control and freedom: support save, revert, reload, and test without forcing accidental writes.
- Error prevention: validate before save where possible and label destructive or advanced controls clearly.
- Recognition rather than recall: show category navigation and contextual help instead of requiring users to know config key names.
- Flexibility and efficiency: preserve keyboard shortcuts and provide fast search/filter once the category model exists.

## Layout

Use the existing destination-native terminal workbench style.

```text
+--------------------------------------------------------------------------------+
| Home Console Library Artifacts Personas Watchlists Schedules Workflows MCP ... |
+--------------------------------------------------------------------------------+
| Settings | Application configuration hub | Local profile | Unsaved: no          |
| Scope: Global app prefs | Runtime ops stay in MCP / ACP / Workflows            |
+----------------------+--------------------------------------+------------------+
| Categories           | Overview                             | Impact / Status  |
| > Overview           | Provider readiness: ready            | Affects Console  |
|   Providers & Models | Storage: writable                    | Provider: ready  |
|   Appearance         | Privacy: encryption off              | Last test: pass  |
|   Storage            | Console paste collapse: enabled      | Source: config   |
|   Privacy/Security   |                                      | Restart: no      |
|   Console Behavior   | [Open blocked provider] [Diagnostics]| Config path ...  |
|   Diagnostics        |                                      |                  |
|   Advanced Config    |                                      |                  |
+----------------------+--------------------------------------+------------------+
| Help: Save writes to config.toml. Revert restores last loaded values.            |
| Footer: S save | R revert | T test selected                                |
+--------------------------------------------------------------------------------+
```

Required regions:

- Header: destination identity, local/server profile if available, unsaved state.
- Mode strip: scope and ownership boundary.
- Left pane: interactive category list.
- Middle pane: selected category form/detail.
- Right pane: impact, validation, status, and diagnostics.
- Footer/status bar: current shortcuts and compact state.

## Category Model

Settings can use sub-screens when a category becomes too large for a single detail pane. The top-level Settings destination remains the hub, while sub-screens provide focused depth for complex domains.

Sub-screen rules:

- A sub-screen must keep the same destination-native visual language.
- A sub-screen must preserve a clear route back to the Settings category list.
- A sub-screen should be used for complex editors such as provider matrices, encryption setup, raw TOML, or storage diagnostics.
- A sub-screen should not be used for simple toggles or short forms that fit in the main workbench.
- Sub-screen routes must remain explicit and testable; hidden modal-only workflows are not sufficient for core configuration.

### Overview

Purpose: make Settings understandable on arrival and show the user what needs attention.

Overview is the default selected category. It should summarize the most important configuration state without forcing users into a specific setup path.

Initial fields/status cards:

- Provider/model readiness.
- Storage/config writability.
- Privacy/security status.
- Console behavior summary.
- Diagnostics summary.

Actions:

- Open blocked provider setup.
- Open Diagnostics.
- Open Privacy/Security if unencrypted secrets are detected.
- Open Storage if paths are unavailable or unwritable.

Inspector should show:

- Overall app configuration health.
- Current profile/config source.
- Highest-severity blocked or degraded state.
- Next-best settings action.

### Providers & Models

Purpose: make model readiness and default provider setup discoverable and testable.

Provider/model values must use the same canonical effective provider/model logic as Console readiness. Settings must not write or display values that contradict what Console will use.

Implementation should introduce or reuse a shared resolver for:

1. Explicit draft value in Settings.
2. Current app reactive/UI selection where applicable.
3. `app_config` loaded defaults.
4. `config.toml` defaults.

The Settings inspector should state which source currently wins.

Initial fields:

- Default provider.
- Default model.
- Provider endpoint or base URL when relevant.
- API key source/status, shown safely without exposing secrets.
- Streaming default.
- Temperature/default generation values.

Actions:

- Test provider.
- Save.
- Revert.
- Open provider diagnostics.

Inspector should show:

- Affects Console readiness.
- Affects RAG generation when applicable.
- Last test result.
- Config source.
- Restart/apply impact.

### Appearance

Purpose: global visual preferences without duplicating the full customize surface.

Initial fields/actions:

- Current theme summary.
- Open Appearance/customization surface.
- Palette/default visual preference summary when available.
- Save/revert global defaults if fields are exposed inline.

Inspector should show:

- Visual changes apply immediately or after restart.
- Config source.
- Whether unsaved changes exist.

### Storage

Purpose: show where data and config live, and provide safe recovery paths.

Initial fields/actions:

- Config path.
- Database paths summary.
- Backup path summary.
- Open/copy path actions where safe.
- Backup status where available.

Do not expose destructive database operations as default actions.

Inspector should show:

- Writable/readable state.
- Last backup or integrity status when available.
- Whether a restart is required.

### Privacy / Security

Purpose: make sensitive-data handling visible and recoverable.

Initial fields/actions:

- Config encryption status.
- API key storage mode/status.
- Network/local-only policy summary if present.
- Open encryption setup/change path.

Inspector should show:

- Sensitive values are never printed in full.
- Whether unencrypted secrets are detected.
- Recovery guidance for blocked encryption or key access.

### Console Behavior

Purpose: house app-level Console behavior defaults.

Initial fields:

- Collapse large pasted chunks over 50 characters.
- Future Console input/display defaults.
- Future default chat/session behavior that is global, not per-chat state.

Actions:

- Save.
- Revert.

Inspector should show:

- Affects new and current Console composition behavior.
- Source of value.
- Apply impact.

### Diagnostics

Purpose: explain current config state and provide non-destructive checks.

Initial fields/actions:

- Reload config.
- Validate config.
- Show config path.
- Provider readiness summary.
- Database status summary.
- Recent settings save/test error summary.

Inspector should show:

- Current app profile.
- Config load status.
- Last validation result.
- Recovery suggestions.

### Advanced Config

Purpose: preserve expert access without making raw TOML the default experience.

Initial behavior:

- Show explicit warning that raw TOML editing bypasses guided validation.
- Provide raw TOML editor only after entering Advanced Config.
- Provide save/reload/validate actions with clear error feedback.
- Validate TOML before writing.
- Save atomically.
- Create or preserve a recoverable backup before overwriting existing config.
- Redact secret values from validation errors, logs, and notifications.
- Refuse to save if parsing succeeds but the top-level structure is not a mapping.

This can reuse selected logic from `Tools_Settings_Window.py`, but should not mount the full old window into Settings.

## Interaction Model

Category selection:

- Mouse click and keyboard navigation select a category.
- The active category is visibly highlighted.
- The middle and right panes update together.

Editing:

- Field edits mark the screen as dirty.
- Dirty state appears in the header and/or footer.
- Save is per selected category by default.
- A future global `Save all` action may appear only after multi-category draft state is implemented.
- Revert restores last loaded values for the selected category.
- Switching categories with unsaved changes must preserve that category draft and show an unsaved marker in the category list.
- Reload discards staged changes only after confirmation if unsaved changes exist.
- Navigating away from Settings with unsaved changes must warn or preserve drafts, not silently discard edits.

Validation:

- Validate on save for typed fields.
- Testable settings expose `Test` actions.
- Failed validation should identify the field and provide recovery copy.
- Secret values must not appear in logs, notifications, screenshots, or error text.

Keyboard:

- Tab moves between category list, form pane, inspector/actions, and footer actions.
- `S` saves when valid.
- `R` reverts.
- `T` tests the selected category if supported.
- `/` is reserved for settings search, but should not appear in the footer until search is implemented.

## Data Model Direction

Do not let `SettingsScreen` grow into another monolithic UI file.

Introduce small units during implementation:

- A settings category definition model.
- A setting row/view model.
- A draft/staged settings state model.
- Category-specific panel builders.
- A persistence adapter over `load_cli_config_and_ensure_existence()` and `save_setting_to_cli_config()`.

Each setting row should carry:

- Label.
- Config section/key or computed source.
- Control type.
- Current value.
- Draft value.
- Help text.
- Validation state.
- Source of value.
- Apply/restart impact.
- Secret/sensitive handling policy.

## Migration Strategy

Phase 1 should not rewrite all settings at once.

Recommended first implementation slice:

1. Make the category list interactive.
2. Add `Overview` as the default category with provider, storage, privacy, Console behavior, and diagnostics summaries.
3. Implement `Providers & Models` with testable read/write defaults using shared effective provider/model resolution.
4. Keep existing `Console Behavior` large-paste toggle, but move it into the category system.
5. Implement `Appearance` as a clean route/action to the existing customization surface.
6. Implement `Storage` as a read-only first-slice status category showing config/database path availability.
7. Implement `Privacy/Security` as a read-only first-slice status category showing encryption and secret-storage state.
8. Implement `Diagnostics` with config path, reload, and validate.
9. Preserve current sync-safety copy as a status card in Diagnostics or Privacy/Security, not as the only active section.

Later slices:

- Storage safe backup/integrity checks and editable path defaults.
- Privacy/security encryption setup and recovery workflows.
- Advanced raw TOML editor.
- Settings search.
- Full provider coverage and per-provider advanced fields.

## Testing Requirements

Automated tests:

- Settings mounts with the three-column workbench.
- Overview is the default selected category.
- Category selection changes visible detail and inspector content.
- Console large-paste setting persists through the category system.
- Provider default edits stage, save, and revert correctly.
- Provider values use the same effective provider/model source as Console readiness.
- Provider test reports success/failure without exposing secrets.
- Appearance action still routes to `customize`.
- `tools_settings` still resolves to MCP, not Settings.
- Dirty state appears after edit and clears after save/revert.
- Switching categories preserves dirty category drafts and displays an unsaved marker.
- Validation errors show field-specific recovery copy.
- Invalid typed provider/model fields block save with recoverable error copy.
- Advanced raw TOML validates before save, writes atomically, and redacts secrets from errors.

Manual/CDP QA:

- Capture actual rendered screenshots before approval.
- Capture screenshots for Overview and each first-slice category before implementation PR completion.
- Verify keyboard focus order.
- Verify mouse category switching.
- Verify save/revert/test flows.
- Verify category switching with unsaved changes.
- Verify long labels and small terminal sizes do not break layout.
- Verify no secret values are visible in screenshots or logs.

## Risks

- Reusing the full legacy `Tools_Settings_Window` would quickly expose many controls but reintroduce confusing IA and mixed ownership.
- Editing raw config by default could make Settings powerful but unsafe for first-time users.
- Provider settings touch app readiness and can easily produce contradictory Console state if not driven by the same effective provider/model logic.
- A monolithic SettingsScreen rewrite would be hard to test and maintain.

## Acceptance Criteria

- The Settings destination is clearly recognizable as the application-wide configuration hub.
- Users can select a category and understand what is configurable there.
- Users can edit, save, revert, and validate at least the first-slice settings.
- The screen clearly communicates ownership boundaries for MCP, ACP, Skills, Personas, Schedules, and Workflows.
- No secrets are exposed in visible UI, logs, notifications, or test assertions.
- The implementation remains compatible with existing route IDs and shell navigation.
- Actual rendered screenshots are captured and approved before implementation PR completion.
