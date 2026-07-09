# Inline API-Key field under the Chat-Defaults provider picker

**Date:** 2026-07-05
**Status:** Approved (design)
**Area:** Settings → General Settings → 💬 Chat Defaults

## Why

A brand-new user who opens Settings (or is directed there by the Chat readiness
message *"…add api_key under [api_settings.openai]"*) has no obvious place to
enter an API key. The keys today are buried under **Configuration File Settings
→ API Settings**, mixed per-provider with model/temperature/timeout fields. The
Chat-Defaults tab already shows the provider they will use, but offers no way to
authenticate it. The result is a dead end: pick a provider, then hunt through a
second settings area to make it work.

The goal is that the provider the user picks and the key that makes it work sit
together, and saving the key lets them start chatting immediately.

## What (scope)

Add a single **API Key** input directly beneath the existing **Provider**
dropdown (`#general-chat-provider`) in the "🤖 Default Provider & Model" group of
`_compose_chat_defaults_settings` (`Tools_Settings_Window.py`). New field order:

```
Provider:     [ OpenAI            ▼ ]
API Key:      [ ••••••••••••••••••••• ]   ← new
Model:        [ gpt-4o               ]
Temperature:  [ 0.6 ]
```

No new block, no new button, no new config keys, no new provider list. This is a
promoted, provider-contextual shortcut onto the key the app already reads.

Out of scope: the standalone API-Settings config tab (unchanged), the identical
Character-Defaults picker (deferred follow-up), any multi-provider "quick start"
wizard.

## Where the key is read (why this works)

Both live send paths resolve a provider's key from
`api_settings.<normalized-provider>.api_key`, where the provider name is
normalized to lowercase with spaces/dashes → underscores:

- Main send: `chat_events.py:923` → `get_provider_readiness(provider, app.app_config)`
  reads `api_settings.<key>.api_key` first, then the `PROVIDER_API_KEY` env var.
- Continuation: `chat_events.py:4366` reads
  `app.app_config["api_settings"][provider_key]["api_key"]` directly.

Normalization is done by `provider_config_key()` in
`Chat/provider_readiness.py`. `"OpenAI"` → `openai`, `"MistralAI"` → `mistralai`,
matching the `[api_settings.*]` sub-tables. `save_setting_to_cli_config` supports
nested sections (`"api_settings.openai"`).

## Behavior

The field's state is driven by a single call to
`get_provider_readiness(selected_provider, self.config_data)`, so provider logic
is never duplicated:

1. **Masked password `Input`** (`password=True`), id `general-chat-api-key`,
   placed between the Provider `Select` and the Model `Input` in the
   `settings-form-grid`.
2. **Valid config key present** → pre-fill the box (masked) with the stored
   `api_settings.<key>.api_key`, gated by `is_valid_provider_api_key()` so the
   literal placeholder `<API_KEY_HERE>` is shown as empty, not pre-filled.
3. **Key satisfied by env var** (`api_key_source` starts with `env:`) → box empty,
   placeholder hint `Detected from $OPENAI_API_KEY — leave blank to keep it`.
4. **Keyless provider** (`requires_api_key == False`, e.g. Ollama, llama.cpp) →
   box `disabled`, hint `No API key needed for this provider.`
5. **Encryption enabled but locked** (stored value looks encrypted / no session
   password) → box `disabled`, hint `Unlock config to edit keys.`

### Reactivity

`@on(Select.Changed, "#general-chat-provider")` reloads the API-Key box for the
newly selected provider (re-running the state logic above). Scoped by id so it
does not fire for the window's other `Select` widgets. An unsaved typed value is
discarded on provider change (acceptable).

### Save

Within the existing `_save_general_settings` (near the Chat-Defaults block,
`Tools_Settings_Window.py:3074`), after saving `chat_defaults.provider`:

- Read `provider_key = provider_config_key(<selected provider>)`.
- Read the box value. **Skip** the save when: the box is disabled (keyless /
  locked), the value is blank, the value equals the currently stored value, or
  the value fails `is_valid_provider_api_key()`. This prevents clobbering an
  env-based or encrypted setup with an empty/placeholder string.
- Otherwise `save_setting_to_cli_config(f"api_settings.{provider_key}", "api_key", value)`.
- After a successful save, **surgically refresh the live config** so no restart
  is needed: reload settings and replace `app_instance.app_config["api_settings"]`
  in place (mutate the existing dict rather than reassigning `app_config`, since
  other components hold the original reference; the send paths read
  `app.app_config["api_settings"]` fresh each call).

The existing "Save Settings" button (`save-general-settings`) triggers all of the
above — no dedicated save button.

### Reset

`_reset_general_settings` does **not** clear or overwrite the API-Key field
(credentials are preserved on a settings reset).

## Security

- API-key values are already redacted in save logs (`_setting_value_for_log` →
  `<redacted>`).
- The field never renders a secret in plaintext (masked input), and env/locked
  keys are never copied into the config file.
- Encryption-at-rest is unchanged: plaintext entered here is encrypted by the
  existing on-exit mechanism, exactly as the current API-Settings tab behaves.

## Testing

- Unit: state resolution (valid config key → pre-fill; env-satisfied → empty +
  hint; keyless → disabled; placeholder/encrypted → not saved). These can target
  a small helper that maps `ProviderReadiness` → field state, kept pure and
  testable independent of the widget.
- Widget/integration: mount `ToolsSettingsWindow`, assert the API-Key `Input`
  exists under the provider picker, changing the provider reloads it, and saving
  writes to `api_settings.<provider>.api_key`.
- Regression: run `Tests/UI/test_tools_settings_window.py` (no test currently
  pins Chat-Defaults widget order).

## Risks / limitations (accepted for v1)

- Cannot blank out a stored key from this box (skip-on-blank); still removable in
  the full API-Settings tab.
- Provider change discards an unsaved typed key.
- Character-Defaults picker gets no equivalent field yet.
