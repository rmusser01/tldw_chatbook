# Inline API-Key Field (Chat Defaults) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-contextual API-Key input directly beneath the Provider dropdown in Settings → General Settings → 💬 Chat Defaults, so a new user can authenticate the provider they picked and start chatting without hunting through the config tab or restarting.

**Architecture:** Two pure, UI-agnostic helper functions in `Chat/provider_readiness.py` map a `ProviderReadiness` (the app's single source of truth for how a provider's key is resolved) into the field's render/persist state. `Tools_Settings_Window.py` renders the field from that state, reloads it when the provider dropdown changes, saves the entered key to `api_settings.<provider>.api_key` (exactly where both live send paths read), and refreshes the live `app.app_config` in place so no restart is needed.

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0 (installed 8.2.7 — `textual.app.AppTest` is NOT available; use `app.run_test()` real-pilot harness), TOML config, pytest / pytest-asyncio.

## Global Constraints

- Provider-name normalization MUST use `provider_config_key()` from `tldw_chatbook/Chat/provider_readiness.py` (lowercase, spaces/dashes → underscores). Do not hand-roll `.lower().replace(...)`.
- Save target section is the nested `api_settings.<provider_key>` — use `save_setting_to_cli_config(f"api_settings.{provider_key}", "api_key", value)`.
- Never pre-fill or persist the placeholder `<API_KEY_HERE>`, an env-provided key, or an encrypted/locked value. Gate with `is_valid_provider_api_key()` and the field-state helpers.
- The new input's id is `general-chat-api-key`; the existing provider select id is `general-chat-provider`.
- Tests run from the project venv: `source .venv/bin/activate` then `pytest`. The `timeout` command is unavailable.
- Follow existing style: Google-style docstrings, early returns, `from textual import on` is already imported in the settings window.

---

### Task 1: Pure field-state helpers in `provider_readiness.py`

Add two UI-agnostic functions that turn a `ProviderReadiness` into the inline field's state, and decide what (if anything) to persist. No Textual imports.

**Files:**
- Modify: `tldw_chatbook/Chat/provider_readiness.py` (append after `get_provider_readiness`, currently ends line 255)
- Test: `Tests/Chat/test_chat_api_key_field.py` (create)

**Interfaces:**
- Consumes: existing `ProviderReadiness` dataclass, `is_valid_provider_api_key(value) -> bool` (already in this module).
- Produces:
  - `ChatApiKeyFieldState` — frozen dataclass with fields `value: str`, `disabled: bool`, `placeholder: str`, `can_persist: bool`.
  - `chat_api_key_field_state(readiness: ProviderReadiness, *, locked: bool) -> ChatApiKeyFieldState`
  - `chat_api_key_value_to_persist(new_value: object, field_state: ChatApiKeyFieldState) -> Optional[str]`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_chat_api_key_field.py`:

```python
"""Unit tests for the inline Chat-Defaults API-key field state helpers."""

import pytest

from tldw_chatbook.Chat.provider_readiness import (
    ProviderReadiness,
    ChatApiKeyFieldState,
    chat_api_key_field_state,
    chat_api_key_value_to_persist,
)


def _readiness(
    *,
    requires_api_key=True,
    ready=False,
    api_key=None,
    api_key_source=None,
    env_var=None,
    provider="OpenAI",
    provider_key="openai",
):
    return ProviderReadiness(
        provider=provider,
        provider_key=provider_key,
        requires_api_key=requires_api_key,
        ready=ready,
        api_key=api_key,
        api_key_source=api_key_source,
        env_var=env_var,
        reason="test",
        recovery=None,
    )


def test_keyless_provider_is_disabled_and_not_persistable():
    state = chat_api_key_field_state(
        _readiness(requires_api_key=False, ready=True, provider="Ollama", provider_key="ollama"),
        locked=False,
    )
    assert state.disabled is True
    assert state.can_persist is False
    assert state.value == ""
    assert "No API key needed" in state.placeholder


def test_locked_config_is_disabled_with_unlock_hint():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="sk-secret", api_key_source="config:api_settings.openai.api_key"),
        locked=True,
    )
    assert state.disabled is True
    assert state.can_persist is False
    assert state.value == ""
    assert "Unlock config" in state.placeholder


def test_config_key_is_prefilled():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="sk-abc123", api_key_source="config:api_settings.openai.api_key"),
        locked=False,
    )
    assert state.disabled is False
    assert state.value == "sk-abc123"
    assert state.can_persist is True


def test_env_key_shows_hint_and_empty_value():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="sk-env", api_key_source="env:OPENAI_API_KEY", env_var="OPENAI_API_KEY"),
        locked=False,
    )
    assert state.value == ""
    assert state.disabled is False
    assert state.can_persist is True
    assert "OPENAI_API_KEY" in state.placeholder


def test_missing_key_is_empty_and_persistable():
    state = chat_api_key_field_state(_readiness(ready=False), locked=False)
    assert state.value == ""
    assert state.disabled is False
    assert state.can_persist is True


def test_persist_skips_when_not_persistable():
    state = ChatApiKeyFieldState(value="", disabled=True, placeholder="", can_persist=False)
    assert chat_api_key_value_to_persist("sk-new", state) is None


def test_persist_skips_blank_and_placeholder():
    state = ChatApiKeyFieldState(value="", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("   ", state) is None
    assert chat_api_key_value_to_persist("<API_KEY_HERE>", state) is None


def test_persist_skips_unchanged_config_value():
    state = ChatApiKeyFieldState(value="sk-abc123", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("sk-abc123", state) is None


def test_persist_returns_stripped_new_value():
    state = ChatApiKeyFieldState(value="sk-old", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("  sk-new  ", state) == "sk-new"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_chat_api_key_field.py -v`
Expected: FAIL — `ImportError: cannot import name 'ChatApiKeyFieldState'`.

- [ ] **Step 3: Implement the helpers**

Append to `tldw_chatbook/Chat/provider_readiness.py` (after line 255). Note `dataclass`, `Optional` are already imported at the top of the file.

```python
@dataclass(frozen=True)
class ChatApiKeyFieldState:
    """Render + persistence state for the inline Chat-Defaults API-key input."""

    value: str          # masked prefill value; "" when nothing should be shown
    disabled: bool      # True for keyless providers or a locked/encrypted config
    placeholder: str    # hint shown when the box is empty
    can_persist: bool   # whether a user-entered value should be written on save


def chat_api_key_field_state(
    readiness: ProviderReadiness,
    *,
    locked: bool,
) -> ChatApiKeyFieldState:
    """Map provider readiness to the inline API-key field's UI/persistence state.

    Args:
        readiness: Resolved readiness for the currently selected provider.
        locked: True when config encryption is enabled but no session password is
            available (stored values are ciphertext and must not be shown/saved).

    Returns:
        The field state to render and the flag for whether a typed value is savable.
    """
    if not readiness.requires_api_key:
        return ChatApiKeyFieldState(
            value="",
            disabled=True,
            placeholder="No API key needed for this provider.",
            can_persist=False,
        )
    if locked:
        return ChatApiKeyFieldState(
            value="",
            disabled=True,
            placeholder="Unlock config to edit keys.",
            can_persist=False,
        )
    source = readiness.api_key_source or ""
    if source.startswith("config:") and readiness.api_key:
        return ChatApiKeyFieldState(
            value=readiness.api_key,
            disabled=False,
            placeholder="Enter API key",
            can_persist=True,
        )
    if source.startswith("env:") and readiness.env_var:
        return ChatApiKeyFieldState(
            value="",
            disabled=False,
            placeholder=f"Detected from ${readiness.env_var} — leave blank to keep it",
            can_persist=True,
        )
    return ChatApiKeyFieldState(
        value="",
        disabled=False,
        placeholder="Enter your API key to start using this provider",
        can_persist=True,
    )


def chat_api_key_value_to_persist(
    new_value: object,
    field_state: ChatApiKeyFieldState,
) -> Optional[str]:
    """Return the API-key value to persist, or None to skip the write.

    Skips when the field is non-persistable, blank, a placeholder, or unchanged
    from the currently displayed value.
    """
    if not field_state.can_persist:
        return None
    candidate = new_value.strip() if isinstance(new_value, str) else ""
    if not is_valid_provider_api_key(candidate):
        return None
    if candidate == field_state.value:
        return None
    return candidate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_chat_api_key_field.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/provider_readiness.py Tests/Chat/test_chat_api_key_field.py
git commit -m "feat(chat): add inline api-key field-state helpers to provider_readiness"
```

---

### Task 2: Render the field + config-lock helper

Insert the API-Key input between the Provider select and the Model input in the Chat-Defaults tab, driven by the Task 1 helpers. Add a `_config_is_locked()` helper.

**Files:**
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py` — imports (line 25-30), `_compose_chat_defaults_settings` (line 692-727), add `_config_is_locked`
- Test: `Tests/UI/test_tools_settings_window.py` (add two tests)

**Interfaces:**
- Consumes: `get_provider_readiness`, `provider_config_key`, `chat_api_key_field_state`, `ChatApiKeyFieldState` (Task 1) from `..Chat.provider_readiness`; `get_encryption_password` (already imported).
- Produces: an `Input#general-chat-api-key` in the Chat-Defaults tab; `ToolsSettingsWindow._config_is_locked(self) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/UI/test_tools_settings_window.py`. These build a tailored config, mount the window, and assert the field renders with the right state. Follow the file's existing `create_dummy_config` / `AppTest` / `MagicMock(spec=App)` patterns.

```python
@pytest.mark.asyncio
async def test_chat_api_key_field_prefilled_for_config_key(monkeypatch, temp_config_path, mock_app_instance):
    create_dummy_config(temp_config_path, {
        "providers": {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {"openai": {"api_key": "sk-configured"}},
    })
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", temp_config_path)
    window = ToolsSettingsWindow(app_instance=mock_app_instance)
    async with AppTest(app=mock_app_instance, driver_class=None) as pilot:
        mock_app_instance.mount(window)
        await pilot.pause()
        field = window.query_one("#general-chat-api-key", Input)
        assert field.password is True
        assert field.value == "sk-configured"
        assert field.disabled is False


@pytest.mark.asyncio
async def test_chat_api_key_field_disabled_for_keyless_provider(monkeypatch, temp_config_path, mock_app_instance):
    create_dummy_config(temp_config_path, {
        "providers": {"Ollama": ["llama3"], "OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "Ollama", "model": "llama3"},
        "api_settings": {},
    })
    monkeypatch.setattr(tldw_chatbook.config, "DEFAULT_CONFIG_PATH", temp_config_path)
    window = ToolsSettingsWindow(app_instance=mock_app_instance)
    async with AppTest(app=mock_app_instance, driver_class=None) as pilot:
        mock_app_instance.mount(window)
        await pilot.pause()
        field = window.query_one("#general-chat-api-key", Input)
        assert field.disabled is True
        assert "No API key needed" in field.placeholder
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_tools_settings_window.py -k chat_api_key_field -v`
Expected: FAIL — `NoMatches`/`QueryError` for `#general-chat-api-key`.

- [ ] **Step 3a: Add imports**

In `tldw_chatbook/UI/Tools_Settings_Window.py`, extend the config import block (line 25-30) to include `load_settings`:

```python
from tldw_chatbook.config import (
    load_cli_config_and_ensure_existence, DEFAULT_CONFIG_PATH, save_setting_to_cli_config, 
    API_MODELS_BY_PROVIDER, check_encryption_needed, get_detected_api_providers,
    enable_config_encryption, disable_config_encryption, change_encryption_password,
    get_encryption_password, get_cli_setting, get_prompts_db_path, load_settings
)
```

Add a new import line after the existing `from ..DB...`/`from .` imports block (near line 40):

```python
from ..Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
    chat_api_key_field_state,
)
```

- [ ] **Step 3b: Add the `_config_is_locked` helper**

Add this method to `ToolsSettingsWindow` (place it just above `_compose_chat_defaults_settings`, line 692):

```python
    def _config_is_locked(self) -> bool:
        """True when encryption is on but no session password is available."""
        encryption_config = self.config_data.get("encryption", {})
        if not encryption_config.get("enabled", False):
            return False
        return not get_encryption_password()
```

- [ ] **Step 3c: Insert the field in `_compose_chat_defaults_settings`**

In `_compose_chat_defaults_settings` (line 692), inside the `settings-form-grid` container, insert the API-Key label + input immediately after the Provider `Select` (line 718) and before the `Label("Model:", ...)` (line 720):

```python
                    yield Select(
                        options=provider_options,
                        value=current_chat_provider,
                        id="general-chat-provider",
                        classes="settings-select",
                        tooltip="Default AI provider for chat conversations"
                    )

                    # Inline API key for the selected provider (new-user quick start)
                    _readiness = get_provider_readiness(current_chat_provider, self.config_data)
                    _key_state = chat_api_key_field_state(_readiness, locked=self._config_is_locked())
                    yield Label("API Key:", classes="settings-label")
                    yield Input(
                        value=_key_state.value,
                        password=True,
                        disabled=_key_state.disabled,
                        id="general-chat-api-key",
                        classes="settings-input",
                        placeholder=_key_state.placeholder,
                        tooltip="API key for the selected provider. Saved to [api_settings.<provider>]."
                    )

                    yield Label("Model:", classes="settings-label")
```

(Leave the existing Model/Temperature widgets untouched below.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_tools_settings_window.py -k chat_api_key_field -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py Tests/UI/test_tools_settings_window.py
git commit -m "feat(settings): render inline api-key field under Chat-Defaults provider picker"
```

---

### Task 3: Reload the field when the provider changes

Add a `Select.Changed` handler scoped to the provider dropdown that recomputes the field state for the newly selected provider.

**Files:**
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py` (add handler method)
- Test: `Tests/UI/test_tools_settings_window.py` (add one test)

**Interfaces:**
- Consumes: `get_provider_readiness`, `chat_api_key_field_state`, `_config_is_locked` (Task 2). `on`, `Select`, `Input`, `QueryError` already imported.
- Produces: `ToolsSettingsWindow._on_chat_provider_changed(self, event: Select.Changed) -> None`.

> **Test harness note (IMPORTANT):** `textual.app.AppTest` does NOT exist in this repo's Textual (8.2.7). Task 2 established the real-pilot helper `mount_settings_window(config_dict, temp_config_path, monkeypatch)` (an `@asynccontextmanager`) plus `_ToolsSettingsHostApp(App)` at the top of `Tests/UI/test_tools_settings_window.py`. Use that helper for all UI tests here — do NOT use `AppTest` or `mock_app_instance` for these tests.

- [ ] **Step 1: Evolve the mount helper to also yield the pilot**

Task 3 must flush a `Select.Changed` message, which needs `pilot.pause()`. Update `mount_settings_window` to yield both the window and the pilot, and update Task 2's two existing call sites accordingly:

```python
# in mount_settings_window(...):  change the final yield
    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(ToolsSettingsWindow)
        yield window, pilot
```

Update the two existing Task 2 tests from `async with mount_settings_window(...) as window:` to `async with mount_settings_window(...) as (window, pilot):` (they ignore `pilot`). Re-run `pytest Tests/UI/test_tools_settings_window.py -k chat_api_key_field -v` and confirm the two still pass before proceeding.

- [ ] **Step 2: Write the failing test**

Add to `Tests/UI/test_tools_settings_window.py`:

```python
@pytest.mark.asyncio
async def test_chat_api_key_field_reloads_on_provider_change(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"], "Ollama": ["llama3"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {"openai": {"api_key": "sk-configured"}},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        field = window.query_one("#general-chat-api-key", Input)
        assert field.value == "sk-configured"

        # Switch to a keyless provider -> field disables and clears
        window.query_one("#general-chat-provider", Select).value = "Ollama"
        await pilot.pause()
        assert field.disabled is True
        assert field.value == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest Tests/UI/test_tools_settings_window.py -k reloads_on_provider_change -v`
Expected: FAIL — field stays enabled with `sk-configured` (no handler yet).

- [ ] **Step 4: Add the handler**

Add to `ToolsSettingsWindow` (place directly after `_compose_chat_defaults_settings`):

```python
    @on(Select.Changed, "#general-chat-provider")
    def _on_chat_provider_changed(self, event: Select.Changed) -> None:
        """Reload the inline API-key field for the newly selected chat provider."""
        try:
            api_key_input = self.query_one("#general-chat-api-key", Input)
        except QueryError:
            return
        provider = event.value
        if not provider or provider is Select.BLANK:
            return
        readiness = get_provider_readiness(str(provider), self.config_data)
        state = chat_api_key_field_state(readiness, locked=self._config_is_locked())
        api_key_input.value = state.value
        api_key_input.disabled = state.disabled
        api_key_input.placeholder = state.placeholder
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest Tests/UI/test_tools_settings_window.py -k reloads_on_provider_change -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py Tests/UI/test_tools_settings_window.py
git commit -m "feat(settings): reload inline api-key field when chat provider changes"
```

---

### Task 4: Save the key + refresh live config (no restart)

Persist the entered key to `api_settings.<provider>.api_key` and update the running `app.app_config` in place so the key is usable immediately. Wire it into the existing "Save Settings" flow.

**Files:**
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py` — add `_save_chat_api_key` + `_refresh_live_api_settings`; call `_save_chat_api_key` from `_save_general_settings` (after the `chat_defaults.provider` save, line 3075)
- Test: `Tests/UI/test_tools_settings_window.py` (add one test)

**Interfaces:**
- Consumes: `chat_api_key_field_state`, `chat_api_key_value_to_persist` (Task 1), `provider_config_key`, `get_provider_readiness`, `save_setting_to_cli_config`, `load_settings`, `load_cli_config_and_ensure_existence`, `_config_is_locked`.
- Produces: `ToolsSettingsWindow._save_chat_api_key(self) -> bool`, `ToolsSettingsWindow._refresh_live_api_settings(self) -> None`.

- [ ] **Step 1: Add the missing import for Task 1's persist helper**

In the `from ..Chat.provider_readiness import (...)` block added in Task 2, add `chat_api_key_value_to_persist`:

```python
from ..Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
    chat_api_key_field_state,
    chat_api_key_value_to_persist,
)
```

- [ ] **Step 2: Write the failing test**

Add to `Tests/UI/test_tools_settings_window.py`:

Use the `mount_settings_window` helper (yields `(window, pilot)`). The window's `app_instance` is the real host app, so set/read `app_config` on `window.app_instance`:

```python
@pytest.mark.asyncio
async def test_chat_api_key_save_writes_config_and_updates_live_config(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        window.app_instance.app_config = {"api_settings": {}}
        window.query_one("#general-chat-api-key", Input).value = "sk-brand-new"

        saved = window._save_chat_api_key()
        assert saved is True

        # Written to the on-disk config under the normalized provider key
        written = toml.load(temp_config_path)
        assert written["api_settings"]["openai"]["api_key"] == "sk-brand-new"

        # Live app config updated in place (no restart needed)
        assert window.app_instance.app_config["api_settings"]["openai"]["api_key"] == "sk-brand-new"


@pytest.mark.asyncio
async def test_chat_api_key_save_skips_blank(monkeypatch, temp_config_path):
    config = {
        "providers": {"OpenAI": ["gpt-4o"]},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4o"},
        "api_settings": {},
    }
    async with mount_settings_window(config, temp_config_path, monkeypatch) as (window, pilot):
        window.app_instance.app_config = {"api_settings": {}}
        window.query_one("#general-chat-api-key", Input).value = "   "
        assert window._save_chat_api_key() is False
        written = toml.load(temp_config_path)
        assert written.get("api_settings", {}).get("openai", {}).get("api_key") is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest Tests/UI/test_tools_settings_window.py -k "chat_api_key_save" -v`
Expected: FAIL — `AttributeError: 'ToolsSettingsWindow' object has no attribute '_save_chat_api_key'`.

- [ ] **Step 4: Implement the save + refresh methods**

Add both methods to `ToolsSettingsWindow` (place after `_on_chat_provider_changed`):

```python
    def _refresh_live_api_settings(self) -> None:
        """Push freshly-saved api_settings into the live app config (no restart).

        Mutates the existing ``app_config`` dict in place so components that hold
        a reference to it — including the Chat send path — observe the new key.
        """
        try:
            reloaded = load_settings(force_reload=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not refresh live api_settings after save: {exc}")
            return
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            app_config["api_settings"] = reloaded.get("api_settings", {})
        # Keep this window's own snapshot consistent.
        self.config_data = load_cli_config_and_ensure_existence(force_reload=True)

    def _save_chat_api_key(self) -> bool:
        """Persist the inline Chat-Defaults API key for the selected provider.

        Returns:
            True if a key was written (and the live config refreshed), else False.
        """
        try:
            provider_value = self.query_one("#general-chat-provider", Select).value
            api_key_widget = self.query_one("#general-chat-api-key", Input)
        except QueryError:
            return False
        if not provider_value or provider_value is Select.BLANK:
            return False
        readiness = get_provider_readiness(str(provider_value), self.config_data)
        field_state = chat_api_key_field_state(readiness, locked=self._config_is_locked())
        key_to_persist = chat_api_key_value_to_persist(api_key_widget.value, field_state)
        if key_to_persist is None:
            return False
        provider_key = provider_config_key(str(provider_value))
        if not save_setting_to_cli_config(f"api_settings.{provider_key}", "api_key", key_to_persist):
            return False
        self._refresh_live_api_settings()
        return True
```

- [ ] **Step 5: Wire it into the Save Settings flow**

In `_save_general_settings`, immediately after the `chat_defaults.provider` save (line 3075-3076):

```python
            # Chat Defaults
            if save_setting_to_cli_config("chat_defaults", "provider", self.query_one("#general-chat-provider", Select).value):
                saved_count += 1
            # Inline API key for the selected provider (contextual quick-start field)
            if self._save_chat_api_key():
                saved_count += 1
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest Tests/UI/test_tools_settings_window.py -k "chat_api_key_save" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py Tests/UI/test_tools_settings_window.py
git commit -m "feat(settings): save inline chat api-key and refresh live config without restart"
```

---

### Task 5: Regression check + reset verification

Confirm the full settings-window suite and provider-readiness suite pass, and that a settings reset does not wipe the key.

**Files:**
- Test only: `Tests/UI/test_tools_settings_window.py`, `Tests/Chat/test_provider_readiness.py`, `Tests/Chat/test_chat_api_key_field.py`

- [ ] **Step 1: Run the affected suites**

Run: `pytest Tests/UI/test_tools_settings_window.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_chat_api_key_field.py -v`
Expected: PASS (pre-existing failures unrelated to this change, if any, are noted in the dev-environment baseline — new tests must all pass).

- [ ] **Step 2: Manually verify reset leaves the key intact**

Read `_reset_general_settings` in `tldw_chatbook/UI/Tools_Settings_Window.py`. Confirm it does not query or clear `#general-chat-api-key`. If it recomposes the tab or sets specific fields, the key re-derives from config (unchanged) — no code change needed. If (and only if) it explicitly clears the field, remove that clearing so credentials are preserved. Record the finding in the task notes.

- [ ] **Step 3: Commit (only if a change was needed in Step 2)**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py
git commit -m "fix(settings): preserve inline chat api-key on general-settings reset"
```

---

## Self-Review

**Spec coverage**
- Field location under provider picker → Task 2 (Step 3c). ✓
- Reads `api_settings.<provider>.api_key` via `provider_config_key` → Global Constraints + Task 4. ✓
- Masked, provider-aware, env-hint, keyless-disable, locked-disable → Task 1 helper + Task 2 render + Task 3 reload. ✓
- Placeholder `<API_KEY_HERE>` / env / encrypted never pre-filled or saved → Task 1 (`is_valid_provider_api_key`, locked branch) + tests. ✓
- Save via existing "Save Settings" button, skip blank/unchanged → Task 4. ✓
- Live `app.app_config` refresh, no restart → Task 4 `_refresh_live_api_settings`. ✓
- Reset preserves key → Task 5. ✓
- Security (redaction, no plaintext render) → satisfied by existing `_setting_value_for_log` + `password=True` (no new code; noted). ✓
- Tests (pure state + widget/integration + regression) → Tasks 1–5. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step shows full content. ✓

**Type consistency:** `ChatApiKeyFieldState(value, disabled, placeholder, can_persist)`, `chat_api_key_field_state(readiness, *, locked)`, `chat_api_key_value_to_persist(new_value, field_state)`, `_config_is_locked()`, `_save_chat_api_key()`, `_refresh_live_api_settings()` — names/signatures match across all tasks and tests. Field id `general-chat-api-key` and provider select id `general-chat-provider` used consistently. ✓
