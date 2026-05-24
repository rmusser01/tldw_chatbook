# Settings Configuration Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Settings destination into the application-wide configuration hub for first-slice global preferences: Overview, Providers & Models, Console Behavior, Appearance, Storage status, Privacy/Security status, and Diagnostics.

**Architecture:** Keep `SettingsScreen` as the routed destination, but move configuration state, provider/model resolution, draft handling, and persistence into small focused modules. The UI composes category-specific panes from those models and saves through a narrow adapter over existing config helpers. Preserve `tools_settings` as an MCP alias.

**Tech Stack:** Python 3.11+, Textual, pytest, TOML config via `tldw_chatbook.config`, source TCSS in `tldw_chatbook/css/components/_agentic_terminal.tcss`, generated CSS via `python tldw_chatbook/css/build_css.py`.

---

## Spec And Constraints

Read first:

- `Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/config.py`
- `Tests/UI/test_destination_shells.py`
- `Tests/UI/test_destination_visual_parity_correction.py`
- `Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md`

Hard constraints:

- Do not mount or wrap the full legacy `Tools_Settings_Window` in Settings.
- Do not break `tools_settings` routing to MCP.
- Do not expose full secret values in UI, logs, notifications, screenshots, or tests.
- Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly. Edit source TCSS, then run `python tldw_chatbook/css/build_css.py`.
- Do not claim a screen is approved without an actual rendered screenshot and user approval.

## File Structure

Create:

- `tldw_chatbook/UI/Screens/provider_model_resolution.py`
  - Shared effective provider/model resolver used by Console and Settings.

- `tldw_chatbook/UI/Screens/settings_config_models.py`
  - Dataclasses/enums for categories, setting rows, validation state, impact summaries, and category drafts.

- `tldw_chatbook/UI/Screens/settings_config_adapter.py`
  - Config load/save/validate/redaction adapter over `load_cli_config_and_ensure_existence()` and `save_setting_to_cli_config()`.

- `Tests/UI/test_settings_configuration_hub.py`
  - Focused mounted and unit regressions for the new Settings hub behavior.

Modify:

- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Replace static category shell with interactive categories and dynamic detail/inspector panes.

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Delegate effective provider/model resolution to the shared helper while preserving current behavior.

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add minimal Settings category/dirty/status styles if existing destination classes are insufficient.

- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate only via `python tldw_chatbook/css/build_css.py` if TCSS changes.

- `Tests/UI/test_destination_shells.py`
  - Update existing Settings contract tests only where the new behavior intentionally replaces old static copy.

- `Tests/UI/test_destination_visual_parity_correction.py`
  - Update visual contract selectors only if IDs/classes change.

## Task 1: Shared Effective Provider/Model Resolver

**Files:**
- Create: `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing resolver tests**

Add tests that encode the current Console precedence and source labeling.

```python
from types import SimpleNamespace

from tldw_chatbook.UI.Screens.provider_model_resolution import resolve_effective_provider_model


def test_effective_provider_model_prefers_console_overrides():
    app = SimpleNamespace(
        chat_api_provider_value="OpenAI",
        chat_api_model_value="gpt-4.1",
        chat_model_value=None,
        app_config={"chat_defaults": {"provider": "llama_cpp", "model": "qwen"}},
    )

    result = resolve_effective_provider_model(
        app,
        console_provider="Anthropic",
        console_model="claude",
    )

    assert result.provider == "Anthropic"
    assert result.model == "claude"
    assert result.provider_source == "console_control"
    assert result.model_source == "console_control"


def test_effective_provider_model_preserves_configured_provider_when_reactive_is_default_openai():
    app = SimpleNamespace(
        chat_api_provider_value="OpenAI",
        chat_api_model_value=None,
        chat_model_value=None,
        app_config={"chat_defaults": {"provider": "llama_cpp", "model": "qwen"}},
    )

    result = resolve_effective_provider_model(app)

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"
    assert result.model == "qwen"


def test_effective_provider_model_prefers_settings_draft_values():
    app = SimpleNamespace(
        chat_api_provider_value="OpenAI",
        chat_api_model_value="gpt-4.1",
        chat_model_value=None,
        app_config={"chat_defaults": {"provider": "llama_cpp", "model": "qwen"}},
    )

    result = resolve_effective_provider_model(
        app,
        settings_provider="Ollama",
        settings_model="llama3.1",
    )

    assert result.provider == "Ollama"
    assert result.model == "llama3.1"
    assert result.provider_source == "settings_draft"
    assert result.model_source == "settings_draft"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_effective_provider_model_prefers_console_overrides Tests/UI/test_settings_configuration_hub.py::test_effective_provider_model_preserves_configured_provider_when_reactive_is_default_openai Tests/UI/test_settings_configuration_hub.py::test_effective_provider_model_prefers_settings_draft_values --tb=short
```

Expected: FAIL because `provider_model_resolution.py` does not exist.

- [ ] **Step 3: Implement resolver**

Create `provider_model_resolution.py` with a small dataclass and helper.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EffectiveProviderModel:
    provider: Any
    model: Any
    provider_source: str
    model_source: str


def _selected_text(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "None"}


def _chat_default(app_instance: Any, key: str) -> Any:
    config = getattr(app_instance, "app_config", {}) or {}
    defaults = config.get("chat_defaults", {})
    return defaults.get(key) if isinstance(defaults, dict) else None


def resolve_effective_provider_model(
    app_instance: Any,
    *,
    console_provider: Any = None,
    console_model: Any = None,
    settings_provider: Any = None,
    settings_model: Any = None,
) -> EffectiveProviderModel:
    configured_provider = _chat_default(app_instance, "provider")
    reactive_provider = getattr(app_instance, "chat_api_provider_value", None)

    if settings_provider is not None:
        provider = settings_provider
        provider_source = "settings_draft"
    elif console_provider is not None:
        provider = console_provider
        provider_source = "console_control"
    elif (
        _selected_text(configured_provider)
        and str(reactive_provider or "").strip() == "OpenAI"
        and str(configured_provider).strip() != "OpenAI"
    ):
        provider = configured_provider
        provider_source = "chat_defaults"
    elif reactive_provider is not None:
        provider = reactive_provider
        provider_source = "app_reactive"
    else:
        provider = configured_provider
        provider_source = "chat_defaults"

    reactive_model = (
        getattr(app_instance, "chat_api_model_value", None)
        or getattr(app_instance, "chat_model_value", None)
    )
    configured_model = _chat_default(app_instance, "model")

    if settings_model is not None:
        model = settings_model
        model_source = "settings_draft"
    elif console_model is not None:
        model = console_model
        model_source = "console_control"
    elif reactive_model is not None:
        model = reactive_model
        model_source = "app_reactive"
    else:
        model = configured_model
        model_source = "chat_defaults"

    return EffectiveProviderModel(provider, model, provider_source, model_source)
```

- [ ] **Step 4: Update `ChatScreen` to delegate**

Modify `ChatScreen._effective_console_provider_model()` so it calls the shared helper and returns `(provider, model)` exactly as before.

```python
from .provider_model_resolution import resolve_effective_provider_model
```

```python
def _effective_console_provider_model(self) -> tuple[Any, Any]:
    result = resolve_effective_provider_model(
        self.app_instance,
        console_provider=self._console_control_provider,
        console_model=self._console_control_model,
    )
    return result.provider, result.model
```

- [ ] **Step 5: Run focused resolver and Console tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_chat_destination_uses_console_workbench_contract --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_settings_configuration_hub.py
git commit -m "Add shared provider model resolution"
```

## Task 2: Settings Config Models And Adapter

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_config_models.py`
- Create: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing model/adapter tests**

Add unit tests for category identity, draft dirty state, redaction, and safe config validation.

```python
from tldw_chatbook.UI.Screens.settings_config_adapter import (
    SettingsConfigAdapter,
    redact_secret_text,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId, SettingsDraft


def test_settings_draft_tracks_dirty_values():
    draft = SettingsDraft(category=SettingsCategoryId.CONSOLE_BEHAVIOR)
    draft.set_value("collapse_large_pastes", True, False)

    assert draft.is_dirty
    assert draft.dirty_keys == {"collapse_large_pastes"}


def test_redact_secret_text_removes_api_key_like_values():
    text = "failed with OPENAI_API_KEY=sk-secret-token and token abc"

    redacted = redact_secret_text(text)

    assert "sk-secret-token" not in redacted
    assert "OPENAI_API_KEY=<redacted>" in redacted


def test_adapter_rejects_non_mapping_toml():
    adapter = SettingsConfigAdapter()

    result = adapter.validate_raw_toml('"not a mapping"')

    assert not result.valid
    assert "top-level TOML value must be a table" in result.message
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_draft_tracks_dirty_values Tests/UI/test_settings_configuration_hub.py::test_redact_secret_text_removes_api_key_like_values Tests/UI/test_settings_configuration_hub.py::test_adapter_rejects_non_mapping_toml --tb=short
```

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement `settings_config_models.py`**

Implement minimal enums/dataclasses:

- `SettingsCategoryId`
- `SettingsValidationState`
- `SettingsValidationResult`
- `SettingsDraft`
- `SettingsCategorySummary`
- `SettingsImpactSummary`

Keep the module free of Textual imports.

- [ ] **Step 4: Implement `settings_config_adapter.py`**

Implement:

- `SettingsConfigAdapter.load()`
- `SettingsConfigAdapter.save_values(section: str, values: Mapping[str, Any])`
- `SettingsConfigAdapter.validate_raw_toml(text: str)`
- `redact_secret_text(text: str)`

Use existing config helpers where possible. Do not write raw TOML save yet unless Advanced Config is implemented in the same task.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: PASS for model/adapter tests and resolver tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_config_adapter.py Tests/UI/test_settings_configuration_hub.py
git commit -m "Add Settings configuration models"
```

## Task 3: Interactive Categories And Overview

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write failing mounted tests**

Add tests for default Overview and category switching.

```python
@pytest.mark.asyncio
async def test_settings_defaults_to_overview_category():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Overview" in text
        assert "Provider readiness" in text
        assert "Storage" in text
        assert "Privacy" in text
        assert "Console paste collapse" in text


@pytest.mark.asyncio
async def test_settings_category_selection_updates_detail_and_inspector():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

    assert "Console Behavior" in text
    assert "Collapse large pasted chunks" in text
    assert "Affects Console" in text


@pytest.mark.asyncio
async def test_settings_tab_focus_and_enter_select_categories():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        screen = _active_destination_screen(host)

        assert "Providers & Models" in _visible_text(screen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_defaults_to_overview_category Tests/UI/test_settings_configuration_hub.py::test_settings_category_selection_updates_detail_and_inspector --tb=short
```

Expected: FAIL because categories are static.

- [ ] **Step 3: Refactor `SettingsScreen` state**

Add:

- `active_category: reactive[str]`
- category metadata list in one method.
- `compose_content()` that renders category buttons and calls detail/inspector render helpers.
- `@on(Button.Pressed, ".settings-category-button")` handler.
- keyboard navigation that lets users Tab into the category list, arrow between categories, and press Enter to select.

Use stable IDs:

- `settings-category-overview`
- `settings-category-providers-models`
- `settings-category-appearance`
- `settings-category-storage`
- `settings-category-privacy-security`
- `settings-category-console-behavior`
- `settings-category-diagnostics`
- `settings-category-advanced-config`

- [ ] **Step 4: Implement Overview detail and inspector**

Overview should show first-slice summaries using safe fallback values:

- Provider readiness from shared resolver.
- Storage/config writable status.
- Privacy/encryption status summary.
- Console large-paste setting summary.
- Diagnostics summary.

- [ ] **Step 5: Update existing Settings contract test**

Update `test_settings_destination_uses_three_column_workbench_contract` so it expects the new Overview identity instead of hardcoded Sync Safety as the only active section.

Keep assertions for:

- three-column workbench.
- category pane narrower than detail/impact.
- dividers.
- boundary copy.

- [ ] **Step 6: Run focused mounted tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py
git commit -m "Make Settings categories interactive"
```

## Task 4: Console Behavior Category With Per-Category Save/Revert

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write failing save/revert tests**

Add mounted tests for draft state and persistence.

```python
@pytest.mark.asyncio
async def test_settings_console_behavior_stages_save_and_revert(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    saved = []

    monkeypatch.setattr(
        settings_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
        raising=False,
    )

    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        screen = _active_destination_screen(host)
        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-save-category")

    assert saved == [("console", "collapse_large_pastes", False)]
    assert app.app_config["console"]["collapse_large_pastes"] is False
```

Add a second test that toggles, clicks `#settings-revert-category`, and asserts no save occurs.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_stages_save_and_revert --tb=short
```

Expected: FAIL because save/revert action IDs do not exist.

- [ ] **Step 3: Implement selected-category draft state**

Add a category draft map keyed by category ID.

Rules:

- A control edit updates draft state only.
- Category labels show an unsaved marker when draft differs from loaded value.
- Switching categories preserves that category's unsaved draft instead of discarding it.
- `Save` persists only the selected category.
- `Revert` clears only the selected category draft.
- `S` invokes selected-category save, `R` invokes selected-category revert, and `T` invokes the selected category test action when available.

- [ ] **Step 4: Move existing large-paste toggle into the category system**

Preserve existing config behavior:

- Section: `console`
- Key: `collapse_large_pastes`
- Default: `True`

Keep existing button ID for compatibility:

- `settings-console-collapse-large-pastes-toggle`

Add action IDs:

- `settings-save-category`
- `settings-revert-category`

Add Textual actions/bindings:

- `action_settings_save_category`
- `action_settings_revert_category`
- `action_settings_test_category`

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config --tb=short
```

Expected: PASS. Update the old destination-shell toggle test only if it needs to click the Console Behavior category first.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_config_adapter.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py
git commit -m "Add Settings category draft save flow"
```

## Task 5: Providers & Models Category

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Modify: `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing provider category tests**

Add tests for visible fields, source labeling, safe save, and test diagnostics.

```python
@pytest.mark.asyncio
async def test_settings_provider_category_uses_effective_console_source():
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "llama_cpp" in text
        assert "qwen" in text
        assert "Source: chat_defaults" in text


@pytest.mark.asyncio
async def test_settings_provider_test_redacts_secrets(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)

        assert "sk-" not in _visible_text(screen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_provider_category_uses_effective_console_source Tests/UI/test_settings_configuration_hub.py::test_settings_provider_test_redacts_secrets --tb=short
```

Expected: FAIL because the Providers & Models detail is not implemented.

- [ ] **Step 3: Implement provider/model form**

First-slice fields:

- Provider text/select field, depending on available provider data.
- Model text/select field, depending on available provider data.
- Streaming default.
- Temperature default.
- Endpoint/base URL summary if present in provider config.
- API key source/status only, never full value.

Stable IDs:

- `settings-provider-value`
- `settings-model-value`
- `settings-streaming-default`
- `settings-temperature-default`
- `settings-provider-endpoint`
- `settings-provider-key-status`
- `settings-test-provider`

- [ ] **Step 4: Implement provider readiness test**

First slice should be deterministic and non-network by default:

- Validate provider exists.
- Validate model is non-empty when provider requires one.
- Validate endpoint/base URL is present for local providers that need one.
- Validate API key env var/key status for providers that need one.

Do not make real network calls in unit/mounted tests.

- [ ] **Step 5: Implement provider save/revert**

Persist to `chat_defaults` for global defaults unless a more canonical existing key is identified during implementation.

If saving to `chat_defaults`, use:

- `save_setting_to_cli_config("chat_defaults", "provider", value)`
- `save_setting_to_cli_config("chat_defaults", "model", value)`
- `save_setting_to_cli_config("chat_defaults", "streaming", value)`
- `save_setting_to_cli_config("chat_defaults", "temperature", value)`

Also update `app_instance.app_config["chat_defaults"]` after successful save so Console reflects the change in-session.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_config_adapter.py tldw_chatbook/UI/Screens/provider_model_resolution.py Tests/UI/test_settings_configuration_hub.py
git commit -m "Add Settings provider model category"
```

## Task 6: Appearance, Storage, Privacy/Security, Diagnostics, And Advanced Config Categories

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write failing category coverage tests**

Add tests that each first-slice category shows useful, non-placeholder content.

```python
@pytest.mark.parametrize(
    ("button_id", "expected"),
    [
        ("#settings-category-appearance", "Open Appearance"),
        ("#settings-category-storage", "Config path"),
        ("#settings-category-privacy-security", "Encryption"),
        ("#settings-category-diagnostics", "Validate config"),
        ("#settings-category-advanced-config", "Raw TOML"),
    ],
)
@pytest.mark.asyncio
async def test_settings_first_slice_categories_have_real_content(button_id, expected):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click(button_id)
        screen = _active_destination_screen(host)

        assert expected in _visible_text(screen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_first_slice_categories_have_real_content --tb=short
```

Expected: FAIL because these categories do not yet render real content.

- [ ] **Step 3: Implement Appearance**

Requirements:

- Show current theme/config source if available.
- Keep `settings-open-appearance` action and route to `customize`.
- Do not duplicate the full customization surface in first slice.

- [ ] **Step 4: Implement Storage status**

Requirements:

- Show config path.
- Show known database paths when available.
- Show readable/writable status if it can be checked safely.
- Do not expose destructive backup/vacuum/restore actions yet.

- [ ] **Step 5: Implement Privacy/Security status**

Requirements:

- Show encryption enabled/disabled.
- Show unencrypted secret detection status if available through existing config helpers.
- Redact any secret-like values.
- Show next action copy, but do not implement encryption setup unless explicitly scoped in a later task.

- [ ] **Step 6: Implement Diagnostics**

Requirements:

- Show config path.
- Add `settings-validate-config`.
- Add `settings-reload-config`.
- Show validation/reload result in the inspector.
- Preserve sync-safety state here or in Privacy/Security.

- [ ] **Step 7: Implement Advanced Config**

Requirements:

- Show explicit warning copy that raw TOML bypasses guided validation.
- Provide a raw TOML editor only inside the Advanced Config category.
- Add `settings-advanced-validate-config`.
- Add `settings-advanced-save-config`.
- Validate TOML before save and refuse non-mapping top-level data.
- Redact secret-like values from validation/save errors.
- Write atomically and create a recoverable backup before replacing an existing config file.

- [ ] **Step 8: Add Advanced Config safety tests**

Add mounted tests that:

- selecting Advanced Config shows warning copy and the raw editor.
- invalid TOML blocks save with recoverable error copy.
- non-mapping TOML blocks save.
- secret-like values are redacted from visible validation errors.

- [ ] **Step 9: Update appearance route test**

Keep or update `test_settings_appearance_action_routes_to_customize_surface` so it selects Appearance before clicking `settings-open-appearance` if needed.

- [ ] **Step 10: Run focused tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface Tests/UI/test_destination_shells.py::test_settings_sync_safety_state_failure_logs_context --tb=short
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_config_adapter.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py
git commit -m "Add Settings status diagnostics and advanced config"
```

## Task 7: Styling And Visual Contract Updates

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_destination_visual_parity_correction.py`
- Test: `Tests/UI/test_master_shell_design_system_contract.py`

- [ ] **Step 1: Write or update visual contract tests**

Update existing tests only for intentional selector/content changes.

Required selectors:

- `#settings-workbench`
- `#settings-category-pane`
- `#settings-detail-pane`
- `#settings-impact-pane`
- `#settings-category-overview`
- `#settings-category-advanced-config`
- `#settings-save-category`
- `#settings-revert-category`
- `#settings-advanced-validate-config`
- `#settings-advanced-save-config`

- [ ] **Step 2: Run visual contract tests to verify current failures**

Run:

```bash
python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_default_states_preserve_workbench_geometry --tb=short
```

Expected: FAIL only if selectors/content changed and tests need update.

- [ ] **Step 3: Add minimal source TCSS**

Only add styles that are necessary for:

- active category highlight.
- dirty category marker.
- validation/status rows.
- action row layout.

Do not create broad new visual language. Preserve the terminal grid style.

- [ ] **Step 4: Rebuild CSS**

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: exits 0. If the known missing `features/_evaluation_v2.tcss` warning appears and exits 0, record it in implementation notes.

- [ ] **Step 5: Run focused style/contract tests**

Run:

```bash
python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_default_states_preserve_workbench_geometry --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_destination_visual_parity_correction.py
git commit -m "Polish Settings hub visual contract"
```

## Task 8: CDP Screenshot QA And Documentation

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/settings-configuration-hub/2026-05-24-settings-configuration-hub.md`
- Modify if needed: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify if needed: relevant `backlog/tasks/task-*.md`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract Tests/UI/test_destination_shells.py::test_settings_appearance_action_routes_to_customize_surface Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config Tests/UI/test_destination_shells.py::test_legacy_tools_settings_route_opens_mcp_not_global_settings --tb=short
```

Expected: PASS.

- [ ] **Step 2: Launch Textual Web/CDP using documented process**

Follow:

```text
Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md
```

Use a per-run HOME/XDG profile and route to Settings by default.

- [ ] **Step 3: Capture actual screenshots**

Capture and save screenshots for:

- Overview.
- Providers & Models.
- Console Behavior.
- Appearance.
- Storage.
- Privacy/Security.
- Diagnostics.
- Advanced Config.

Do not use generated SVGs or code mockups as approval evidence.

- [ ] **Step 4: Request user approval**

Present screenshot paths and wait for approval.

If the user rejects a screen:

- Record the issue.
- Fix the screen.
- Re-run focused tests.
- Re-capture screenshot.
- Ask again.

- [ ] **Step 5: Write QA evidence**

Document:

- Automated commands run.
- CDP environment.
- Screenshot paths.
- User approval status.
- Known residual risks.

- [ ] **Step 6: Commit QA/docs**

```bash
git add Docs/superpowers/qa/product-maturity/settings-configuration-hub/2026-05-24-settings-configuration-hub.md Docs/superpowers/trackers/product-maturity-roadmap.md backlog/tasks
git commit -m "Document Settings hub QA evidence"
```

Only include tracker/backlog paths if actually modified.

## Task 9: Final Verification And PR Prep

**Files:**
- No expected source changes unless verification finds issues.

- [ ] **Step 1: Run full focused verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_destinations_use_pane_layouts Tests/UI/test_destination_visual_parity_correction.py::test_runtime_and_settings_default_states_preserve_workbench_geometry Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected:

- `git diff --check` has no output.
- `git status --short` shows only intentional changes before final commit, or clean after final commit.

- [ ] **Step 3: Self-review**

Review:

```bash
git diff origin/dev...HEAD --stat
git diff origin/dev...HEAD -- tldw_chatbook/UI/Screens/settings_screen.py
git diff origin/dev...HEAD -- Tests/UI/test_settings_configuration_hub.py
```

Check specifically:

- No full secret values are rendered.
- `tools_settings` still maps to MCP.
- Settings does not mount `Tools_Settings_Window`.
- Generated CSS was not edited manually.
- Category IDs are stable and testable.

- [ ] **Step 4: Final commit if needed**

If any final fixes were made:

```bash
git add <changed-files>
git commit -m "Finalize Settings configuration hub"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin <branch-name>
gh pr create --base dev --head <branch-name> --title "Make Settings the configuration hub" --body-file <pr-body-file>
```

PR body must include:

- Summary.
- Test commands and results.
- CDP screenshot approvals.
- Known limitations: Storage/Privacy are first-slice status categories unless later implementation completed deeper flows.
