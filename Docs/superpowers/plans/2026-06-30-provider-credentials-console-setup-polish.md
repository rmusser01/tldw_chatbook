# Provider Credentials And Console Setup Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore an in-app provider API-key setup path and polish Console blocked setup so users can recover without hidden command-palette knowledge or obscured composer controls.

**Architecture:** Settings remains the durable provider configuration owner and saves endpoint, env-var name, and local config API-key values under `api_settings.<provider>`. Console owns setup blocker detection and navigation context, sending missing-key recovery to the provider credential controls. The Console composer stops rendering long blocked/recovery copy inline and relies on the Workbench recovery callout plus button tooltip for detailed recovery.

**Tech Stack:** Python 3.11, Textual, Rich, pytest, TCSS.

---

## Scope And ADR

ADR required: yes.

ADR path: `backlog/decisions/012-provider-credential-settings-boundary.md`.

Reason: this task exposes direct in-app mutation of local provider API keys, which is a credential and privacy boundary even though config fallback support already exists.

## Files

- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `backlog/tasks/task-145 - Restore-provider-credential-onboarding-and-polish-Console-setup-UX.md`
- Reference only: `tldw_chatbook/Chat/provider_readiness.py`
- Reference only: `tldw_chatbook/config.py`

## Task 1: Settings Tests For Provider API-Key Controls

- [x] **Step 1: Write failing tests**

Add tests near the existing provider credential env-var tests in `Tests/UI/test_settings_configuration_hub.py`:

```python
@pytest.mark.asyncio
async def test_settings_provider_category_renders_local_api_key_setup_without_revealing_secret():
    app = _build_test_app()
    fake_key = "sk-test-visible-redaction-source"
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY", "api_key": fake_key}
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        api_key = screen.query_one("#settings-provider-api-key", Input)
        assert api_key.password is True
        assert api_key.value == ""
        assert "API key" in _visible_text(screen)
        assert "local config key saved" in _visible_text(screen).lower()
        assert fake_key not in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_provider_category_saves_and_clears_local_api_key(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "OPENAI_API_KEY"}
    }
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        api_key.value = "sk-test-new-local-key"
        screen.handle_provider_api_key_changed(Input.Changed(api_key, api_key.value))
        await pilot.click("#settings-save-category")
        assert ("api_settings.openai", "api_key", "sk-test-new-local-key") in saved
        assert app.app_config["api_settings"]["openai"]["api_key"] == "sk-test-new-local-key"

        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        await pilot.click(clear)
        await pilot.click("#settings-save-category")

    assert ("api_settings.openai", "api_key", "") in saved
    assert app.app_config["api_settings"]["openai"]["api_key"] == ""
```

- [x] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py::test_settings_provider_category_renders_local_api_key_setup_without_revealing_secret Tests/UI/test_settings_configuration_hub.py::test_settings_provider_category_saves_and_clears_local_api_key -q`

Expected: fail because `#settings-provider-api-key`, clear button, and change handler do not exist.

- [x] **Step 3: Implement minimal Settings UI and save behavior**

In `tldw_chatbook/UI/Screens/settings_screen.py`:

- Add an `Input` with id `settings-provider-api-key`, `password=True`, empty value by default, and placeholder copy that says it saves a local config key.
- Add a `Button` with id `settings-provider-api-key-clear`.
- Add a status row id such as `settings-provider-credential-status` that reports `API key source: local config key saved`, `API key source: env:<VAR>`, `API key source: missing`, or `API key source: not required`.
- Add `_provider_saved_api_key_present(provider: str) -> bool` and `_provider_api_key_placeholder(provider: str) -> str`.
- Add `@on(Input.Changed, "#settings-provider-api-key")` handler `handle_provider_api_key_changed`.
- Add `@on(Button.Pressed, "#settings-provider-api-key-clear")` handler that stages `api_key` as `""`, clears the input, and updates dynamic widgets.
- Extend `_provider_form_values_from_widgets()` to return `api_key`.
- Extend provider save logic to persist dirty `api_key` through `SettingsConfigAdapter().save_values(f"api_settings.{provider_save_key}", {"api_key": api_key})`.
- Update `app.app_config["api_settings"][provider_save_key]["api_key"]` after save, but never put raw key into status copy.

- [x] **Step 4: Verify green**

Run the same focused Settings tests. Expected: pass.

## Task 2: Settings Navigation Context Focuses Credential Field

- [x] **Step 1: Write failing test**

Add a test near existing Settings navigation-context tests:

```python
@pytest.mark.asyncio
async def test_settings_provider_navigation_context_focuses_api_key_field():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "openai",
                "model": "gpt-4.1",
                "field": "api_key",
            }
        )
        await pilot.pause(0.1)

        api_key = screen.query_one("#settings-provider-api-key", Input)
        assert api_key.has_focus
```

- [x] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py::test_settings_provider_navigation_context_focuses_api_key_field -q`

Expected: fail because field focus intent is not handled.

- [x] **Step 3: Implement navigation field intent**

Update `SettingsScreen.apply_navigation_context()` to read `field`. After category/provider/model sync, focus:

- `api_key` -> `#settings-provider-api-key`
- `endpoint` -> `#settings-provider-endpoint-value`
- `credential_env_var` -> `#settings-provider-credential-env-var`

Use a safe `try/except QueryError` path and avoid remounting the screen.

- [x] **Step 4: Verify green**

Run the focused test. Expected: pass.

## Task 3: Console Recovery Context Tests

- [x] **Step 1: Write failing tests**

Update `Tests/UI/test_console_native_chat_flow.py::test_console_add_api_key_recovery_targets_provider_settings_category` to expect:

```python
assert message.screen_context == {
    "category": SettingsCategoryId.PROVIDERS_MODELS.value,
    "provider": "huggingface",
    "model": "meta-llama/test-model",
    "field": "api_key",
}
```

Add or update an endpoint recovery test to expect `"field": "endpoint"`.

- [x] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_native_chat_flow.py::test_console_add_api_key_recovery_targets_provider_settings_category -q`

Expected: fail because Console currently omits `field`.

- [x] **Step 3: Implement Console recovery field intent**

In `tldw_chatbook/UI/Screens/chat_screen.py`:

- Add a helper returning recovery field intent from the same readiness branch used by `_console_provider_recovery_action`.
- Add `"field": "api_key"` for missing API key.
- Add `"field": "endpoint"` for endpoint recovery.
- Keep choose-provider and choose-model recovery inside the Console settings modal.
- Refine `_console_setup_blocked_reason()` copy to name Settings > Providers & Models for missing keys.

- [x] **Step 4: Verify green**

Run the focused Console recovery test. Expected: pass.

## Task 4: Console Composer And Visual Contract Tests

- [x] **Step 1: Write failing tests**

Update tests that currently require inline composer recovery/reason. The new contract is:

```python
assert not _is_displayed(console.query_one("#console-composer-recovery"))
assert not _is_displayed(console.query_one("#console-send-disabled-reason"))
visible_draft = console.query_one("#console-command-visible-text")
assert visible_draft.region.width >= 32
```

Keep send button tooltip assertions so blocked details are still accessible:

```python
send = console.query_one("#console-send-message", Button)
assert "Settings" in (send.tooltip or "")
```

- [x] **Step 2: Verify red**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_composer_shows_send_disabled_reason_near_send Tests/UI/test_console_workbench_contract.py::test_console_composer_actions_remain_visible_inside_composer_bounds Tests/UI/test_console_internals_decomposition.py::test_console_composer_empty_setup_blocked_state_shows_reason -q`

Expected: fail because inline recovery/reason widgets are currently displayed and take width.

- [x] **Step 3: Implement composer polish**

In `tldw_chatbook/Widgets/Console/console_composer_bar.py`:

- Keep `#console-composer-recovery` and `#console-send-disabled-reason` mounted for compatibility.
- Do not display or assign fixed width to either widget in setup-blocked states.
- Keep `send_button.tooltip` as the detailed blocker copy.
- Keep empty-draft placeholder concise. Do not inject long setup copy into the draft field.
- Keep `console-composer-setup-blocked` class for styling.

In `tldw_chatbook/css/components/_agentic_terminal.tcss`:

- Ensure `.console-composer-recovery` and `.console-send-disabled-reason` remain hidden by default and do not consume composer width.
- Strengthen useful separators with existing semantic tokens such as `$ds-column-line` or `$ds-input-border`.
- Avoid visible blank bands. Leave hidden compatibility selectors at height 0.

- [x] **Step 4: Verify green**

Run the focused composer tests. Expected: pass.

## Task 5: Focused Regression And Task Closeout

- [x] **Step 1: Run focused Settings and Console suites**

Run: `PATH=.venv/bin:$PATH pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_privacy_security.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_internals_decomposition.py -q`

- [x] **Step 2: Run CSS build/check if required**

Run: `PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py`

- [x] **Step 3: Run diff hygiene**

Run: `git diff --check`

- [x] **Step 4: Update Backlog task**

Check all TASK-145 acceptance criteria and add implementation notes with:

- Settings API-key controls added.
- Console recovery field routing added.
- Composer blocked-state clutter removed.
- Tests and verification commands run.

- [ ] **Step 5: Commit**

Stage only owned files and commit:

```bash
git add \
  "backlog/tasks/task-145 - Restore-provider-credential-onboarding-and-polish-Console-setup-UX.md" \
  backlog/decisions/012-provider-credential-settings-boundary.md \
  Docs/superpowers/specs/2026-06-30-provider-credentials-console-setup-polish-design.md \
  Docs/superpowers/plans/2026-06-30-provider-credentials-console-setup-polish.md \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_internals_decomposition.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Widgets/Console/console_composer_bar.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss
git commit -m "Restore provider credential setup and polish Console blockers"
```
