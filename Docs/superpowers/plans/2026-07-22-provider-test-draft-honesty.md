# TASK-432 Provider Test draft-honesty — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Settings ▸ Providers & Models "Test Provider" check evaluates the unsaved **draft** provider configuration and reports the exact draft values it used — tagged as unsaved — so the evidence never contradicts what was tested.

**Architecture:** Feed `get_provider_readiness` a deep-copied `app_config` with the *dirty* draft provider fields overlaid (endpoint / api_key_env_var / api_key), then assemble provenance-tagged evidence. The value logic lives in pure/resolved-input helpers so it is unit-testable without mounting the settings screen.

**Tech Stack:** Python 3.11+, Textual, pytest (+ pytest-asyncio for the pilot tests).

## Global Constraints

- All changes are in `tldw_chatbook/UI/Screens/settings_screen.py` (plus tests). No change to `tldw_chatbook/Chat/provider_readiness.py`.
- The draft API-key **value** is never placed in a finding — only a source label; every findings/summary string stays wrapped in `redact_secret_text(...)`.
- `dirty_keys()` from `self._provider_draft()` is the sole "unsaved" signal (the app's own dirty tracking). Fields absent from a draft (or no draft at all) are treated as saved.
- No behavior change when there is no draft: the evidence must match today's saved-value output exactly (no `(draft)` tags).
- Provider-provided helpers to reuse (do not reimplement): `_provider_config_entry`, `_provider_endpoint_setting_key`, `_provider_endpoint_summary(provider, endpoint=...)`, `_provider_widget_value`, `_provider_display_name`, `provider_config_key`, `get_provider_readiness`, `redact_secret_text`. `import copy`, `import os`, `Mapping` are already imported.

---

### Task 1: Draft-overlaid staged config

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (add module-level `overlay_provider_draft_config` near the other module helpers, e.g. just above `class SettingsScreen`; add `_provider_test_staged_config` method next to `_provider_discovery_staged_settings`, ~line 5137).
- Test: `Tests/UI/test_settings_provider_test_draft.py` (new).

**Interfaces:**
- Produces (pure, module-level):
  ```python
  def overlay_provider_draft_config(
      app_config: Mapping[str, object],
      *,
      provider_save_key: str,
      endpoint_key: str,
      draft_endpoint: str | None,
      draft_env_var: str | None,
      draft_api_key: str | None,
  ) -> dict:
      ...
  ```
  A field whose argument is `None` is left untouched; a field passed as `""` is overlaid as empty (models an explicit clear). Returns a deep copy; the input is never mutated.
- Produces (method): `SettingsScreen._provider_test_staged_config(self, provider: str) -> Mapping[str, object]`.

- [ ] **Step 1: Write failing tests for `overlay_provider_draft_config`**

```python
# Tests/UI/test_settings_provider_test_draft.py
from tldw_chatbook.UI.Screens.settings_screen import overlay_provider_draft_config


def _base_config():
    return {
        "api_settings": {
            "llama_cpp": {"api_url": "http://localhost:8080/completion", "api_key": "saved-key"},
            "openai": {"api_key": "other-saved"},
        }
    }


def test_overlay_endpoint_only_deep_copies_and_preserves_others():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint="http://localhost:9099",
        draft_env_var=None,
        draft_api_key=None,
    )
    # draft endpoint overlaid
    assert merged["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:9099"
    # saved key + other provider preserved
    assert merged["api_settings"]["llama_cpp"]["api_key"] == "saved-key"
    assert merged["api_settings"]["openai"]["api_key"] == "other-saved"
    # input not mutated
    assert base["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:8080/completion"


def test_overlay_api_key_and_env_var():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var="MY_LLAMA_KEY",
        draft_api_key="draft-secret",
    )
    section = merged["api_settings"]["llama_cpp"]
    assert section["api_key"] == "draft-secret"
    assert section["api_key_env_var"] == "MY_LLAMA_KEY"


def test_overlay_api_key_clear_sets_empty():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key="",
    )
    assert merged["api_settings"]["llama_cpp"]["api_key"] == ""


def test_overlay_creates_missing_section():
    merged = overlay_provider_draft_config(
        {"api_settings": {}},
        provider_save_key="newprov",
        endpoint_key="api_base_url",
        draft_endpoint="http://x:1/v1",
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged["api_settings"]["newprov"]["api_base_url"] == "http://x:1/v1"


def test_overlay_no_fields_is_a_faithful_copy():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged == base
    assert merged is not base
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py -q`
Expected: FAIL with `ImportError` / `AttributeError` (function not defined).

- [ ] **Step 3: Implement `overlay_provider_draft_config` (module level)**

```python
def overlay_provider_draft_config(
    app_config,
    *,
    provider_save_key: str,
    endpoint_key: str,
    draft_endpoint: str | None,
    draft_env_var: str | None,
    draft_api_key: str | None,
) -> dict:
    """Return a deep copy of ``app_config`` with unsaved draft provider fields overlaid.

    Args:
        app_config: The loaded application configuration.
        provider_save_key: The ``api_settings`` section key to overlay onto.
        endpoint_key: The endpoint setting key for this provider (e.g. ``api_url``).
        draft_endpoint: Draft endpoint, or ``None`` to leave the saved endpoint.
        draft_env_var: Draft credential env-var name, or ``None`` to leave saved.
        draft_api_key: Draft API key (``""`` models an explicit clear), or ``None``.

    Returns:
        A new config dict; ``app_config`` is never mutated.
    """
    merged = copy.deepcopy(dict(app_config)) if isinstance(app_config, Mapping) else {}
    api_settings = merged.get("api_settings")
    if not isinstance(api_settings, dict):
        api_settings = {}
        merged["api_settings"] = api_settings
    section = api_settings.get(provider_save_key)
    if not isinstance(section, dict):
        section = {}
        api_settings[provider_save_key] = section
    if draft_endpoint is not None:
        section[endpoint_key] = draft_endpoint
    if draft_env_var is not None:
        section["api_key_env_var"] = draft_env_var
    if draft_api_key is not None:
        section["api_key"] = draft_api_key
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Add `_provider_test_staged_config` method**

Insert next to `_provider_discovery_staged_settings` (mirrors its widget reads):

```python
def _provider_test_staged_config(self, provider: str) -> Mapping[str, object]:
    """Return app_config with the unsaved draft provider fields overlaid.

    Only dirty fields are overlaid, so a provider with no unsaved edits tests
    exactly the saved config (task-432).

    Args:
        provider: The provider whose Test is running (the draft widget value).

    Returns:
        A config mapping the Test's readiness check can evaluate.
    """
    app_config = getattr(self.app_instance, "app_config", {}) or {}
    draft = self._provider_draft()
    dirty = draft.dirty_keys if draft is not None else set()  # dirty_keys is a @property
    if not ({"endpoint", "credential_env_var", "api_key"} & dirty):
        return app_config
    provider_save_key, _config = self._provider_config_entry(provider)
    provider_save_key = provider_save_key or provider_config_key(provider)
    if not provider_save_key:
        return app_config
    try:
        endpoint = self.query_one("#settings-provider-endpoint-value", Input).value.strip()
        env_var = self.query_one("#settings-provider-credential-env-var", Input).value.strip()
        api_key = self.query_one("#settings-provider-api-key", Input).value.strip()
    except QueryError:
        values = self._provider_setting_values_mapping()
        endpoint = str(values.get("endpoint") or "").strip()
        env_var = str(values.get("credential_env_var") or "").strip()
        api_key = str(values.get("api_key") or "").strip()
    return overlay_provider_draft_config(
        app_config,
        provider_save_key=provider_save_key,
        endpoint_key=self._provider_endpoint_setting_key(provider),
        draft_endpoint=endpoint if "endpoint" in dirty else None,
        draft_env_var=env_var if "credential_env_var" in dirty else None,
        draft_api_key=api_key if "api_key" in dirty else None,
    )
```

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_settings_provider_test_draft.py tldw_chatbook/UI/Screens/settings_screen.py
git commit -m "feat(settings): draft-overlaid config for the provider Test (task-432)"
```

---

### Task 2: Provenance-tagged, draft-aware evidence

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` — rewrite `_provider_readiness_test_report` (~line 5497) to build the staged config, gather `dirty`, and delegate findings assembly to a new resolved-input method `_build_provider_readiness_findings`.
- Test: `Tests/UI/test_settings_provider_test_draft.py` (extend).

**Interfaces:**
- Consumes: `overlay_provider_draft_config`, `_provider_test_staged_config` (Task 1).
- Produces (method): `_build_provider_readiness_findings(self, provider: str, model: str, readiness, *, draft_endpoint: str, dirty: set[str]) -> tuple[str, str, bool]` — reads only `app_config` (via existing helpers) and `os.environ`, never widgets, so it runs on a bare `SettingsScreen` instance.

- [ ] **Step 1: Write failing bare-instance tests**

```python
# add to Tests/UI/test_settings_provider_test_draft.py
import os
from unittest.mock import patch
from types import SimpleNamespace

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness


def _bare_settings_screen(app_config):
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config=app_config)
    return screen


def test_findings_show_draft_endpoint_tagged():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:9099"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:9099", dirty={"endpoint"},
    )
    assert "http://localhost:9099 (draft)" in detail
    assert "8080" not in detail


def test_findings_relabel_draft_api_key_source_and_hide_value():
    app_config = {"api_settings": {"openai": {"api_key": "draft-secret"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty={"api_key"},
    )
    assert "api_key_source=draft api_key (unsaved)" in detail
    assert "draft-secret" not in detail and "draft-secret" not in summary


def test_findings_tag_draft_env_var():
    app_config = {"api_settings": {"openai": {"api_key_env_var": "MY_KEY"}}}
    screen = _bare_settings_screen(app_config)
    with patch.dict(os.environ, {"MY_KEY": "envval"}, clear=False):
        readiness = get_provider_readiness("OpenAI", app_config)
        detail, _summary, _passed = screen._build_provider_readiness_findings(
            "OpenAI", "gpt-4o", readiness,
            draft_endpoint="", dirty={"credential_env_var"},
        )
    assert "(draft env var)" in detail


def test_findings_no_draft_has_no_tags():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:8080", dirty=set(),
    )
    assert "(draft)" not in detail and "(unsaved)" not in detail
    assert "http://localhost:8080" in detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py -k findings -q`
Expected: FAIL (`AttributeError: _build_provider_readiness_findings`).

- [ ] **Step 3: Implement `_build_provider_readiness_findings` + rewire the report**

Replace the body of `_provider_readiness_test_report` so it resolves inputs and delegates:

```python
def _provider_readiness_test_report(self) -> tuple[str, str, bool]:
    """Run the local provider readiness test against the DRAFT config."""
    try:
        provider = self._provider_widget_value()
        model = self.query_one("#settings-model-value", Input).value.strip()
        draft_endpoint = self.query_one(
            "#settings-provider-endpoint-value", Input
        ).value.strip()
    except QueryError:
        values = self._provider_setting_values()
        provider = str(values.get("provider") or "").strip()
        model = str(values.get("model") or "").strip()
        draft_endpoint = str(values.get("endpoint") or "").strip()
    draft = self._provider_draft()
    dirty = draft.dirty_keys if draft is not None else set()  # dirty_keys is a @property
    readiness = get_provider_readiness(
        provider, self._provider_test_staged_config(provider)
    )
    return self._build_provider_readiness_findings(
        provider, model, readiness, draft_endpoint=draft_endpoint, dirty=dirty
    )
```

Add the resolved-input builder (moves today's findings logic here, adds provenance tags):

```python
def _build_provider_readiness_findings(
    self,
    provider: str,
    model: str,
    readiness,
    *,
    draft_endpoint: str,
    dirty: set[str],
) -> tuple[str, str, bool]:
    """Assemble the Test evidence line + toast from resolved inputs.

    Reads only ``app_config`` (via helpers) and ``os.environ`` -- never widgets
    -- so it is unit-testable on a bare screen instance.

    Args:
        provider: Provider under test (draft widget value).
        model: Model under test (draft widget value).
        readiness: ``ProviderReadiness`` from the draft-overlaid config.
        draft_endpoint: The endpoint the test used (draft widget, may be empty).
        dirty: The provider draft's dirty field keys.

    Returns:
        Tuple of (redacted detail line, redacted toast summary, passed).
    """
    provider_key = provider_config_key(provider)
    findings: list[str] = ["Provider test", readiness.user_message]

    if not model:
        findings.append("model=missing")
    else:
        findings.append(f"model={model}{' (draft)' if 'model' in dirty else ''}")

    if readiness.api_key_source:
        if (
            "api_key" in dirty
            and readiness.api_key_source
            == f"config:api_settings.{provider_key}.api_key"
        ):
            findings.append("api_key_source=draft api_key (unsaved)")
        else:
            findings.append(f"api_key_source={readiness.api_key_source}")
    if readiness.env_var:
        raw_value = os.environ.get(readiness.env_var)
        env_tag = " (draft env var)" if "credential_env_var" in dirty else ""
        findings.append(
            f"{readiness.env_var}={raw_value if raw_value else 'missing'}{env_tag}"
        )
    elif not readiness.requires_api_key:
        findings.append("api_key=not required")

    endpoint_summary = self._provider_endpoint_summary(provider, endpoint=draft_endpoint)
    if "endpoint" in dirty:
        endpoint_summary = f"{endpoint_summary} (draft)"
    findings.append(endpoint_summary)

    passed = bool(readiness.ready and model)
    findings.append(f"status={'ready' if passed else 'blocked'}")

    display_name = self._provider_display_name(provider) if provider else "Provider"
    if passed:
        summary = f"Provider test passed: {display_name} is ready; model {model}."
    elif not readiness.ready:
        summary = f"Provider test failed: {readiness.user_message}"
        if not model:
            summary += " Also set a default model."
    else:
        summary = (
            f"Provider test failed: {display_name} is ready but no default model is set."
        )
    return (
        redact_secret_text(" | ".join(findings)),
        redact_secret_text(summary),
        passed,
    )
```

Note: the api-key-source relabel condition needs no draft-key value — when `api_key` is dirty *and* readiness resolved to `config:...api_key`, the overlaid (non-empty) draft key produced it; a cleared draft (`api_key=""`) never resolves to `config:...api_key`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py -q`
Expected: PASS (Task 1 + Task 2 tests).

- [ ] **Step 5: Run the existing settings suite for regressions**

Run: `python -m pytest Tests/UI/test_settings_configuration_hub.py -q`
Expected: PASS (no regression in the heavyweight settings tests).

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_settings_provider_test_draft.py tldw_chatbook/UI/Screens/settings_screen.py
git commit -m "feat(settings): provider Test reports draft values, tagged unsaved (task-432 AC#1)"
```

---

### Task 3: Pin AC#2 / AC#3 (button path) + end-to-end wiring

**Files:**
- Test: `Tests/UI/test_settings_provider_test_draft.py` (extend with pilot tests).
- No production change expected (AC#2/AC#3 already delivered by the `#settings-test-provider` button); if a pilot test reveals a genuine gap, fix it and note it.

**Interfaces:**
- Consumes: the wired `handle_test_provider` / `action_settings_test_category` and `#settings-provider-test-result`.

- [ ] **Step 1: Write pilot tests (AC#2, AC#3, wiring)**

Reuse the harness helpers from `Tests/UI/test_settings_configuration_hub.py` (`host = ...`, `run_test`, `_open_settings_category`, `_wait_for_selector`). Sketch (fill in with the suite's exact harness imports when implementing):

```python
import pytest

@pytest.mark.asyncio
async def test_test_provider_button_runs_the_check(...):
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers_models")
        screen = pilot.app.screen
        # AC#2: clicking the button (not the 't' hotkey) runs the test.
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        result = screen.query_one("#settings-provider-test-result", Static)
        assert "Provider test" in str(result.render())

@pytest.mark.asyncio
async def test_test_provider_button_runs_with_input_focused(...):
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers_models")
        screen = pilot.app.screen
        # AC#3: focus a provider input (the 't' hotkey would no-op here) then click.
        screen.query_one("#settings-model-value", Input).focus()
        await pilot.pause()
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        assert "Provider test" in str(
            screen.query_one("#settings-provider-test-result", Static).render()
        )

@pytest.mark.asyncio
async def test_test_provider_result_shows_draft_endpoint(...):
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers_models")
        screen = pilot.app.screen
        # Pick a URL-based provider, type a draft endpoint, run the test.
        # (Set the provider via its widget, then:)
        ep = screen.query_one("#settings-provider-endpoint-value", Input)
        ep.value = "http://localhost:9099"
        await pilot.pause()
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        detail = str(screen.query_one("#settings-provider-test-result", Static).render())
        assert "http://localhost:9099 (draft)" in detail
```

- [ ] **Step 2: Run and iterate the pilot tests**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py -k "button or draft_endpoint" -q`
Expected: PASS. If the draft-endpoint pilot flakes on the async endpoint probe, assert on the pre-probe `_provider_test_result` (the detail line is set before the probe worker) or poll `#settings-provider-test-result` with a short bounded loop like `_wait_for_selector`.

- [ ] **Step 3: Full regression + commit**

Run: `python -m pytest Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_configuration_hub.py -q`
Expected: PASS.

```bash
git add Tests/UI/test_settings_provider_test_draft.py
git commit -m "test(settings): pin Test-button path + draft-endpoint wiring (task-432 AC#2/AC#3)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 → Tasks 1+2 (draft-overlaid readiness + provenance tags). AC#2 → Task 3 button-runs test. AC#3 → Task 3 button-with-input-focused (non-hotkey path). All covered.
- **No placeholders:** the only intentionally-sketched code is the Task 3 pilot harness wiring, which must adopt the exact fixture/import names from `test_settings_configuration_hub.py` at implementation time (that file's harness is the source of truth).
- **Type/name consistency:** `overlay_provider_draft_config` signature and `_build_provider_readiness_findings` signature are identical across the tasks that define and call them; `dirty` is a `set[str]` throughout; field keys (`endpoint`, `credential_env_var`, `api_key`, `model`) match `_provider_form_values_from_widgets`.
