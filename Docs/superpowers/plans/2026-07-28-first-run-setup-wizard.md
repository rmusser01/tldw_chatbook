# First-Run Setup Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A skippable, re-runnable first-run setup wizard (hermes-agent's process in chatbook's wizard chrome) that walks a new user through provider/key, model, RAG, tools, notes sync, appearance, and key encryption, then lands them in a working app.

**Architecture:** A pure-logic state module (`first_run_setup_state.py`) owns all decisions and config mutations; a `WizardScreen` subclass with a `SetupWizardContainer(WizardContainer)` renders it. `BaseWizard.py` is never modified. Commits happen per-step (commit-on-Next) through one exclusive worker. The wizard auto-offers once via a guard in `_push_initial_screen()` (covers both splash paths).

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest + pytest-asyncio (Textual Pilot), httpx.

**Spec:** `Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md` — read it before starting.

## Global Constraints

- Never modify `tldw_chatbook/UI/Wizards/BaseWizard.py` — all behavior changes live in subclasses.
- The wizard writes ONLY these config sections: `api_settings.<provider>`, `chat_defaults`, `embedding_config`, `tools`, `notes`, `general`, `splash_screen`, and its own `first_run` section. Encryption goes through `enable_config_encryption()`, never direct key writes.
- All config writes go through `save_settings_to_cli_config(section_values)` (batch form) — never loop `save_setting_to_cli_config` per key.
- Secrets never round-trip into inputs: prefill readers return presence metadata only.
- Two probe timeout budgets: 2.5s localhost (existing constants), 8.0s cloud.
- Steps never import from Settings screens; the state module never imports Textual or does I/O.
- Known-broken helpers — do NOT use: `config.get_detected_api_providers()` (matches `"api_settings.<p>"` as a top-level key; always returns `[]`) and `get_api_key()`'s `api_settings` branch (same bug). Walk the nested `config["api_settings"]` dict instead.
- Test runner: `pytest Tests/...`. If `pytest` is missing from the venv, install dev deps with `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` (the venv is uv-managed and has no pip).
- Async tests need explicit `@pytest.mark.asyncio` (no global asyncio_mode at root).
- Repo hygiene: this work is tracked as a backlog task (created in Task 1); update its status/plan/notes per CLAUDE.md as you go.

---

### Task 1: Fix task-740 (splash config read bug) + backlog hygiene

The Appearance step writes `[splash_screen]`; today both readers discard that section because they call `get_cli_setting("splash_screen", <dict>)`, and the non-string second positional is coerced to `default` and returned unconditionally.

**Files:**
- Modify: `tldw_chatbook/Widgets/splash_screen.py:196`
- Modify: `tldw_chatbook/Widgets/settings_splash_screen_viewer.py:55`
- Test: `Tests/Widgets/test_splash_screen_config_read.py` (create)

**Interfaces:**
- Consumes: `config.get_cli_setting(section: str, key: str = None, default: Any = None)` (config.py:4358); nested-section fallback supports `"splash_screen.effects"` dotted sections.
- Produces: no new API — both call sites read per-key so `[splash_screen]` TOML values are honored.

- [ ] **Step 1: Create the backlog task for the wizard feature and start task-740**

```bash
backlog task create "First-run setup wizard (hermes-modeled onboarding)" \
  -d "Guided, skippable, re-runnable setup wizard per Docs/superpowers/specs/2026-07-28-first-run-setup-wizard-design.md" \
  --ac "New user with fresh config is offered the wizard once after startup" \
  --ac "Quick and Full tracks both complete and land in a working app" \
  --ac "Every step is skippable; Esc asks for confirmation and completed steps stay saved" \
  --ac "Wizard is re-runnable from Settings and the command palette with current values prefilled" \
  --ac "Secrets are masked everywhere and encryption offer uses the existing mechanism" \
  -s "In Progress"
backlog task edit 740 -s "In Progress"
```

- [ ] **Step 2: Write the failing test**

Create `Tests/Widgets/test_splash_screen_config_read.py`:

```python
"""Regression tests for task-740: [splash_screen] config must be honored."""

from unittest.mock import patch


def test_splash_screen_reads_configured_duration():
    """A configured value must win over the hardcoded default."""

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "duration":
            return 9.5
        return default

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.splash_screen import SplashScreen

        splash = SplashScreen()
        assert splash.config["duration"] == 9.5


def test_splash_screen_default_applies_only_when_key_absent():
    def fake_get_cli_setting(section, key=None, default=None):
        return default

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.splash_screen import SplashScreen

        splash = SplashScreen()
        assert splash.config["duration"] == 1.5


def test_settings_splash_viewer_reads_configured_card_selection():
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "card_selection":
            return "matrix"
        return default

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
            SettingsSplashScreenViewer,
        )

        viewer = SettingsSplashScreenViewer()
        assert viewer.splash_config["card_selection"] == "matrix"
```

Adjust the constructor calls if either class requires arguments — read the `__init__` of each first; the assertion targets (`splash.config`, `viewer.splash_config`) must match the attribute each class stores its merged config in (verify attribute names at `splash_screen.py:196-201` and `settings_splash_screen_viewer.py:50-60` and fix the test to match reality, not vice versa).

- [ ] **Step 3: Run to verify failure**

Run: `pytest Tests/Widgets/test_splash_screen_config_read.py -v`
Expected: FAIL — configured values (9.5, "matrix") are ignored because the dict-as-second-arg call returns defaults unconditionally.

- [ ] **Step 4: Fix both call sites (per-key reads)**

In `tldw_chatbook/Widgets/splash_screen.py`, replace line 196 (`config = get_cli_setting("splash_screen", default_config)`) and the merge loop that follows with:

```python
            config = {
                key: get_cli_setting("splash_screen", key, value)
                for key, value in default_config.items()
            }
```

If `default_config` contains effects keys that live in the nested `[splash_screen.effects]` TOML table (`fade_in_duration`, `fade_out_duration`, `animation_speed`), read those with the dotted section instead:

```python
            _EFFECTS_KEYS = {"fade_in_duration", "fade_out_duration", "animation_speed"}
            config = {
                key: get_cli_setting(
                    "splash_screen.effects" if key in _EFFECTS_KEYS else "splash_screen",
                    key,
                    value,
                )
                for key, value in default_config.items()
            }
```

Apply the same shape to `settings_splash_screen_viewer.py:55` against `DEFAULT_SPLASH_CONFIG` (it flattens effects keys — use the `_EFFECTS_KEYS` variant there).

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest Tests/Widgets/test_splash_screen_config_read.py -v`
Expected: PASS (3/3).

- [ ] **Step 6: Close task-740 and commit**

```bash
backlog task edit 740 -s Done --notes "Per-key get_cli_setting reads in splash_screen.py and settings_splash_screen_viewer.py; regression tests in Tests/Widgets/test_splash_screen_config_read.py"
git add tldw_chatbook/Widgets/splash_screen.py tldw_chatbook/Widgets/settings_splash_screen_viewer.py Tests/Widgets/test_splash_screen_config_read.py backlog/
git commit -m "fix: honor [splash_screen] config via per-key get_cli_setting reads (TASK-740)"
```

---

### Task 2: State module — offer gating and wizard flags

**Files:**
- Create: `tldw_chatbook/UI/Wizards/first_run_setup_state.py`
- Test: `Tests/Wizards/test_first_run_setup_state.py` (create)

**Interfaces:**
- Consumes: nothing from the app — pure module (stdlib only), mirroring `Chat/console_onboarding_state.py` (frozen dataclasses, keyword-only `build_*`/`coerce_*` functions, Google-style docstrings).
- Produces (later tasks rely on these exact names):
  - `WIZARD_STATE_SECTION = "first_run"`, `SETUP_STARTED_KEY = "setup_started"`, `SETUP_COMPLETED_KEY = "setup_completed"`
  - `coerce_wizard_flag(raw: Any) -> bool`
  - `any_provider_configured(app_config: Mapping[str, object], environ: Mapping[str, str]) -> bool`
  - `should_offer_wizard(app_config, environ) -> bool`
  - `should_show_resume_toast(app_config, environ) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Wizards/test_first_run_setup_state.py`:

```python
"""Unit tests for the pure first-run setup wizard state module."""

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    any_provider_configured,
    coerce_wizard_flag,
    should_offer_wizard,
    should_show_resume_toast,
)


def _config(api_settings=None, first_run=None):
    cfg = {}
    if api_settings is not None:
        cfg["api_settings"] = api_settings
    if first_run is not None:
        cfg["first_run"] = first_run
    return cfg


class TestCoerceWizardFlag:
    def test_truthy_values(self):
        assert coerce_wizard_flag(True) is True
        assert coerce_wizard_flag("true") is True
        assert coerce_wizard_flag(1) is True

    def test_falsy_and_garbage_values(self):
        assert coerce_wizard_flag(False) is False
        assert coerce_wizard_flag(None) is False
        assert coerce_wizard_flag("nope") is False
        assert coerce_wizard_flag({}) is False


class TestAnyProviderConfigured:
    def test_empty_config_is_unconfigured(self):
        assert any_provider_configured(_config(), {}) is False

    def test_placeholder_key_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key": "<API_KEY_HERE>"}})
        assert any_provider_configured(cfg, {}) is False

    def test_real_inline_key_counts(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert any_provider_configured(cfg, {}) is True

    def test_env_var_present_counts(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {"OPENAI_API_KEY": "sk-x"}) is True

    def test_env_var_declared_but_unset_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {}) is False

    def test_local_endpoint_url_counts(self):
        cfg = _config(api_settings={"llama_cpp": {"api_url": "http://127.0.0.1:8080"}})
        assert any_provider_configured(cfg, {}) is True


class TestShouldOfferWizard:
    def test_fresh_config_offers(self):
        assert should_offer_wizard(_config(), {}) is True

    def test_configured_provider_blocks_offer(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert should_offer_wizard(cfg, {}) is False

    def test_completed_blocks_offer(self):
        cfg = _config(first_run={"setup_completed": True})
        assert should_offer_wizard(cfg, {}) is False

    def test_started_but_not_completed_blocks_reoffer(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_offer_wizard(cfg, {}) is False


class TestShouldShowResumeToast:
    def test_started_not_completed_shows_toast(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_show_resume_toast(cfg, {}) is True

    def test_completed_never_shows_toast(self):
        cfg = _config(first_run={"setup_started": True, "setup_completed": True})
        assert should_show_resume_toast(cfg, {}) is False

    def test_never_started_never_shows_toast(self):
        assert should_show_resume_toast(_config(), {}) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v`
Expected: FAIL with `ModuleNotFoundError: ... first_run_setup_state`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/UI/Wizards/first_run_setup_state.py`:

```python
"""Pure state contracts for the first-run setup wizard.

No Textual imports, no I/O — every function is a pure transform over the
in-memory app config, mirroring Chat/console_onboarding_state.py. The wizard
Screen owns rendering and persistence; this module owns every decision.
"""

from __future__ import annotations

from typing import Any, Mapping

WIZARD_STATE_SECTION = "first_run"
SETUP_STARTED_KEY = "setup_started"
SETUP_COMPLETED_KEY = "setup_completed"

# Endpoint keys a local provider may use (mirrors
# Chat/local_server_discovery._ENDPOINT_CONFIG_KEYS).
_ENDPOINT_KEYS = ("api_url", "api_base_url", "api_base", "base_url", "api_endpoint", "endpoint")

_PLACEHOLDER_MARKERS = ("<", ">")


def coerce_wizard_flag(raw: Any) -> bool:
    """Tolerantly parse a persisted wizard flag.

    Args:
        raw: Whatever the TOML loader produced for the key.

    Returns:
        True only for bool True, int 1, or the string "true" (case-insensitive).
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw == 1
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _is_real_secret(value: Any) -> bool:
    """A non-empty string that is not a <PLACEHOLDER> template value."""
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    return not (stripped.startswith(_PLACEHOLDER_MARKERS[0]) and stripped.endswith(_PLACEHOLDER_MARKERS[1]))


def any_provider_configured(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Return True when any provider has usable credentials or an endpoint.

    Walks the NESTED ``app_config["api_settings"]`` dict. Do not replace this
    with config.get_detected_api_providers(): that helper matches
    "api_settings.<p>" as a top-level key and always returns [].
    """
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if not isinstance(settings, Mapping):
            continue
        if _is_real_secret(settings.get("api_key")):
            return True
        env_var = settings.get("api_key_env_var")
        if isinstance(env_var, str) and env_var.strip() and environ.get(env_var.strip()):
            return True
        for endpoint_key in _ENDPOINT_KEYS:
            if _is_real_secret(settings.get(endpoint_key)):
                return True
    return False


def _wizard_flag(app_config: Mapping[str, object], key: str) -> bool:
    section = app_config.get(WIZARD_STATE_SECTION)
    if not isinstance(section, Mapping):
        return False
    return coerce_wizard_flag(section.get(key))


def should_offer_wizard(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Auto-offer once: no wizard state keys AND nothing configured."""
    if _wizard_flag(app_config, SETUP_STARTED_KEY):
        return False
    if _wizard_flag(app_config, SETUP_COMPLETED_KEY):
        return False
    return not any_provider_configured(app_config, environ)


def should_show_resume_toast(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Started but never finished: point at Settings, never re-push."""
    return _wizard_flag(app_config, SETUP_STARTED_KEY) and not _wizard_flag(
        app_config, SETUP_COMPLETED_KEY
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py Tests/Wizards/test_first_run_setup_state.py
git commit -m "feat: first-run wizard offer gating and flag state (pure module)"
```

---

### Task 3: State module — tracks, commit builders, dependency invalidation, section allowlist

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py`
- Test: `Tests/Wizards/test_first_run_setup_state.py` (append)

**Interfaces:**
- Produces (exact names later tasks use):
  - `TRACK_QUICK = "quick"`, `TRACK_FULL = "full"`
  - `STEP_WELCOME/STEP_PROVIDER/STEP_MODEL/STEP_RAG/STEP_TOOLS/STEP_NOTES/STEP_APPEARANCE/STEP_PROTECT/STEP_SUMMARY` (string ids: `"welcome"`, `"provider"`, `"model"`, `"rag"`, `"tools"`, `"notes"`, `"appearance"`, `"protect-keys"`, `"summary"`)
  - `active_step_ids(track: str, *, key_entered: bool) -> tuple[str, ...]`
  - `build_provider_commit(*, provider_key: str, api_key: str | None, api_url: str | None) -> dict[str, dict[str, Any]]`
  - `build_model_commit(*, provider_value: str, model_id: str) -> dict[str, dict[str, Any]]`
  - `build_rag_commit(*, default_model_id: str) -> dict[str, dict[str, Any]]`
  - `build_tools_commit(*, gate_values: Mapping[str, bool]) -> dict[str, dict[str, Any]]`
  - `build_notes_commit(*, sync_directory: str, auto_sync_enabled: bool) -> dict[str, dict[str, Any]]`
  - `build_appearance_commit(*, default_theme: str, splash_card: str | None) -> dict[str, dict[str, Any]]`
  - `build_wizard_state_commit(*, started: bool | None = None, completed: bool | None = None) -> dict[str, dict[str, Any]]`
  - `invalidate_model_for_provider_change(commit: dict[str, dict[str, Any]], *, previous_provider_value: str | None, new_provider_value: str) -> dict[str, dict[str, Any]]`
  - `WIZARD_OWNED_SECTIONS: frozenset[str]` and `commit_sections_allowed(section_values: Mapping[str, Mapping[Any, Any]]) -> bool`

- [ ] **Step 1: Write the failing tests (append to the test file)**

```python
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_APPEARANCE,
    STEP_MODEL,
    STEP_NOTES,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SUMMARY,
    STEP_TOOLS,
    STEP_WELCOME,
    TRACK_FULL,
    TRACK_QUICK,
    active_step_ids,
    build_appearance_commit,
    build_model_commit,
    build_notes_commit,
    build_provider_commit,
    build_rag_commit,
    build_tools_commit,
    build_wizard_state_commit,
    commit_sections_allowed,
    invalidate_model_for_provider_change,
)


class TestActiveStepIds:
    def test_full_track_without_key(self):
        assert active_step_ids(TRACK_FULL, key_entered=False) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_RAG, STEP_TOOLS,
            STEP_NOTES, STEP_APPEARANCE, STEP_SUMMARY,
        )

    def test_full_track_with_key_includes_protect(self):
        assert STEP_PROTECT in active_step_ids(TRACK_FULL, key_entered=True)

    def test_quick_track(self):
        assert active_step_ids(TRACK_QUICK, key_entered=False) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_SUMMARY,
        )

    def test_quick_track_with_key(self):
        assert active_step_ids(TRACK_QUICK, key_entered=True) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_PROTECT, STEP_SUMMARY,
        )


class TestCommitBuilders:
    def test_provider_commit_cloud(self):
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
        assert commit == {"api_settings.openai": {"api_key": "sk-x"}}

    def test_provider_commit_local(self):
        commit = build_provider_commit(
            provider_key="llama_cpp", api_key=None, api_url="http://127.0.0.1:8080"
        )
        assert commit == {"api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"}}

    def test_provider_commit_env_key_writes_nothing_secret(self):
        commit = build_provider_commit(provider_key="openai", api_key=None, api_url=None)
        assert commit == {}

    def test_model_commit(self):
        commit = build_model_commit(provider_value="OpenAI", model_id="gpt-5.6-terra")
        assert commit == {"chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}}

    def test_tools_commit_only_gate_keys(self):
        commit = build_tools_commit(gate_values={"read_file_enabled": True, "write_file_enabled": False})
        assert commit == {"tools": {"read_file_enabled": True, "write_file_enabled": False}}

    def test_notes_commit(self):
        commit = build_notes_commit(sync_directory="~/Notes", auto_sync_enabled=True)
        assert commit == {"notes": {"sync_directory": "~/Notes", "auto_sync_enabled": True}}

    def test_appearance_commit_with_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card="matrix")
        assert commit == {
            "general": {"default_theme": "textual-dark"},
            "splash_screen": {"card_selection": "matrix"},
        }

    def test_appearance_commit_without_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card=None)
        assert commit == {"general": {"default_theme": "textual-dark"}}

    def test_rag_commit(self):
        commit = build_rag_commit(default_model_id="e5-small-v2")
        assert commit == {"embedding_config": {"default_model_id": "e5-small-v2"}}

    def test_state_commit(self):
        assert build_wizard_state_commit(started=True) == {"first_run": {"setup_started": True}}
        assert build_wizard_state_commit(completed=True) == {"first_run": {"setup_completed": True}}


class TestDependencyInvalidation:
    def test_provider_change_clears_stale_model(self):
        commit = build_provider_commit(provider_key="anthropic", api_key="sk-a", api_url=None)
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="Anthropic"
        )
        assert merged["chat_defaults"] == {"provider": "Anthropic", "model": ""}

    def test_same_provider_leaves_model_alone(self):
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="OpenAI"
        )
        assert "chat_defaults" not in merged


class TestSectionAllowlist:
    def test_all_builders_stay_in_allowlist(self):
        commits = [
            build_provider_commit(provider_key="openai", api_key="sk", api_url=None),
            build_model_commit(provider_value="OpenAI", model_id="m"),
            build_rag_commit(default_model_id="e5-small-v2"),
            build_tools_commit(gate_values={"read_file_enabled": True}),
            build_notes_commit(sync_directory="~/n", auto_sync_enabled=False),
            build_appearance_commit(default_theme="t", splash_card="c"),
            build_wizard_state_commit(started=True),
        ]
        for commit in commits:
            assert commit_sections_allowed(commit), commit

    def test_foreign_section_rejected(self):
        assert commit_sections_allowed({"database": {"x": 1}}) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v -k "ActiveStep or Commit or Invalidation or Allowlist"`
Expected: FAIL with ImportError on the new names.

- [ ] **Step 3: Implement (append to `first_run_setup_state.py`)**

```python
TRACK_QUICK = "quick"
TRACK_FULL = "full"

STEP_WELCOME = "welcome"
STEP_PROVIDER = "provider"
STEP_MODEL = "model"
STEP_RAG = "rag"
STEP_TOOLS = "tools"
STEP_NOTES = "notes"
STEP_APPEARANCE = "appearance"
STEP_PROTECT = "protect-keys"
STEP_SUMMARY = "summary"

_FULL_TRACK = (
    STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_RAG, STEP_TOOLS,
    STEP_NOTES, STEP_APPEARANCE, STEP_PROTECT, STEP_SUMMARY,
)
_QUICK_TRACK = (STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_PROTECT, STEP_SUMMARY)

WIZARD_OWNED_SECTIONS = frozenset(
    {"chat_defaults", "embedding_config", "tools", "notes", "general",
     "splash_screen", WIZARD_STATE_SECTION}
)
_API_SETTINGS_PREFIX = "api_settings."


def active_step_ids(track: str, *, key_entered: bool) -> tuple[str, ...]:
    """Resolve the ordered active step ids for a track.

    Args:
        track: TRACK_QUICK or TRACK_FULL (anything else falls back to full).
        key_entered: Whether any secret was entered this run; gates STEP_PROTECT.
    """
    base = _QUICK_TRACK if track == TRACK_QUICK else _FULL_TRACK
    if key_entered:
        return base
    return tuple(step for step in base if step != STEP_PROTECT)


def build_provider_commit(
    *, provider_key: str, api_key: str | None, api_url: str | None
) -> dict[str, dict[str, Any]]:
    """Mutation for the provider step. Empty when the key lives in the env."""
    values: dict[str, Any] = {}
    if api_key:
        values["api_key"] = api_key
    if api_url:
        values["api_url"] = api_url
    if not values:
        return {}
    return {f"{_API_SETTINGS_PREFIX}{provider_key}": values}


def build_model_commit(*, provider_value: str, model_id: str) -> dict[str, dict[str, Any]]:
    return {"chat_defaults": {"provider": provider_value, "model": model_id}}


def build_rag_commit(*, default_model_id: str) -> dict[str, dict[str, Any]]:
    return {"embedding_config": {"default_model_id": default_model_id}}


def build_tools_commit(*, gate_values: Mapping[str, bool]) -> dict[str, dict[str, Any]]:
    return {"tools": {key: bool(value) for key, value in gate_values.items()}}


def build_notes_commit(
    *, sync_directory: str, auto_sync_enabled: bool
) -> dict[str, dict[str, Any]]:
    return {"notes": {"sync_directory": sync_directory, "auto_sync_enabled": auto_sync_enabled}}


def build_appearance_commit(
    *, default_theme: str, splash_card: str | None
) -> dict[str, dict[str, Any]]:
    commit: dict[str, dict[str, Any]] = {"general": {"default_theme": default_theme}}
    if splash_card:
        commit["splash_screen"] = {"card_selection": splash_card}
    return commit


def build_wizard_state_commit(
    *, started: bool | None = None, completed: bool | None = None
) -> dict[str, dict[str, Any]]:
    values: dict[str, Any] = {}
    if started is not None:
        values[SETUP_STARTED_KEY] = started
    if completed is not None:
        values[SETUP_COMPLETED_KEY] = completed
    return {WIZARD_STATE_SECTION: values} if values else {}


def invalidate_model_for_provider_change(
    commit: dict[str, dict[str, Any]],
    *,
    previous_provider_value: str | None,
    new_provider_value: str,
) -> dict[str, dict[str, Any]]:
    """Supersede a stale model when the committed provider changes.

    Without this, Back-and-switch leaves chat_defaults pairing the new
    provider with the old provider's model.
    """
    if previous_provider_value and previous_provider_value != new_provider_value:
        merged = dict(commit)
        merged["chat_defaults"] = {"provider": new_provider_value, "model": ""}
        return merged
    return commit


def commit_sections_allowed(section_values: Mapping[str, Mapping[Any, Any]]) -> bool:
    """The invariant oracle: wizard commits touch only wizard-owned sections."""
    for section in section_values:
        if section in WIZARD_OWNED_SECTIONS:
            continue
        if section.startswith(_API_SETTINGS_PREFIX) and len(section) > len(_API_SETTINGS_PREFIX):
            continue
        return False
    return True
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v`
Expected: PASS (all, including Task 2's).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py Tests/Wizards/test_first_run_setup_state.py
git commit -m "feat: wizard tracks, commit builders, model invalidation, section allowlist"
```

---

### Task 4: State module — prefill readers and summary matrix

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py`
- Test: `Tests/Wizards/test_first_run_setup_state.py` (append)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SecretPresence: configured: bool; env_var: str | None; env_var_set: bool` — never carries the secret value
  - `read_provider_secret_presence(app_config, environ, *, provider_key: str) -> SecretPresence`
  - `@dataclass(frozen=True) WizardPrefill: provider_value: str; model_id: str; sync_directory: str; auto_sync_enabled: bool; default_theme: str; tool_gates: tuple[tuple[str, bool], ...]`
  - `read_wizard_prefill(app_config) -> WizardPrefill`
  - `@dataclass(frozen=True) SummaryRow: label: str; ok: bool; detail: str`
  - `build_summary_rows(app_config, environ, *, rag_deps_installed: bool) -> tuple[SummaryRow, ...]`

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    build_summary_rows,
    read_provider_secret_presence,
    read_wizard_prefill,
)


class TestSecretPresence:
    def test_inline_key_is_configured_without_value(self):
        cfg = {"api_settings": {"openai": {"api_key": "sk-secret"}}}
        presence = read_provider_secret_presence(cfg, {}, provider_key="openai")
        assert presence.configured is True
        assert "sk-secret" not in repr(presence)

    def test_env_var_reported(self):
        cfg = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
        presence = read_provider_secret_presence(
            cfg, {"OPENAI_API_KEY": "sk-x"}, provider_key="openai"
        )
        assert presence.env_var == "OPENAI_API_KEY"
        assert presence.env_var_set is True
        assert presence.configured is True

    def test_unconfigured(self):
        presence = read_provider_secret_presence({}, {}, provider_key="openai")
        assert presence.configured is False


class TestWizardPrefill:
    def test_reads_current_values(self):
        cfg = {
            "chat_defaults": {"provider": "Anthropic", "model": "claude-opus-5"},
            "notes": {"sync_directory": "~/N", "auto_sync_enabled": True},
            "general": {"default_theme": "textual-light"},
            "tools": {"read_file_enabled": True},
        }
        prefill = read_wizard_prefill(cfg)
        assert prefill.provider_value == "Anthropic"
        assert prefill.model_id == "claude-opus-5"
        assert prefill.sync_directory == "~/N"
        assert prefill.auto_sync_enabled is True
        assert prefill.default_theme == "textual-light"
        assert ("read_file_enabled", True) in prefill.tool_gates

    def test_empty_config_yields_empty_strings(self):
        prefill = read_wizard_prefill({})
        assert prefill.provider_value == ""
        assert prefill.model_id == ""


class TestSummaryRows:
    def test_rows_reflect_persisted_state(self):
        cfg = {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
            "encryption": {"enabled": True},
        }
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
        assert rows["Provider"].ok is True
        assert rows["Default model"].ok is True
        assert rows["RAG"].ok is False
        assert "not installed" in rows["RAG"].detail
        assert rows["Key encryption"].ok is True

    def test_empty_config_all_missing(self):
        rows = build_summary_rows({}, {}, rag_deps_installed=True)
        by_label = {row.label: row for row in rows}
        assert by_label["Provider"].ok is False
        assert by_label["Tools"].ok is False  # no gates on => off is reported honestly
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v -k "SecretPresence or WizardPrefill or SummaryRows"`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement (append)**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecretPresence:
    """Whether a provider secret exists — never the secret itself."""

    configured: bool
    env_var: str | None = None
    env_var_set: bool = False


@dataclass(frozen=True)
class WizardPrefill:
    """Current config values for re-run prefill (no secrets)."""

    provider_value: str = ""
    model_id: str = ""
    sync_directory: str = ""
    auto_sync_enabled: bool = False
    default_theme: str = ""
    tool_gates: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class SummaryRow:
    """One ✓/✗ line of the final summary matrix."""

    label: str
    ok: bool
    detail: str = ""


def _section(app_config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = app_config.get(name)
    return section if isinstance(section, Mapping) else {}


def read_provider_secret_presence(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    provider_key: str,
) -> SecretPresence:
    settings = _section(_section(app_config, "api_settings"), provider_key)
    env_var_raw = settings.get("api_key_env_var")
    env_var = env_var_raw.strip() if isinstance(env_var_raw, str) and env_var_raw.strip() else None
    env_var_set = bool(env_var and environ.get(env_var))
    inline = _is_real_secret(settings.get("api_key"))
    return SecretPresence(
        configured=inline or env_var_set, env_var=env_var, env_var_set=env_var_set
    )


def read_wizard_prefill(app_config: Mapping[str, object]) -> WizardPrefill:
    chat_defaults = _section(app_config, "chat_defaults")
    notes = _section(app_config, "notes")
    general = _section(app_config, "general")
    tools = _section(app_config, "tools")
    return WizardPrefill(
        provider_value=str(chat_defaults.get("provider") or ""),
        model_id=str(chat_defaults.get("model") or ""),
        sync_directory=str(notes.get("sync_directory") or ""),
        auto_sync_enabled=coerce_wizard_flag(notes.get("auto_sync_enabled")),
        default_theme=str(general.get("default_theme") or ""),
        tool_gates=tuple(
            (str(key), coerce_wizard_flag(value)) for key, value in tools.items()
        ),
    )


def build_summary_rows(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    rag_deps_installed: bool,
) -> tuple[SummaryRow, ...]:
    """Build the ✓/✗ matrix strictly from persisted config (never step memory)."""
    prefill = read_wizard_prefill(app_config)
    provider_ok = any_provider_configured(app_config, environ)
    tools_on = [key for key, value in prefill.tool_gates if value]
    notes_on = prefill.auto_sync_enabled and bool(prefill.sync_directory)
    encryption_on = coerce_wizard_flag(_section(app_config, "encryption").get("enabled"))
    rag_model = str(_section(app_config, "embedding_config").get("default_model_id") or "")
    if not rag_deps_installed:
        rag_row = SummaryRow("RAG", False, "embeddings deps not installed")
    elif rag_model:
        rag_row = SummaryRow("RAG", True, f"embedding model: {rag_model}")
    else:
        rag_row = SummaryRow("RAG", False, "no embedding model selected")
    return (
        SummaryRow("Provider", provider_ok, "" if provider_ok else "no credentials or endpoint"),
        SummaryRow(
            "Default model",
            bool(prefill.model_id),
            prefill.model_id or "not selected",
        ),
        rag_row,
        SummaryRow(
            "Tools",
            bool(tools_on),
            f"{len(tools_on)} enabled" if tools_on else "all off (default)",
        ),
        SummaryRow(
            "Notes sync",
            notes_on,
            prefill.sync_directory if notes_on else "off",
        ),
        SummaryRow(
            "Theme", bool(prefill.default_theme), prefill.default_theme or "default"
        ),
        SummaryRow(
            "Key encryption", encryption_on, "" if encryption_on else "off"
        ),
    )
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `pytest Tests/Wizards/test_first_run_setup_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py Tests/Wizards/test_first_run_setup_state.py
git commit -m "feat: wizard prefill readers (secret presence only) and summary matrix"
```

---

### Task 5: Wizard skeleton — screen, SetupWizardContainer, Welcome step, Esc confirm

**Files:**
- Create: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Modify: `tldw_chatbook/css/features/_wizards.tcss` (append)
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (create)

**Interfaces:**
- Consumes: `WizardScreen`, `WizardContainer`, `WizardStep`, `WizardStepConfig`, `WizardNavigation`, `WizardProgress` from `tldw_chatbook.UI.Wizards.BaseWizard` (`WizardScreen`/`WizardStepConfig` are NOT in the package `__init__` — import from `.BaseWizard` directly). `active_step_ids`, `TRACK_*`, `STEP_*`, `build_wizard_state_commit`, `commit_sections_allowed` from Task 3. `ConfirmationDialog` from `tldw_chatbook.Widgets.confirmation_dialog` (dismisses `True`/`False`).
- Produces:
  - `class FirstRunSetupWizard(WizardScreen)` — constructor `FirstRunSetupWizard(app_instance, rerun: bool = False)`; dismisses `dict | None`: `None` = cancelled/finish-later, `{"completed": True, "exit_route": str | None}` on finish or explicit skip.
  - `class SetupWizardContainer(WizardContainer)` — internal; navigates over active step ids; `commit_config(self, section_values: dict) -> bool` is the ONLY config write path for steps (serialized worker).
  - `class SetupStep(WizardStep)` — base for all wizard steps with `async def commit(self) -> tuple[bool, str]` (default `(True, "")`).

**Framework facts the implementation must respect** (verified against `BaseWizard.py`):
- `handle_next()`/`complete_wizard()` read `get_step_data()`, not `get_data()` — steps override `get_step_data()`.
- `complete_wizard()` does NOT dismiss — the `on_complete` callback must dismiss the screen.
- Steps are fixed at construction (`steps=` list); there is no `add_step()`.
- `WizardProgress` has no reactive watchers — on track change, remove and re-mount a fresh one.
- `WizardContainer.on_mount` calls `show_step(0)` then `set_timer(0.1, self.validate_step)` — Pilot tests must `await pilot.pause(0.15)` after mount.
- `WizardScreen.__init__` swallows kwargs into `self.wizard_kwargs` and its Escape binding calls `self.dismiss(None)` — override `action_cancel` on BOTH screen and container.

- [ ] **Step 1: Write the failing Pilot tests**

Create `Tests/Wizards/test_first_run_setup_wizard.py`:

```python
"""Pilot tests for the first-run setup wizard skeleton."""

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SUMMARY,
    TRACK_FULL,
    TRACK_QUICK,
)


class _HostApp(App):
    def __init__(self, wizard: FirstRunSetupWizard):
        super().__init__()
        self._wizard = wizard
        self.wizard_result = "UNSET"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self._capture)

    def _capture(self, result) -> None:
        self.wizard_result = result


def _make_wizard(**kwargs) -> FirstRunSetupWizard:
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance, **kwargs)
    return wizard


@pytest.mark.asyncio
async def test_welcome_track_choice_activates_quick_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_QUICK)
        assert STEP_PROVIDER in container.active_ids
        assert STEP_RAG not in container.active_ids
        assert container.active_ids[-1] == STEP_SUMMARY


@pytest.mark.asyncio
async def test_welcome_full_track_activates_all_non_conditional_steps():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(TRACK_FULL)
        assert STEP_RAG in container.active_ids


@pytest.mark.asyncio
async def test_escape_asks_for_confirmation_instead_of_dismissing():
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        # The wizard must still be open (confirm dialog on top), not dismissed.
        assert app.wizard_result == "UNSET"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: FAIL with `ModuleNotFoundError: ... FirstRunSetupWizard`.

- [ ] **Step 3: Implement the skeleton**

Create `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`:

```python
"""First-run setup wizard: hermes-agent's setup process in chatbook chrome.

Screen + container subclass over BaseWizard (which is never modified).
All decisions and config mutations are built by first_run_setup_state;
this module renders them and owns persistence via one exclusive worker.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static, Switch

from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardContainer,
    WizardNavigation,
    WizardProgress,
    WizardScreen,
    WizardStep,
    WizardStepConfig,
)
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class SetupStep(WizardStep):
    """Base step: adds an awaitable commit hook and an inline error line."""

    async def commit(self) -> tuple[bool, str]:
        """Persist this step's data. Return (ok, error_message)."""
        return True, ""

    def show_step_error(self, message: str) -> None:
        try:
            self.query_one(".setup-step-error", Static).update(message)
        except Exception:
            logger.warning("Setup step error had nowhere to render: {}", message)


class WelcomeStep(SetupStep):
    """Track choice: Quick / Full / Skip."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-welcome"):
            yield Static("Welcome to tldw chatbook", classes="setup-title")
            yield Static(
                "Let's get you set up. Pick a path — everything here can be "
                "changed later in Settings, and every step can be skipped.",
                classes="setup-subtitle",
            )
            with RadioSet(id="setup-track-choice"):
                yield RadioButton(
                    "Quick setup — provider & model (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield RadioButton("Full setup — configure everything", id="setup-track-full")
            yield Button(
                "Skip — explore on my own", id="setup-skip-entirely", variant="default"
            )
            yield Static("", classes="setup-step-error")

    def get_step_data(self) -> Dict[str, Any]:
        return {"track": self.chosen_track()}

    def chosen_track(self) -> str:
        try:
            full = self.query_one("#setup-track-full", RadioButton).value
        except Exception:
            full = False
        return wizard_state.TRACK_FULL if full else wizard_state.TRACK_QUICK


class SetupWizardContainer(WizardContainer):
    """Navigates over the active-step subset; commits on Next via one worker."""

    def __init__(self, app_instance, rerun: bool = False, **kwargs):
        self.rerun = rerun
        self.key_entered = False
        self.track = wizard_state.TRACK_FULL
        steps = self._create_steps()
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title="Set up tldw chatbook",
            on_complete=self._handle_complete,
            **kwargs,
        )
        self.active_ids: tuple[str, ...] = wizard_state.active_step_ids(
            self.track, key_entered=self.key_entered
        )
        self._advancing = False

    # -- step construction -------------------------------------------------
    def _create_steps(self) -> List[WizardStep]:
        # Later tasks append real steps here; the skeleton ships Welcome +
        # placeholder SetupSteps so navigation is testable end to end.
        def cfg(step_id: str, title: str, number: int) -> WizardStepConfig:
            return WizardStepConfig(id=step_id, title=title, step_number=number)

        return [
            WelcomeStep(wizard=self, config=cfg(wizard_state.STEP_WELCOME, "Welcome", 1)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_PROVIDER, "Provider", 2)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_MODEL, "Model", 3)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_RAG, "RAG", 4)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_TOOLS, "Tools", 5)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_NOTES, "Notes sync", 6)),
            SetupStep(
                wizard=self, config=cfg(wizard_state.STEP_APPEARANCE, "Appearance", 7)
            ),
            SetupStep(
                wizard=self, config=cfg(wizard_state.STEP_PROTECT, "Protect keys", 8)
            ),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_SUMMARY, "Summary", 9)),
        ]

    # -- active-step navigation --------------------------------------------
    def select_track(self, track: str) -> None:
        """Recompute the active subset after the Welcome choice."""
        self.track = track
        self._refresh_active_ids()

    def note_key_entered(self) -> None:
        if not self.key_entered:
            self.key_entered = True
            self._refresh_active_ids()

    def _refresh_active_ids(self) -> None:
        self.active_ids = wizard_state.active_step_ids(
            self.track, key_entered=self.key_entered
        )
        self._rebuild_progress()

    def _step_index_for_id(self, step_id: str) -> Optional[int]:
        for index, step in enumerate(self.steps):
            if step.config and step.config.id == step_id:
                return index
        return None

    def _active_position(self, absolute_index: int) -> int:
        step = self.steps[absolute_index]
        step_id = step.config.id if step.config else ""
        return self.active_ids.index(step_id) if step_id in self.active_ids else 0

    def _next_active_index(self, absolute_index: int) -> Optional[int]:
        position = self._active_position(absolute_index)
        if position + 1 >= len(self.active_ids):
            return None
        return self._step_index_for_id(self.active_ids[position + 1])

    def _previous_active_index(self, absolute_index: int) -> Optional[int]:
        position = self._active_position(absolute_index)
        if position <= 0:
            return None
        return self._step_index_for_id(self.active_ids[position - 1])

    def update_progress(self) -> None:
        """Recount against the ACTIVE subset, not the full step list."""
        try:
            position = self._active_position(self.current_step or 0)
            nav = self.query_one(".wizard-navigation", WizardNavigation)
            nav.total_steps = len(self.active_ids)
            nav.current_step = position + 1
            nav.can_go_back = position > 0
            nav.can_go_forward = self.can_proceed
        except Exception:
            pass

    def _rebuild_progress(self) -> None:
        # WizardProgress has no watchers; replace it wholesale on track change.
        try:
            old = self.query_one(".wizard-progress", WizardProgress)
            parent = old.parent
            old.remove()
            fresh = WizardProgress(classes="wizard-progress")
            fresh.total_steps = len(self.active_ids)
            fresh.current_step = self._active_position(self.current_step or 0) + 1
            fresh.step_titles = [
                self.steps[self._step_index_for_id(step_id)].config.title
                for step_id in self.active_ids
                if self._step_index_for_id(step_id) is not None
            ]
            if parent is not None:
                parent.mount(fresh)
        except Exception:
            logger.debug("Wizard progress rebuild skipped", exc_info=True)

    # -- commit-on-Next ----------------------------------------------------
    @on(Button.Pressed, "#wizard-next")
    def handle_next(self) -> None:  # overrides base; guard prevents double fire
        if self._advancing or not self.can_proceed:
            return
        self._advancing = True
        self.run_worker(self._advance(), exclusive=True, group="setup-wizard-advance")

    async def _advance(self) -> None:
        try:
            step = self.steps[self.current_step]
            if isinstance(step, SetupStep):
                ok, error = await step.commit()
                if not ok:
                    step.show_step_error(f"{error}  (Retry, or Skip this step.)")
                    return
            if isinstance(step, WelcomeStep):
                self.select_track(step.chosen_track())
            step_id = step.config.id if step.config else f"step_{self.current_step}"
            self.wizard_data[step_id] = step.get_step_data()
            step.is_complete = True
            next_index = self._next_active_index(self.current_step)
            if next_index is None:
                self.complete_wizard()
            else:
                self.show_step(next_index)
        finally:
            self._advancing = False

    @on(Button.Pressed, "#wizard-back")
    def handle_back(self) -> None:
        previous = self._previous_active_index(self.current_step)
        if previous is not None:
            self.show_step(previous)

    # -- explicit whole-wizard skip ---------------------------------------
    @on(Button.Pressed, "#setup-skip-entirely")
    def handle_skip_entirely(self) -> None:
        self.run_worker(self._skip_entirely(), exclusive=True, group="setup-wizard-advance")

    async def _skip_entirely(self) -> None:
        await self.commit_config(
            wizard_state.build_wizard_state_commit(completed=True)
        )
        self._dismiss_screen({"completed": True, "exit_route": None})

    # -- persistence (the only write path for steps) -----------------------
    async def commit_config(self, section_values: dict) -> bool:
        """Serialize every config write through one worker-side call."""
        if not section_values:
            return True
        if not wizard_state.commit_sections_allowed(section_values):
            logger.error("Wizard commit rejected non-owned sections: {}", list(section_values))
            return False
        import asyncio

        from tldw_chatbook.config import save_settings_to_cli_config

        def _write() -> bool:
            return save_settings_to_cli_config(section_values)

        ok = await asyncio.get_running_loop().run_in_executor(None, _write)
        if ok:
            self._mirror_into_app_config(section_values)
        return ok

    def _mirror_into_app_config(self, section_values: dict) -> None:
        """Keep the in-memory app_config consistent (chat_screen.py pattern)."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        for dotted_section, values in section_values.items():
            target = app_config
            for part in dotted_section.split("."):
                nxt = target.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    target[part] = nxt
                target = nxt
            target.update(values)

    # -- completion / cancel ----------------------------------------------
    def _handle_complete(self, wizard_data: Dict[str, Any]) -> None:
        summary_data = wizard_data.get(wizard_state.STEP_SUMMARY, {})
        exit_route = summary_data.get("exit_route")
        self.run_worker(
            self._finalize(exit_route), exclusive=True, group="setup-wizard-advance"
        )

    async def _finalize(self, exit_route: Optional[str]) -> None:
        await self.commit_config(wizard_state.build_wizard_state_commit(completed=True))
        self._dismiss_screen({"completed": True, "exit_route": exit_route})

    def _dismiss_screen(self, result: Optional[dict]) -> None:
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.dismiss(result)

    def action_cancel(self) -> None:
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.action_cancel()


class FirstRunSetupWizard(WizardScreen):
    """Full-screen first-run setup wizard. Dismisses dict | None."""

    def __init__(self, app_instance, rerun: bool = False):
        super().__init__(app_instance)
        self.rerun = rerun

    def compose(self) -> ComposeResult:
        yield SetupWizardContainer(self.app_instance, rerun=self.rerun)

    def on_mount(self) -> None:
        if not self.rerun:
            self._persist_started_flag()

    @work(thread=True, group="setup-wizard-started-flag")
    def _persist_started_flag(self) -> None:
        from tldw_chatbook.config import save_settings_to_cli_config

        try:
            save_settings_to_cli_config(
                wizard_state.build_wizard_state_commit(started=True)
            )
        except Exception as exc:
            logger.warning("Failed to persist wizard started flag: {}", exc)
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            app_config.setdefault(wizard_state.WIZARD_STATE_SECTION, {})[
                wizard_state.SETUP_STARTED_KEY
            ] = True

    def action_cancel(self) -> None:
        dialog = ConfirmationDialog(
            title="Finish setup later?",
            message=(
                "Steps you've already completed are saved. You can finish "
                "setup any time from Settings ▸ Diagnostics."
            ),
            confirm_label="Finish later",
            cancel_label="Keep going",
        )
        self.app.push_screen(dialog, self._handle_cancel_confirm)

    def _handle_cancel_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self.dismiss(None)
```

- [ ] **Step 4: Append wizard CSS**

Append to `tldw_chatbook/css/features/_wizards.tcss`:

```css
/* ── First-run setup wizard ─────────────────────────────────────────── */
.setup-welcome { padding: 2 4; }
.setup-title { text-style: bold; text-align: center; margin-bottom: 1; }
.setup-subtitle { color: $text-muted; margin-bottom: 2; }
.setup-step-error { color: $error; margin-top: 1; }
.setup-field-label { margin-top: 1; }
.setup-probe-status { color: $text-muted; margin-top: 1; }
.setup-summary-row { height: 1; }
.setup-summary-ok { color: $success; }
.setup-summary-missing { color: $error; }
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: PASS (3/3). If the escape test fails because the container's escape binding fires `WizardContainer.action_cancel` from the base class binding table, the container override above handles it — debug from there, do not touch BaseWizard.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: first-run wizard skeleton — track branching, commit-on-Next, esc confirm"
```

---

### Task 6: Provider step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append)

**Interfaces:**
- Consumes:
  - `supported_console_provider_catalog()` from `tldw_chatbook.Chat.console_provider_support` — returns `ConsoleProviderCatalogEntry` tuples with `readiness_key`, `display_name`, `requires_api_key` (grouping rule: `requires_api_key` ⇒ Cloud; keys in `PROVIDER_CUSTOM_GROUP_KEYS` ⇒ Custom; else Local — mirror `settings_screen.py:6423`).
  - `discover_local_servers(app_config, *, http_client=None, timeout=2.5)` and `DiscoveredLocalServer` from `tldw_chatbook.Chat.local_server_discovery` (async).
  - `probe_settings_endpoint(base_url, *, timeout=..., http_client=...)` from `tldw_chatbook.UI.Screens.settings_endpoint_probe` — NO auth param; for cloud checks pass `http_client=httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"})` and close it after.
  - `read_provider_secret_presence`, `build_provider_commit`, `invalidate_model_for_provider_change` from the state module.
  - Provider-value form for `chat_defaults.provider`: read `chat_screen.py:9137` (`_apply_detected_local_server`) once and mirror the exact string form it persists — the step stores that value in `self.provider_value_for_chat_defaults`.
- Produces: `class ProviderStep(SetupStep)` with constructor `ProviderStep(wizard, config, *, discover=discover_local_servers, probe=probe_settings_endpoint, environ=os.environ)` (injected seams for tests); `get_step_data()` returns `{"provider_key": str, "provider_value": str, "entered_key": bool}`.

**Behavior (all from the spec):** grouped provider list (Cloud/Local); masked `Input(password=True)` for the key; "Found in your environment ✓ — nothing to store" when the provider's `api_key_env_var` is set; Keep/Replace/Clear buttons when a key is already configured (never showing the value); a one-click row per discovered local server; async probe with generation token, 8.0s cloud budget, and "Couldn't verify — save anyway" (validation never blocks Next); commit via `wizard.commit_config(build_provider_commit(...))` merged through `invalidate_model_for_provider_change` when the provider changed since the last commit.

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProviderStep
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig


def _provider_step(wizard=None, environ=None, discover=None, probe=None):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = wizard or SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    return ProviderStep(
        wizard=wizard,
        config=WizardStepConfig(id="provider", title="Provider", step_number=2),
        discover=discover or AsyncMock(return_value=()),
        probe=probe or AsyncMock(),
        environ=environ or {},
    )


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


@pytest.mark.asyncio
async def test_provider_step_env_key_shows_found_in_environment():
    step = _provider_step(environ={"OPENAI_API_KEY": "sk-x"})
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        status = step.query_one("#setup-provider-key-status", Static)
        assert "environment" in str(status.render()).lower()


@pytest.mark.asyncio
async def test_provider_step_stale_probe_result_is_discarded():
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        generation_before = step.probe_generation
        step.select_provider("anthropic")
        # A result stamped with the old generation must not render.
        step.apply_probe_result(generation_before, reachable=True, summary="stale ok")
        status = step.query_one("#setup-provider-probe-status", Static)
        assert "stale ok" not in str(status.render())


@pytest.mark.asyncio
async def test_provider_step_commit_writes_key_and_notes_key_entered():
    from unittest.mock import AsyncMock

    from types import SimpleNamespace

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        note_key_entered=MagicMock(),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _provider_step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        step.query_one("#setup-provider-key-input", Input).value = "sk-new"
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed["api_settings.openai"]["api_key"] == "sk-new"
        wizard.note_key_entered.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v -k provider`
Expected: FAIL (ImportError: `ProviderStep`).

- [ ] **Step 3: Implement `ProviderStep`**

Add to `FirstRunSetupWizard.py` (and replace the placeholder `SetupStep` for `STEP_PROVIDER` in `_create_steps` with `ProviderStep(wizard=self, config=cfg(...), environ=os.environ)`):

```python
CLOUD_PROBE_TIMEOUT_SECONDS = 8.0


class ProviderStep(SetupStep):
    """Choose a provider, supply credentials, verify without blocking."""

    def __init__(self, wizard=None, config=None, *, discover=None, probe=None,
                 environ=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        from tldw_chatbook.Chat.local_server_discovery import discover_local_servers
        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            probe_settings_endpoint,
        )

        self._discover = discover or discover_local_servers
        self._probe = probe or probe_settings_endpoint
        self._environ = dict(environ) if environ is not None else dict(os.environ)
        self.probe_generation = 0
        self.selected_provider_key: str = ""
        self.provider_value_for_chat_defaults: str = ""
        self._last_committed_provider_value: Optional[str] = None
        self._entered_key = False

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Chat.console_provider_support import (
            supported_console_provider_catalog,
        )

        entries = supported_console_provider_catalog()
        with Vertical(classes="setup-provider"):
            yield Static("Connect a provider", classes="setup-title")
            yield Static(
                "Cloud providers need an API key. Local servers just need to "
                "be running — we'll look for them.",
                classes="setup-subtitle",
            )
            with RadioSet(id="setup-provider-choice"):
                for entry in self._grouped(entries):
                    yield RadioButton(
                        entry.display_name, id=f"setup-provider-{entry.readiness_key}"
                    )
            yield Static("", id="setup-provider-detected", classes="setup-probe-status")
            yield Button(
                "Use this server", id="setup-provider-use-detected",
                classes="hidden", variant="primary",
            )
            yield Label("API key", classes="setup-field-label")
            yield Input(password=True, id="setup-provider-key-input",
                        placeholder="Paste your API key")
            yield Static("", id="setup-provider-key-status", classes="setup-probe-status")
            with Horizontal(id="setup-provider-key-actions", classes="hidden"):
                yield Button("Keep current", id="setup-provider-key-keep")
                yield Button("Replace", id="setup-provider-key-replace")
                yield Button("Clear", id="setup-provider-key-clear")
            yield Static("", id="setup-provider-probe-status", classes="setup-probe-status")
            yield Static("", classes="setup-step-error")

    @staticmethod
    def _grouped(entries):
        from tldw_chatbook.Chat.provider_catalog import (
            PROVIDER_CUSTOM_GROUP_KEYS,
        )

        def group_rank(entry):
            if entry.readiness_key in PROVIDER_CUSTOM_GROUP_KEYS:
                return 2
            return 0 if entry.requires_api_key else 1

        return sorted(entries, key=lambda e: (group_rank(e), e.display_name.lower()))

    def on_show(self) -> None:
        super().on_show()
        self._start_discovery()

    def _start_discovery(self) -> None:
        self.run_worker(self._discover_servers(), exclusive=True,
                        group="setup-provider-discovery")

    async def _discover_servers(self) -> None:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        try:
            servers = tuple(await self._discover(app_config) or ())
        except Exception:
            logger.debug("Wizard local discovery failed", exc_info=True)
            return
        if not servers:
            return
        self.detected_server = servers[0]
        self.query_one("#setup-provider-detected", Static).update(
            f"Found a local server at {self.detected_server.base_url} "
            f"({self.detected_server.provider_key})."
        )
        use_button = self.query_one("#setup-provider-use-detected", Button)
        use_button.remove_class("hidden")

    @on(Button.Pressed, "#setup-provider-use-detected")
    def _on_use_detected(self) -> None:
        """One-click connect: adopt the discovered server as the provider."""
        server = getattr(self, "detected_server", None)
        if server is None:
            return
        self.select_provider(server.provider_key)
        self.detected_base_url = server.base_url
        self.query_one("#setup-provider-detected", Static).update(
            f"✓ Using {server.base_url} ({server.provider_key})."
        )

    def select_provider(self, provider_key: str) -> None:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            read_provider_secret_presence,
        )

        self.selected_provider_key = provider_key
        self.probe_generation += 1
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        presence = read_provider_secret_presence(
            app_config, self._environ, provider_key=provider_key
        )
        status = self.query_one("#setup-provider-key-status", Static)
        actions = self.query_one("#setup-provider-key-actions", Horizontal)
        key_input = self.query_one("#setup-provider-key-input", Input)
        if presence.env_var_set:
            status.update(f"Found {presence.env_var} in your environment ✓ — nothing to store.")
            key_input.display = False
            actions.add_class("hidden")
        elif presence.configured:
            status.update("An API key is already configured for this provider.")
            key_input.display = False
            actions.remove_class("hidden")
        else:
            status.update("")
            key_input.display = True
            actions.add_class("hidden")
        self.query_one("#setup-provider-probe-status", Static).update("")

    @on(RadioSet.Changed, "#setup-provider-choice")
    def _on_provider_chosen(self, event: RadioSet.Changed) -> None:
        pressed_id = event.pressed.id or ""
        self.select_provider(pressed_id.removeprefix("setup-provider-"))

    @on(Button.Pressed, "#setup-provider-key-replace")
    def _on_replace(self) -> None:
        self.query_one("#setup-provider-key-input", Input).display = True

    @on(Input.Submitted, "#setup-provider-key-input")
    def _on_key_submitted(self, event: Input.Submitted) -> None:
        """Live-but-never-blocking verification: probe on Enter in the key field."""
        if event.value.strip():
            self._launch_probe(api_key=event.value.strip())

    def _launch_probe(self, *, api_key: str | None = None) -> None:
        self.probe_generation += 1
        generation = self.probe_generation
        base_url = getattr(self, "detected_base_url", None)
        self.query_one("#setup-provider-probe-status", Static).update("Testing…")
        self.run_worker(
            self._run_probe(generation, base_url=base_url, api_key=api_key),
            exclusive=True,
            group="setup-provider-probe",
        )

    async def _run_probe(
        self, generation: int, *, base_url: str | None, api_key: str | None
    ) -> None:
        import httpx

        # Local servers probe their own base URL; cloud keys probe the
        # provider's OpenAI-compatible endpoint with the key as a bearer
        # header via the http_client seam (probe_settings_endpoint has no
        # auth parameter by design). Providers without a known compatible
        # endpoint resolve to "couldn't verify — save anyway".
        target = base_url or self._cloud_probe_base_url(self.selected_provider_key)
        if not target:
            self.apply_probe_result(
                generation, reachable=False, summary="No test endpoint for this provider."
            )
            return
        client = None
        try:
            if api_key:
                client = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {api_key}"}
                )
            outcome = await self._probe(
                target,
                timeout=CLOUD_PROBE_TIMEOUT_SECONDS if api_key else 2.5,
                http_client=client,
            )
            self.apply_probe_result(
                generation, reachable=outcome.reachable, summary=outcome.summary
            )
        except Exception:
            logger.debug("Wizard provider probe failed", exc_info=True)
            self.apply_probe_result(
                generation, reachable=False, summary="Probe errored."
            )
        finally:
            if client is not None:
                await client.aclose()

    @staticmethod
    def _cloud_probe_base_url(provider_key: str) -> str:
        """OpenAI-compatible base URLs for cloud-key verification (v1 fence:
        only providers with a known compatible /v1/models endpoint)."""
        return {
            "openai": "https://api.openai.com",
            "openrouter": "https://openrouter.ai/api",
            "groq": "https://api.groq.com/openai",
            "deepseek": "https://api.deepseek.com",
            "mistral": "https://api.mistral.ai",
        }.get(provider_key, "")

    def apply_probe_result(self, generation: int, *, reachable: bool, summary: str) -> None:
        """Render a probe outcome only if it is still current (no stale ✓)."""
        if generation != self.probe_generation:
            return
        prefix = "✓ " if reachable else "✗ "
        suffix = "" if reachable else "  Couldn't verify — you can save anyway."
        self.query_one("#setup-provider-probe-status", Static).update(
            f"{prefix}{summary}{suffix}"
        )

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_provider_commit,
            invalidate_model_for_provider_change,
        )

        if not self.selected_provider_key:
            return True, ""  # skipping the step entirely is legal
        key_input = self.query_one("#setup-provider-key-input", Input)
        api_key = key_input.value.strip() if key_input.display and key_input.value else None
        commit = build_provider_commit(
            provider_key=self.selected_provider_key,
            api_key=api_key,
            api_url=getattr(self, "detected_base_url", None),
        )
        # Resolve the exact chat_defaults.provider value form the same way
        # chat_screen._apply_detected_local_server does (read it once; mirror it).
        self.provider_value_for_chat_defaults = self._display_value_for(
            self.selected_provider_key
        )
        commit = invalidate_model_for_provider_change(
            commit,
            previous_provider_value=self._last_committed_provider_value,
            new_provider_value=self.provider_value_for_chat_defaults,
        )
        ok = await self.wizard.commit_config(commit)
        if not ok:
            return False, "Saving the provider settings failed."
        self._last_committed_provider_value = self.provider_value_for_chat_defaults
        if api_key:
            self._entered_key = True
            self.wizard.note_key_entered()
        return True, ""

    @staticmethod
    def _display_value_for(provider_key: str) -> str:
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        return provider_display_name(provider_key)

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "provider_key": self.selected_provider_key,
            "provider_value": self.provider_value_for_chat_defaults,
            "entered_key": self._entered_key,
        }
```

Before finishing: open `tldw_chatbook/UI/Screens/chat_screen.py:9137` (`_apply_detected_local_server`) and confirm what string form it writes to `chat_defaults.provider` (display name vs readiness key). If it differs from `provider_display_name(...)`, change `_display_value_for` to match it exactly and note the finding in the backlog task.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wizard provider step — grouped picker, env detection, tokened probes"
```

---

### Task 7: Model step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append)

**Interfaces:**
- Consumes: `get_cli_providers_and_models()` from `tldw_chatbook.config` (curated fallback); `app_instance.llm_provider_catalog_scope_service.discover_models(mode="local", provider=..., staged_settings=...)` (async, returns object with `.status`/`.models` — see `settings_screen.py:7079` for the exact call shape); `ProviderStep` data via `wizard.wizard_data[STEP_PROVIDER]` (`provider_key`, `provider_value`); `build_model_commit`.
- Produces: `class ModelStep(SetupStep)` — constructor `ModelStep(wizard, config, *, discover_models=None)` (injectable); `get_step_data()` returns `{"model_id": str}`.

**Behavior:** on `on_show()`, read the provider from `wizard.wizard_data`; if it changed since last shown, clear the current selection (the UI half of dependency invalidation — the config half already happened in ProviderStep's commit). Fetch models: try the scope service (worker, 8s guard, injectable), fall back to `get_cli_providers_and_models().get(provider_value, [])`, always include an "enter custom name" `Input`. Commit `build_model_commit(provider_value=..., model_id=...)`; empty selection commits nothing (skip-safe).

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ModelStep


def _model_step(wizard, discover_models=None):
    from unittest.mock import AsyncMock

    return ModelStep(
        wizard=wizard,
        config=WizardStepConfig(id="model", title="Model", step_number=3),
        discover_models=discover_models or AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_model_step_provider_change_resets_selection():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        step.set_selected_model("gpt-5.6-terra")
        assert step.selected_model_id == "gpt-5.6-terra"
        wizard.wizard_data["provider"] = {
            "provider_key": "anthropic", "provider_value": "Anthropic",
        }
        step.on_show()
        assert step.selected_model_id == ""


@pytest.mark.asyncio
async def test_model_step_commit_writes_chat_defaults():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data={"provider": {"provider_key": "openai", "provider_value": "OpenAI"}},
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = _model_step(wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        step.set_selected_model("gpt-5.6-terra")
        ok, error = await step.commit()
        assert ok, error
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}
        }
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v -k model`
Expected: FAIL (ImportError: `ModelStep`).

- [ ] **Step 3: Implement `ModelStep`**

```python
class ModelStep(SetupStep):
    """Pick a default model for the chosen provider."""

    def __init__(self, wizard=None, config=None, *, discover_models=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._discover_models = discover_models
        self._shown_for_provider: Optional[str] = None
        self.selected_model_id: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-model"):
            yield Static("Pick a default model", classes="setup-title")
            yield Static("", id="setup-model-provider-line", classes="setup-subtitle")
            with RadioSet(id="setup-model-choice"):
                yield RadioButton("(loading models…)", id="setup-model-loading")
            yield Label("Or enter a model name", classes="setup-field-label")
            yield Input(id="setup-model-custom", placeholder="model-id")
            yield Static("", classes="setup-step-error")

    def _current_provider(self) -> tuple[str, str]:
        data = (self.wizard.wizard_data or {}).get(
            wizard_state.STEP_PROVIDER, {}
        )
        return str(data.get("provider_key", "")), str(data.get("provider_value", ""))

    def on_show(self) -> None:
        super().on_show()
        provider_key, provider_value = self._current_provider()
        if provider_key != self._shown_for_provider:
            self.selected_model_id = ""
            self._shown_for_provider = provider_key
            try:
                self.query_one("#setup-model-custom", Input).value = ""
            except Exception:
                pass
        try:
            self.query_one("#setup-model-provider-line", Static).update(
                f"Models for {provider_value or 'your provider'}."
            )
        except Exception:
            pass
        if provider_key:
            self.run_worker(self._load_models(provider_key, provider_value),
                            exclusive=True, group="setup-model-load")

    async def _load_models(self, provider_key: str, provider_value: str) -> None:
        models: list[str] = []
        discover = self._discover_models
        if discover is None:
            service = getattr(
                self.wizard.app_instance, "llm_provider_catalog_scope_service", None
            )
            if service is not None:
                async def discover(pk=provider_key, svc=service):
                    result = await svc.discover_models(mode="local", provider=pk,
                                                       staged_settings=None)
                    if str(getattr(result, "status", "")) == "success":
                        return list(getattr(result, "models", ()) or ())
                    return []
        if discover is not None:
            try:
                models = list(await discover(provider_key))
            except Exception:
                logger.debug("Wizard model discovery failed", exc_info=True)
        if not models:
            from tldw_chatbook.config import get_cli_providers_and_models

            models = list(get_cli_providers_and_models().get(provider_value, []))
        self._render_models(models[:20])

    def _render_models(self, models: list[str]) -> None:
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        radio_set.remove_children()
        for index, model_id in enumerate(models):
            radio_set.mount(RadioButton(model_id, id=f"setup-model-option-{index}"))
        if not models:
            radio_set.mount(
                RadioButton("(no models found — enter one below)", disabled=True)
            )

    @on(RadioSet.Changed, "#setup-model-choice")
    def _on_model_chosen(self, event: RadioSet.Changed) -> None:
        self.set_selected_model(str(event.pressed.label))

    @on(Input.Changed, "#setup-model-custom")
    def _on_custom_model(self, event: Input.Changed) -> None:
        if event.value.strip():
            self.selected_model_id = event.value.strip()

    def set_selected_model(self, model_id: str) -> None:
        self.selected_model_id = model_id

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_model_commit

        _, provider_value = self._current_provider()
        if not (provider_value and self.selected_model_id):
            return True, ""  # skip-safe
        ok = await self.wizard.commit_config(
            build_model_commit(
                provider_value=provider_value, model_id=self.selected_model_id
            )
        )
        return (True, "") if ok else (False, "Saving the model choice failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"model_id": self.selected_model_id}
```

Replace the `STEP_MODEL` placeholder in `_create_steps` with `ModelStep(wizard=self, config=cfg(...))`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wizard model step with discovery, curated fallback, provider-change reset"
```

---

### Task 8: RAG step and Tools step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append)

**Interfaces:**
- Consumes: `embeddings_rag_deps_installed()` from `tldw_chatbook.Utils.optional_deps` (cheap find_spec probe — do NOT use `check_embeddings_rag_deps()` or read `DEPENDENCIES_AVAILABLE` directly); `[embedding_config].models` sub-tables from app_config for choices; `gateable_builtin_tools()` from `tldw_chatbook.Agents.tool_catalog` (rows have `.gate_key`, `.tool_name`); `build_rag_commit`, `build_tools_commit`.
- Produces: `class RagStep(SetupStep)` (constructor `RagStep(wizard, config, *, deps_installed=embeddings_rag_deps_installed)`), `class ToolsStep(SetupStep)`.

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import RagStep, ToolsStep


@pytest.mark.asyncio
async def test_rag_step_missing_deps_shows_install_copy_and_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="RAG", step_number=4),
        deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        body = str(step.query_one("#setup-rag-status", Static).render())
        assert "tldw_chatbook[embeddings_rag]" in body
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_tools_step_commits_only_changed_gates():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
        assert switches, "tools step must render one switch per gateable tool"
        assert all(sw.value is False for sw in switches)  # default OFF
        switches[0].value = True
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["tools"][step.gate_key_for(switches[0])] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v -k "rag or tools"`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement both steps**

```python
class RagStep(SetupStep):
    """RAG/embeddings: report dep status; pick a default embedding model."""

    def __init__(self, wizard=None, config=None, *, deps_installed=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if deps_installed is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps_installed = embeddings_rag_deps_installed
        self._deps_installed = deps_installed
        self.selected_embedding_model: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-rag"):
            yield Static("Search & RAG", classes="setup-title")
            yield Static("", id="setup-rag-status", classes="setup-subtitle")
            with RadioSet(id="setup-rag-model-choice"):
                for model_id in self._embedding_model_ids():
                    yield RadioButton(model_id)
            yield Static("", classes="setup-step-error")

    def _embedding_model_ids(self) -> list[str]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        embedding_config = app_config.get("embedding_config", {})
        models = embedding_config.get("models", {}) if isinstance(embedding_config, dict) else {}
        return sorted(models) if isinstance(models, dict) else []

    def on_mount(self) -> None:
        status = self.query_one("#setup-rag-status", Static)
        if self._deps_installed():
            status.update("Embedding dependencies are installed. Pick a default model, or skip.")
        else:
            status.update(
                "RAG needs optional dependencies that aren't installed. Install the "
                "extras package `tldw_chatbook[embeddings_rag]` with your package "
                "manager, then revisit Settings ▸ RAG. Skipping for now is fine."
            )
            try:
                self.query_one("#setup-rag-model-choice", RadioSet).disabled = True
            except Exception:
                pass

    @on(RadioSet.Changed, "#setup-rag-model-choice")
    def _on_model(self, event: RadioSet.Changed) -> None:
        self.selected_embedding_model = str(event.pressed.label)

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_rag_commit

        if not (self._deps_installed() and self.selected_embedding_model):
            return True, ""
        ok = await self.wizard.commit_config(
            build_rag_commit(default_model_id=self.selected_embedding_model)
        )
        return (True, "") if ok else (False, "Saving the embedding model failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"embedding_model": self.selected_embedding_model}


class ToolsStep(SetupStep):
    """Enable built-in tools (all default OFF; risk-tagged ones still ask per call)."""

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Agents.tool_catalog import gateable_builtin_tools

        self._entries = list(gateable_builtin_tools())
        with Vertical(classes="setup-tools"):
            yield Static("Built-in tools", classes="setup-title")
            yield Static(
                "Everything is off by default. Tools that read or change your "
                "files still show an approval card every time they run.",
                classes="setup-subtitle",
            )
            for entry in self._entries:
                with Horizontal(classes="setup-tool-row"):
                    yield Switch(value=False, id=f"setup-tool-{entry.tool_name}")
                    yield Label(entry.tool_name.replace("_", " "))
            yield Static("", classes="setup-step-error")

    def gate_key_for(self, switch: Switch) -> str:
        tool_name = (switch.id or "").removeprefix("setup-tool-")
        for entry in self._entries:
            if entry.tool_name == tool_name:
                return entry.gate_key
        return ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_tools_commit

        gate_values: dict[str, bool] = {}
        for switch in self.query(Switch):
            gate_key = self.gate_key_for(switch)
            if gate_key and switch.value:  # only persist enables; absent == off
                gate_values[gate_key] = True
        if not gate_values:
            return True, ""
        ok = await self.wizard.commit_config(build_tools_commit(gate_values=gate_values))
        return (True, "") if ok else (False, "Saving tool settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"enabled_gates": [
            self.gate_key_for(sw) for sw in self.query(Switch) if sw.value
        ]}
```

Replace the `STEP_RAG` and `STEP_TOOLS` placeholders in `_create_steps`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wizard RAG and tools steps"
```

---

### Task 9: Notes sync step and Appearance step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append)

**Interfaces:**
- Consumes: `build_notes_commit`, `build_appearance_commit`, `read_wizard_prefill`; theme list from `self.app.available_themes` (Textual App API — dict of registered themes); splash card names from `tldw_chatbook.Utils.Splash_Screens.card_definitions.get_all_card_definitions()` (returns a dict keyed by card name).
- Produces: `class NotesSyncStep(SetupStep)`, `class AppearanceStep(SetupStep)`.

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import AppearanceStep, NotesSyncStep


@pytest.mark.asyncio
async def test_notes_step_commit_writes_directory_and_toggle():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-notes-enable", Switch).value = True
        step.query_one("#setup-notes-directory", Input).value = "~/MyNotes"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {
            "notes": {"sync_directory": "~/MyNotes", "auto_sync_enabled": True}
        }


@pytest.mark.asyncio
async def test_notes_step_disabled_commits_nothing():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = NotesSyncStep(
        wizard=wizard,
        config=WizardStepConfig(id="notes", title="Notes sync", step_number=6),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_appearance_step_commits_theme_and_card():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=7),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.selected_theme = "textual-light"
        step.selected_splash_card = "matrix"
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed["general"] == {"default_theme": "textual-light"}
        assert committed["splash_screen"] == {"card_selection": "matrix"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v -k "notes or appearance"`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement both steps**

```python
class NotesSyncStep(SetupStep):
    """Optional bidirectional notes sync: a directory and a toggle."""

    def compose(self) -> ComposeResult:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_wizard_prefill

        prefill = read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        with Vertical(classes="setup-notes"):
            yield Static("Notes sync", classes="setup-title")
            yield Static(
                "Keep a folder of Markdown files in sync with your notes. "
                "Skip if you only want in-app notes.",
                classes="setup-subtitle",
            )
            with Horizontal(classes="setup-tool-row"):
                yield Switch(value=prefill.auto_sync_enabled, id="setup-notes-enable")
                yield Label("Enable notes sync")
            yield Label("Notes directory", classes="setup-field-label")
            yield Input(
                value=prefill.sync_directory or "~/Documents/Notes",
                id="setup-notes-directory",
            )
            yield Static("", classes="setup-step-error")

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_notes_commit

        enabled = self.query_one("#setup-notes-enable", Switch).value
        directory = self.query_one("#setup-notes-directory", Input).value.strip()
        if not enabled:
            return True, ""
        if not directory:
            return False, "Pick a directory or turn sync off."
        ok = await self.wizard.commit_config(
            build_notes_commit(sync_directory=directory, auto_sync_enabled=True)
        )
        return (True, "") if ok else (False, "Saving notes sync settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"auto_sync_enabled": self.query_one("#setup-notes-enable", Switch).value}


class AppearanceStep(SetupStep):
    """Theme and splash card. Applies the theme live on commit (best effort)."""

    selected_theme: str = ""
    selected_splash_card: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-appearance"):
            yield Static("Appearance", classes="setup-title")
            yield Label("Theme", classes="setup-field-label")
            with RadioSet(id="setup-theme-choice"):
                for theme_name in self._theme_names():
                    yield RadioButton(theme_name)
            yield Label("Splash screen card", classes="setup-field-label")
            with RadioSet(id="setup-splash-choice"):
                yield RadioButton("Surprise me (random)", value=True)
                for card_name in self._card_names()[:10]:
                    yield RadioButton(card_name)
            yield Static("", classes="setup-step-error")

    def _theme_names(self) -> list[str]:
        try:
            return sorted(self.app.available_themes)
        except Exception:
            return ["textual-dark", "textual-light"]

    @staticmethod
    def _card_names() -> list[str]:
        try:
            from tldw_chatbook.Utils.Splash_Screens.card_definitions import (
                get_all_card_definitions,
            )

            return sorted(get_all_card_definitions())
        except Exception:
            return []

    @on(RadioSet.Changed, "#setup-theme-choice")
    def _on_theme(self, event: RadioSet.Changed) -> None:
        self.selected_theme = str(event.pressed.label)

    @on(RadioSet.Changed, "#setup-splash-choice")
    def _on_card(self, event: RadioSet.Changed) -> None:
        label = str(event.pressed.label)
        self.selected_splash_card = "" if label.startswith("Surprise me") else label

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_appearance_commit,
        )

        if not self.selected_theme and not self.selected_splash_card:
            return True, ""
        theme = self.selected_theme or "textual-dark"
        ok = await self.wizard.commit_config(
            build_appearance_commit(
                default_theme=theme,
                splash_card=self.selected_splash_card or None,
            )
        )
        if ok and self.selected_theme:
            try:
                self.app.theme = self.selected_theme
            except Exception:
                logger.debug("Live theme apply failed; persisted value still wins")
        return (True, "") if ok else (False, "Saving appearance settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"theme": self.selected_theme, "splash_card": self.selected_splash_card}
```

Replace the `STEP_NOTES` and `STEP_APPEARANCE` placeholders in `_create_steps`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wizard notes-sync and appearance steps"
```

---

### Task 10: Protect-keys step and Summary step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append)

**Interfaces:**
- Consumes: `check_encryption_needed()`, `enable_config_encryption(password: str) -> bool` from `tldw_chatbook.config` (thread-safe; rewrites the whole config under the config RLock); `PasswordDialog` from `tldw_chatbook.Widgets.password_dialog` (dismisses `str | None`; read the existing setup-mode caller at `UI/Tools_Settings_Window.py:7284` for the exact constructor mode/kwargs and mirror them); `build_summary_rows`, `embeddings_rag_deps_installed`; `TAB_CHAT`, `TAB_HOME` from `tldw_chatbook.Constants`.
- Produces: `class ProtectKeysStep(SetupStep)` (constructor takes `enable_encryption=None` injectable, defaulting to `enable_config_encryption`); `class SummaryStep(SetupStep)` — `get_step_data()` returns `{"exit_route": str | None}`.

**Behavior:**
- ProtectKeys: explain plainly ("You'll be asked for this password each time chatbook starts"); a "Set a password" button pushes `PasswordDialog` (setup mode); on a password, run `enable_config_encryption(password)` inside `wizard.commit_config`'s executor pattern (`asyncio.get_running_loop().run_in_executor` — same serialization guarantee: the config RLock plus the exclusive advance worker mean no interleaving with step commits). Failure → inline error, keys stay plaintext, step remains skippable.
- Summary: on `on_show()`, force-reload persisted config (`load_cli_config_and_ensure_existence(force_reload=True)` in a worker) and render `build_summary_rows(...)` — read-back, never step memory. Quick track prepends the "Left at recommended defaults" notice (tools off, RAG off, default theme, notes sync off — with Settings pointers). Exits: first-run (`wizard.rerun False`) → "Start chatting" (`exit_route=TAB_CHAT`) / "Explore on my own" (`exit_route=TAB_HOME`); re-run → "Done" (`exit_route=None`) / "Go to Chat" (`exit_route=TAB_CHAT`). Each button records `exit_route` then presses Next programmatically (`self.wizard.handle_next()`); Summary is the last active step so Next triggers `complete_wizard()` → `_finalize` → dismiss.

- [ ] **Step 1: Write the failing tests (append)**

```python
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProtectKeysStep, SummaryStep


@pytest.mark.asyncio
async def test_protect_keys_enables_encryption_via_injected_callable():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    calls = []
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True), rerun=False,
    )
    step = ProtectKeysStep(
        wizard=wizard,
        config=WizardStepConfig(id="protect-keys", title="Protect keys", step_number=8),
        enable_encryption=lambda pw: calls.append(pw) or True,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok = await step.apply_password("hunter2-long-password")
        assert ok is True
        assert calls == ["hunter2-long-password"]


@pytest.mark.asyncio
async def test_summary_step_renders_rows_from_read_back():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
        },
        rag_deps_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause()
        rendered = str(step.query_one("#setup-summary-rows", Static).render())
        assert "Provider" in rendered
        assert "✓" in rendered and "✗" in rendered
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Wizards/test_first_run_setup_wizard.py -v -k "protect or summary"`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement both steps**

```python
class ProtectKeysStep(SetupStep):
    """Offer config encryption for any keys entered this run."""

    def __init__(self, wizard=None, config=None, *, enable_encryption=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._enable_encryption = enable_encryption
        self.encryption_enabled = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-protect"):
            yield Static("Protect your keys", classes="setup-title")
            yield Static(
                "Encrypt the API keys in your config file with a password. "
                "You'll be asked for this password each time chatbook starts. "
                "Skip to leave keys as plain text (you can enable this later "
                "in Settings ▸ Privacy & Security).",
                classes="setup-subtitle",
            )
            yield Button("Set a password", id="setup-protect-set-password",
                         variant="primary")
            yield Static("", id="setup-protect-status", classes="setup-probe-status")
            yield Static("", classes="setup-step-error")

    @on(Button.Pressed, "#setup-protect-set-password")
    def _on_set_password(self) -> None:
        from tldw_chatbook.Widgets.password_dialog import PasswordDialog

        # Mirror the setup-mode constructor used at Tools_Settings_Window.py:7284.
        dialog = PasswordDialog(
            mode="setup",
            title="Choose a master password",
            message="This password decrypts your config at startup.",
        )
        self.app.push_screen(dialog, self._on_password_result)

    def _on_password_result(self, password: str | None) -> None:
        if not password:
            return
        self.run_worker(self._apply_password_worker(password), exclusive=True,
                        group="setup-wizard-advance")

    async def _apply_password_worker(self, password: str) -> None:
        ok = await self.apply_password(password)
        status = self.query_one("#setup-protect-status", Static)
        if ok:
            status.update("✓ Encryption enabled.")
        else:
            self.show_step_error(
                "Enabling encryption failed — your keys are unchanged (plain text)."
            )

    async def apply_password(self, password: str) -> bool:
        import asyncio

        enable = self._enable_encryption
        if enable is None:
            from tldw_chatbook.config import enable_config_encryption

            enable = enable_config_encryption
        ok = bool(
            await asyncio.get_running_loop().run_in_executor(None, enable, password)
        )
        self.encryption_enabled = ok
        return ok

    def get_step_data(self) -> Dict[str, Any]:
        return {"encryption_enabled": self.encryption_enabled}


class SummaryStep(SetupStep):
    """Read-back ✓/✗ matrix plus mode-dependent exits."""

    def __init__(self, wizard=None, config=None, *, load_config=None,
                 rag_deps_installed=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._load_config = load_config
        self._rag_deps_installed = rag_deps_installed
        self.exit_route: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-summary"):
            yield Static("Setup summary", classes="setup-title")
            yield Static("", id="setup-summary-defaults-note", classes="setup-subtitle")
            yield Static("", id="setup-summary-rows")
            yield Static("", id="setup-summary-footer", classes="setup-subtitle")
            with Horizontal(classes="setup-summary-actions"):
                if getattr(self.wizard, "rerun", False):
                    yield Button("Done", id="setup-exit-done", variant="primary")
                    yield Button("Go to Chat", id="setup-exit-chat")
                else:
                    yield Button("Start chatting", id="setup-exit-chat", variant="primary")
                    yield Button("Explore on my own", id="setup-exit-home")

    def on_show(self) -> None:
        super().on_show()
        track = (self.wizard.wizard_data or {}).get(
            wizard_state.STEP_WELCOME, {}
        ).get("track")
        if track == wizard_state.TRACK_QUICK:
            self.query_one("#setup-summary-defaults-note", Static).update(
                "Left at recommended defaults: tools off, RAG off, default theme, "
                "notes sync off — each lives in Settings when you want it."
            )
        self.run_worker(self._render_rows(), exclusive=True, group="setup-summary-load")

    async def _render_rows(self) -> None:
        import asyncio

        load = self._load_config
        if load is None:
            from tldw_chatbook.config import load_cli_config_and_ensure_existence

            def load():
                return load_cli_config_and_ensure_existence(force_reload=True)

        deps = self._rag_deps_installed
        if deps is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps = embeddings_rag_deps_installed
        config = await asyncio.get_running_loop().run_in_executor(None, load)
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_summary_rows

        rows = build_summary_rows(config, dict(os.environ), rag_deps_installed=deps())
        lines = [
            f"{'✓' if row.ok else '✗'} {row.label}"
            + (f" — {row.detail}" if row.detail else "")
            for row in rows
        ]
        self.query_one("#setup-summary-rows", Static).update("\n".join(lines))
        from tldw_chatbook.config import get_cli_config_path

        try:
            self.query_one("#setup-summary-footer", Static).update(
                f"Config file: {get_cli_config_path()}\n"
                "Re-run setup any time: Settings ▸ Diagnostics ▸ Run setup wizard."
            )
        except Exception:
            pass

    @on(Button.Pressed, "#setup-exit-chat")
    def _exit_chat(self) -> None:
        from tldw_chatbook.Constants import TAB_CHAT

        self._finish(TAB_CHAT)

    @on(Button.Pressed, "#setup-exit-home")
    def _exit_home(self) -> None:
        from tldw_chatbook.Constants import TAB_HOME

        self._finish(TAB_HOME)

    @on(Button.Pressed, "#setup-exit-done")
    def _exit_done(self) -> None:
        self._finish(None)

    def _finish(self, exit_route: Optional[str]) -> None:
        self.exit_route = exit_route
        self.wizard.handle_next()

    def get_step_data(self) -> Dict[str, Any]:
        return {"exit_route": self.exit_route}
```

Replace the `STEP_PROTECT` and `STEP_SUMMARY` placeholders in `_create_steps`. Verify the `PasswordDialog` setup-mode constructor kwargs against `UI/Tools_Settings_Window.py:7284` and adjust the `_on_set_password` call to match exactly.

- [ ] **Step 4: Run the whole wizard test file**

Run: `pytest Tests/Wizards/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wizard protect-keys and read-back summary steps"
```

---

### Task 11: App wiring — auto-offer, resume toast, Settings button, command palette

**Files:**
- Modify: `tldw_chatbook/app.py` (`_push_initial_screen` ~:6936, remove first-run toast call at :6786-6789 and `_show_first_run_notification` at :6907, add palette provider near `LibraryIngestProvider` ~:1359, register in `COMMANDS` ~:3287)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (Diagnostics pane ~:10796; handlers ~:13522; enter-activation allowlist ~:14947)
- Test: `Tests/Wizards/test_first_run_setup_wizard.py` (append unit tests for the gating call path)

**Interfaces:**
- Consumes: `should_offer_wizard`, `should_show_resume_toast` (Task 2); `FirstRunSetupWizard` (Task 5); `NavigateToScreen` from `tldw_chatbook.UI.Navigation.main_navigation`.
- Produces: `TldwCli._maybe_offer_first_run_wizard()`, `TldwCli._handle_first_run_wizard_result(result)`; Settings button id `settings-run-setup-wizard`; palette provider `SetupWizardProvider`.

- [ ] **Step 1: Add the offer hook to `app.py`**

At the END of `_push_initial_screen()` (after `self._initial_screen_pushed = True`), add:

```python
        self._maybe_offer_first_run_wizard()
```

Add these methods to `TldwCli` (near `_show_first_run_notification`'s old location):

```python
    def _maybe_offer_first_run_wizard(self) -> None:
        """Offer the setup wizard once; otherwise nudge unfinished setups."""
        try:
            from tldw_chatbook.UI.Wizards.first_run_setup_state import (
                should_offer_wizard,
                should_show_resume_toast,
            )

            if should_offer_wizard(self.app_config, os.environ):
                self.call_after_refresh(self._push_first_run_wizard)
            elif should_show_resume_toast(self.app_config, os.environ):
                self.notify(
                    "Setup isn't finished — run it any time from "
                    "Settings ▸ Diagnostics ▸ Run setup wizard.",
                    title="Finish setup",
                    severity="information",
                    timeout=8,
                )
        except Exception as e:
            logger.error(f"First-run wizard offer failed: {e}")

    def _push_first_run_wizard(self) -> None:
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        self.push_screen(FirstRunSetupWizard(self), self._handle_first_run_wizard_result)

    def _handle_first_run_wizard_result(self, result: dict | None) -> None:
        if not isinstance(result, dict):
            return  # cancelled / finish-later: resume toast handles next launch
        exit_route = result.get("exit_route")
        if exit_route:
            from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

            self.post_message(NavigateToScreen(str(exit_route)))
```

Then DELETE the `_first_run` toast call site (`app.py:6786-6789`) and the `_show_first_run_notification` method (`app.py:6907-6920`) — the wizard and resume toast replace them. (`_resolve_initial_shell_route`'s `_first_run → TAB_HOME` behavior stays: the wizard sits on top of Home.)

- [ ] **Step 2: Add the command palette provider**

Next to `LibraryIngestProvider` (`app.py:~1359`), following its exact shape:

```python
class SetupWizardProvider(Provider):
    """Provider for re-running the first-run setup wizard."""

    COMMANDS = (
        (
            "Setup: Run setup wizard…",
            "run_setup_wizard",
            "Walk through providers, models, and app configuration",
        ),
    )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for command_text, action_id, help_text in self.COMMANDS:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setup_wizard_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for command_text, action_id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setup_wizard_action, action_id),
                help=help_text,
            )

    def handle_setup_wizard_action(self, action_id: str) -> None:
        try:
            if action_id == "run_setup_wizard":
                from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                    FirstRunSetupWizard,
                )

                self.app.push_screen(FirstRunSetupWizard(self.app, rerun=True))
        except Exception as e:
            self.app.notify(f"Failed to open setup wizard: {e}", severity="error")
```

Add `SetupWizardProvider` to the `COMMANDS` set at `app.py:~3287`.

- [ ] **Step 3: Add the Settings button**

In `settings_screen.py`, inside the Diagnostics actions row (`#settings-diagnostics-actions`, ~:10796), add:

```python
                    yield Button(
                        "Run Setup Wizard",
                        id="settings-run-setup-wizard",
                        tooltip="Re-run the guided first-run setup with current values.",
                    )
```

Next to the other Diagnostics handlers (~:13522):

```python
    @on(Button.Pressed, "#settings-run-setup-wizard")
    def handle_run_setup_wizard(self, event: Button.Pressed) -> None:
        event.stop()
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        self.app.push_screen(FirstRunSetupWizard(self.app_instance, rerun=True))
```

Add `"settings-run-setup-wizard"` to the enter-activation allowlist at ~:14947 (without this, Enter on the focused button falls through to category selection).

- [ ] **Step 4: Write unit tests for the gating path (append to wizard test file)**

```python
class TestAppOfferGating:
    """The app hook is thin; assert the state functions drive it correctly."""

    def test_fresh_config_offers_and_upgrader_does_not(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import should_offer_wizard

        assert should_offer_wizard({}, {}) is True
        upgrader = {"api_settings": {"openai": {"api_key": "sk-x"}}}
        assert should_offer_wizard(upgrader, {}) is False

    def test_rerun_flag_reaches_container(self):
        wizard = _make_wizard(rerun=True)
        assert wizard.rerun is True
```

- [ ] **Step 5: Run tests, then boot-smoke the app**

Run: `pytest Tests/Wizards/ Tests/Widgets/test_splash_screen_config_read.py -v`
Expected: PASS.

Boot smoke (import-level regressions in app.py are the risk here):

```bash
python3 -c "import tldw_chatbook.app"
```

Expected: clean import, no traceback.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat: wire first-run wizard offer, resume toast, Settings + palette re-entry"
```

---

### Task 12: Integration tests, docs, live verification, close-out

**Files:**
- Test: `Tests/Wizards/test_first_run_setup_integration.py` (create)
- Create: `Docs/User_Guide/First_Run_Setup.md`
- Modify: the backlog task from Task 1 (status/notes)

**Interfaces:**
- Consumes: everything above; `TLDW_CONFIG_PATH` env override (config.py honors it via `_get_effective_config_path`); `load_cli_config_and_ensure_existence(force_reload=True)`.

- [ ] **Step 1: Write the integration tests**

Create `Tests/Wizards/test_first_run_setup_integration.py`:

```python
"""Integration tests: wizard commit plans against a real TOML config file."""

import os
from pathlib import Path

import pytest

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


@pytest.fixture()
def temp_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    from tldw_chatbook import config as config_module

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    yield config_path
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def _reload():
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    return load_cli_config_and_ensure_existence(force_reload=True)


def _write(section_values):
    from tldw_chatbook.config import save_settings_to_cli_config

    assert wizard_state.commit_sections_allowed(section_values), section_values
    assert save_settings_to_cli_config(section_values) is True


class TestCommitRoundTrip:
    def test_provider_and_model_commits_land_in_toml(self, temp_config):
        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-integration", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        assert config["api_settings"]["openai"]["api_key"] == "sk-integration"
        assert config["chat_defaults"]["provider"] == "OpenAI"
        assert config["chat_defaults"]["model"] == "gpt-5.6-terra"

    def test_wizard_state_flags_land_and_gate_offers(self, temp_config):
        _write(wizard_state.build_wizard_state_commit(started=True))
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False
        assert wizard_state.should_show_resume_toast(config, {}) is True
        _write(wizard_state.build_wizard_state_commit(completed=True))
        config = _reload()
        assert wizard_state.should_show_resume_toast(config, {}) is False

    def test_rerun_prefill_round_trip_without_secret_leak(self, temp_config):
        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-secret", api_url=None
        ))
        _write(wizard_state.build_model_commit(
            provider_value="OpenAI", model_id="gpt-5.6-terra"
        ))
        config = _reload()
        prefill = wizard_state.read_wizard_prefill(config)
        assert prefill.provider_value == "OpenAI"
        assert "sk-secret" not in repr(prefill)
        presence = wizard_state.read_provider_secret_presence(
            config, {}, provider_key="openai"
        )
        assert presence.configured is True
        assert "sk-secret" not in repr(presence)

    def test_upgrader_config_never_auto_offers(self, temp_config):
        _write({"api_settings.anthropic": {"api_key": "sk-upgrader"}})
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False

    def test_summary_rows_match_persisted_state(self, temp_config):
        _write(wizard_state.build_notes_commit(
            sync_directory="~/N", auto_sync_enabled=True
        ))
        config = _reload()
        rows = {r.label: r for r in wizard_state.build_summary_rows(
            config, {}, rag_deps_installed=False
        )}
        assert rows["Notes sync"].ok is True


class TestEncryptionAtRest:
    def test_enable_encryption_encrypts_stored_key(self, temp_config):
        from tldw_chatbook.config import enable_config_encryption

        _write(wizard_state.build_provider_commit(
            provider_key="openai", api_key="sk-to-encrypt", api_url=None
        ))
        assert enable_config_encryption("integration-test-password") is True
        raw = Path(os.environ["TLDW_CONFIG_PATH"]).read_text()
        assert "sk-to-encrypt" not in raw
        assert "enc:" in raw or "password_verifier" in raw
```

If `enable_config_encryption` requires the optional crypto dependency and it is absent in the venv, mark `TestEncryptionAtRest` with `@pytest.mark.optional` and note it in the backlog task instead of skipping silently.

- [ ] **Step 2: Run the integration tests**

Run: `pytest Tests/Wizards/test_first_run_setup_integration.py -v`
Expected: PASS. If cached config state leaks between tests, the `force_reload=True` in the fixture teardown is the intended reset — debug there first.

- [ ] **Step 3: Run the full affected suite**

Run: `pytest Tests/Wizards/ Tests/Widgets/test_splash_screen_config_read.py Tests/Chatbooks/ -v`
Expected: PASS — Chatbooks tests prove the untouched BaseWizard framework still serves the existing wizards.

- [ ] **Step 4: Write the user guide page**

Create `Docs/User_Guide/First_Run_Setup.md`:

```markdown
# First-Run Setup

> Verified against: first-run wizard implementation, 2026-07 (this page ships with it).

On your first launch, chatbook offers a guided setup. It is entirely optional —
every step has a Skip, Escape asks before closing, and anything you configure
(or don't) can be changed later in Settings.

## The two tracks

- **Quick setup (recommended)** — connect one provider, pick a default model,
  done. Everything else stays at recommended defaults (tools off, RAG off,
  default theme, notes sync off).
- **Full setup** — also walks through RAG/embeddings, built-in tools, notes
  sync, appearance, and key encryption.

## What each step does

| Step | What it configures | Where it lives in Settings |
|---|---|---|
| Provider | API key or local server endpoint | Providers & Models |
| Model | Default chat model | Providers & Models |
| RAG | Embedding model (needs the `embeddings_rag` extras) | RAG |
| Tools | Built-in tool gates (all off by default) | Tools |
| Notes sync | Folder + on/off toggle | Notes |
| Appearance | Theme and splash screen card | Appearance |
| Protect keys | Config encryption (password at startup) | Privacy & Security |

The final summary shows a ✓/✗ line per area, read back from what was actually
saved. Local servers (Ollama, llama.cpp) are auto-detected on localhost; no
probe traffic leaves your machine without your action.

## Running it again

- **Settings ▸ Diagnostics ▸ Run Setup Wizard**, or
- Command palette: "Setup: Run setup wizard…"

On a re-run, current values are prefilled and stored API keys are shown only
as "configured" — never displayed.
```

- [ ] **Step 5: Live verification (per `backlog/docs/lessons-live-verification.md` — evidence, not vibes)**

Run the real app against a scratch config and walk the checklist; record outcomes in the backlog task notes:

```bash
TLDW_CONFIG_PATH=/tmp/wizard-live-test/config.toml python3 -m tldw_chatbook.app
```

Checklist (each item is pass/fail evidence for the backlog notes):
1. Fresh config, splash ON: splash plays, Home renders, wizard appears on top.
2. Quick track end-to-end with a real or dummy key → summary shows honest ✓/✗ → "Start chatting" lands in Chat.
3. Fresh config, splash OFF (`enabled = false` pre-seeded in the scratch TOML): wizard still auto-offers.
4. Esc mid-wizard → confirm dialog → "Finish later" → relaunch app → resume toast (no re-push).
5. Full track: skip every step → app fully usable; summary all ✗ but honest.
6. Re-run from Settings ▸ Diagnostics: prefilled values, key shown as "configured", exits return to Settings ("Done").
7. 80×24 terminal: wizard renders without clipped navigation.

- [ ] **Step 6: Close out the backlog task and commit**

```bash
backlog task edit <wizard-task-id> -s Done --notes "Implemented per Docs/superpowers/plans/2026-07-28-first-run-setup-wizard.md: pure state module + WizardScreen subclass, commit-on-Next, auto-offer via _push_initial_screen, Settings/palette re-entry, integration + live verification evidence recorded here."
git add Tests/Wizards/test_first_run_setup_integration.py Docs/User_Guide/First_Run_Setup.md backlog/
git commit -m "test+docs: wizard integration suite, user guide page, task close-out"
```

If any lesson generalizes (a trap that cost real time), add it to the matching `backlog/docs/lessons-*.md` with the incident, per CLAUDE.md.

---

## Plan Self-Review Notes

- **Spec coverage:** entry guard + resume toast (T2, T11), two tracks + conditional ProtectKeys (T3, T5), provider step with env detection/discovery/tokened probes (T6), model step with staged discovery + dependency invalidation (T3, T7), RAG/tools (T8), notes/appearance (T9, splash write unblocked by T1), encryption via existing mechanism serialized with commits (T10), read-back summary + mode-dependent exits (T10), both startup paths via `_push_initial_screen` (T11), re-entry ×2 (T11), section-allowlist oracle (T3, enforced at runtime in `commit_config`, asserted in T12 integration), secrets-never-round-trip (T4, T12), 80×24 + splash-off live checks (T12).
- **Known deliberate deviations:** cloud-key probe with auth header is wired through the `http_client` seam but only exercised when a step chooses to probe a cloud provider — v1 renders "couldn't verify — save anyway" whenever the probe can't confirm, which satisfies the never-blocking spec rule.
- **Verify-don't-assume points called out in tasks:** `chat_defaults.provider` value form (T6), `PasswordDialog` setup-mode kwargs (T10), splash config attribute names (T1).
