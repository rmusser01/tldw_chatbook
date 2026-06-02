# Console Background Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional snow, rain, and matrix-style background effects for Console, defaulting to off and scoped to the transcript/event stream by default.

**Architecture:** Keep background effects as app-level Console behavior, not per-session chat state. Add a pure settings normalizer, extend config and Settings persistence, then wrap the existing `ConsoleTranscript` with a non-focusable effect surface that does not change transcript row reconciliation, focus, scrolling, or exports. Implement transcript scope first; only enable workbench scope after selector and contrast checks prove it does not interfere with controls.

**Tech Stack:** Python 3.11+, Textual widgets and timers, TCSS, existing `SettingsConfigAdapter`, pytest Textual harness tests.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-02-console-background-effects-design.md`
- Relevant existing Console design: `Docs/superpowers/specs/2026-05-21-console-native-chat-core-design.md`
- CSS build note: `Docs/superpowers/specs/2026-05-28-non-obscuring-focus-selection-design.md`

## Plan Review

Local plan review passed on 2026-06-02 against the required completeness, spec alignment, task decomposition, and buildability checklist. The formal plan-document-reviewer subagent was not dispatched because the available subagent tool is restricted unless the user explicitly asks for delegation.

## Preflight

- Create or identify the Backlog.md task before implementation.
- Move the task to `In Progress`.
- Add this implementation plan path to that task's `## Implementation Plan` section.
- Do not mark the Backlog.md task `Done` until acceptance criteria are checked, implementation notes are added, tests pass, CSS is regenerated when source TCSS changes, and self-review is complete.

## File Structure

- Create: `tldw_chatbook/Utils/console_background_effects.py`
  - Pure constants, `ConsoleBackgroundEffectSettings`, normalization helpers, and config serialization.
  - No imports from UI, Textual, `ChatScreen`, or Console widgets.
- Modify: `tldw_chatbook/config.py`
  - Add defaults and load-time normalization for `[console.background_effects]`.
  - Add generated config comments documenting effect, scope, intensity, and fps.
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
  - Add Console Behavior settings controls.
  - Extend Settings ownership and save path for nested `console.background_effects`.
  - Update in-memory `app_config`.
- Create: `tldw_chatbook/Widgets/Console/console_background_effect.py`
  - Non-focusable Textual renderer, frame state, bounded timers, and effect generation.
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Yield a transcript surface wrapper while keeping `#console-native-transcript` queryable.
  - Sync effect settings when Console config changes.
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
  - Export new Console effect widgets if tests or other Console modules need imports.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  - Build effect settings from `app_config` and pass them into `ConsoleSessionSurface`.
  - Refresh the surface after Settings changes or Console mount.
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add stable classes for transcript effect surface and renderer.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Run `python tldw_chatbook/css/build_css.py` after TCSS source edits.
- Test: `Tests/test_config_console_defaults.py`
  - Config defaults and normalization.
- Test: `Tests/UI/test_settings_configuration_hub.py`
  - Settings controls, ownership, nested persistence, in-memory update.
- Test: `Tests/UI/test_console_background_effects.py`
  - Effect renderer model/timer behavior and Console wrapper query/focus behavior.
- Test: `Tests/UI/test_console_native_transcript.py`
  - Add focused regression if wrapper changes transcript harness behavior.
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
  - Preserve required Console selectors.

---

### Task 1: Config Model And Defaults

**Files:**
- Create: `tldw_chatbook/Utils/console_background_effects.py`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/test_config_console_defaults.py`

- [ ] **Step 1: Write failing tests for default and invalid Console background config**

Append tests to `Tests/test_config_console_defaults.py`:

```python
def test_console_background_effect_defaults_disabled():
    background = config_module.DEFAULT_CONFIG_FROM_TOML["console"]["background_effects"]

    assert background == {
        "enabled": False,
        "effect": "none",
        "scope": "transcript",
        "intensity": "low",
        "fps": 6,
    }


def test_load_settings_normalizes_console_background_effects(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[console.background_effects]",
                'enabled = "true"',
                'effect = "fire"',
                'scope = "everywhere"',
                'intensity = "extreme"',
                "fps = 99",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["background_effects"] == {
        "enabled": True,
        "effect": "none",
        "scope": "transcript",
        "intensity": "low",
        "fps": 12,
    }
```

- [ ] **Step 2: Run failing config tests**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_console_defaults.py -q
```

Expected: the new tests fail because `background_effects` does not exist yet.

- [ ] **Step 3: Add pure settings model and normalization helper**

Create `tldw_chatbook/Utils/console_background_effects.py`:

```python
"""Console background effect settings and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CONSOLE_BACKGROUND_EFFECTS = frozenset({"none", "snow", "rain", "matrix"})
CONSOLE_BACKGROUND_SCOPES = frozenset({"transcript", "workbench"})
CONSOLE_BACKGROUND_INTENSITIES = frozenset({"low", "medium", "high"})
DEFAULT_CONSOLE_BACKGROUND_FPS = 6
MIN_CONSOLE_BACKGROUND_FPS = 1
MAX_CONSOLE_BACKGROUND_FPS = 12


@dataclass(frozen=True)
class ConsoleBackgroundEffectSettings:
    enabled: bool = False
    effect: str = "none"
    scope: str = "transcript"
    intensity: str = "low"
    fps: int = DEFAULT_CONSOLE_BACKGROUND_FPS

    def to_config(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "effect": self.effect,
            "scope": self.scope,
            "intensity": self.intensity,
            "fps": self.fps,
        }

    @property
    def active(self) -> bool:
        return self.enabled and self.effect != "none"


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_choice(value: object, allowed: frozenset[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else default


def _coerce_fps(value: object) -> int:
    try:
        fps = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CONSOLE_BACKGROUND_FPS
    return max(MIN_CONSOLE_BACKGROUND_FPS, min(MAX_CONSOLE_BACKGROUND_FPS, fps))


def normalize_console_background_effects(
    values: Mapping[str, object] | None,
) -> ConsoleBackgroundEffectSettings:
    raw = values if isinstance(values, Mapping) else {}
    effect = _coerce_choice(raw.get("effect"), CONSOLE_BACKGROUND_EFFECTS, "none")
    return ConsoleBackgroundEffectSettings(
        enabled=_coerce_bool(raw.get("enabled"), False),
        effect=effect,
        scope=_coerce_choice(raw.get("scope"), CONSOLE_BACKGROUND_SCOPES, "transcript"),
        intensity=_coerce_choice(
            raw.get("intensity"),
            CONSOLE_BACKGROUND_INTENSITIES,
            "low",
        ),
        fps=_coerce_fps(raw.get("fps")),
    )
```

- [ ] **Step 4: Wire config defaults and load normalization**

Modify `tldw_chatbook/config.py`:

```python
from tldw_chatbook.Utils.console_background_effects import (
    normalize_console_background_effects,
)
```

In the `[console]` default TOML string, add:

```toml
[console.background_effects]
enabled = false  # Optional Console ambience. Off by default for readability.
effect = "none"  # none, snow, rain, matrix
scope = "transcript"  # transcript, workbench
intensity = "low"  # low, medium, high
fps = 6  # 1-12
```

After existing `collapse_large_pastes` and `paste_collapse_threshold` normalization:

```python
    background_effects = final_console_settings_cli.get("background_effects")
    if not isinstance(background_effects, dict):
        background_effects = {}
    final_console_settings_cli["background_effects"] = (
        normalize_console_background_effects(background_effects).to_config()
    )
```

- [ ] **Step 5: Run config tests**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_console_defaults.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 6: Commit config slice**

```bash
git add tldw_chatbook/Utils/console_background_effects.py tldw_chatbook/config.py Tests/test_config_console_defaults.py
git commit -m "Add console background effect config defaults"
```

---

### Task 2: Settings Ownership, Controls, And Persistence

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing Settings ownership and rendering tests**

Append focused tests to `Tests/UI/test_settings_configuration_hub.py`:

```python
@pytest.mark.asyncio
async def test_settings_console_behavior_renders_background_effect_controls():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": False,
            "effect": "none",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-console-background-effect-enabled", Button)
        assert screen.query_one("#settings-console-background-effect-type", Select)
        assert screen.query_one("#settings-console-background-effect-scope", Select)
        assert screen.query_one("#settings-console-background-effect-intensity", Select)
        assert screen.query_one("#settings-console-background-effect-fps", Input)
        assert "Transcript (recommended)" in _visible_text(screen)


def test_settings_console_behavior_owns_background_effect_settings():
    app = _build_test_app()
    screen = SettingsScreen(app)
    ownership = screen._ownership_record(SettingsCategoryId.CONSOLE_BEHAVIOR)

    assert "console.background_effects.*" in ownership.owns_config_sections
```

Add `Select` and `SettingsScreen`, `SettingsCategoryId` imports if they are not already available in the test file.

- [ ] **Step 2: Write failing nested save test**

Append:

```python
@pytest.mark.asyncio
async def test_settings_console_background_effects_save_nested_config(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {
        "collapse_large_pastes": True,
        "background_effects": {
            "enabled": False,
            "effect": "none",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        },
    }
    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        enabled = screen.query_one("#settings-console-background-effect-enabled", Button)
        effect = screen.query_one("#settings-console-background-effect-type", Select)
        fps = screen.query_one("#settings-console-background-effect-fps", Input)

        enabled.press()
        effect.value = "matrix"
        fps.value = "10"
        screen.handle_console_background_effect_fps_changed(Input.Changed(fps, fps.value))

        await pilot.click("#settings-save-category")
        await _wait_for_settings_text(screen, pilot, "Console behavior settings saved.")

    assert saved == [
        {
            "console": {
                "background_effects": {
                    "enabled": True,
                    "effect": "matrix",
                    "scope": "transcript",
                    "intensity": "low",
                    "fps": 10,
                }
            }
        }
    ]
    assert app.app_config["console"]["background_effects"]["enabled"] is True
    assert app.app_config["console"]["background_effects"]["effect"] == "matrix"
```

- [ ] **Step 3: Run failing Settings tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_renders_background_effect_controls Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_owns_background_effect_settings Tests/UI/test_settings_configuration_hub.py::test_settings_console_background_effects_save_nested_config -q
```

Expected: fail on missing controls/ownership/save handler.

- [ ] **Step 4: Add Settings imports and constants**

Modify `tldw_chatbook/UI/Screens/settings_screen.py` imports:

```python
from textual.widgets import Button, Input, Rule, Select, Static, TextArea
from ...Utils.console_background_effects import (
    CONSOLE_BACKGROUND_EFFECTS,
    CONSOLE_BACKGROUND_INTENSITIES,
    CONSOLE_BACKGROUND_SCOPES,
    DEFAULT_CONSOLE_BACKGROUND_FPS,
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
    normalize_console_background_effects,
)
```

Add draft keys:

```python
CONSOLE_BACKGROUND_EFFECT_KEYS = frozenset(
    {
        "background_effects.enabled",
        "background_effects.effect",
        "background_effects.scope",
        "background_effects.intensity",
        "background_effects.fps",
    }
)
CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER = (
    "background_effects.enabled",
    "background_effects.effect",
    "background_effects.scope",
    "background_effects.intensity",
    "background_effects.fps",
)
```

Extend `CONSOLE_BEHAVIOR_SAVE_ORDER` by appending `CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER`.

- [ ] **Step 5: Extend ownership record**

Add `"console.background_effects.*"` to the Console Behavior `owns_config_sections` tuple.

- [ ] **Step 6: Add loaded/draft helpers**

Add methods near existing Console behavior helpers:

```python
    def _loaded_console_background_effects(self) -> dict[str, object]:
        return normalize_console_background_effects(
            self._console_settings().get("background_effects")
        ).to_config()

    def _console_background_effect_value(self, key: str) -> object:
        draft_key = f"background_effects.{key}"
        draft = self._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
        if draft is not None and draft_key in draft.values:
            return draft.values[draft_key]
        return self._loaded_console_background_effects().get(key)

    def _stage_console_background_effect_value(self, key: str, value: object) -> None:
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        draft_key = f"background_effects.{key}"
        draft = self._settings_drafts.setdefault(category, SettingsDraft(category=category))
        draft.set_value(
            draft_key,
            self._loaded_console_background_effects().get(key),
            value,
        )
        if not draft.is_dirty:
            self._settings_drafts.pop(category, None)
```

- [ ] **Step 7: Add render controls**

Inside `_render_console_behavior_card`, after composer paste handling or before it, add:

```python
            yield Static("Background effects", classes="destination-section")
            yield Button(
                "Enabled" if self._console_background_effect_value("enabled") else "Disabled",
                id="settings-console-background-effect-enabled",
                tooltip="Toggle optional Console transcript background effects.",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Effect", classes="settings-input-label")
                yield Select(
                    [(value, value) for value in ("none", "snow", "rain", "matrix")],
                    value=str(self._console_background_effect_value("effect") or "none"),
                    allow_blank=False,
                    id="settings-console-background-effect-type",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Scope", classes="settings-input-label")
                yield Select(
                    [
                        ("Transcript (recommended)", "transcript"),
                        ("Workbench (advanced)", "workbench"),
                    ],
                    value=str(self._console_background_effect_value("scope") or "transcript"),
                    allow_blank=False,
                    id="settings-console-background-effect-scope",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Intensity", classes="settings-input-label")
                yield Select(
                    [(value, value) for value in ("low", "medium", "high")],
                    value=str(self._console_background_effect_value("intensity") or "low"),
                    allow_blank=False,
                    id="settings-console-background-effect-intensity",
                    classes="settings-compact-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Frame rate", classes="settings-input-label")
                yield Input(
                    value=str(self._console_background_effect_value("fps") or DEFAULT_CONSOLE_BACKGROUND_FPS),
                    id="settings-console-background-effect-fps",
                    classes="settings-compact-input",
                    placeholder=str(DEFAULT_CONSOLE_BACKGROUND_FPS),
                    restrict=r"^[0-9]*$",
                )
```

- [ ] **Step 8: Add event handlers**

Add handlers:

```python
    @on(Button.Pressed, "#settings-console-background-effect-enabled")
    def handle_console_background_effect_enabled_changed(self, event: Button.Pressed) -> None:
        event.stop()
        next_value = not bool(self._console_background_effect_value("enabled"))
        self._stage_console_background_effect_value("enabled", next_value)
        event.button.label = "Enabled" if next_value else "Disabled"
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Select.Changed, "#settings-console-background-effect-type")
    def handle_console_background_effect_type_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._stage_console_background_effect_value("effect", str(event.value))
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Select.Changed, "#settings-console-background-effect-scope")
    def handle_console_background_effect_scope_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._stage_console_background_effect_value("scope", str(event.value))
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Select.Changed, "#settings-console-background-effect-intensity")
    def handle_console_background_effect_intensity_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._stage_console_background_effect_value("intensity", str(event.value))
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)

    @on(Input.Changed, "#settings-console-background-effect-fps")
    def handle_console_background_effect_fps_changed(self, event: Input.Changed) -> None:
        self._stage_console_background_effect_value("fps", event.value)
        self._console_behavior_result = "Console behavior settings staged."
        self._set_static_text("#settings-console-behavior-result", self._console_behavior_result)
        self._update_draft_status_widgets(SettingsCategoryId.CONSOLE_BEHAVIOR)
```

- [ ] **Step 9: Normalize and save nested values**

In `action_settings_save_category`, when Console Behavior is active:

```python
background_dirty = any(key.startswith("background_effects.") for key in draft.dirty_keys)
if background_dirty:
    background_values = self._loaded_console_background_effects()
    for key in CONSOLE_BACKGROUND_EFFECT_SAVE_ORDER:
        if key in draft.values:
            background_values[key.removeprefix("background_effects.")] = draft.values[key]
    normalized_background = normalize_console_background_effects(background_values).to_config()
else:
    normalized_background = None
```

Before saving:

```python
if normalized_background is not None:
    console_values["background_effects"] = normalized_background
```

Update `_apply_console_behavior_save_result` to merge nested background settings:

```python
if "background_effects" in console_values:
    self._console_settings()["background_effects"] = dict(console_values["background_effects"])
    console_values = {
        key: value
        for key, value in console_values.items()
        if key != "background_effects"
    }
self._console_settings().update(console_values)
```

- [ ] **Step 10: Sync widgets after save**

Extend `_sync_console_behavior_widgets` for the new controls. Keep this robust with `try/except QueryError`, matching existing style.

- [ ] **Step 11: Run Settings tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_renders_background_effect_controls Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_owns_background_effect_settings Tests/UI/test_settings_configuration_hub.py::test_settings_console_background_effects_save_nested_config -q
```

Expected: pass.

- [ ] **Step 12: Commit Settings slice**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py
git commit -m "Add console background effect settings controls"
```

---

### Task 3: Background Effect Renderer

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_background_effect.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Test: `Tests/UI/test_console_background_effects.py`

- [ ] **Step 1: Write failing renderer tests**

Create `Tests/UI/test_console_background_effects.py`:

```python
"""Console background effect widget tests."""

import pytest

from textual.app import App, ComposeResult

from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Console.console_background_effect import (
    ConsoleBackgroundEffect,
)


class EffectHarness(App[None]):
    def __init__(self, settings: ConsoleBackgroundEffectSettings) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        yield ConsoleBackgroundEffect(self.settings, id="console-background-effect")


def test_console_background_effect_disabled_is_inactive():
    effect = ConsoleBackgroundEffect(
        ConsoleBackgroundEffectSettings(enabled=False, effect="matrix")
    )

    assert effect.can_focus is False
    assert effect.is_effect_active is False


@pytest.mark.asyncio
async def test_console_background_effect_enabled_renders_frame():
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(
            enabled=True,
            effect="matrix",
            scope="transcript",
            intensity="low",
            fps=6,
        )
    )

    async with app.run_test(size=(60, 18)) as pilot:
        effect = app.query_one("#console-background-effect", ConsoleBackgroundEffect)
        await pilot.pause(0.2)

        assert effect.is_effect_active is True
        assert effect.frame_text(width=40, height=8).strip()
```

Mark async test with `pytest.mark.asyncio` if this repo style requires it.

- [ ] **Step 2: Run failing renderer tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_background_effects.py -q
```

Expected: fail because `ConsoleBackgroundEffect` does not exist.

- [ ] **Step 3: Implement renderer**

Create `tldw_chatbook/Widgets/Console/console_background_effect.py`:

```python
"""Console ambient background effect widgets."""

from __future__ import annotations

import random
from dataclasses import dataclass

from rich.segment import Segment
from rich.style import Style
from textual.app import ComposeResult
from textual.containers import Container
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget

from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


_INTENSITY_DENSITY = {"low": 0.035, "medium": 0.07, "high": 0.11}
_EFFECT_CHARS = {
    "snow": ".*+",
    "rain": ".:|",
    "matrix": "abcdefghijklmnopqrstuvwxyz0123456789",
}


@dataclass
class _Particle:
    x: int
    y: int
    speed: int
    char: str


class ConsoleBackgroundEffect(Widget):
    """Non-focusable cell-based Console background animation."""

    can_focus = False

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self._random = random.Random(seed)
        self._particles: list[_Particle] = []
        self._timer: Timer | None = None

    @property
    def is_effect_active(self) -> bool:
        return self.settings.active

    def on_mount(self) -> None:
        self._sync_timer()

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        self.settings = settings
        self._particles.clear()
        self._sync_timer()
        self.refresh()

    def _sync_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if not self.settings.active or not self.is_mounted:
            return
        self._timer = self.set_interval(
            1 / max(1, self.settings.fps),
            self._advance_frame,
            name="console-background-effect",
        )

    def _advance_frame(self) -> None:
        if not self.settings.active:
            self._sync_timer()
            return
        self._step_particles(max(1, self.size.width), max(1, self.size.height))
        self.refresh(layout=False)

    def _target_particle_count(self, width: int, height: int) -> int:
        density = _INTENSITY_DENSITY.get(self.settings.intensity, _INTENSITY_DENSITY["low"])
        return max(1, int(width * height * density))

    def _new_particle(self, width: int, height: int) -> _Particle:
        chars = _EFFECT_CHARS.get(self.settings.effect, ".")
        speed = 1 if self.settings.effect in {"rain", "matrix"} else self._random.choice((1, 1, 2))
        return _Particle(
            x=self._random.randrange(max(1, width)),
            y=self._random.randrange(max(1, height)),
            speed=speed,
            char=self._random.choice(chars),
        )

    def _step_particles(self, width: int, height: int) -> None:
        target_count = self._target_particle_count(width, height)
        while len(self._particles) < target_count:
            self._particles.append(self._new_particle(width, height))
        self._particles = self._particles[:target_count]
        for particle in self._particles:
            particle.y = (particle.y + particle.speed) % max(1, height)
            if self.settings.effect == "snow":
                particle.x = max(0, min(width - 1, particle.x + self._random.choice((-1, 0, 1))))
            if self.settings.effect == "matrix":
                particle.char = self._random.choice(_EFFECT_CHARS["matrix"])

    def frame_text(self, *, width: int, height: int) -> str:
        if not self.settings.active or width <= 0 or height <= 0:
            return ""
        self._step_particles(width, height)
        grid = [[" " for _ in range(width)] for _ in range(height)]
        for particle in self._particles:
            if 0 <= particle.x < width and 0 <= particle.y < height:
                grid[particle.y][particle.x] = particle.char
        return "\n".join("".join(row) for row in grid)

    def render_line(self, y: int) -> Strip:
        width = max(0, self.size.width)
        height = max(0, self.size.height)
        frame = self.frame_text(width=width, height=height).splitlines()
        line = frame[y] if 0 <= y < len(frame) else ""
        style = Style(dim=True)
        return Strip([Segment(line.ljust(width), style)])


class ConsoleTranscriptSurface(Container):
    """Host an optional background effect without changing transcript identity."""

    can_focus = False

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        transcript: ConsoleTranscript | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self._transcript = transcript or ConsoleTranscript(id="console-native-transcript")

    def compose(self) -> ComposeResult:
        yield ConsoleBackgroundEffect(
            self.settings,
            id="console-transcript-background-effect",
            classes="console-background-effect",
        )
        yield self._transcript

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        self.settings = settings
        try:
            self.query_one(
                "#console-transcript-background-effect",
                ConsoleBackgroundEffect,
            ).update_settings(settings)
        except Exception:
            pass
```

Adjust `render_line` if Textual requires `Strip.blank(width, style)` or a different `Strip` constructor in the installed version.

- [ ] **Step 4: Export widgets**

Modify `tldw_chatbook/Widgets/Console/__init__.py`:

```python
from .console_background_effect import ConsoleBackgroundEffect, ConsoleTranscriptSurface
```

Add both names to `__all__`.

- [ ] **Step 5: Run renderer tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_background_effects.py -q
```

Expected: renderer tests pass.

- [ ] **Step 6: Commit renderer slice**

```bash
git add tldw_chatbook/Widgets/Console/console_background_effect.py tldw_chatbook/Widgets/Console/__init__.py Tests/UI/test_console_background_effects.py
git commit -m "Add console background effect renderer"
```

---

### Task 4: Console Transcript Wiring And CSS

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_background_effects.py`
- Test: `Tests/UI/test_console_native_transcript.py`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`

- [ ] **Step 1: Write failing Console wrapper tests**

Append to `Tests/UI/test_console_background_effects.py`:

```python
import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Widgets.Console import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_background_effect import ConsoleBackgroundEffect


@pytest.mark.asyncio
async def test_console_transcript_scope_mounts_effect_without_hiding_transcript():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "matrix",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-transcript-background-effect")

        assert console.query_one("#console-native-transcript", ConsoleTranscript)
        effect = console.query_one("#console-transcript-background-effect", ConsoleBackgroundEffect)
        assert effect.is_effect_active is True
        assert not console.query("#console-left-rail #console-transcript-background-effect")


@pytest.mark.asyncio
async def test_console_background_disabled_does_not_start_active_effect():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": False,
            "effect": "matrix",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        effects = list(console.query("#console-transcript-background-effect"))
        if effects:
            assert effects[0].is_effect_active is False
```

- [ ] **Step 2: Run failing Console wrapper tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_background_effects.py::test_console_transcript_scope_mounts_effect_without_hiding_transcript Tests/UI/test_console_background_effects.py::test_console_background_disabled_does_not_start_active_effect -q
```

Expected: fail until `ConsoleSessionSurface` yields the wrapper and `ChatScreen` passes settings.

- [ ] **Step 3: Pass settings into ConsoleSessionSurface**

Modify `ConsoleSessionSurface.__init__` in `tldw_chatbook/Widgets/Console/console_session_surface.py`:

```python
from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Console.console_background_effect import ConsoleTranscriptSurface
```

Add parameter and field:

```python
    def __init__(
        self,
        app_instance: Any,
        *,
        background_effect_settings: ConsoleBackgroundEffectSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.background_effect_settings = (
            background_effect_settings or ConsoleBackgroundEffectSettings()
        )
        self._session_sync_lock = asyncio.Lock()
```

Replace direct transcript yield:

```python
        yield ConsoleTranscriptSurface(
            self.background_effect_settings,
            id="console-transcript-surface",
            classes="console-transcript-surface",
        )
```

Add method:

```python
    def sync_background_effect_settings(
        self,
        settings: ConsoleBackgroundEffectSettings,
    ) -> None:
        self.background_effect_settings = settings
        try:
            self.query_one("#console-transcript-surface", ConsoleTranscriptSurface).update_settings(settings)
        except Exception:
            return
```

- [ ] **Step 4: Build settings in ChatScreen**

Modify `tldw_chatbook/UI/Screens/chat_screen.py` imports:

```python
from tldw_chatbook.Utils.console_background_effects import (
    normalize_console_background_effects,
)
```

Add helper near `_chat_default_value`:

```python
    def _console_background_effect_settings(self):
        config = getattr(self.app_instance, "app_config", {}) or {}
        console = config.get("console", {}) if isinstance(config, dict) else {}
        background = console.get("background_effects", {}) if isinstance(console, dict) else {}
        settings = normalize_console_background_effects(background)
        if settings.scope == "workbench":
            # Workbench mode is enabled only after Task 5 validates the wider scope.
            return settings
        return settings
```

Pass settings in `_ensure_console_session_surface`:

```python
                background_effect_settings=self._console_background_effect_settings(),
```

When the surface already exists, sync settings before returning:

```python
        else:
            self.console_session_surface.sync_background_effect_settings(
                self._console_background_effect_settings()
            )
```

- [ ] **Step 5: Add TCSS source styles**

Edit `tldw_chatbook/css/components/_agentic_terminal.tcss`:

```css
#console-transcript-surface {
    height: 1fr;
    min-height: 0;
    layers: console-background console-content;
}

#console-transcript-background-effect {
    layer: console-background;
    width: 100%;
    height: 100%;
    color: $ds-text-muted;
    background: transparent;
}

#console-transcript-surface #console-native-transcript {
    layer: console-content;
    height: 1fr;
    min-height: 0;
    background: transparent;
}

.console-transcript-message,
.console-transcript-action-row {
    background: $ds-surface-panel;
}
```

If `layers` still causes layout participation in local testing, replace with a simpler version: mount the background renderer only for empty transcript space and keep it hidden when messages exist. Do not compromise transcript focus.

- [ ] **Step 6: Regenerate bundled CSS**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: exits 0. Existing missing-module warnings are acceptable only if they match known repo baseline.

- [ ] **Step 7: Run Console wrapper and existing transcript tests**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_background_effects.py Tests/UI/test_console_native_transcript.py::test_console_transcript_keyboard_selects_messages_and_enter_shows_actions Tests/UI/test_console_native_transcript.py::test_console_screen_exposes_native_transcript Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions -q
```

Expected: pass.

- [ ] **Step 8: Commit Console wiring slice**

```bash
git add tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_background_effects.py
git commit -m "Wire background effects behind console transcript"
```

---

### Task 5: Workbench Scope Gate

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_background_effect.py`
- Test: `Tests/UI/test_console_background_effects.py`
- Test: `Tests/UI/test_destination_visual_parity_correction.py` or a focused Console selector test if the larger parity file is too broad.

- [ ] **Step 1: Write failing workbench scope behavior test**

Append:

```python
@pytest.mark.asyncio
async def test_console_workbench_scope_is_explicitly_gated_without_breaking_controls():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "rain",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-staged-context-tray")

        assert console.query_one("#console-native-transcript")
        assert console.query_one("#console-native-composer")
        assert console.query_one("#console-staged-context-tray")
```

This test is intentionally selector-focused. If a real workbench layer is implemented, add `#console-workbench-background-effect`. If workbench is gated off, assert visible fallback copy in Settings and transcript effect remains active.

- [ ] **Step 2: Choose one of two implementation paths**

Path A, implement workbench effect only if local Textual layering keeps controls readable:

- Add `ConsoleWorkbenchBackgroundSurface`.
- Wrap `#console-workspace-grid` or its inner content without changing existing selectors.
- Add `#console-workbench-background-effect`.
- Verify rails, composer, inspector, and transcript selectors remain visible.

Path B, gate workbench off in this slice:

- Keep `workbench` as a normalized config value.
- Settings scope control still shows `Workbench (advanced)` but disables it or normalizes back to transcript with visible result copy:

```text
Workbench scope is not available in this build; using Transcript scope.
```

- Do not silently claim workbench mode is active.

Use Path B if Path A causes focus, selector, layout, or contrast regressions.

- [ ] **Step 3: Run focused workbench gate test**

Run:

```bash
.venv/bin/python -m pytest Tests/UI/test_console_background_effects.py::test_console_workbench_scope_is_explicitly_gated_without_breaking_controls -q
```

Expected: pass with either validated workbench scope or explicit fallback.

- [ ] **Step 4: Commit workbench gate slice**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_background_effect.py Tests/UI/test_console_background_effects.py
git commit -m "Gate console background workbench scope"
```

---

### Task 6: Final Verification And Documentation

**Files:**
- Modify: Backlog.md task file under `backlog/tasks/`
- Optional modify: `Docs/superpowers/qa/console/2026-06-02-console-background-effects-qa.md`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
.venv/bin/python -m pytest Tests/test_config_console_defaults.py Tests/UI/test_settings_configuration_hub.py::test_settings_console_behavior_renders_background_effect_controls Tests/UI/test_settings_configuration_hub.py::test_settings_console_background_effects_save_nested_config Tests/UI/test_console_background_effects.py Tests/UI/test_console_native_transcript.py::test_console_transcript_keyboard_selects_messages_and_enter_shows_actions Tests/UI/test_console_native_transcript.py::test_console_tab_reaches_major_console_screen_regions Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run CSS build if TCSS changed**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: exits 0. If output changes, review and include `tldw_chatbook/css/tldw_cli_modular.tcss`.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Self-review code paths**

Check:

- `ConsoleBackgroundEffectSettings` is pure and not UI-coupled.
- Config defaults keep effects disabled.
- Settings saves nested `background_effects`.
- `#console-native-transcript` remains queryable.
- Background effect widget is non-focusable.
- Effect timer stops on unmount and disabled settings.
- Workbench scope is either implemented safely or visibly gated off.

- [ ] **Step 5: Update Backlog task**

Update the task:

- Check all acceptance criteria.
- Add `## Implementation Notes`.
- Include test commands and results.
- Move status to `Done` only after all Definition of Done items are complete.

- [ ] **Step 6: Final commit**

```bash
git status --short
git add <changed implementation, tests, CSS, docs, backlog task files>
git commit -m "Add optional console background effects"
```

Expected: only intentional files are staged. Do not stage unrelated untracked screenshots or local tool directories.
