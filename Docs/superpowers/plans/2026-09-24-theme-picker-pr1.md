# Theme Picker (PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settings ▸ Theme opens as a picker (list with colour strips, filter, preview card, Use/Try/Revert, live active/launch markers); the existing editor is reached through Clone/New; Appearance shows a read-only theme row.

**Architecture:** A pure `theme_catalog` module (catalog entries + `use_theme`/`revert_theme`) is the single owner of "which themes exist, which is active, which is the launch default". A new `ThemePicker` widget and a `ThemePane` (`ContentSwitcher`: picker ↔ editor) replace the direct editor mount in `SettingsScreen`. The palette's `switch_theme` and the Appearance row read/write through the same module.

**Tech Stack:** Python 3.12, Textual 8.2.8 (`OptionList`, `ContentSwitcher`, `theme_changed_signal`), Rich `Text`, pytest + pytest-asyncio, `app.run_test`.

**Spec:** `Docs/superpowers/specs/2026-09-24-theme-picker-redesign-design.md` (§4, §5, §8, §10 PR 1, §11). Read it first.

## Global Constraints

- **Base branch:** create `feat/theme-picker-pr1` off `fix/theme-ux-wave` (PR #2830). If #2830 has merged, branch off `origin/dev` instead. Work in a worktree under `.claude/worktrees/`, and prefix every shell command with `cd <worktree> &&`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=<worktree>`. Assert that `tldw_chatbook.__file__` resolves inside the worktree; the venv's editable install points at the main checkout.
- **Tests:**
  - Any pilot test that builds `TldwCli` or the Settings screen needs `@private_profile_test` and a `request` argument (`from Tests.private_profile import private_profile_test`). Without it, the test fails locally with `RecoveryRequired`.
  - Isolated-widget tests use `Tests.textual_test_harness.IsolatedWidgetTestApp`.
- **CSS (ADR-161, zero literals):**
  - Use `$ds-size-*` and `$ds-space-*` tokens only. Known sizes: 1, 3, 8, 9, 12, 16, 24. Check `tldw_chatbook/css/` for the token file before adding a size.
  - Never end a selector with a bare `Button`; key a class instead.
  - After any `.tcss` edit, run `./build_css.sh` and commit the regenerated `tldw_chatbook/css/tldw_cli_modular.tcss`. Never hand-edit the bundle.
- **Copy and markers:** the active and launch markers are the words `active` and `launch`. Colour is never the only carrier of state. Use toast verbs exactly as the spec writes them.
- **Keys:** do not rebind `/` (Settings search, `settings_screen.py` `on_key`).
- **Backlog:** use task file `backlog/tasks/task-32948 - Settings-Theme-picker-first-redesign.md`. Before creating it, re-verify that id 32948 is free across `origin/dev`, all branches and all worktrees. PR 1 fills the Implementation Notes and leaves the status In Progress, because PR 2 and PR 3 follow.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Before the PR:**
  - `PYTHON=<venv python> ./scripts/preflight.sh`. Check its exit code directly; never pipe it to `tail`.
  - `ruff check` on the touched files, with no new findings.

## Review Focus

1. **Revert after a Try then a Use.** Revert must restore the theme and launch default from before the *first* change, not the tried theme. This is pinned in Task 2 (`test_merge_keeps_first_previous_values`).
2. **Unknown or unregistered theme names.** A launch default naming a theme that isn't registered, or `use_theme` on a stale entry, must not crash and must not write config. Pinned in Task 2 (`test_use_theme_unknown_name_raises_and_writes_nothing`) and Task 1 (`test_launch_default_not_registered_marks_nothing`).
3. **Filter with zero matches, then Enter.** Must do nothing and show "No themes match". Pinned in Task 4 (`test_filter_no_match_enter_is_inert`).
4. **Config write failure on Use.** The theme still switches for the session, the toast says the launch default was not saved, and Revert does not try to write. Pinned in Task 2 (`test_use_theme_persist_failure_reports_not_persisted`) and Task 4 (`test_use_toast_when_persist_fails`).
5. **Layout at 80×24.** Every picker control must be reachable; this is the regression class that caused the complaints. Pinned in Task 7.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tldw_chatbook/css/Themes/theme_catalog.py` | create | `ThemeEntry`, `build_catalog`, `is_catalog_theme`, `display_name`, `ThemeChange`, `use_theme`, `revert_theme`, `current_launch_default`, `user_theme_names`. No widgets. |
| `tldw_chatbook/Widgets/theme_preview.py` | create | `ThemePreview` widget: the Console-shaped preview stub, painted from a colours dict. It is extracted from the editor, which now uses it. |
| `tldw_chatbook/Widgets/settings_theme_picker.py` | create | `ThemeFilterInput`, `ThemeOptionList`, `ThemePicker`, `ThemePane`. |
| `tldw_chatbook/Widgets/settings_theme_editor.py` | modify | Uses `ThemePreview`; removes the "Set as launch default" button and handler; uses `is_catalog_theme`. |
| `tldw_chatbook/UI/Screens/settings_screen.py` | modify | Mounts `ThemePane`; handles `EditRequested` and Back; drops the Theme state banner; replaces the Appearance theme Select with a read-only row and Open Theme button; trims `default_theme` out of the Appearance draft paths. |
| `tldw_chatbook/UI/Screens/settings_appearance_defaults.py` | modify | The save payload and validation no longer own `default_theme`. |
| `tldw_chatbook/app.py` | modify | `ThemeProvider.switch_theme` calls `use_theme`. |
| `tldw_chatbook/css/components/_settings_splash_theme.tcss` | modify | Picker layout and heights, including compact mode. |
| `Tests/Utils/test_theme_catalog.py` | create | Unit tests for the catalog, `use_theme` and `revert_theme`. |
| `Tests/UI/test_settings_theme_picker.py` | create | Pilot tests for the picker and pane (isolated app). |
| `Tests/UI/test_settings_theme_picker_screen.py` | create | Real Settings screen tests: pane wiring, Appearance row, geometry, contrast. |
| `Docs/User_Guide/settings.md` | modify | Rewrite the Theme and Appearance sections, and update the Verified stamp. |

---

### Task 1: Theme catalog

**Files:**
- Create: `tldw_chatbook/css/Themes/theme_catalog.py`
- Test: `Tests/Utils/test_theme_catalog.py`

**Interfaces:**
- Produces:
  - `ThemeEntry` (frozen dataclass): `id: str`, `display_name: str`, `origin: Literal["yours","shipped","textual"]`, `dark: bool`, `colours: tuple[tuple[str, str], ...]` (the 10 `BASE_KEYS` resolved to uppercase `#RRGGBB`, as pairs so the dataclass stays hashable), `is_active: bool`, `is_launch_default: bool`, `overrides: Literal["shipped","textual"] | None`. Property `strip -> tuple[str, ...]` (the 7 `STRIP_KEYS` colours, in order).
  - `build_catalog(available: Mapping[str, Theme], user_names: Collection[str], active: str, launch_default: str) -> list[ThemeEntry]`. Ordered yours → shipped → textual; within each group, by `display_name.casefold()`.
  - `display_name(theme_id: str) -> str`.
  - `is_catalog_theme(name: str) -> bool`.
  - `STRIP_KEYS = ("background", "surface", "primary", "secondary", "accent", "success", "error")`.
  - `BASE_KEYS`: the editor's 10 base colours (`primary`, `secondary`, `accent`, `background`, `surface`, `panel`, `foreground`, `success`, `warning`, `error`).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Utils/test_theme_catalog.py
from textual.theme import BUILTIN_THEMES, Theme

from tldw_chatbook.css.Themes.theme_catalog import (
    STRIP_KEYS,
    build_catalog,
    display_name,
    is_catalog_theme,
)
from tldw_chatbook.css.Themes.themes import ALL_THEMES

SHIPPED = {t.name: t for t in ALL_THEMES}


def _available(**extra):
    themes = dict(BUILTIN_THEMES)
    themes.update(SHIPPED)
    themes.update(extra)
    return themes


def test_display_name_titles_slug():
    assert display_name("modern_dark_dracula") == "Modern Dark Dracula"
    assert display_name("textual-light") == "Textual Light"


def test_groups_and_order():
    mine = Theme(name="warm_paper", primary="#BD6B32", dark=False)
    entries = build_catalog(_available(warm_paper=mine), {"warm_paper"}, "warm_paper", "textual-dark")
    origins = [e.origin for e in entries]
    assert origins == sorted(origins, key=["yours", "shipped", "textual"].index)
    assert entries[0].id == "warm_paper" and entries[0].origin == "yours"
    assert {e.id for e in entries if e.origin == "textual"} == set(BUILTIN_THEMES)


def test_custom_prefixed_registrations_are_hidden():
    scratch = Theme(name="custom_apricot", primary="#123456")
    entries = build_catalog(_available(custom_apricot=scratch), set(), "custom_apricot", "textual-dark")
    assert all(not e.id.startswith("custom_") for e in entries)


def test_markers():
    entries = build_catalog(_available(), set(), "nord", "apricot")
    by_id = {e.id: e for e in entries}
    assert by_id["nord"].is_active and not by_id["nord"].is_launch_default
    assert by_id["apricot"].is_launch_default and not by_id["apricot"].is_active
    assert sum(e.is_active for e in entries) == 1


def test_custom_prefixed_active_marks_its_base_theme():
    mine = Theme(name="warm_paper", primary="#BD6B32", dark=False)
    scratch = Theme(name="custom_warm_paper", primary="#123456", dark=False)
    entries = build_catalog(
        _available(warm_paper=mine, custom_warm_paper=scratch), {"warm_paper"}, "custom_warm_paper", "x"
    )
    assert [e.id for e in entries if e.is_active] == ["warm_paper"]


def test_launch_default_not_registered_marks_nothing():
    entries = build_catalog(_available(), set(), "textual-dark", "deleted_theme")
    assert not any(e.is_launch_default for e in entries)


def test_user_file_overriding_shipped_and_textual():
    mine_apricot = Theme(name="apricot", primary="#000000")
    mine_nord = Theme(name="nord", primary="#111111")
    entries = build_catalog(
        _available(apricot=mine_apricot, nord=mine_nord), {"apricot", "nord"}, "nord", "nord"
    )
    by_id = {e.id: e for e in entries}
    assert by_id["apricot"].origin == "yours" and by_id["apricot"].overrides == "shipped"
    assert by_id["nord"].origin == "yours" and by_id["nord"].overrides == "textual"
    assert [e.id for e in entries].count("apricot") == 1


def test_strip_is_seven_resolved_hex_colours():
    entry = next(e for e in build_catalog(_available(), set(), "textual-dark", "x") if e.id == "textual-dark")
    assert len(entry.strip) == len(STRIP_KEYS) == 7
    assert all(c.startswith("#") and len(c) == 7 and c == c.upper() for c in entry.strip)


def test_duplicate_display_names_get_id_suffix():
    twin = Theme(name="solarized_light", primary="#268BD2", dark=False)
    entries = build_catalog(_available(solarized_light=twin), set(), "textual-dark", "x")
    names = [e.display_name for e in entries]
    assert len(names) == len(set(names))
    assert any(n.endswith("· solarized-light") for n in names)


def test_is_catalog_theme():
    assert is_catalog_theme("textual-dark") and is_catalog_theme("nord") and is_catalog_theme("apricot")
    assert not is_catalog_theme("warm_paper")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD $PY -m pytest Tests/Utils/test_theme_catalog.py -q -p no:cacheprovider`
Expected: FAIL, `ModuleNotFoundError: tldw_chatbook.css.Themes.theme_catalog`

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/css/Themes/theme_catalog.py
"""Single owner of the theme catalog and of switching themes (TASK-32948).

Pure data plus two small app-touching functions (``use_theme`` /
``revert_theme``); the picker, the palette and Appearance all read and
write through here so they can never disagree about which theme is
active or which one loads at launch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

from .themes import ALL_THEMES

Origin = Literal["yours", "shipped", "textual"]
STRIP_KEYS = ("background", "surface", "primary", "secondary", "accent", "success", "error")
BASE_KEYS = (
    "primary", "secondary", "accent", "background", "surface",
    "panel", "foreground", "success", "warning", "error",
)
_ORIGIN_ORDER: dict[str, int] = {"yours": 0, "shipped": 1, "textual": 2}
_SHIPPED_NAMES = frozenset(t.name for t in ALL_THEMES if getattr(t, "name", None))


@dataclass(frozen=True)
class ThemeEntry:
    id: str
    display_name: str
    origin: Origin
    dark: bool
    colours: tuple[tuple[str, str], ...]
    is_active: bool
    is_launch_default: bool
    overrides: Literal["shipped", "textual"] | None = None

    @property
    def strip(self) -> tuple[str, ...]:
        palette = dict(self.colours)
        return tuple(palette[key] for key in STRIP_KEYS)


def display_name(theme_id: str) -> str:
    return theme_id.replace("_", " ").replace("-", " ").title()


def is_catalog_theme(name: str) -> bool:
    """True for a shipped or Textual built-in theme name."""
    return name in _SHIPPED_NAMES or name in BUILTIN_THEMES


def _colours(theme: Theme) -> tuple[tuple[str, str], ...]:
    # Same resolution as the editor (TASK-31255): explicit colours byte-exact,
    # unset ones from the generated colour system.
    try:
        generated = theme.to_color_system().generate()
    except Exception:  # noqa: BLE001 - a malformed theme must not break the list
        generated = {}
    pairs: list[tuple[str, str]] = []
    for key in BASE_KEYS:
        raw = getattr(theme, key, None) or generated.get(key)
        try:
            pairs.append((key, Color.parse(str(raw)).hex.upper() if raw else "#808080"))
        except Exception:  # noqa: BLE001
            pairs.append((key, "#808080"))
    return tuple(pairs)


def _origin(name: str, user_names: Collection[str]) -> tuple[Origin, Literal["shipped", "textual"] | None]:
    if name in user_names:
        if name in _SHIPPED_NAMES:
            return "yours", "shipped"
        if name in BUILTIN_THEMES:
            return "yours", "textual"
        return "yours", None
    if name in BUILTIN_THEMES and name not in _SHIPPED_NAMES:
        return "textual", None
    return "shipped", None


def build_catalog(
    available: Mapping[str, Theme],
    user_names: Collection[str],
    active: str,
    launch_default: str,
) -> list[ThemeEntry]:
    """List every registered theme once, grouped yours → shipped → textual."""
    rows: list[tuple[str, Origin, Literal["shipped", "textual"] | None, Theme]] = []
    for name, theme in available.items():
        if name.startswith("custom_"):
            continue  # Apply's process-only registration (PR #2375 #8)
        origin, overrides = _origin(name, user_names)
        rows.append((name, origin, overrides, theme))
    counts = Counter(display_name(name) for name, *_ in rows)
    entries = [
        ThemeEntry(
            id=name,
            display_name=(
                f"{display_name(name)} · {name}" if counts[display_name(name)] > 1 else display_name(name)
            ),
            origin=origin,
            dark=bool(getattr(theme, "dark", True)),
            colours=_colours(theme),
            # Editor Apply runs `custom_<name>`; that is still <name> in use.
            is_active=active in (name, f"custom_{name}"),
            is_launch_default=name == launch_default,
            overrides=overrides,
        )
        for name, origin, overrides, theme in rows
    ]
    entries.sort(key=lambda e: (_ORIGIN_ORDER[e.origin], e.display_name.casefold()))
    return entries
```

Note: the `solarized_light` / `solarized-light` pair both render as "Solarized Light", so the test above passes with the real Textual built-in; the `twin` just guarantees the collision.

- [ ] **Step 4: Run the tests to verify they pass**

Run the same command. Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/Themes/theme_catalog.py Tests/Utils/test_theme_catalog.py
git commit -m "feat(theme): catalog of registered themes with origins and markers (TASK-32948)"
```

---

### Task 2: `use_theme`, `revert_theme`, and the palette

**Files:**
- Modify: `tldw_chatbook/css/Themes/theme_catalog.py`
- Modify: `tldw_chatbook/app.py:1214-1226` (`ThemeProvider.switch_theme`)
- Test: `Tests/Utils/test_theme_catalog.py`

**Interfaces:**
- Consumes: `display_name` (Task 1).
- Produces:
  - `ThemeChange` (frozen dataclass): `previous_active: str`, `previous_launch_default: str`, `persisted: bool`. Method `merge(later: ThemeChange) -> ThemeChange` keeps self's previous values and ORs `persisted`.
  - `current_launch_default() -> str`.
  - `use_theme(app, name: str, *, persist: bool) -> ThemeChange`. It raises `InvalidThemeError` for an unknown name *before* writing anything. In Textual 8.2.8 that class is `textual.app.InvalidThemeError`; `textual.theme` has no such name (verified).
  - `revert_theme(app, change: ThemeChange) -> None`.
  - `user_theme_names(directory: Path) -> set[str]`.

- [ ] **Step 1: Write the failing tests.** Append them to `Tests/Utils/test_theme_catalog.py`, with the new imports moved to the top of the file.

```python
import pytest
from types import SimpleNamespace

from textual.app import InvalidThemeError

from tldw_chatbook.css.Themes import theme_catalog as tc


class _FakeApp:
    def __init__(self, theme="textual-dark", known=("textual-dark", "nord", "apricot")):
        self._theme = theme
        self._known = set(known)
        self.app_config = {"general": {"default_theme": "textual-dark"}}

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if value not in self._known:
            raise InvalidThemeError(value)
        self._theme = value


@pytest.fixture
def writes(monkeypatch):
    calls = []

    def fake_apply(mutation):
        calls.append(mutation)
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", fake_apply)
    monkeypatch.setattr(tc, "current_launch_default", lambda: "textual-dark")
    return calls


def test_use_theme_persist_writes_config_and_app_config(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    assert app.theme == "nord"
    assert writes == [{"general": {"default_theme": "nord"}}]
    assert app.app_config["general"]["default_theme"] == "nord"
    assert change == tc.ThemeChange("textual-dark", "textual-dark", True)


def test_try_does_not_write(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=False)
    assert app.theme == "nord" and writes == [] and change.persisted is False


def test_use_theme_unknown_name_raises_and_writes_nothing(writes):
    app = _FakeApp()
    with pytest.raises(InvalidThemeError):
        tc.use_theme(app, "no_such_theme", persist=True)
    assert writes == [] and app.theme == "textual-dark"


def test_use_theme_persist_failure_reports_not_persisted(monkeypatch):
    monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=False, caches_reloaded=False))
    monkeypatch.setattr(tc, "current_launch_default", lambda: "textual-dark")
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    assert app.theme == "nord" and change.persisted is False
    assert app.app_config["general"]["default_theme"] == "textual-dark"


def test_revert_restores_active_and_launch_default(writes):
    app = _FakeApp()
    change = tc.use_theme(app, "nord", persist=True)
    tc.revert_theme(app, change)
    assert app.theme == "textual-dark"
    assert writes[-1] == {"general": {"default_theme": "textual-dark"}}


def test_revert_of_try_writes_nothing(writes):
    app = _FakeApp()
    tc.revert_theme(app, tc.use_theme(app, "nord", persist=False))
    assert app.theme == "textual-dark" and writes == []


def test_merge_keeps_first_previous_values(writes):
    app = _FakeApp()
    first = tc.use_theme(app, "apricot", persist=False)   # Try apricot
    second = tc.use_theme(app, "nord", persist=True)      # then Use nord
    merged = first.merge(second)
    assert merged == tc.ThemeChange("textual-dark", "textual-dark", True)
    tc.revert_theme(app, merged)
    assert app.theme == "textual-dark"
    assert writes[-1] == {"general": {"default_theme": "textual-dark"}}


def test_user_theme_names_reads_toml_stems(tmp_path):
    (tmp_path / "warm_paper.toml").write_text("[theme]\nname='warm_paper'\n")
    (tmp_path / "notes.txt").write_text("x")
    assert tc.user_theme_names(tmp_path) == {"warm_paper"}
    assert tc.user_theme_names(tmp_path / "missing") == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL, `AttributeError: ... has no attribute 'use_theme'`.

- [ ] **Step 3: Implement** (append to `theme_catalog.py`, and add `from pathlib import Path` and `from typing import Any` to its imports)

```python
@dataclass(frozen=True)
class ThemeChange:
    previous_active: str
    previous_launch_default: str
    persisted: bool

    def merge(self, later: "ThemeChange") -> "ThemeChange":
        """Chain a later change after this one; Revert goes back to the first."""
        return ThemeChange(self.previous_active, self.previous_launch_default, self.persisted or later.persisted)


def current_launch_default() -> str:
    from ...config import get_cli_setting

    return str(get_cli_setting("general", "default_theme", "textual-dark"))


def _apply_config_mutation(mutation: dict) -> Any:
    from ...config import apply_settings_mutation_to_cli_config

    return apply_settings_mutation_to_cli_config(mutation)


def _persist_launch_default(app: Any, name: str) -> bool:
    result = _apply_config_mutation({"general": {"default_theme": name}})
    if not getattr(result, "file_replaced", False):
        return False
    # The in-memory copy Settings reads (was handle_theme_launch_default_changed).
    config = getattr(app, "app_config", None)
    if isinstance(config, dict):
        general = dict(config.get("general", {}))
        general["default_theme"] = name
        config["general"] = general
    return True


def use_theme(app: Any, name: str, *, persist: bool) -> ThemeChange:
    """Switch the app theme; with ``persist`` also make it the launch default.

    Raises textual.app.InvalidThemeError (before any write) for an
    unregistered name.
    """
    previous_active = str(app.theme)
    previous_launch = current_launch_default()
    app.theme = name
    persisted = _persist_launch_default(app, name) if persist else False
    return ThemeChange(previous_active, previous_launch, persisted)


def revert_theme(app: Any, change: ThemeChange) -> None:
    app.theme = change.previous_active
    if change.persisted:
        _persist_launch_default(app, change.previous_launch_default)


def user_theme_names(directory: Path) -> set[str]:
    """Stems of the saved theme files (the editor saves ``<name>.toml``)."""
    try:
        return {path.stem for path in directory.glob("*.toml")}
    except OSError:
        return set()
```

- [ ] **Step 4: Route the palette through `use_theme`.** Replace the body of `ThemeProvider.switch_theme` in `tldw_chatbook/app.py`:

```python
    def switch_theme(self, theme_name: str) -> None:
        """Switch to the specified theme and keep it as the launch default."""
        from .css.Themes.theme_catalog import display_name, use_theme

        try:
            change = use_theme(self.app, theme_name, persist=True)
        except Exception as e:  # noqa: BLE001 - palette commands must not raise
            self.app.notify(f"Failed to apply theme: {e}", severity="error")
            return
        if change.persisted:
            self.app.notify(f"{display_name(theme_name)} is now your theme", severity="information")
        else:
            self.app.notify(
                f"{display_name(theme_name)} applied; the launch default was not saved",
                severity="warning",
            )
```

Then remove `save_setting_to_cli_config` from `app.py`'s imports if nothing else in the file uses it (`grep -n save_setting_to_cli_config tldw_chatbook/app.py`).

- [ ] **Step 5: Run the tests** (the catalog tests, plus `grep -rln "switch_theme" Tests` and run those files). Expected: all pass. If an existing palette test asserts the old toast `"Theme changed to"`, update the assertion to the new copy.

- [ ] **Step 6: Commit** with the message `feat(theme): use_theme/revert_theme own theme switching; palette routes through it (TASK-32948)`.

---

### Task 3: Extract `ThemePreview`

**Files:**
- Create: `tldw_chatbook/Widgets/theme_preview.py`
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`_PREVIEW_ROWS`, `_PREVIEW_STYLE`, `_compose_preview_section`, `_refresh_preview`)
- Test: `Tests/UI/test_settings_theme_picker.py` (new file)

**Interfaces:**
- Produces: `ThemePreview(Vertical)` with `ThemePreview(prefix: str, *, compact: bool = False, **kwargs)`.
  - Row ids are `f"{prefix}-{suffix}"`, keeping the editor's existing `settings-theme-preview-<suffix>` ids.
  - `paint(colours: Mapping[str, str]) -> None`.
  - `compact=True` renders only the `rail` and `accent` rows. That is the 2-row narrow preview from spec §5.

- [ ] **Step 1: Write the failing test**

```python
# Tests/UI/test_settings_theme_picker.py
import pytest
from textual.app import ComposeResult

from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Widgets.theme_preview import ThemePreview


def _app(*widgets):
    def compose() -> ComposeResult:
        yield from widgets

    return IsolatedWidgetTestApp(compose)


@pytest.mark.asyncio
async def test_theme_preview_paints_rows_from_colours():
    preview = ThemePreview("pv")
    async with _app(preview).run_test(size=(80, 20)) as pilot:
        preview.paint({"panel": "#112233", "foreground": "#EEEEEE", "accent": "#FF8800", "background": "#000000"})
        await pilot.pause()
        rail = preview.query_one("#pv-rail")
        assert rail.styles.background.hex.upper() == "#112233"
        assert "[ Send ]" in str(preview.query_one("#pv-accent").render())


@pytest.mark.asyncio
async def test_compact_preview_has_two_rows():
    preview = ThemePreview("pv", compact=True)
    async with _app(preview).run_test(size=(80, 20)):
        assert [w.id for w in preview.children] == ["pv-rail", "pv-accent"]
```

- [ ] **Step 2: Run the test.** Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement** by moving the editor's two tables verbatim into the new module:

```python
# tldw_chatbook/Widgets/theme_preview.py
"""Console-shaped theme preview, painted from a colours mapping (TASK-31259, TASK-32948)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

PREVIEW_ROWS = (
    ("rail", " Console ▸ Conversation · ready"),
    ("user", " You: summarise the attached paper"),
    ("assistant", " Assistant: Here is the summary…"),
    ("success", " ✓ tool web_search finished"),
    ("warning", " ! approval needed before the next call"),
    ("error", " ✗ provider returned 401"),
    ("accent", " [ Send ]   Ctrl+P palette"),
)
PREVIEW_STYLE = {
    "rail": ("panel", "foreground"),
    "user": ("primary", "foreground"),
    "assistant": ("surface", "foreground"),
    "success": ("background", "success"),
    "warning": ("background", "warning"),
    "error": ("background", "error"),
    "accent": ("background", "accent"),
}
_COMPACT_ROWS = ("rail", "accent")


class ThemePreview(Vertical):
    def __init__(self, prefix: str, *, compact: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("classes", "settings-theme-preview")
        super().__init__(**kwargs)
        self._prefix = prefix
        self._rows = tuple(r for r in PREVIEW_ROWS if not compact or r[0] in _COMPACT_ROWS)

    def compose(self) -> ComposeResult:
        for suffix, text in self._rows:
            yield Static(
                text,
                id=f"{self._prefix}-{suffix}",
                classes="settings-theme-preview-row",
                markup=False,  # task-32946: "[ Send ]" is literal text
            )

    def paint(self, colours: Mapping[str, str]) -> None:
        for suffix, _text in self._rows:
            background_key, foreground_key = PREVIEW_STYLE[suffix]
            try:
                row = self.query_one(f"#{self._prefix}-{suffix}", Static)
            except Exception:  # noqa: BLE001 - not mounted yet
                return
            try:
                if background := colours.get(background_key):
                    row.set_styles(background=background)  # ds-runtime: previewed palette
                if foreground := colours.get(foreground_key):
                    row.set_styles(color=foreground)  # ds-runtime: previewed palette
            except Exception:  # noqa: BLE001 - a half-typed hex must not break painting
                continue
```

In the editor:
- Delete `_PREVIEW_ROWS` and `_PREVIEW_STYLE`.
- Replace the `Vertical`/`Static` loop in `_compose_preview_section` with `yield ThemePreview("settings-theme-preview", id="settings-theme-preview")`.
- Replace `_refresh_preview`'s body with:

```python
        try:
            self.query_one("#settings-theme-preview", ThemePreview).paint(self.current_theme_data)
        except QueryError:
            return
```

Also add `from .theme_preview import ThemePreview` to the editor's imports.

- [ ] **Step 4: Run** `Tests/UI/test_settings_theme_picker.py`, `Tests/UI/test_settings_theme_editor.py` and `Tests/UI/test_settings_theme_editor_render.py`. Expected: all pass. The row ids are unchanged, and `ThemePreview` sets the `settings-theme-preview` class the CSS already targets.

- [ ] **Step 5: Commit** with the message `refactor(theme): extract ThemePreview from the editor for reuse by the picker (TASK-32948)`.

---

### Task 4: `ThemePicker`

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_picker.py` (create it here)
- Test: `Tests/UI/test_settings_theme_picker.py`

**Interfaces:**
- Consumes: `build_catalog`, `ThemeEntry`, `display_name`, `use_theme`, `revert_theme`, `ThemeChange`, `current_launch_default`, `user_theme_names` (Tasks 1–2), and `ThemePreview` (Task 3).
- Produces:
  - `ThemePicker(Vertical)`, which has `id="settings-theme-picker"` when mounted by the pane.
  - Message `ThemePicker.EditRequested(theme_id: str, mode: Literal["clone","new"])`.
  - Attributes `entries: list[ThemeEntry]` and `highlighted_id: str | None`.
  - Method `refresh_catalog()`.
  - Child ids:
    - `#settings-theme-filter` (a `ThemeFilterInput`)
    - `#settings-theme-list` (a `ThemeOptionList`)
    - `#settings-theme-card-title` (a `Static`)
    - `#settings-theme-picker-preview` (a `ThemePreview`)
    - Buttons: `#settings-theme-use`, `#settings-theme-try`, `#settings-theme-clone`, `#settings-theme-new`, and `#settings-theme-revert` (hidden until there is something to revert)
    - `#settings-theme-empty` (a `Static`, shown when the filter matches nothing)

- [ ] **Step 1: Write the failing tests.** Append the tests to `Tests/UI/test_settings_theme_picker.py`, and move the new import lines to the top of the file, next to Task 3's imports (ruff E402).

```python
from types import SimpleNamespace

from textual import on
from textual.widgets import Button, OptionList

from tldw_chatbook.css.Themes import theme_catalog as tc
from tldw_chatbook.css.Themes.themes import ALL_THEMES
from tldw_chatbook.Widgets.settings_theme_picker import ThemePicker


@pytest.fixture
def config_writes(monkeypatch, tmp_path):
    calls = []
    state = {"launch": "textual-dark"}

    def fake_apply(mutation):
        calls.append(mutation)
        state["launch"] = mutation["general"]["default_theme"]
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", fake_apply)
    monkeypatch.setattr(tc, "current_launch_default", lambda: state["launch"])
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.settings_theme_picker.get_user_themes_dir", lambda: tmp_path
    )
    # The picker imported the name, so patch its copy too.
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.settings_theme_picker.current_launch_default", lambda: state["launch"]
    )
    return calls


async def _picker_app(size=(160, 45)):
    picker = ThemePicker(id="settings-theme-picker")
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    return app, picker


def _option_ids(picker):
    lst = picker.query_one("#settings-theme-list", OptionList)
    return [lst.get_option_at_index(i).id for i in range(lst.option_count)]


@pytest.mark.asyncio
async def test_rows_show_markers_as_words(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        active = next(lst.get_option_at_index(i) for i in range(lst.option_count) if lst.get_option_at_index(i).id == app.theme)
        assert "active" in str(active.prompt) and "launch" in str(active.prompt)


@pytest.mark.asyncio
async def test_filter_narrows_and_enter_uses(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"apricot")
        await pilot.pause()
        assert "apricot" in _option_ids(picker)
        assert "nord" not in _option_ids(picker)
        await pilot.press("enter")          # filter -> list
        await pilot.pause()
        assert app.focused.id == "settings-theme-list"
        await pilot.press("enter")          # Use
        await pilot.pause()
        assert app.theme == picker.highlighted_id
        assert config_writes[-1] == {"general": {"default_theme": app.theme}}
        assert picker.query_one("#settings-theme-revert", Button).display


@pytest.mark.asyncio
async def test_filter_no_match_enter_is_inert(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        before = app.theme
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"zzzqqq", "enter")
        await pilot.pause()
        assert app.theme == before and config_writes == []
        assert picker.query_one("#settings-theme-empty").display
        assert "No themes match 'zzzqqq'" in str(picker.query_one("#settings-theme-empty").render())


@pytest.mark.asyncio
async def test_highlight_repaints_preview_not_app(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        before = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        await pilot.press("down", "down")
        await pilot.pause()
        assert app.theme == before
        assert picker.highlighted_id != before
        assert (
            display_name_for(picker, picker.highlighted_id)
            in str(picker.query_one("#settings-theme-card-title").render())
        )


def display_name_for(picker, theme_id):
    return next(e.display_name for e in picker.entries if e.id == theme_id)


@pytest.mark.asyncio
async def test_try_then_use_then_revert_restores_original(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        original = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        await pilot.press("down", "t")      # Try
        await pilot.pause()
        tried = app.theme
        assert tried != original and config_writes == []
        await pilot.press("down", "enter")  # Use another
        await pilot.pause()
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert app.theme == original
        assert config_writes[-1] == {"general": {"default_theme": "textual-dark"}}
        assert not picker.query_one("#settings-theme-revert", Button).display


@pytest.mark.asyncio
async def test_up_on_first_row_returns_to_filter(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index(_option_ids(picker)[1])  # first enabled row
        await pilot.press("up")
        await pilot.pause()
        assert app.focused.id == "settings-theme-filter"


@pytest.mark.asyncio
async def test_markers_follow_external_theme_change(config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        app.theme = "nord"                  # e.g. the palette
        await pilot.pause()
        assert next(e for e in picker.entries if e.id == "nord").is_active


class _CaptureEditApp(IsolatedWidgetTestApp):
    def __init__(self, compose):
        super().__init__(compose)
        self.edits: list[tuple[str, str]] = []

    @on(ThemePicker.EditRequested)
    def _capture(self, message: ThemePicker.EditRequested) -> None:
        self.edits.append((message.mode, message.theme_id))


@pytest.mark.asyncio
async def test_clone_and_new_post_edit_requested(config_writes):
    picker = ThemePicker(id="settings-theme-picker")

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list", OptionList).focus()
        await pilot.press("c", "n")
        await pilot.pause()
        assert [mode for mode, _ in app.edits] == ["clone", "new"]
        assert app.edits[0][1] == picker.highlighted_id


@pytest.mark.asyncio
async def test_use_toast_when_persist_fails(monkeypatch, config_writes):
    monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=False, caches_reloaded=False))
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append(message)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert any("launch default was not saved" in n for n in notes)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL, `ImportError: ThemePicker`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Widgets/settings_theme_picker.py
"""Settings ▸ Theme picker and the picker/editor pane (TASK-32948)."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from loguru import logger
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..config import get_user_themes_dir
from ..css.Themes.theme_catalog import (
    ThemeChange,
    ThemeEntry,
    build_catalog,
    current_launch_default,
    display_name,
    revert_theme,
    use_theme,
    user_theme_names,
)
from .theme_preview import ThemePreview

_GROUP_TITLES = {"yours": "YOUR THEMES", "shipped": "SHIPPED", "textual": "TEXTUAL"}


def _row(entry: ThemeEntry) -> Text:
    text = Text(f"{entry.display_name}  ")
    for colour in entry.strip:
        text.append("▮", Style(color=colour))
    markers = [m for m, on_ in (("active", entry.is_active), ("launch", entry.is_launch_default)) if on_]
    if entry.overrides:
        markers.append(f"overrides {entry.overrides}")
    if markers:
        text.append("  " + " · ".join(markers))
    return text


class ThemeFilterInput(Input):
    BINDINGS: ClassVar[list[Binding]] = [Binding("down", "to_list", "List", show=False)]

    def action_to_list(self) -> None:
        self.parent_picker().focus_list()

    def parent_picker(self) -> "ThemePicker":
        return self.query_ancestor(ThemePicker)


class ThemeOptionList(OptionList):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "try_theme", "Try"),
        Binding("c", "clone_theme", "Clone"),
        Binding("n", "new_theme", "New"),
    ]

    def _first_enabled_index(self) -> int | None:
        for index in range(self.option_count):
            if not self.get_option_at_index(index).disabled:
                return index
        return None

    def action_cursor_up(self) -> None:
        # OptionList wraps at the ends (find_next_enabled); the picker wants
        # the top row to hand focus back to the filter instead (spec §5).
        if self.highlighted is None or self.highlighted == self._first_enabled_index():
            self.query_ancestor(ThemePicker).query_one("#settings-theme-filter").focus()
            return
        super().action_cursor_up()

    def action_try_theme(self) -> None:
        self.query_ancestor(ThemePicker).try_highlighted()

    def action_clone_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("clone")

    def action_new_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("new")


class ThemePicker(Vertical):
    class EditRequested(Message):
        def __init__(self, theme_id: str, mode: Literal["clone", "new"]) -> None:
            self.theme_id = theme_id
            self.mode = mode
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entries: list[ThemeEntry] = []
        self.highlighted_id: str | None = None
        self._revert: ThemeChange | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings-theme-picker-columns"):
            with Vertical(id="settings-theme-list-column"):
                yield ThemeFilterInput(placeholder="Filter themes", id="settings-theme-filter")
                yield ThemeOptionList(id="settings-theme-list")
                yield Static("", id="settings-theme-empty", classes="settings-help-copy", markup=False)
            with Vertical(id="settings-theme-card-column"):
                yield Static("", id="settings-theme-card-title", classes="destination-section", markup=False)
                yield ThemePreview("settings-theme-picker-preview", id="settings-theme-picker-preview")
                with Horizontal(classes="settings-action-row"):
                    yield Button("Use this theme", id="settings-theme-use", variant="primary", classes="theme-editor-action")
                    yield Button("Try", id="settings-theme-try", classes="theme-editor-action")
                    yield Button("Clone", id="settings-theme-clone", classes="theme-editor-action")
                    yield Button("New", id="settings-theme-new", classes="theme-editor-action")
                    yield Button("Revert", id="settings-theme-revert", classes="theme-editor-action")

    def on_mount(self) -> None:
        self.query_one("#settings-theme-revert").display = False
        self.query_one("#settings-theme-empty").display = False
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_catalog())
        self.refresh_catalog(highlight=str(self.app.theme))

    # -- catalog -------------------------------------------------------
    def refresh_catalog(self, highlight: str | None = None) -> None:
        self.entries = build_catalog(
            self.app.available_themes,
            user_theme_names(get_user_themes_dir()),
            str(self.app.theme),
            current_launch_default(),
        )
        self._render_list(highlight or self.highlighted_id)

    def _render_list(self, highlight: str | None) -> None:
        query = self.query_one("#settings-theme-filter", Input).value.strip().casefold()
        shown = [e for e in self.entries if not query or query in e.display_name.casefold() or query in e.id.casefold()]
        options: list[Option] = []
        for origin, title in _GROUP_TITLES.items():
            group = [e for e in shown if e.origin == origin]
            if not group:
                continue
            header = f"{title} ({len(group)})" if query else title
            options.append(Option(header, disabled=True))
            options.extend(Option(_row(e), id=e.id) for e in group)
        lst = self.query_one("#settings-theme-list", OptionList)
        lst.clear_options()
        lst.add_options(options)
        empty = self.query_one("#settings-theme-empty", Static)
        empty.display = not shown
        empty.update(f"No themes match '{query}'" if not shown else "")
        ids = [e.id for e in shown]
        target = highlight if highlight in ids else (ids[0] if ids else None)
        if target is not None:
            lst.highlighted = lst.get_option_index(target)
        self._show(target)

    def _show(self, theme_id: str | None) -> None:
        self.highlighted_id = theme_id
        entry = next((e for e in self.entries if e.id == theme_id), None)
        title = self.query_one("#settings-theme-card-title", Static)
        for button_id in ("#settings-theme-use", "#settings-theme-try", "#settings-theme-clone"):
            self.query_one(button_id, Button).disabled = entry is None
        if entry is None:
            title.update("")
            return
        tone = "dark" if entry.dark else "light"
        title.update(f"{entry.display_name}  ·  {tone} · {entry.origin}")
        self.query_one(ThemePreview).paint(dict(entry.colours))

    def focus_list(self) -> None:
        lst = self.query_one("#settings-theme-list", OptionList)
        if self.highlighted_id is not None:
            lst.focus()

    # -- events --------------------------------------------------------
    @on(Input.Changed, "#settings-theme-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._render_list(self.highlighted_id)

    @on(Input.Submitted, "#settings-theme-filter")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.focus_list()

    @on(OptionList.OptionHighlighted, "#settings-theme-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self._show(event.option.id)

    @on(OptionList.OptionSelected, "#settings-theme-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.use_highlighted()

    @on(Button.Pressed, "#settings-theme-use")
    def _use_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.use_highlighted()

    @on(Button.Pressed, "#settings-theme-try")
    def _try_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.try_highlighted()

    @on(Button.Pressed, "#settings-theme-clone")
    def _clone_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("clone")

    @on(Button.Pressed, "#settings-theme-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("new")

    @on(Button.Pressed, "#settings-theme-revert")
    def _revert_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._revert is None:
            return
        change, self._revert = self._revert, None
        try:
            revert_theme(self.app, change)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Theme revert failed: {exc}")
            self.app.notify(f"Could not revert the theme: {exc}", severity="error")
        self.query_one("#settings-theme-revert").display = False
        self.refresh_catalog()

    # -- actions -------------------------------------------------------
    def use_highlighted(self) -> None:
        self._switch(persist=True)

    def try_highlighted(self) -> None:
        self._switch(persist=False)

    def request_edit(self, mode: Literal["clone", "new"]) -> None:
        if self.highlighted_id is not None:
            self.post_message(self.EditRequested(self.highlighted_id, mode))

    def _switch(self, *, persist: bool) -> None:
        theme_id = self.highlighted_id
        if theme_id is None:
            return
        try:
            change = use_theme(self.app, theme_id, persist=persist)
        except Exception as exc:  # noqa: BLE001 - stale entry / unregistered theme
            self.app.notify(f"Could not apply {display_name(theme_id)}: {exc}", severity="error")
            self.refresh_catalog()
            return
        self._revert = change if self._revert is None else self._revert.merge(change)
        was = display_name(self._revert.previous_active)
        revert = self.query_one("#settings-theme-revert", Button)
        revert.label = f"Revert to {was}"
        revert.display = True
        if not persist:
            self.app.notify(f"Trying {display_name(theme_id)} for this session", severity="information")
        elif change.persisted:
            self.app.notify(f"{display_name(theme_id)} is now your theme (was: {display_name(change.previous_active)})", severity="information")
        else:
            self.app.notify(f"{display_name(theme_id)} applied; the launch default was not saved", severity="warning")
        self.refresh_catalog(highlight=theme_id)
```

Notes for the implementer:
- `Option(prompt, id=None, disabled=True)` is the Textual 8 signature. Group headers have no id, so `OptionHighlighted` never fires for them (they are disabled).
- `query_ancestor` exists on `DOMNode` in Textual 8.2.8 (verified).
- `theme_changed_signal.subscribe(self, …)` is released automatically when the widget unmounts; Textual signals hold weak references to their subscribers.

- [ ] **Step 4: Run** `Tests/UI/test_settings_theme_picker.py`. Expected: all pass. Then run `Tests/Architecture/test_no_blocking_io_on_message_pump.py`. That guard walks the call graph from message handlers and flags calls such as `.glob(`.
  - The picker's `on_mount` → `refresh_catalog` → `user_theme_names` path globs the themes directory. The glob lives in `css/Themes/`, outside the guard's scanned `UI`/`Widgets` directories, so it is expected to pass.
  - If the guard flags it anyway, do **not** add a baseline entry blindly. Read the file's header on scope. Measure the glob on a directory of 50 files (expect well under 1 ms, the "small local filesystem operation" class the header exempts). Follow the file's documented procedure for that class, and cite the measurement in the Notes.

- [ ] **Step 5: Commit** with the message `feat(theme): ThemePicker — filter, colour strips, preview card, Use/Try/Revert, live markers (TASK-32948)`.

---

### Task 5: `ThemePane` and the Settings wiring

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_picker.py` (add `ThemePane`)
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py`:
  - Remove the "Set as launch default" button (`_compose_actions_section`) and `on_set_launch_default`.
  - Change the apply hint copy.
  - `_is_catalog_theme` → `is_catalog_theme`.
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`:
  - The Theme branch of `_render_detail_pane` (~21385) mounts `ThemePane`.
  - The detail-pane region (~22315) drops the pinned banner for Theme.
  - `_category_state_banner_text` and `_category_state_scope_text` lose their THEME branches.
  - The inspector copy at ~2152 and ~16048 mentions Use/Try.
- Test: `Tests/UI/test_settings_theme_picker_screen.py` (new)

**Interfaces:**
- Consumes: `ThemePicker`, `ThemePicker.EditRequested` (Task 4); the existing `SettingsThemeEditor` methods `load_theme(name)`, `load_user_theme(name)`, `on_clone_theme()`, `on_new_theme()` and `is_modified`; `ThemeLeaveModal` (TASK-32941).
- Produces:
  - `ThemePane(ContentSwitcher)` with `id="settings-theme-pane"`. Its children are `ThemePicker(id="settings-theme-picker")` and `Vertical(id="settings-theme-editor-view")`. The editor view holds `Button("Back to themes", id="settings-theme-back")` and `SettingsThemeEditor(id="settings-theme-editor")`.
  - `show_picker()`.
  - `open_editor(theme_id, mode)`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_settings_theme_picker_screen.py
import pytest
from textual.widgets import ContentSwitcher

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.css.Themes.themes import ALL_THEMES


def _host():
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    for theme in ALL_THEMES:
        host.register_theme(theme)
    return host


@pytest.mark.asyncio
@private_profile_test
async def test_theme_category_opens_on_the_picker_without_state_banner(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"
        assert not host.screen.query("#settings-category-state-banner")


@pytest.mark.asyncio
@private_profile_test
async def test_clone_opens_editor_and_back_returns(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"
        editor = host.screen.query_one("#settings-theme-editor")
        assert editor.current_theme_name.endswith("_copy")
        assert not host.screen.query("#settings-theme-set-default")
        editor.is_modified = False
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_back_with_unsaved_edits_asks_first(request):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        host.screen.query_one("#settings-theme-editor").is_modified = True
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert isinstance(host.screen, ThemeLeaveModal)
        await pilot.press("escape")          # Stay
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"
```

- [ ] **Step 2: Run the tests.** Expected: FAIL, because there is no `#settings-theme-pane`.

- [ ] **Step 3: Implement `ThemePane`** (append to `settings_theme_picker.py`, and import `ContentSwitcher` from `textual.widgets`):

```python
class ThemePane(ContentSwitcher):
    """Picker first; the editor swaps in for Clone/New (spec §4, D3)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(initial="settings-theme-picker", **kwargs)

    def compose(self) -> ComposeResult:
        from .settings_theme_editor import SettingsThemeEditor

        yield ThemePicker(id="settings-theme-picker")
        with Vertical(id="settings-theme-editor-view"):
            with Horizontal(classes="settings-action-row"):
                yield Button("Back to themes", id="settings-theme-back", classes="theme-editor-action")
            yield SettingsThemeEditor(id="settings-theme-editor")

    def show_picker(self) -> None:
        self.current = "settings-theme-picker"
        picker = self.query_one(ThemePicker)
        picker.refresh_catalog()
        picker.focus_list()

    def open_editor(self, theme_id: str, mode: Literal["clone", "new"]) -> None:
        from .settings_theme_editor import SettingsThemeEditor

        editor = self.query_one(SettingsThemeEditor)
        picker = self.query_one(ThemePicker)
        entry = next((e for e in picker.entries if e.id == theme_id), None)
        if entry is not None and entry.origin == "yours":
            editor.load_user_theme(theme_id)
        else:
            editor.load_theme(theme_id)
        if mode == "clone":
            editor.on_clone_theme()
        else:
            editor.on_new_theme()
        self.current = "settings-theme-editor-view"
        editor.query_one("#settings-theme-name").focus()

    @on(ThemePicker.EditRequested)
    def _edit_requested(self, event: ThemePicker.EditRequested) -> None:
        event.stop()
        self.open_editor(event.theme_id, event.mode)
```

- [ ] **Step 4: Handle Back in `SettingsScreen`.** Back reuses the existing leave-guard pattern, so the modal and the Save path stay in one place. Next to `_confirm_theme_category_leave`, add:

```python
    @on(Button.Pressed, "#settings-theme-back")
    def handle_theme_back(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            editor = self.query_one("#settings-theme-editor", SettingsThemeEditor)
            pane = self.query_one("#settings-theme-pane", ThemePane)
        except QueryError:
            return
        if not editor.is_modified:
            pane.show_picker()
            return
        self.run_worker(self._confirm_theme_back(pane, editor), group="theme-back", exit_on_error=False)

    async def _confirm_theme_back(self, pane: ThemePane, editor: SettingsThemeEditor) -> None:
        choice = await self.app.push_screen_wait(ThemeLeaveModal())
        if choice == "cancel":
            return
        if choice == "save":
            editor.on_save_theme()
            if editor.is_modified:
                return  # refused name or pending overwrite confirmation: stay
        else:
            editor.is_modified = False
            self.theme_editor_modified = False
            self._refresh_theme_modified_widgets()
        pane.show_picker()
```

- [ ] **Step 5: Mount the pane and drop the banner.**
  - Change the import to `from ...Widgets.settings_theme_picker import ThemePane`.
  - In `_render_detail_pane`, replace `yield SettingsThemeEditor(id="settings-theme-editor")` with `yield ThemePane(id="settings-theme-pane")`.
  - In the detail-pane region method (~22315), change the unconditional banner line to:

```python
        if active_summary.category is not SettingsCategoryId.THEME:
            yield self._render_category_state_banner(active_summary.category)
```

  - Delete the two `if category is SettingsCategoryId.THEME:` branches in `_category_state_banner_text` and `_category_state_scope_text`.
  - Search the file for other THEME-specific banner references (`grep -n "Managed in editor\|Use the editor's Apply" tldw_chatbook/UI/Screens/settings_screen.py`) and remove them. `_update_category_state_banner` already swallows `QueryError`; confirm this at ~9351.
  - Rewrite both copies of the inspector string `"Apply changes this session; Save stores a theme file; Set as launch default updates general.default_theme"` as `"Use applies a theme and keeps it at launch; Try applies it for this session; Clone or New opens the editor"`.

- [ ] **Step 6: Clean up the editor.**
  - Delete the `Button("Set as launch default", id="settings-theme-set-default", …)` yield and the `on_set_launch_default` handler.
  - **Keep** `_save_launch_default`, because Delete still uses it until PR 2.
  - Change the apply hint to `"Apply previews this palette now - no Save needed; Save stores the theme file."`
  - Replace the editor's local `_is_catalog_theme` body with `return is_catalog_theme(name)` (`from ..css.Themes.theme_catalog import is_catalog_theme`).
  - Run `grep -rn "settings-theme-set-default\|on_set_launch_default" Tests tldw_chatbook`. For each hit:
    - A test of the removed button: delete it, and add a line to the Implementation Notes' "Retired tests" list naming it and its replacement, `test_try_then_use_then_revert_restores_original` / `test_filter_narrows_and_enter_uses`.
    - The contrast test's `PLAIN` tuple: drop `"set-default"`.

- [ ] **Step 7: Run** the new screen tests, the picker tests, `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_theme_card_contrast.py`, and `Tests/UI/test_settings_configuration_hub.py -k "theme or Theme"` (the `@private_profile_test` ones). Expected: all pass.
  - If a hub test asserts the old Theme banner text or the direct `#settings-theme-editor` mount at the pane root, update it. The editor is still findable at `#settings-theme-editor`.
  - Record every test you change in the Notes.

- [ ] **Step 8: Commit** with the message `feat(settings): Theme opens on the picker; editor behind Clone/New with guarded Back (TASK-32948)`.

---

### Task 6: Appearance read-only theme row

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`:
  - The Appearance compose (~21150): replace the theme `Select` row.
  - `_appearance_field_selector` (~9427), `_update_appearance_validation_classes` (~9457), `_appearance_invalid_field_key` (~9385): drop `default_theme`.
  - `_sync_appearance_widgets` (~31857): drop the Select sync.
  - `handle_appearance_theme_changed` (~24363): delete it.
  - The field guidance for `settings-appearance-theme` (~15774): retarget it to the new row.
  - `handle_theme_launch_default_changed` (~24231): keep only the category label refresh.
  - The Appearance preview (~30773): stop setting `app.theme`.
  - `_appearance_theme_options` (~9511): delete it if it's unused after this.
- Modify: `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`: the save payload (~438) no longer writes `default_theme`; validation (~291) no longer requires it.
- Test: `Tests/UI/test_settings_theme_picker_screen.py`

**Interfaces:**
- Consumes: `current_launch_default`, `display_name` (Task 1/2), `ThemePane` (Task 5).
- Produces: `#settings-appearance-theme-summary` (a `Static`) and `#settings-appearance-open-theme` (a `Button`).

- [ ] **Step 1: Write the failing tests** (append)

```python
@pytest.mark.asyncio
@private_profile_test
async def test_appearance_shows_read_only_theme_row_and_opens_picker(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        assert not host.screen.query("#settings-appearance-theme")
        summary = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "launch default" in summary and "active" in summary
        host.theme = "nord"  # e.g. from the palette, while Appearance is open
        await pilot.pause(0.2)
        assert "active: Nord" in str(host.screen.query_one("#settings-appearance-theme-summary").render())
        await pilot.click("#settings-appearance-open-theme")
        await pilot.pause(0.3)
        assert host.screen.query_one("#settings-theme-pane").current == "settings-theme-picker"


def test_appearance_save_never_writes_default_theme():
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="textual-light")
    sections = sad.build_appearance_save_sections({"general": {"default_theme": "nord"}}, values)
    # The existing launch default passes through untouched; the draft value never lands.
    assert sections["general"]["default_theme"] == "nord"


def test_appearance_validation_ignores_theme():
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="")
    assert sad.validate_appearance_defaults(values).valid
```

- [ ] **Step 2: Run the tests.** Expected: FAIL, because `#settings-appearance-theme` still exists.

- [ ] **Step 3: Implement the row.** In the Appearance compose, replace the `with Horizontal(... settings-select-row): Static("Theme"…); Select(...)` block with:

```python
                with Horizontal(classes="settings-input-row"):
                    yield Static("Theme", classes="settings-input-label")
                    yield Static(
                        self._appearance_theme_summary(),
                        id="settings-appearance-theme-summary",
                        classes="settings-detail-row",
                        markup=False,
                    )
                    yield Button(
                        "Open Theme",
                        id="settings-appearance-open-theme",
                        classes="theme-editor-action",
                    )
```

Then add these methods:

```python
    def _appearance_theme_summary(self) -> str:
        from ...css.Themes.theme_catalog import current_launch_default, display_name

        launch = current_launch_default()
        active = str(getattr(self.app_instance, "theme", launch))
        registered = getattr(self.app_instance, "available_themes", {}) or {}
        if launch not in registered:
            return f"launch default missing: {launch} · active: {display_name(active)}"
        return f"{display_name(launch)} (launch default) · active: {display_name(active)}"

    @on(Button.Pressed, "#settings-appearance-open-theme")
    def handle_appearance_open_theme(self, event: Button.Pressed) -> None:
        event.stop()
        self._select_category(SettingsCategoryId.THEME.value, restore_focus=True)
```

Wire it so the summary refreshes on theme changes:
- In `_sync_appearance_widgets`, replace the Select sync block with `self._set_static_text("#settings-appearance-theme-summary", self._appearance_theme_summary())`. `_set_static_text` already exists; confirm with grep.
- In `SettingsScreen.on_mount`, subscribe once: `self.app.theme_changed_signal.subscribe(self, lambda _theme: self._set_static_text("#settings-appearance-theme-summary", self._appearance_theme_summary()))`. `_set_static_text` must tolerate a missing widget; confirm it swallows `QueryError`, or wrap the call. This keeps the row live when the palette changes the theme while Appearance is open.
- In `handle_theme_launch_default_changed`, keep only `event.stop()` and `self._refresh_category_button_label(SettingsCategoryId.APPEARANCE)`. `use_theme` now owns the in-memory config update; the palette and the picker both go through it.

- [ ] **Step 4: Remove `default_theme` from the Appearance draft paths.**
  - Delete the `"default_theme"` entries from the selector map (~9427) and the validation-class key tuple (~9457).
  - Delete the `if message.startswith("Theme"): return "default_theme"` branch (~9385).
  - Delete `handle_appearance_theme_changed`.
  - In the preview at ~30773, delete the `setattr(self.app_instance, "theme", ...)` block, and base `preview_applied` on the remaining runtime-safe values. Read the surrounding function first and keep its result copy truthful.
  - In `settings_appearance_defaults.py`:
    - Remove `"default_theme": ...` from `general.update({...})`, so an existing value passes through the deep-merged copy untouched.
    - In `validate_appearance_defaults`, delete the "Theme is required" and length checks.
    - Leave the `default_theme` dataclass field and its `_normalise_theme` load (read-only uses).
  - Retarget the field guidance for `settings-appearance-theme` to `settings-appearance-open-theme`, as `("Focused setting", "Theme")`, `("Purpose", "Opens Settings ▸ Theme, where themes are chosen.")` and `("Saved as", "general.default_theme (set by Use in Theme)")`.
  - Grep for the Settings search index entry that names `settings-appearance-theme` (`grep -n "settings-appearance-theme" tldw_chatbook/UI/Screens/*.py`). Point it at the Theme category, so searching "theme" lands on the picker.

- [ ] **Step 5: Update the existing tests that pin the dropdown.** Run `grep -rln "settings-appearance-theme\|_appearance_theme_options\|default_theme" Tests/UI Tests/Settings* 2>/dev/null`.
  - Delete tests that only assert the Select's options or labels, including the TASK-32945 "(saved)"/"(Textual)" options tests. Their guarantee moves to the catalog tests in Task 1: origins, dedupe and override markers.
  - Rewrite tests that assert Appearance saves or validates `default_theme` to assert the opposite, as above.
  - List each retired or rewritten test, with its reason, in the Implementation Notes.

- [ ] **Step 6: Run** Task 6's tests plus every test file touched in Step 5, then `Tests/UI/test_settings_save_commit_models.py` and `Tests/UI/test_settings_interface_keyboard_journeys.py`. Expected: all pass. Tests without `@private_profile_test` that fail with `RecoveryRequired` are CI-only; note them.

- [ ] **Step 7: Commit** with the message `feat(settings): Appearance shows the theme read-only and links to the picker; draft no longer owns default_theme (TASK-32948)`.

---

### Task 7: Layout CSS, geometry and contrast at 80×24 and 190×55

**Files:**
- Modify: `tldw_chatbook/css/components/_settings_splash_theme.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` via `./build_css.sh`
- Test: `Tests/UI/test_settings_theme_picker_screen.py`, `Tests/UI/test_settings_theme_card_contrast.py`

- [ ] **Step 1: Write the failing geometry test** (append)

```python
PICKER_CONTROLS = (
    "#settings-theme-filter",
    "#settings-theme-list",
    "#settings-theme-picker-preview",
    "#settings-theme-use",
    "#settings-theme-try",
    "#settings-theme-clone",
    "#settings-theme-new",
)


def _visible(host, widget):
    geometry = host.screen._compositor.find_widget(widget)
    return geometry.region.intersection(geometry.clip)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (190, 55)])
@private_profile_test
async def test_every_picker_control_is_reachable(request, theme, size):
    host = _host()
    async with host.run_test(size=size) as pilot:
        host.theme = theme
        await _category(host, pilot, "Theme")
        for selector in PICKER_CONTROLS:
            widget = host.screen.query_one(selector)
            widget.scroll_visible(animate=False)
            await pilot.pause(0.1)
            region = _visible(host, widget)
            assert region.height > 0 and region.width > 0, f"{selector} unreachable at {size}"
        lst = host.screen.query_one("#settings-theme-list")
        assert _visible(host, lst).height >= 5, f"list shows <5 rows at {size}"
```

- [ ] **Step 2: Run it.** Expected: FAIL, at least on the list height at 80×24, and possibly on the card column being clipped.

- [ ] **Step 3: Add the CSS.** Use tokens only; if a size you need is missing, use the nearest existing `$ds-size-*`.

```css
/* TASK-32948: picker layout. The pane sits inside the VerticalScroll detail
   body, so every container here must be height:auto with bounded children;
   a 1fr child inside that scroll is the 07-19..09-17 clipping regression. */
#settings-theme-pane,
#settings-theme-picker,
#settings-theme-editor-view,
#settings-theme-picker-columns,
#settings-theme-list-column,
#settings-theme-card-column {
    height: auto;
}
#settings-theme-list-column,
#settings-theme-card-column {
    width: 1fr;
}
#settings-theme-list {
    height: auto;
    min-height: $ds-size-8;
    max-height: $ds-size-24;
}
#settings-workbench.settings-workbench-compact #settings-theme-picker-columns {
    layout: vertical;
}
/* The highlighted row must not be carried by colour alone (PRODUCT.md
   accessibility): bold as well as the highlight fill. */
ThemeOptionList > .option-list--option-highlighted {
    text-style: bold;
}
#settings-workbench.settings-workbench-compact #settings-theme-list {
    max-height: $ds-size-9;
}
```

At compact width the preview should shrink to its two rows. `ThemePreview(compact=True)` is chosen at compose time, and a CSS-only alternative is simpler: add `.settings-theme-preview-row { display: none; }` for rows other than rail and accent under `settings-workbench-compact`, keyed on the row ids:

```css
#settings-workbench.settings-workbench-compact #settings-theme-picker-preview-user,
#settings-workbench.settings-workbench-compact #settings-theme-picker-preview-assistant,
#settings-workbench.settings-workbench-compact #settings-theme-picker-preview-success,
#settings-workbench.settings-workbench-compact #settings-theme-picker-preview-warning,
#settings-workbench.settings-workbench-compact #settings-theme-picker-preview-error {
    display: none;
}
```

Use the CSS approach, and remove the `compact` parameter from `ThemePreview` and its Task 3 test (YAGNI). Note the removal in the Implementation Notes.

- [ ] **Step 4: Rebuild and run.** `./build_css.sh`, then the geometry test, then `Tests/UI/test_css_build_integrity.py`. That test pins the theme module's selectors (`#settings-theme-tree`, `.settings-theme-preview`); both survive PR 1. If it pins a selector list, add the new picker selectors there too. Expected: PASS at both sizes and in both themes. If 80×24 still gives fewer than 5 list rows, first check the page with `pilot.app.save_screenshot()` and look for a leftover banner or header. Then reduce the list's `min-height` or the chip row, **not** the assertion.

- [ ] **Step 5: Extend the contrast test.** In `Tests/UI/test_settings_theme_card_contrast.py`, `_open_theme_card` must now return the picker. Add a case that measures `#settings-theme-use` (filled chip: label ≥ 4.5:1 against the card) and the bracket edges of `#settings-theme-try` (≥ 3:1 against the card), using the file's existing `_label_colors`/`_edges` helpers, for all 4 themes. The picker's buttons reuse the `theme-editor-action` class, so the TASK-32947 chip rules should already apply. If a case fails, fix the CSS, not the threshold.

Also assert the highlighted list row carries a non-colour cue. Take the cells of the row at `lst.region.y + (lst.highlighted - lst.scroll_offset.y)` via `_cells`, and require `style.bold` for its alphanumeric cells. This replaces the spec's `›` cursor glyph: Textual's `OptionList` has no per-row cursor slot, and bold serves the same purpose (recorded as a deviation in the Implementation Notes).

- [ ] **Step 6: Commit** with the message `feat(theme): picker layout bounded inside the scrolling pane; geometry + contrast pinned at 80x24/190x55 (TASK-32948)`.

---

### Task 8: Docs, task file, full checks, PR

**Files:**
- Modify: `Docs/User_Guide/settings.md`: the Theme and Appearance sections, plus a new Verified stamp.
- Create: `backlog/tasks/task-32948 - Settings-Theme-picker-first-redesign.md`

- [ ] **Step 1: User Guide.** Rewrite the Settings ▸ Theme section around the picker:
  - filter, groups, and the markers `active`, `launch` and `overrides shipped/Textual`
  - keys: Enter = Use, `t` Try, `c` Clone, `n` New, ↑ from the top row returns to the filter, F6
  - Revert
  - the editor behind Clone/New, with Back and its leave prompt

  Replace the Appearance theme paragraph with the read-only row and Open Theme button. Add the stamp: `*Verified against feat/theme-picker-pr1 @ <sha> — <date> (TASK-32948 PR 1): pinned by pilot tests at 80x24 and 190x55; <driven live | not driven live>.*`

- [ ] **Step 2: Task file.**
  - Re-verify that id 32948 is free: `git fetch origin`, then sweep all refs and worktrees as in `backlog/docs/lessons-backlog-hygiene.md`.
  - Write the task with status In Progress, a Description (why), and ACs covering all three PRs, each AC as its own `- [ ]`. Tick the PR 1 ACs.
  - Write the Implementation Notes: what PR 1 shipped, the retired or rewritten tests with their reasons, and every deviation from this plan (for example the `foreground` strip change or removing the `compact` parameter).

- [ ] **Step 3: Full checks.**
  - `ruff check` on every touched `.py`, with no new findings compared to the base.
  - `PYTHON=<venv python> ./scripts/preflight.sh`, exit code 0. If the diagnostic inventory flags a new `logger.warning`, read the flagged statement before running `--write`.
  - Run the theme suites: `Tests/Utils/test_theme_catalog.py`, `Tests/UI/test_settings_theme_picker.py`, `Tests/UI/test_settings_theme_picker_screen.py`, `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_theme_editor_render.py`, `Tests/UI/test_settings_theme_card_contrast.py`, `Tests/UI/test_theme_contrast.py`, `Tests/Utils/test_user_theme_loader.py`, `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`. Use `-n 4`.
  - Then `Tests/UI/test_settings_configuration_hub.py` on the branch and on the base, comparing the FAILED sets. Report any failure that appears only on the branch.

- [ ] **Step 4: Live check (strongly recommended).** Invoke the `verify` skill. Launch against a scratch `TLDW_CONFIG_PATH` profile at 80×24 and 190×55. Check:
  - filter plus Enter Enter uses a theme;
  - Try, then Revert;
  - Clone opens the editor, and Back with edits prompts;
  - Appearance shows the row, and Open Theme lands on the picker;
  - after a relaunch, the Use'd theme loads.

  Save the `tmux capture-pane` evidence to the session scratchpad and cite it in the Notes.

- [ ] **Step 5: Commit, push, PR.** Commit the docs and the task file. Push `feat/theme-picker-pr1`, then run `gh pr create --base dev`. If #2830 has not merged yet, target the `fix/theme-ux-wave` branch and say so in the body. The body covers the summary, the test plan, the retired tests, and the deliberate behaviour changes: Appearance no longer edits the theme, and the palette toast wording changed. End the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Do not merge.
