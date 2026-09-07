# Theme Editor UX Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the twelve defects found in the 2026-09-03 Settings ▸ Theme editor review (backlog TASK-31250 … TASK-31280) so a saved theme actually persists, every action names the right theme, and every state the code sets is visible.

**Architecture:** All fixes stay inside the existing Settings-native editor (`tldw_chatbook/Widgets/settings_theme_editor.py`) plus three small seams it already touches: the theme catalog module (`css/Themes/themes.py`, gains one loader), app startup registration (`app.py`), and the Settings screen's Appearance options / dirty flag / inspector rows (`UI/Screens/settings_screen.py`). No new widgets, no new dependencies. The one new runtime concept is that user TOML themes are registered with the app like shipped ones.

**Tech Stack:** Python 3.12, Textual 8.2.8 (`App.available_themes`, `Theme`, `Select`, `Tree`), pytest + `app.run_test()`, toml.

**Spec:** `.impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md` (critique snapshot) and the twelve task files `backlog/tasks/task-3125[0-9]*.md`, `task-31279*.md`, `task-31280*.md`. Each task below names the backlog id it closes; tick its ACs and add Implementation Notes when done.

## Global Constraints

- Work in the worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/theme-editor-fixes` (branch `fix/theme-editor-ux`, based on origin/dev). Every path below is relative to it. Subagents start in the MAIN checkout: `cd` there first or use absolute paths.
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <path> -q -p no:cacheprovider` from the worktree root (system python is 3.9). Never verify with `-k`.
- Pinned copy that must survive (tests assert these fragments): apply hint contains `applies immediately - no Save needed`; tree hint contains `New` and `theme`; delete toasts `'<n>' is a built-in theme and cannot be deleted`, `'<n>' is a shipped theme and cannot be deleted`, `No saved custom theme named '<n>'`, `Deleted theme '<n>'`; `Creating new theme`; confirm label `Discard changes`; reset toast `Theme reset to original values` (Task 12 may change it together with its test).
- Pinned behaviour: mount posts NO `ThemeModifiedStatus` and `is_modified = reactive(False, init=False)` stays (recompose-storm guard, `test_theme_category_settles_without_recompose_storm`); Reset without edits skips the confirmation; Delete refuses built-in/shipped names; a user file shadowing a shipped name is deletable.
- After ANY change to `tldw_chatbook/css/components/*.tcss` or a widget `DEFAULT_CSS`: run `python3 tldw_chatbook/css/build_css.py` and commit `tldw_chatbook/css/tldw_cli_modular.tcss` (and `tldw_chatbook/css/widget_defaults_self.tcss` + `widget_defaults_scoped.tcss` when a `DEFAULT_CSS` changed). Never hand-edit the bundle.
- Never `git add -A`. Stage explicit paths. Commit messages end with the two trailers used in this session (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01BarRiaGxPsRjSdCQzPZXgf`).
- Test harness for the editor: `_isolated_editor_app(editor)` in `Tests/UI/test_settings_theme_editor.py` (real `App`, `app.notify` is a `MagicMock`, `editor.custom_themes_path = tmp_path`). Painted-frame assertions use the production bundle host pattern from `Tests/UI/test_checkbox_height_render.py` (`_rendered_text(app)` over `app.screen._compositor.render_strips()`).
- Backlog hygiene per task: `backlog-py task edit <id> -s "In Progress"` when starting; when done, tick ACs (`--check-ac N`), add `--notes`, set `-s Done`. Use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/backlog-py`.

---

### Task 1: Name box drives the theme name (TASK-31251)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (handler near `on_color_input_changed`, `_reset_theme`, `on_save_theme`)
- Test: `Tests/UI/test_settings_theme_editor.py`

**Interfaces:**
- Produces: `SettingsThemeEditor.current_theme_name` always equals the trimmed Name box text once the user edits it. Later tasks rely on this.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_name_box_drives_apply_save_reset_delete(tmp_path):
    """TASK-31251: New -> rename -> Apply/Save/Reset/Delete all use the typed name."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        name_input = editor.query_one("#settings-theme-name", Input)
        name_input.value = "ocean"
        await pilot.pause()
        assert editor.current_theme_name == "ocean"

        editor.on_apply_theme()
        await pilot.pause()
        assert app.notify.call_args.args[0] == "Theme 'ocean' applied"

        editor.on_save_theme()
        await pilot.pause()
        assert (tmp_path / "ocean.toml").exists()
        assert editor.current_theme_name == "ocean"

        editor.on_reset_theme()
        await pilot.pause()
        assert name_input.value == "ocean"

        editor.on_delete_theme()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        assert "ocean" in app.screen.message
```

- [ ] **Step 2: Run it, expect FAIL** on `editor.current_theme_name == "ocean"` (still `new_theme`).

Run: `.../python -m pytest Tests/UI/test_settings_theme_editor.py::test_settings_theme_editor_name_box_drives_apply_save_reset_delete -q -p no:cacheprovider`

- [ ] **Step 3: Implement**

Add after `on_color_input_changed`:

```python
    @on(Input.Changed, "#settings-theme-name")
    def on_theme_name_changed(self, event: Input.Changed) -> None:
        """Keep current_theme_name in step with the Name box (TASK-31251).

        Programmatic loads set the box to the name already held, so the
        equality guard makes those echoes no-ops.
        """
        name = event.value.strip()
        if name and name != self.current_theme_name:
            self.current_theme_name = name
```

In `_reset_theme`, replace the body with:

```python
        user_theme_path = self.custom_themes_path / f"{self.current_theme_name}.toml"
        if user_theme_path.exists():
            self.load_user_theme(self.current_theme_name)
        elif self._is_catalog_theme(self.current_theme_name):
            self.load_theme(self.current_theme_name)
        else:
            self.app.notify(
                f"No saved version of '{self.current_theme_name}' to reset to",
                severity="warning",
            )
            return
        self.app.notify("Theme reset to original values", severity="information")
```

and add the helper:

```python
    def _is_catalog_theme(self, name: str) -> bool:
        """True for Textual built-ins and shipped ALL_THEMES names."""
        return name in ("textual-dark", "textual-light") or any(
            getattr(t, "name", None) == name for t in ALL_THEMES
        )
```

- [ ] **Step 4: Run the new test and the whole file, expect PASS** (20 tests).
- [ ] **Step 5: Backlog + commit**

```bash
git add tldw_chatbook/Widgets/settings_theme_editor.py Tests/UI/test_settings_theme_editor.py
git commit -m "fix(theme-editor): Name box edits drive Apply/Export/Reset/Delete (TASK-31251)"
```

---

### Task 2: Loading never re-themes the app; colours come from the real Theme objects (TASK-31255)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`load_theme`, delete `_extract_current_theme_colors`, `_delete_user_theme`)
- Test: `Tests/UI/test_settings_theme_editor.py`

**Interfaces:**
- Produces: `load_theme(name)` resolves `name` through `self.app.available_themes` first, then `ALL_THEMES`; it never assigns `self.app.theme`. Task 4 relies on the `available_themes` lookup.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_selecting_builtin_leaf_does_not_retheme_app(tmp_path):
    """TASK-31255: browsing the tree is read-only for the running app."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-light"
        await pilot.pause()
        editor.load_theme("textual-dark")
        await pilot.pause()
        assert app.theme == "textual-light"
        assert editor.color_inputs["primary"].value.upper() == "#004578"
        assert editor.color_inputs["background"].value.upper() == "#121212"


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_keeps_app_theme(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    _write_user_theme(tmp_path, "my_custom_theme")
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-light"
        editor.load_user_theme("my_custom_theme")
        await pilot.pause()
        editor.on_delete_theme()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert app.theme == "textual-light"
        assert editor.current_theme_name == "textual-dark"
```

- [ ] **Step 2: Run, expect FAIL** (`app.theme` flips to `textual-dark`; background reads the hardcoded `#0C0C0C`).
- [ ] **Step 3: Implement.** Replace the body of `load_theme` from the `if theme_name in [...]` block onward with:

```python
        theme = self.app.available_themes.get(theme_name) or next(
            (t for t in ALL_THEMES if getattr(t, "name", None) == theme_name), None
        )
        if theme is None:
            logger.warning(f"Theme editor: unknown theme '{theme_name}', keeping current palette")
        else:
            self.current_theme_data = self._extract_theme_colors(theme)
            self.is_dark_theme = bool(getattr(theme, "dark", True))
        self._update_color_inputs()
        self._update_dark_mode_checkbox()
        self.is_modified = False
```

Delete `_extract_current_theme_colors` entirely (its two hardcoded tables). `_delete_user_theme` keeps `self.load_theme("textual-dark")` — it now only touches the editor.

- [ ] **Step 4: Run the file, expect PASS** (22 tests; the two existing delete tests still assert `current_theme_name == "textual-dark"`).
- [ ] **Step 5: Commit** `fix(theme-editor): tree browsing and Delete no longer re-theme the app (TASK-31255)`.

---

### Task 3: Saved themes load at startup, appear in Appearance and the palette, can be the launch default (TASK-31250)

**Files:**
- Modify: `tldw_chatbook/config.py` (add `get_user_themes_dir()` next to `get_cli_config_path`)
- Modify: `tldw_chatbook/css/Themes/themes.py` (add `load_user_themes`)
- Modify: `tldw_chatbook/app.py:14335-14337` (register user themes) and `app.py:1146-1150` (`ThemeProvider` list)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:8012-8033` (`_appearance_theme_options`), `:588-590` (`_theme_save_target` → use `get_user_themes_dir`)
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`__init__` dir, `on_save_theme` registers, new `Set as launch default` button + handler, apply hint copy)
- Modify: `Docs/User_Guide/settings.md` (§ Interface — Theme, quick recipe 3, quirk "The theme didn't change after saving", Verified-against stamp)
- Test: `Tests/Utils/test_user_theme_loader.py` (new), `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_configuration_hub.py`

**Interfaces:**
- Produces: `config.get_user_themes_dir() -> Path`; `themes.load_user_themes(themes_dir: Path) -> list[Theme]`; button id `settings-theme-set-default`.

- [ ] **Step 1: Failing loader test** (`Tests/Utils/test_user_theme_loader.py`)

```python
from pathlib import Path
from tldw_chatbook.css.Themes.themes import load_user_themes


def _write(dir_: Path, name: str, body: str) -> None:
    (dir_ / f"{name}.toml").write_text(body, encoding="utf-8")


def test_load_user_themes_reads_good_files_and_skips_bad_ones(tmp_path):
    _write(tmp_path, "ocean", '[theme]\nname = "ocean"\ndark = true\n[colors]\nprimary = "#9966FF"\n')
    _write(tmp_path, "broken", "this is not toml = = =\n")
    _write(tmp_path, "nocolors", '[theme]\nname = "bare"\n')
    themes = load_user_themes(tmp_path)
    names = sorted(t.name for t in themes)
    assert names == ["bare", "ocean"]
    ocean = next(t for t in themes if t.name == "ocean")
    assert ocean.primary.hex.upper() == "#9966FF"
    assert ocean.dark is True


def test_load_user_themes_missing_dir_returns_empty(tmp_path):
    assert load_user_themes(tmp_path / "nope") == []
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: cannot import name 'load_user_themes'`).
- [ ] **Step 3: Implement the loader** in `themes.py` (after `create_theme_from_dict`):

```python
def load_user_themes(themes_dir) -> list[Theme]:
    """Read every ``*.toml`` under ``themes_dir`` into Theme objects.

    The Settings theme editor writes ``[theme] name/dark`` + ``[colors]``.
    Unreadable files are skipped with a warning so one bad file cannot
    block startup.
    """
    from pathlib import Path
    import toml
    from loguru import logger

    themes: list[Theme] = []
    root = Path(themes_dir)
    if not root.is_dir():
        return themes
    for path in sorted(root.glob("*.toml")):
        try:
            data = toml.load(path)
            meta = data.get("theme", {}) or {}
            name = str(meta.get("name") or path.stem).strip() or path.stem
            colors = dict(data.get("colors", {}) or {})
            colors["dark"] = bool(meta.get("dark", True))
            themes.append(create_theme_from_dict(name, colors))
        except Exception as exc:  # noqa: BLE001 - one bad file must not block startup
            logger.warning(f"Skipping unreadable user theme {path}: {exc}")
    return themes
```

Add to `config.py` next to `get_cli_config_path`:

```python
def get_user_themes_dir() -> Path:
    """Directory holding the user's saved theme TOML files (active profile)."""
    return get_cli_config_path().parent / "themes"
```

Wire it: in `app.py` after `for theme_name in ALL_THEMES: self.register_theme(theme_name)` add

```python
        from .css.Themes.themes import load_user_themes
        from .config import get_user_themes_dir
        for user_theme in load_user_themes(get_user_themes_dir()):
            self.register_theme(user_theme)
```

In `ThemeProvider` (app.py ~1146) replace the hand-built `available_themes` list with `available_themes = list(self.app.available_themes)`.

In `settings_screen.py` `_appearance_theme_options`, after the `ALL_THEMES` loop and before the `current_theme` block, add:

```python
        registered = getattr(getattr(self, "app_instance", None), "available_themes", None) or {}
        for theme_name in registered:
            if theme_name in seen or theme_name in ("textual-dark", "textual-light"):
                continue
            seen.add(theme_name)
            options.append((f"{theme_name.replace('_', ' ').replace('-', ' ').title()} (saved)", theme_name))
```

(`test_settings_appearance_theme_options_use_specific_import_fallback` pins the `ALL_THEMES` import lines — keep them unchanged.) Replace `_theme_save_target`'s body with `return get_user_themes_dir()` and the editor's `__init__` dir with `self.custom_themes_path = get_user_themes_dir()` (import from `..config`).

In the editor: `on_save_theme` after the file write adds

```python
            self.app.register_theme(
                create_theme_from_dict(theme_name, {**self.current_theme_data, "dark": self.is_dark_theme})
            )
```

Add the button to the Actions row: `yield Button("Set as launch default", id="settings-theme-set-default")` and the handler:

```python
    @on(Button.Pressed, "#settings-theme-set-default")
    def on_set_launch_default(self) -> None:
        """Make the saved theme the startup theme (TASK-31250)."""
        name = self.current_theme_name
        if not (self.custom_themes_path / f"{name}.toml").exists() and not self._is_catalog_theme(name):
            self.app.notify("Save the theme first, then set it as the launch default", severity="warning")
            return
        from ..config import save_setting_to_cli_config

        save_setting_to_cli_config("general", "default_theme", name)
        self.app.notify(f"'{name}' will load at the next launch", severity="success")
```

Apply hint copy becomes: `"Apply applies immediately - no Save needed; Save stores the theme; Set as launch default makes it load at startup."`

- [ ] **Step 4: Editor test** (append to `test_settings_theme_editor.py`)

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_save_registers_theme_with_app(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.query_one("#settings-theme-name", Input).value = "ocean"
        await pilot.pause()
        editor.on_save_theme()
        await pilot.pause()
        assert "ocean" in app.available_themes
```

and a `_appearance_theme_options` unit test in `test_settings_configuration_hub.py` using a `types.SimpleNamespace` stub with `_appearance_setting_values=lambda: {"default_theme": "textual-dark"}` and `app_instance=SimpleNamespace(available_themes={"ocean": object()})`, asserting `("Ocean (saved)", "ocean")` is in `SettingsScreen._appearance_theme_options(stub)`.

- [ ] **Step 5: Run** the three test files, expect PASS. Also run `Tests/UI/test_command_palette_shell_routes.py` (ThemeProvider consumers).
- [ ] **Step 6: Docs.** In `Docs/User_Guide/settings.md` § Interface — Theme: replace "**This editor never sets the launch default** — press **Save** here, then pick the theme in **Appearance** → **Theme**." with "**Save** stores the theme and registers it at once, so it appears in **Appearance** → **Theme** and the palette; **Set as launch default** makes it the startup theme." Update recipe 3 and the quirk paragraph accordingly; bump the stamp line to `*Verified against fix/theme-editor-ux @ <sha> — 2026-09-04.*`.
- [ ] **Step 7: Commit** `feat(theme-editor): register saved user themes at startup and after Save (TASK-31250)`.

---

### Task 4: Re-entering Theme after Apply shows the applied palette; stale dirty flag cleared (TASK-31252)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`_initialize_editor`)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (`_select_category`, ~20118-20232)
- Test: `Tests/UI/test_settings_theme_editor.py`, `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Failing editor test**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_remount_after_apply_restores_palette(tmp_path):
    """TASK-31252: app.theme == 'custom_<name>' must load, not blank the editor."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.color_inputs["primary"].value = "#123456"
        await pilot.pause()
        editor.on_apply_theme()
        await pilot.pause()
        assert app.theme == "custom_new_theme"
        editor._initialize_editor()  # what a remount does
        await pilot.pause()
        assert editor.current_theme_name == "new_theme"
        assert editor.color_inputs["primary"].value.upper() == "#123456"
```

- [ ] **Step 2: Run, expect FAIL** (name `custom_new_theme`, inputs unchanged/empty).
- [ ] **Step 3: Implement.** In `_initialize_editor`, replace `self.load_theme(self.app.theme)` with:

```python
            app_theme = str(self.app.theme)
            self.load_theme(app_theme)
            if app_theme.startswith("custom_"):
                # Apply registers the working palette under a custom_ prefix so
                # it never clobbers a shipped registration; the editor shows the
                # user-facing name (TASK-31252).
                self.current_theme_name = app_theme[len("custom_"):]
                self.query_one("#settings-theme-name", Input).value = self.current_theme_name
                self.query_one("#settings-theme-name", Input).disabled = False
```

Screen side: in `_select_category`, right before `self.active_category = category_value` add

```python
        if (
            self.active_category == SettingsCategoryId.THEME.value
            and category_value != SettingsCategoryId.THEME.value
            and self.theme_editor_modified
        ):
            # Leaving Theme remounts the editor and drops in-progress edits, so
            # the dirty displays must not survive the departure (TASK-31252).
            self.theme_editor_modified = False
            self._refresh_theme_modified_widgets()
```

- [ ] **Step 4: Screen test** (in `test_settings_configuration_hub.py`, modelled on `test_theme_user_edit_does_not_remount_editor` at ~10785): edit a colour so the flag is True, `screen._select_category(SettingsCategoryId.APPEARANCE.value)`, pause, `screen._select_category(SettingsCategoryId.THEME.value)`, pause, assert `screen.theme_editor_modified is False` and the `#settings-theme-unsaved-note` renderable contains `No`.
- [ ] **Step 5: Run** both files plus `test_theme_category_settles_without_recompose_storm`; expect PASS.
- [ ] **Step 6: Commit** `fix(theme-editor): restore applied palette on remount, clear stale dirty flag (TASK-31252)`.

---

### Task 5: Generate from Primary honours the hue (TASK-31253)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`_generate_theme_from_primary`, `_adjust_color`)
- Test: `Tests/UI/test_settings_theme_editor.py`

- [ ] **Step 1: Failing test** (pure function, no app needed)

```python
def _hue_deg(hex_value: str) -> float:
    from textual.color import Color
    return Color.parse(hex_value).hsl.h * 360


@pytest.mark.parametrize("primary", ["#9966FF", "#00CC66", "#FF9900"])
def test_generate_from_primary_keeps_the_primary_hue(primary):
    editor = SettingsThemeEditor.__new__(SettingsThemeEditor)
    editor.is_dark_theme = True
    from textual.color import Color
    palette = editor._generate_theme_from_primary(Color.parse(primary))
    base = _hue_deg(primary)
    for key in ("secondary", "background", "surface", "panel"):
        delta = abs(_hue_deg(palette[key]) - base) % 360
        assert min(delta, 360 - delta) <= 30, (key, palette[key])
    accent_delta = abs(_hue_deg(palette["accent"]) - base) % 360
    assert 150 <= min(accent_delta, 360 - accent_delta) <= 180
    assert all(v == v.upper() for v in palette.values())
```

- [ ] **Step 2: Run, expect FAIL** (secondary is red for every primary).
- [ ] **Step 3: Implement.** In `_generate_theme_from_primary` change the first line to `hue, saturation, lightness = primary.hsl` → `hsl = primary.hsl; hue, saturation, lightness = hsl.h * 360, hsl.s, hsl.l`. In `_adjust_color` return `f"#{r:02X}{g:02X}{b:02X}"`.
- [ ] **Step 4: Run, expect PASS.** Also assert `primary.hex` path stays uppercase (Textual returns uppercase already).
- [ ] **Step 5: Commit** `fix(theme-editor): Generate from Primary uses degrees, uppercase hex (TASK-31253)`.

---

### Task 6: Make swatch text, invalid state, preset target and Dark toggle visible (TASK-31254)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (compose: preset target `Select`, checkbox; `_update_color_swatch`, `on_color_input_changed`, `_apply_preset_swatch`, `on_descendant_focus` removed)
- Modify: `tldw_chatbook/css/components/_settings_splash_theme.tcss` (swatch rules, toggle rule)
- Rebuild: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_settings_theme_editor.py`, new `Tests/UI/test_settings_theme_editor_render.py`

**Interfaces:**
- Produces: `Select` with id `settings-theme-preset-target` whose value is a `BASE_COLORS` name (default `"primary"`); `_preset_target()` returns it. Task 7's tests use the id.

- [ ] **Step 1: Failing painted-frame test** (`Tests/UI/test_settings_theme_editor_render.py`, copy the `_BUNDLED_CSS_PATH` / `_rendered_text` / bundle-loading harness from `Tests/UI/test_checkbox_height_render.py`, mounting `SettingsThemeEditor()` inside a `Vertical` at size `(120, 60)`):

```python
@pytest.mark.asyncio
async def test_swatch_paints_hex_text_and_dark_toggle_paints_state():
    app = _BundleHarness()  # CSS_PATH = [str(_BUNDLED_CSS_PATH)], compose yields SettingsThemeEditor()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(SettingsThemeEditor)
        editor.color_inputs["primary"].value = "#9966FF"
        await pilot.pause()
        painted = _rendered_text(app)
        assert "#9966FF" in painted            # swatch text is painted
        assert "▔▔▔" not in painted.split("Dark theme")[1][:12]  # toggle is not a clipped border
        assert "Presets fill" in painted
```

- [ ] **Step 2: Run, expect FAIL** (`#9966FF` appears only once — in the Input — assert on count `>= 2` to be exact).
- [ ] **Step 3: Implement.**

CSS (`_settings_splash_theme.tcss`): replace the `.color-swatch` and `.color-preset-swatch` blocks with

```css
.color-swatch {
    width: 9;
    min-width: 9;
    height: 1;
    min-height: 1;
    border: none;
    content-align: center middle;
    text-style: bold;
}

.color-preset-swatch {
    width: 3;
    min-width: 3;
    height: 1;
    min-height: 1;
    margin: 0 1 0 0;
    border: none;
}

/* task-1369 / TASK-31254: focus ring without a border row -- outline paints
   over the swatch's own cells and keeps its colour visible. */
.color-preset-swatch:focus {
    outline: solid $ds-focus-accent;
    text-style: bold;
}

/* TASK-31254: escape the app-wide `Checkbox { height: 2 }` + tall border
   that leaves zero content rows (same idiom as
   `.settings-imagegen-backend-row Checkbox`, TASK-18960). */
#settings-theme-dark-mode {
    width: auto;
    height: 1;
    min-height: 1;
    margin-bottom: 0;
    border: none;
    padding: 0;
}
```

Widget: in `_compose_palette_section`, replace the "Focus a swatch…" help Static with

```python
        with Horizontal(classes="settings-input-row"):
            yield Static("Presets fill", classes="settings-input-label")
            yield Select(
                [(name.title(), name) for name in self.BASE_COLORS],
                value="primary",
                allow_blank=False,
                id="settings-theme-preset-target",
                classes="settings-compact-select",
            )
        yield Static(
            "Pick the colour above, then click a swatch or focus it and press Enter or Space.",
            classes="settings-help-copy",
        )
```

Add `Select` to the `textual.widgets` import. Replace `last_focused_color_input` usage: delete `on_descendant_focus` and the `selected` class lines in `_initialize_editor`/`_apply_preset_swatch`; add

```python
    def _preset_target(self) -> str:
        try:
            value = self.query_one("#settings-theme-preset-target", Select).value
        except QueryError:
            value = "primary"
        return value if value in self.color_inputs else "primary"
```

and use `target = self._preset_target()` in `_apply_preset_swatch`. Give the Dark toggle a label: `Checkbox("On", value=True, id="settings-theme-dark-mode")` (label reads "On" beside the glyph; the row label stays "Dark theme"). Invalid input: in `on_color_input_changed` use class `settings-invalid-input` (add/remove) instead of `invalid-color`, and replace `self._update_color_swatch(color_name, "#000000")` with

```python
                    self.color_swatches[color_name].update("invalid")
```

(keep the previous background). Drop the now-dead `.invalid-color` references.

- [ ] **Step 4: Rebuild the bundle** `python3 tldw_chatbook/css/build_css.py`, run the new render test + the editor file; `test_settings_theme_editor_preset_swatches_are_keyboard_activatable` must still pass (default target primary).
- [ ] **Step 5: Commit** `fix(theme-editor): paint swatch hex, invalid state, preset target and dark toggle (TASK-31254)` staging the widget, tcss, bundle, and both test files.

---

### Task 7: Actions above presets, User Themes first, stable preset target under Tab (TASK-31256, plus the group naming from TASK-31280)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_theme_editor.py` (`compose` order, `_populate_theme_tree`, `_load_user_themes`, `on_theme_selected`, tree hint copy, `on_save_theme`/`_delete_user_theme` tree bookkeeping)
- Test: `Tests/UI/test_settings_theme_editor.py` (update `_user_theme_labels`)

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_actions_precede_presets_in_focus_order(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ids = [w.id for w in editor.query("*") if getattr(w, "can_focus", False) and w.id]
        assert ids.index("settings-theme-apply") < ids.index("settings-theme-preset-Blues-0")


@pytest.mark.asyncio
async def test_settings_theme_editor_tabbing_does_not_move_preset_target(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.color_inputs["error"].focus()
        await pilot.pause()
        editor.query_one("#settings-theme-preset-Blues-0").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert editor.current_theme_data["primary"] == editor.COLOR_PRESETS["Blues"][0]
        assert editor.color_inputs["error"].value != editor.COLOR_PRESETS["Blues"][0]


@pytest.mark.asyncio
async def test_settings_theme_editor_tree_lists_your_themes_first_and_expanded(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    _write_user_theme(tmp_path, "ocean")
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor._populate_theme_tree()
        tree = editor.query_one("#settings-theme-tree", Tree)
        labels = [str(n.label) for n in tree.root.children]
        assert labels == ["Your themes", "Built-in", "Shipped themes"]
        assert tree.root.is_expanded
        assert tree.root.children[0].is_expanded and not tree.root.children[2].is_expanded
        assert [str(c.label) for c in tree.root.children[0].children] == ["ocean"]
```

- [ ] **Step 2: Run, expect FAIL** on all three.
- [ ] **Step 3: Implement.** `compose`: yield sections in the order library → actions → palette → preview. `_populate_theme_tree`:

```python
        tree.root.remove_children()
        user_node = tree.root.add("Your themes", expand=True)
        self._load_user_themes(user_node)
        builtin_node = tree.root.add("Built-in", expand=True)
        builtin_node.add_leaf("textual-dark", data="catalog")
        builtin_node.add_leaf("textual-light", data="catalog")
        shipped_node = tree.root.add("Shipped themes", expand=False)
        for theme in ALL_THEMES:
            if hasattr(theme, "name"):
                shipped_node.add_leaf(theme.name, data="catalog")
        tree.root.expand()
```

`_load_user_themes` adds `parent_node.add_leaf(theme_name, data="user")`. `on_theme_selected` uses `event.node.data == "user"` to pick `load_user_theme`, else `load_theme(str(event.node.label))`. Save/Delete tree bookkeeping looks for the node labelled `"Your themes"` and leaf label `theme_name` (no prefix). Tree hint copy: `"Your themes come first; expand Shipped themes to browse the catalog. New starts a theme from the current palette."`. Update the test helper `_user_theme_labels` to look for `"Your themes"`, and the three delete tests' `f"user:{...}"` expectations to bare names.

- [ ] **Step 4: Run the file, expect PASS.**
- [ ] **Step 5: Commit** `fix(theme-editor): actions before presets, your themes first, stable preset target (TASK-31256)`.

---

### Task 8: New starts from the current palette (TASK-31257)

**Files:** `tldw_chatbook/Widgets/settings_theme_editor.py` (`_new_theme`), `Tests/UI/test_settings_theme_editor.py`, `Docs/User_Guide/settings.md` (§ Theme: "**New** copies the loaded palette into a theme called new_theme").

- [ ] **Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_new_copies_current_palette(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.load_theme("textual-light")
        await pilot.pause()
        before = dict(editor.current_theme_data)
        editor.on_new_theme()
        await pilot.pause()
        assert editor.current_theme_name == "new_theme"
        assert editor.current_theme_data == before
        assert editor.is_dark_theme is False
```

- [ ] **Step 2: Run, expect FAIL** (palette becomes the hardcoded blue set, dark True).
- [ ] **Step 3: Implement.** In `_new_theme` replace the hardcoded dict with

```python
        defaults = {
            "primary": "#0099FF", "secondary": "#006FB3", "accent": "#FFD700",
            "background": "#1E1E1E", "surface": "#2C2C2C", "panel": "#252525",
            "foreground": "#FFFFFF", "success": "#008000", "warning": "#FFD700", "error": "#FF0000",
        }
        self.current_theme_data = dict(self.current_theme_data) or defaults
        # is_dark_theme keeps the loaded theme's value
```

and delete the `self.is_dark_theme = True` line.

- [ ] **Step 4: Run the file, expect PASS** (`test_theme_tree_has_empty_state_guidance`, `test_settings_theme_editor_new_confirms_before_discarding_edits` unchanged).
- [ ] **Step 5: Commit** `fix(theme-editor): New starts from the current palette (TASK-31257)`.

---

### Task 9: Confirm before Save/Export overwrite (TASK-31258)

**Files:** `tldw_chatbook/Widgets/settings_theme_editor.py` (`on_save_theme`, `on_export_theme`, new `_write_theme_file`, `_loaded_user_theme` attribute), `Tests/UI/test_settings_theme_editor.py`.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_save_confirms_before_overwriting_another_theme(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    _write_user_theme(tmp_path, "ocean")
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.query_one("#settings-theme-name", Input).value = "ocean"
        await pilot.pause()
        editor.on_save_theme()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        assert app.screen.confirm_label == "Overwrite"
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert 'primary = "#0099FF"' in (tmp_path / "ocean.toml").read_text()


@pytest.mark.asyncio
async def test_settings_theme_editor_saving_the_loaded_theme_does_not_confirm(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    _write_user_theme(tmp_path, "ocean")
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.load_user_theme("ocean")
        await pilot.pause()
        editor.color_inputs["primary"].value = "#123456"
        await pilot.pause()
        editor.on_save_theme()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmationDialog)
        assert "#123456" in (tmp_path / "ocean.toml").read_text()
```

- [ ] **Step 2: Run, expect FAIL** (first: no dialog, file overwritten).
- [ ] **Step 3: Implement.** Track `self._loaded_user_theme: str | None = None` in `__init__`; set it in `load_user_theme` and after a successful save; clear it in `load_theme`, `_new_theme`, `on_clone_theme`. Split `on_save_theme` so the write lives in `_write_theme_file(theme_name, theme_path)` (the current try-block including registration and tree bookkeeping), and before writing:

```python
        if theme_path.exists() and self._loaded_user_theme != theme_name:
            async def _confirmed() -> None:
                self._write_theme_file(theme_name, theme_path)

            self.app.push_screen(
                ConfirmationDialog(
                    title="Overwrite theme",
                    message=f"A saved theme named '{theme_name}' already exists. Replace it?",
                    confirm_label="Overwrite",
                    cancel_label="Keep existing",
                    confirm_callback=_confirmed,
                )
            )
            return
        self._write_theme_file(theme_name, theme_path)
```

Same shape for Export: `if export_path.exists():` → dialog "Overwrite export" / "Overwrite" / "Keep existing", else `_write_export(export_path)`.

- [ ] **Step 4: Run, expect PASS.** Commit `fix(theme-editor): confirm before Save/Export overwrite an existing file (TASK-31258)`.

---

### Task 10: Preview repaints from the edited palette and shows a Console-shaped stub (TASK-31259)

**Files:** `tldw_chatbook/Widgets/settings_theme_editor.py` (`_compose_preview_section`, new `_refresh_preview`, call sites), `tldw_chatbook/css/components/_settings_splash_theme.tcss` (preview rows), bundle, `Tests/UI/test_settings_theme_editor.py`, `Docs/User_Guide/settings.md` (drop "decorative").

- [ ] **Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_settings_theme_editor_preview_repaints_from_edits_without_apply(tmp_path):
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.color_inputs["primary"].value = "#123456"
        await pilot.pause()
        row = editor.query_one("#settings-theme-preview-user")
        assert row.styles.background.hex.upper() == "#123456"
```

- [ ] **Step 2: Run, expect FAIL** (`NoMatches`).
- [ ] **Step 3: Implement.** Replace `_compose_preview_section` body with

```python
        yield Static("Live Preview", classes="destination-section")
        with Vertical(id="settings-theme-preview", classes="settings-theme-preview"):
            yield Static(" Console ▸ Conversation · ● ready", id="settings-theme-preview-rail")
            yield Static(" You: summarise the attached paper", id="settings-theme-preview-user")
            yield Static(" Assistant: Here is the summary…", id="settings-theme-preview-assistant")
            yield Static(" ✓ tool web_search finished", id="settings-theme-preview-success")
            yield Static(" ! approval needed before the next call", id="settings-theme-preview-warning")
            yield Static(" ✗ provider returned 401", id="settings-theme-preview-error")
            yield Static(" [ Send ]  Ctrl+P palette", id="settings-theme-preview-accent")
```

and

```python
    _PREVIEW_STYLE = {
        "rail": ("panel", "foreground"),
        "user": ("primary", "foreground"),
        "assistant": ("surface", "foreground"),
        "success": ("background", "success"),
        "warning": ("background", "warning"),
        "error": ("background", "error"),
        "accent": ("background", "accent"),
    }

    def _refresh_preview(self) -> None:
        """Paint the preview rows from the palette being edited (TASK-31259)."""
        for suffix, (bg_key, fg_key) in self._PREVIEW_STYLE.items():
            try:
                row = self.query_one(f"#settings-theme-preview-{suffix}", Static)
            except QueryError:
                continue
            bg = self.current_theme_data.get(bg_key)
            fg = self.current_theme_data.get(fg_key)
            try:
                if bg:
                    row.styles.background = bg
                if fg:
                    row.styles.color = fg
            except Exception:  # noqa: BLE001 - an invalid hex must not break painting
                continue
```

Call `self._refresh_preview()` at the end of `_update_color_inputs`, in `on_color_input_changed` after a valid change, in `_apply_preset_swatch`, and in `on_generate_theme`. CSS: `.settings-theme-preview { height: auto; margin: 1 0; border: solid $ds-grid-line; } .settings-theme-preview Static { height: 1; }` (replace the old `.preview-*` rules, which lose their consumers). Rebuild the bundle.

- [ ] **Step 4: Run, expect PASS.** Docs: § Theme "a **Live Preview** that repaints as you type (a Console-shaped stub)". Commit `feat(theme-editor): preview repaints from edits and shows a Console stub (TASK-31259)`.

---

### Task 11: Compact layout at ≤100 columns and a readable save path (TASK-31279)

**Files:** `tldw_chatbook/css/components/_settings_splash_theme.tcss`, bundle, `tldw_chatbook/UI/Screens/settings_screen.py` (`_theme_save_target` display sites ~4607, ~13607, ~18615 → new `_display_path`), `Tests/UI/test_settings_theme_editor_render.py`.

- [ ] **Step 1: Failing test** in the render file: bundle harness at `size=(110, 36)` mounting `SettingsThemeEditor()` inside a `Vertical` of `width: 45`; assert every `Button` in the editor has `region.right <= container.region.right` and `region.width > 0`.
- [ ] **Step 2: Run, expect FAIL** (Delete/Export exceed the container).
- [ ] **Step 3: Implement.** CSS:

```css
/* TASK-31279: below SETTINGS_COMPACT_WORKBENCH_MAX_WIDTH (100 cols) the
   detail pane is ~45 cells; four 16-cell buttons cannot share a row. */
#settings-workbench.settings-workbench-compact #settings-theme-card .settings-action-row {
    layout: vertical;
    height: auto;
}
#settings-workbench.settings-workbench-compact #settings-theme-card .settings-action-row Button {
    width: 100%;
}
#settings-theme-card .settings-action-row Button {
    min-width: 8;
}
```

Because the render harness has no `#settings-workbench`, wrap the editor in the test inside `Vertical(id="settings-workbench", classes="settings-workbench-compact")` so the rule applies. In `settings_screen.py` add

```python
def _display_path(path: Path) -> str:
    """Shorten a path for the inspector: ~ for home (TASK-31279)."""
    try:
        return "~" + os.sep + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
```

and use `_display_path(_theme_save_target())` at the three sites, dropping the duplicate "Affected config" path row (keep the "Save target" row).

- [ ] **Step 4: Rebuild bundle, run render + hub tests (`test_theme_category_opens_without_crashing`), expect PASS.** Commit `fix(theme-editor): stack action rows in compact mode, shorten inspector path (TASK-31279)`.

---

### Task 12: Polish, docs stamp, full verification (TASK-31280 remainder + Definition of Done)

**Files:** `tldw_chatbook/Widgets/settings_theme_editor.py` (`on_reset_theme`, button variants), `Tests/UI/test_settings_theme_editor.py`, `Docs/User_Guide/settings.md`, all twelve task files.

- [ ] **Step 1:** Update `test_settings_theme_editor_reset_without_edits_skips_confirmation` to expect `"No changes to reset"`; run, expect FAIL.
- [ ] **Step 2:** In `on_reset_theme`, the no-edits branch becomes `self.app.notify("No changes to reset", severity="information"); return` (still no dialog). Button variants: `New` → default, `Generate from Primary` → default, `Apply` stays primary, `Save` success, `Set as launch default` default. Run, expect PASS.
- [ ] **Step 3:** Run the full related suites: `Tests/UI/test_settings_theme_editor.py Tests/UI/test_settings_theme_editor_render.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_css_build_integrity.py Tests/Utils/test_user_theme_loader.py Tests/UI/test_command_palette_shell_routes.py Tests/Performance/test_textual_css_fastpath.py Tests/UI/test_non_obscuring_focus_contract.py` and `python3 Helper_Scripts/check_backlog_task_ids.py` equivalent `python3 scripts/check_backlog_task_ids.py`.
- [ ] **Step 4:** Live verification (verify skill): launch with a scratch `TLDW_CONFIG_PATH`, walk New → rename → edit → Apply → Save → Set as launch default → quit → relaunch, confirm the theme is applied at launch and the editor reopens on it. Record the capture paths in the task notes.
- [ ] **Step 5:** Docs stamp: update `Docs/User_Guide/settings.md` "Verified against" line for the Theme section to the branch sha and date.
- [ ] **Step 6:** Backlog: for each of 31250-31259, 31279, 31280 tick ACs, add Implementation Notes, set Done. Commit `docs(backlog): close Theme editor UX tasks 31250-31259, 31279 and 31280`.
