# Focus Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a chrome-free "focus mode" for the Console — nav bar and workbench header hidden, one-line status bar kept — toggleable at runtime and settable at startup.

**Architecture:** App-level `focus_mode` flag on `TldwCli`; the Console screen mirrors it onto a `-focus` CSS class that suppresses chrome via `display: none`. One exit rule (any navigation away clears the flag) and one re-entry rule (app-level `ctrl+shift+f` toggle). No gating layer, no new screens, `BaseAppScreen.compose()` untouched.

**Tech Stack:** Textual 8.x (Python ≥3.11); pytest with the `ConsolidatedCSSApp` UI harness.

**Spec:** `Docs/superpowers/specs/2026-08-16-focus-mode-design.md` — the plan argues from the spec; read both. ADR: `backlog/decisions/067-focus-mode-chrome-free-console.md`. Backlog task: `task-16320`.

## Global Constraints

- Python ≥3.11, Textual 8.x (≥8.0.0,<9). No new dependencies.
- ADR-031: never bind terminal-convention keys (plain ctrl+c/v/x/s/d/z/a/r/w) or shadow ctrl+p/ctrl+q/f1/f6. `ctrl+shift+f` is verified free. Every footer-advertised hint must be a real binding.
- ADR-042: no raw visual values in new CSS — the only new property is `display: none`.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is **generated** (`GENERATED FILE - DO NOT EDIT DIRECTLY`). Edit source modules under `css/` and regenerate with `python3 tldw_chatbook/css/build_css.py`. After rebuilding, `git diff` the generated files and confirm only the focus rules were added. UI tests load the generated bundle, so rebuild before running them.
- Testing: targeted runs only (this file plus the console contract tests); do NOT run the full suite without asking the owner.
- Commit after every task; conventional commit messages (`feat(console): …`).
- Route constants: `TAB_CHAT == "chat"` (`Constants.py:15`). Import constants; never hardcode route strings.

---

### Task 1: Config template, `--focus` CLI flag, app attributes

**Files:**
- Modify: `tldw_chatbook/config.py:2735` (`default_tab` line in `CONFIG_TOML_CONTENT`'s `[general]` block — add the new key after it)
- Modify: `tldw_chatbook/app.py:12832-12853` (argparse block inside `main_cli_runner()` — extract to module level), `app.py:5605-5612` (initial-tab block in `TldwCli.__init__`), BOTH app-construction sites: `app.py:12607` (under `if __name__ == "__main__":`, which parses NO args today) and `app.py:12887` (in `main_cli_runner()`)
- Test: `Tests/UI/test_focus_mode.py` (new file)

**Interfaces:**
- Consumes: nothing new.
- Produces: module-level `_build_arg_parser() -> argparse.ArgumentParser` in `app.py`; `TldwCli` instance attributes `focus_mode: bool` (always present, default `False`), `_focus_mode_config: bool`, `_cli_focus_override: bool`; config key `[general] focus_mode`.

**Dev-branch note (2026-08-19):** app.py has TWO `TldwCli()` construction sites. Line 12607 is under `if __name__ == "__main__":` (the `python3 -m tldw_chatbook.app` path) and historically parses no arguments; line 12887 is inside `main_cli_runner()` (console-script entry) and parses via an inline argparse block. The parser extraction must serve both: parse in `__main__` (new behavior — wrap in try/except SystemExit and fall back to no-args on parse failure so the module still runs bare) and replace the inline block in `main_cli_runner()`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_focus_mode.py`:

```python
"""Focus mode (task-16320, ADR-067) — config, CLI, and behavior tests."""

import io
from types import SimpleNamespace

import pytest

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app

from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME


class TestFocusCliAndConfig:
    def test_arg_parser_accepts_focus_flag(self):
        from tldw_chatbook.app import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["--focus"])
        assert args.focus is True
        args = parser.parse_args([])
        assert args.focus is False

    def test_config_template_declares_focus_mode(self):
        from tldw_chatbook.config import CONFIG_TOML_CONTENT

        general_block = CONFIG_TOML_CONTENT.split("[general]")[1].split("\n[")[0]
        assert "focus_mode = false" in general_block
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_arg_parser'`.

- [ ] **Step 3: Implement**

In `config.py`, inside `CONFIG_TOML_CONTENT`'s `[general]` block (after the `default_tab` line at `config.py:2679`), add:

```toml
focus_mode = false  # Start the Console chrome-free (no nav bar / workbench header; one-line status bar kept)
```

In `app.py`, extract the argparse block (currently inline in `main_cli_runner()` at `app.py:12832-12853`) into a module-level function placed directly above `if __name__ == "__main__":` (line 12435):

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the tldw-cli argument parser (extracted from main_cli_runner() for testability)."""
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli",
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run the application as a web server"
    )
    parser.add_argument(
        "--host", type=str, help="Host address for web server (default: localhost)"
    )
    parser.add_argument("--port", type=int, help="Port for web server (default: 8000)")
    parser.add_argument("--web-title", type=str, help="Title for the web page")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for web server"
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Start chrome-free in the Console (hides nav bar and workbench header)",
    )
    return parser
```

Replace the inline block in `main_cli_runner()` with `args = _build_arg_parser().parse_args()` (delete the now-duplicated `import argparse`/construction inside the function; add module-level `import argparse` to the stdlib import group at the top of `app.py` — there is none today).

Then wire the `__main__` path (line ~12435 block, before `app_instance = TldwCli()` at `app.py:12607`): that block parses no args today. Add:

```python
    try:
        _main_args = _build_arg_parser().parse_args()
    except SystemExit:
        # Bare `python3 -m tldw_chatbook.app` with an unknown flag must not
        # hard-exit before logging is up; fall back to defaults.
        _main_args = None
```

and use `_main_args` where the override is set (below).

In `TldwCli.__init__`, extend the initial-tab block at `app.py:5605-5612`:

```python
        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = self._normalize_initial_tab_from_config(
            initial_tab_from_config
        )
        # --- Focus mode (task-16320) ---
        self.focus_mode = False
        self._focus_mode_config = bool(
            get_cli_setting("general", "focus_mode", False)
        )
```

At BOTH app-construction sites, after `app_instance = TldwCli()`, set the override:

In `main_cli_runner()` (after `app.py:12887`):

```python
    app_instance._cli_focus_override = bool(args.focus)
```

In the `if __name__ == "__main__":` block (after `app.py:12607`):

```python
    app_instance._cli_focus_override = bool(
        getattr(_main_args, "focus", False) if _main_args is not None else False
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/app.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): add focus_mode config key, --focus flag, and app state"
```

---

### Task 2: Startup route resolution

**Files:**
- Modify: `tldw_chatbook/app.py:8179-8205` (`_resolve_initial_shell_route`)
- Test: `Tests/UI/test_focus_mode.py`

**Interfaces:**
- Consumes: `TldwCli.focus_mode`, `_focus_mode_config`, `_cli_focus_override` (Task 1).
- Produces: `_resolve_initial_shell_route` sets `self.focus_mode = True` and returns `TAB_CHAT` when focus is requested at startup; first-run/wizard branches still win.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_focus_mode.py`:

```python
class TestInitialRouteResolution:
    """Unbound-method tests: _resolve_initial_shell_route reads only
    app_config / _initial_tab_value / the focus attrs, so a stub works."""

    @staticmethod
    def _stub(**overrides):
        stub = SimpleNamespace(
            app_config={"_first_run": False},
            _initial_tab_value="notes",
            _cli_focus_override=False,
            _focus_mode_config=False,
            focus_mode=False,
        )
        for key, value in overrides.items():
            setattr(stub, key, value)
        return stub

    @pytest.fixture(autouse=True)
    def _wizard_off(self, monkeypatch):
        monkeypatch.setattr(
            "tldw_chatbook.UI.Wizards.first_run_setup_state.setup_recovery_action",
            lambda cfg, env: "skip",
        )

    def test_cli_focus_override_forces_chat(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True)
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT
        assert stub.focus_mode is True

    def test_config_focus_mode_forces_chat(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_focus_mode_config=True, _initial_tab_value="notes")
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT
        assert stub.focus_mode is True

    def test_cli_flag_wins_over_false_config(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True)
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT

    def test_no_focus_respects_default_tab(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_initial_tab_value="notes")
        assert TldwCli._resolve_initial_shell_route(stub) == "notes"
        assert stub.focus_mode is False

    def test_first_run_onboarding_beats_focus(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(
            _cli_focus_override=True, app_config={"_first_run": True}
        )
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_HOME
        assert stub.focus_mode is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py::TestInitialRouteResolution -v`
Expected: FAIL — focus stubs resolve to `"notes"`, `focus_mode` stays `False`.

- [ ] **Step 3: Implement**

In `_resolve_initial_shell_route` (`app.py:8206`), insert between the wizard `except` block and the final `return`:

```python
        # task-16320: focus mode is Console-only by definition, so a focus
        # request forces the route — but the first-run/wizard branches ABOVE
        # keep onboarding unbeatable (spec: first-run wins).
        if getattr(self, "_cli_focus_override", False) or getattr(
            self, "_focus_mode_config", False
        ):
            self.focus_mode = True
            return TAB_CHAT
        return getattr(self, "_initial_tab_value", TAB_CHAT)
```

(The existing final `return getattr(self, "_initial_tab_value", TAB_CHAT)` line is replaced by this block.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): resolve startup route to chat when focus mode requested"
```

---

### Task 3: Focus chrome CSS + ChatScreen application

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:5023` (next to the `-console-compact` precedent), regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` via `tldw_chatbook/css/build_css.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:14296` (`on_mount`) and near `chat_screen.py:3023` (`_register_console_footer_shortcuts` neighborhood)
- Test: `Tests/UI/test_focus_mode.py`

**Interfaces:**
- Consumes: `app_instance.focus_mode` (Task 1/2).
- Produces: `ChatScreen._apply_focus_chrome() -> None` (idempotent: mirrors the flag onto the `-focus` class and refreshes footer hints); CSS class `-focus` on `ChatScreen`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_focus_mode.py`:

```python
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class FocusConsoleHarness(ConsolidatedCSSApp):
    """Mounts the real ChatScreen against a fake app, like the neighboring
    console contract harnesses do (see test_console_workbench_contract.py)."""

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _make_app_instance(focus: bool):
    app_instance = _build_test_app()
    app_instance.focus_mode = focus
    return app_instance


class TestFocusChromeSuppression:
    async def test_focus_mode_hides_chrome_keeps_status_line(self):
        app_instance = _make_app_instance(focus=True)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test() as pilot:
            screen = pilot.app.screen
            assert screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is False
            header = screen.query_one("#console-workbench-header")
            assert header.display is False
            # One-line status bar is KEPT (owner decision, ADR-067).
            footer = screen.query_one("#screen-footer-status", AppFooterStatus)
            assert footer.display is not False

    async def test_default_mount_shows_all_chrome(self):
        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test() as pilot:
            screen = pilot.app.screen
            assert not screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is not False
            header = screen.query_one("#console-workbench-header")
            assert header.display is not False
            footer = screen.query_one("#screen-footer-status", AppFooterStatus)
            assert footer.display is not False

    async def test_apply_focus_chrome_flips_in_place(self):
        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test() as pilot:
            screen = pilot.app.screen
            assert not screen.has_class("-focus")
            app_instance.focus_mode = True
            screen._apply_focus_chrome()
            assert screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is False
            app_instance.focus_mode = False
            screen._apply_focus_chrome()
            assert not screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is not False
```

Note: if `_build_test_app()` (from `Tests/UI/test_destination_shells.py`) needs arguments, match however `test_console_workbench_contract.py` builds its `app_instance` — the contract is: an object `ChatScreen` accepts whose `focus_mode` attribute we control.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py::TestFocusChromeSuppression -v`
Expected: FAIL — `has_class("-focus")` is False / no `_apply_focus_chrome`.

- [ ] **Step 3: Implement**

a) In `tldw_chatbook/css/components/_agentic_terminal.tcss`, immediately after the `#console-shell.-console-compact #console-workbench-header` rule (line ~4895-4902), add:

```css
/* task-16320 / ADR-067: focus mode. The -focus class lives on the
   ChatScreen itself (MainNavigationBar is composed by BaseAppScreen as a
   sibling of #console-shell, so #console-shell-rooted selectors cannot
   reach it). display:none only — no raw visual values (ADR-042). The
   one-line #screen-footer-status status bar is intentionally KEPT. */
ChatScreen.-focus MainNavigationBar {
    display: none;
}

ChatScreen.-focus #console-workbench-header {
    display: none;
}
```

b) Regenerate the bundle:

```bash
python3 tldw_chatbook/css/build_css.py
git diff --stat tldw_chatbook/css/
```

Confirm the diff touches `tldw_cli_modular.tcss` with only the focus block (plus any pre-existing timestamp-header line change the generator writes — if the header timestamp churns, that is expected and committable; anything else unexpected, stop and investigate).

c) In `chat_screen.py`, add the method next to `_register_console_footer_shortcuts` (~line 3023):

```python
    def _apply_focus_chrome(self) -> None:
        """Mirror the app-level focus_mode flag onto this screen (task-16320).

        Idempotent: sets/removes the ``-focus`` class that suppresses the
        nav bar and workbench header (CSS: _agentic_terminal.tcss), and
        refreshes the footer hints so the focus toggle's label tracks the
        target state.
        """
        focused = bool(getattr(self.app_instance, "focus_mode", False))
        self.set_class(focused, "-focus")
        self._register_console_footer_shortcuts()
```

d) In `on_mount` (`chat_screen.py:14892`), call it early — right after `self.app_instance._console_h3_image_edit_screen = self` (line 14296):

```python
        self._apply_focus_chrome()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/ tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): suppress nav bar and workbench header under -focus class"
```

---

### Task 4: Footer shortcut hint

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:3095-3100` (`_register_console_footer_shortcuts`)
- Test: `Tests/UI/test_focus_mode.py`

**Interfaces:**
- Consumes: `_apply_focus_chrome` (Task 3), `app_instance.focus_mode`.
- Produces: console footer registration always contains a `("Ctrl+Shift+F", …)` pair whose label is `"exit focus"` when focused, `"focus"` otherwise.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_focus_mode.py`:

```python
class TestFocusFooterHint:
    async def test_footer_advertises_toggle_in_both_states(self):
        app_instance = _make_app_instance(focus=True)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test() as pilot:
            screen = pilot.app.screen
            source, shortcuts = screen._footer_shortcut_registration
            assert source == "console"
            assert ("Ctrl+Shift+F", "exit focus") in shortcuts

            app_instance.focus_mode = False
            screen._register_console_footer_shortcuts()
            _, shortcuts = screen._footer_shortcut_registration
            assert ("Ctrl+Shift+F", "focus") in shortcuts
            assert ("Ctrl+Shift+F", "exit focus") not in shortcuts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py::TestFocusFooterHint -v`
Expected: FAIL — no `Ctrl+Shift+F` pair in the registration.

- [ ] **Step 3: Implement**

In `_register_console_footer_shortcuts` (`chat_screen.py:3095-3100`), replace the body with:

```python
        shortcuts = (
            CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED
            if self._console_setup_modal_blocking()
            else CONSOLE_WORKBENCH_SHORTCUTS
        )
        # task-16320 / ADR-031: advertise the focus toggle in the footer —
        # the only exit affordance visible in focus mode (no nav bar). The
        # label names the action the key will perform, per the truthfulness
        # rule.
        focus_label = (
            "exit focus"
            if bool(getattr(self.app_instance, "focus_mode", False))
            else "focus"
        )
        shortcuts = (*shortcuts, ("Ctrl+Shift+F", focus_label))
        self.register_footer_shortcuts(source="console", shortcuts=shortcuts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): advertise Ctrl+Shift+F focus toggle in footer hints"
```

---

### Task 5: App-level toggle binding and navigation exit choke point

**Files:**
- Modify: `tldw_chatbook/app.py:5253-5259` (`TldwCli.BINDINGS`), `app.py:8205` (new helpers after `_resolve_initial_shell_route`), `app.py:8465-8468` (`_handle_screen_navigation_locked`)
- Test: `Tests/UI/test_focus_mode.py`

**Interfaces:**
- Consumes: `ChatScreen._apply_focus_chrome` (Task 3, invoked duck-typed — no import of `chat_screen` needed, avoiding the circular import the screen registry exists to prevent).
- Produces: app action `toggle_focus_mode` (binding `ctrl+shift+f`, `show=False`); `TldwCli._set_focus_mode(enabled: bool) -> None`; `TldwCli._clear_focus_if_leaving_console(screen_name: str) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_focus_mode.py`:

```python
class TestAppToggleAndNavigationExit:
    def test_ctrl_shift_f_binding_registered(self):
        from tldw_chatbook.app import TldwCli

        assert any(
            binding.key == "ctrl+shift+f" for binding in TldwCli.BINDINGS
        )

    def test_set_focus_mode_applies_to_console_screen(self):
        from tldw_chatbook.app import TldwCli

        calls = []

        class FakeConsoleScreen:
            def _apply_focus_chrome(self):
                calls.append("applied")

        stub = SimpleNamespace(
            focus_mode=False,
            _navigation_outgoing_screen=lambda: FakeConsoleScreen(),
            post_message=lambda msg: calls.append(msg),
        )
        TldwCli._set_focus_mode(stub, True)
        assert stub.focus_mode is True
        assert calls == ["applied"]

    def test_set_focus_mode_navigates_when_elsewhere(self):
        from tldw_chatbook.app import TldwCli

        posted = []
        stub = SimpleNamespace(
            focus_mode=False,
            _navigation_outgoing_screen=lambda: object(),
            post_message=posted.append,
        )
        TldwCli._set_focus_mode(stub, True)
        assert stub.focus_mode is True
        assert len(posted) == 1
        assert posted[0].screen_name == TAB_CHAT

    def test_set_focus_mode_disable_clears_flag(self):
        from tldw_chatbook.app import TldwCli

        posted = []
        stub = SimpleNamespace(
            focus_mode=True,
            _navigation_outgoing_screen=lambda: object(),
            post_message=posted.append,
        )
        TldwCli._set_focus_mode(stub, False)
        assert stub.focus_mode is False
        assert posted == []  # disabling never navigates

    def test_clear_focus_when_leaving_console(self):
        from tldw_chatbook.app import TldwCli

        leaving = SimpleNamespace(focus_mode=True)
        TldwCli._clear_focus_if_leaving_console(leaving, "settings")
        assert leaving.focus_mode is False

        staying = SimpleNamespace(focus_mode=True)
        TldwCli._clear_focus_if_leaving_console(staying, TAB_CHAT)
        assert staying.focus_mode is True

    def test_action_toggle_flips_state(self):
        from tldw_chatbook.app import TldwCli

        stub = SimpleNamespace(
            focus_mode=True,
            _set_focus_mode=lambda enabled: setattr(stub, "focus_mode", enabled),
        )
        TldwCli.action_toggle_focus_mode(stub)
        assert stub.focus_mode is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py::TestAppToggleAndNavigationExit -v`
Expected: FAIL — no `_set_focus_mode` / `_clear_focus_if_leaving_console` / binding.

- [ ] **Step 3: Implement**

a) In `TldwCli.BINDINGS` (`app.py:5246`), add to the app-global cluster (after the `f6` entry at `app.py:5251`):

```python
            Binding(
                "ctrl+shift+f",
                "toggle_focus_mode",
                "Focus Mode",
                show=False,
            ),
```

b) After `_resolve_initial_shell_route` (ends `app.py:8205`), add:

```python
    def _set_focus_mode(self, enabled: bool) -> None:
        """Set focus mode and apply it to the Console if it is on screen.

        task-16320 / ADR-067. Duck-types the content screen (it may or may
        not be the Console — do NOT import ChatScreen here; the screen
        registry keeps app.py free of screen imports for circular-import
        reasons). Enabling while elsewhere navigates to the Console first;
        the screen's mount-time ``_apply_focus_chrome`` read then applies
        the chrome. Disabling only clears the flag.
        """
        self.focus_mode = enabled
        content_screen = self._navigation_outgoing_screen()
        apply_chrome = getattr(content_screen, "_apply_focus_chrome", None)
        if callable(apply_chrome):
            apply_chrome()
        elif enabled:
            self.post_message(NavigateToScreen(TAB_CHAT))

    def action_toggle_focus_mode(self) -> None:
        """Ctrl+Shift+F: toggle the chrome-free Console focus mode."""
        self._set_focus_mode(not self.focus_mode)

    def _clear_focus_if_leaving_console(self, screen_name: str) -> None:
        """Single exit rule (ADR-067): focus mode is Console-only — any
        navigation to another route restores normal chrome on arrival."""
        if screen_name != TAB_CHAT:
            self.focus_mode = False
```

c) In `_handle_screen_navigation_locked`, immediately after the target resolution (the `screen_name, current_tab_value, screen_class = ...` assignment ending at `app.py:8467`, before the `logger.info(f"Navigating to screen...")` line), add:

```python
        self._clear_focus_if_leaving_console(screen_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (17 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): app-level Ctrl+Shift+F focus toggle and navigation exit rule"
```

---

### Task 6: Command palette QuickAction

**Files:**
- Modify: `tldw_chatbook/app.py:1180-1264` (`QuickActionsProvider.search`, `.discover`, `.execute_quick_action`)
- Test: `Tests/UI/test_focus_mode.py`

**Interfaces:**
- Consumes: `TldwCli.action_toggle_focus_mode` (Task 5).
- Produces: palette command `"Quick Actions: Toggle Focus Mode"` (action id `"toggle_focus_mode"`).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_focus_mode.py`:

```python
class TestPaletteQuickAction:
    async def test_focus_toggle_is_searchable_and_executable(self):
        from tldw_chatbook.app import QuickActionsProvider

        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test() as pilot:
            provider = QuickActionsProvider(pilot.app.screen)
            hits = [hit async for hit in provider.search("focus")]
            assert any("Toggle Focus Mode" in hit.display for hit in hits)

            called = []
            pilot.app.action_toggle_focus_mode = lambda: called.append(True)
            provider.execute_quick_action("toggle_focus_mode")
            assert called == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py::TestPaletteQuickAction -v`
Expected: FAIL — no hit matches "Toggle Focus Mode".

- [ ] **Step 3: Implement**

In `QuickActionsProvider.search` (`app.py:1189`), append to the `quick_actions` list:

```python
            (
                "Quick Actions: Toggle Focus Mode",
                "toggle_focus_mode",
                "Hide or restore the Console's nav bar and header (Ctrl+Shift+F)",
            ),
```

In `discover` (`app.py:1224`), append the same 3-tuple to `popular_actions`.

In `execute_quick_action` (`app.py:1251`), add a branch:

```python
            elif action_id == "toggle_focus_mode":
                self.app.action_toggle_focus_mode()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_focus_mode.py -v`
Expected: PASS (18 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py Tests/UI/test_focus_mode.py
git commit -m "feat(console): add Toggle Focus Mode quick action to command palette"
```

---

### Task 7: Documentation and closeout

**Files:**
- Modify: `Docs/User_Guide/console.md` (new "Focus mode" subsection)
- Modify: `backlog/tasks/task-16320 - Focus-mode-chrome-free-Console-presentation.md` (ACs → checked, Implementation Notes)
- Test: no new tests; verification runs.

**Interfaces:**
- Consumes: everything above.
- Produces: shipped docs + closed task.

- [ ] **Step 1: Document focus mode**

In `Docs/User_Guide/console.md`, add a `### Focus mode` subsection after the "Small terminals" section (~line 93), following the file's existing tone:

```markdown
### Focus mode

Focus mode (`Ctrl+Shift+F`, or the palette's "Quick Actions: Toggle Focus
Mode") strips the Console down to the conversation: the navigation bar and
the workbench header disappear; the one-line status bar — token count and
key hints — stays. It is the claude-code-style surface for heads-down
coding, and the comfortable shape on a phone over `--serve`, where fine
pointers and function keys are scarce.

- Start chrome-free every launch: set `[general] focus_mode = true`, or
  launch with `--focus` (which also forces the Console as the startup
  screen, overriding `default_tab`; first-run onboarding still comes
  first).
- Leaving is any navigation — a destination hotkey or a palette jump
  lands you on the target screen with normal chrome, and one
  `Ctrl+Shift+F` brings the focused Console back.
- Context usage remains available in the status line (on wide terminals)
  and via `Ctrl+Shift+P`.
```

- [ ] **Step 2: Run the targeted verification set**

```bash
python3 -m pytest Tests/UI/test_focus_mode.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_scope_row.py -v
```

Expected: all PASS. If contract tests reference the console footer registration shape and break on the new pair, update those fixtures to include the focus pair — the truthfulness rule (ADR-031) means the fixture must match the real registration, not the other way around.

- [ ] **Step 3: Update the backlog task**

In `task-16320`'s file: check all six ACs (`- [x]`), and append to Implementation Notes: approach summary, the files touched (`config.py`, `app.py`, `chat_screen.py`, `css/components/_agentic_terminal.tcss` + regenerated bundle, `Tests/UI/test_focus_mode.py`, `Docs/User_Guide/console.md`), and the deviations from plan (if any). Link ADR-067 (already linked in notes).

```bash
backlog task edit 16320 --notes "..."   # append the closeout summary
backlog task edit 16320 -s Done          # only after review approval
```

- [ ] **Step 4: Commit**

```bash
git add Docs/User_Guide/console.md "backlog/tasks/task-16320 - Focus-mode-chrome-free-Console-presentation.md"
git commit -m "docs(console): document focus mode; close task-16320"
```

---

## Self-Review (completed)

- **Spec coverage:** entry points (Tasks 1–2), chrome suppression + kept status line (Task 3), footer hint truthfulness (Task 4), toggle + single exit rule (Task 5), palette discoverability (Task 6), docs/closeout incl. AC #6 guard tests (Tasks 3 & 7). Spec's "recompose survival" and "toggle during streaming" edge cases need no code — the class lives on the screen and no rebuild occurs; covered by Task 3's flip test.
- **Placeholders:** none; every code step shows the code.
- **Type consistency:** `_apply_focus_chrome()` (Task 3) is consumed duck-typed in `_set_focus_mode` (Task 5); `_footer_shortcut_registration` tuple shape `(source, shortcuts)` matches `base_app_screen.py:242-246`; action id `toggle_focus_mode` consistent across Tasks 5–6.
