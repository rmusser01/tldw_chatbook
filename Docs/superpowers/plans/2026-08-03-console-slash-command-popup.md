# Console Slash-Command Popup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a user types `/` in the native Console composer, show a floating, keyboard-navigable suggestion popup (claude-code style) listing slash commands and skills, filtered live; Enter/Tab inserts the completion.

**Architecture:** A pure suggestion-provider function (`Chat/console_command_suggestions.py`) computes completions from the draft text; a non-focusable overlay widget (`Widgets/Console/console_command_popup.py`) renders them absolutely positioned above the composer; `ChatScreen` owns all wiring (draft-mutation sync hook, key routing in `on_key`, escape/tab actions).

**Tech Stack:** Python 3.11+, Textual 8.2.7 (OptionList, `position: absolute` + `styles.offset`), pytest + Textual `run_test()` pilot tests.

**Spec:** `Docs/superpowers/specs/2026-08-03-console-slash-command-popup-design.md` (ADR: none required — self-contained UI feature, rationale in spec).

---

## Key codebase facts (verified)

- Composer: `ConsoleComposerBar` (`tldw_chatbook/Widgets/Console/console_composer_bar.py`), instantiated in `ChatScreen.compose()` at `tldw_chatbook/UI/Screens/chat_screen.py:7586`, yielded via `yield self._frame_console_region(composer)` at line 7599.
- Registry: `self._console_command_registry` (chat_screen.py:1579) with `/prompt`, `/system`, `/skills`; skill snapshot at `self._console_skill_candidates: tuple[SkillCommandCandidate, ...]` (chat_screen.py:1587).
- Draft-mutation funnel: `_sync_console_workbench_actions_from_draft` (chat_screen.py:11142) is already called after every keystroke-level mutation in `on_key` (chat_screen.py:11307) and `on_paste`.
- Key handling: `ChatScreen.on_key` (chat_screen.py:11307). `enter` branch at 11379 (popup accept must run before it). `up`/`down` are currently unhandled. `tab` is a Binding → `action_focus_next` (chat_screen.py:630). `escape` has two bindings → `action_expand_collapsed_console_composer` (priority, :624) and `action_focus_console_composer_home` (:1198).
- Overlay precedent: `tldw_chatbook/Widgets/tooltip.py:139` positions a floating widget with `self.styles.offset = (x, y)`; `Widget.region` is relative to the Screen, `Widget.content_region` is absolute (verified against installed Textual 8.2.7 docstrings).
- CSS is generated: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, then run `python3 tldw_chatbook/css/build_css.py` to regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` (both are committed).
- `ChatScreen` has no existing `on_resize` handler — safe to add one.
- Test harness for screen pilot tests: `ConsoleHarness`, `_build_test_app`, `_configure_native_ready_console`, `_wait_for_selector` — see `Tests/UI/test_console_command_composer.py:1-30` for exact imports.

---

## Task 1: Suggestion provider (pure logic)

**Files:**
- Create: `tldw_chatbook/Chat/console_command_suggestions.py`
- Test: `Tests/Chat/test_console_command_suggestions.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the pure slash-command suggestion provider."""

from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_command_suggestions import suggestions_for_draft
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate

SKILLS = (
    SkillCommandCandidate(name="web-search", description="Search the web"),
    SkillCommandCandidate(name="summarize", description="Summarize text"),
)


def _labels(result):
    return [s.label for s in result]


def test_bare_slash_lists_commands_then_skills():
    result = suggestions_for_draft("/", default_console_registry(), SKILLS)
    assert _labels(result) == ["/prompt", "/system", "/skills", "/web-search", "/summarize"]


def test_prefix_filters_case_insensitively():
    result = suggestions_for_draft("/SK", default_console_registry(), SKILLS)
    assert _labels(result) == ["/skills"]


def test_skill_entries_insert_bare_slash_name():
    result = suggestions_for_draft("/w", default_console_registry(), SKILLS)
    assert _labels(result) == ["/web-search"]
    assert result[0].insert_text == "/web-search "
    assert result[0].description == "Search the web"


def test_non_command_drafts_return_none():
    registry = default_console_registry()
    assert suggestions_for_draft("hello", registry, SKILLS) is None
    assert suggestions_for_draft("/prompt foo", registry, SKILLS) is None
    assert suggestions_for_draft(" /", registry, SKILLS) is None


def test_empty_filter_returns_empty_list():
    assert suggestions_for_draft("/zzz", default_console_registry(), SKILLS) == []


def test_skills_arg_mode_filters_and_builds_full_replacement():
    result = suggestions_for_draft("/skills w", default_console_registry(), SKILLS)
    assert _labels(result) == ["web-search"]
    assert result[0].insert_text == "/skills web-search "


def test_skills_arg_mode_ends_after_second_argument():
    assert suggestions_for_draft("/skills web-search extra", default_console_registry(), SKILLS) is None


def test_skill_named_like_a_command_is_deduplicated():
    skills = (SkillCommandCandidate(name="prompt", description="clash"),)
    result = suggestions_for_draft("/", default_console_registry(), skills)
    assert _labels(result) == ["/prompt", "/system", "/skills"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_command_suggestions.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_command_suggestions'`

- [ ] **Step 3: Implement the provider**

Create `tldw_chatbook/Chat/console_command_suggestions.py`:

```python
"""Pure slash-command suggestion provider for the Console composer popup.

Mirrors the purity discipline of :mod:`console_command_grammar` and
:mod:`console_skill_resolver`: no Textual, no app state, no I/O. Callers own
all UI wiring and paste-segment gating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .console_command_grammar import (
    COMMAND_PREFIX,
    SKILLS_COMMAND_NAME,
    ConsoleCommandRegistry,
)
from .console_skill_resolver import SkillCommandCandidate

_COMMAND_MODE_PATTERN = re.compile(r"^/(\S*)$")
_SKILLS_ARG_MODE_PATTERN = re.compile(
    rf"^{COMMAND_PREFIX}{SKILLS_COMMAND_NAME}\s+(\S*)$", re.IGNORECASE
)

# `ConsoleCommand` carries no description field, so the three built-ins get
# their popup copy here; skill entries use the resolver snapshot descriptions.
_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "prompt": "Insert a saved prompt into the composer",
    "system": "Apply a saved system prompt to this session",
    "skills": "List or run a skill",
}


@dataclass(frozen=True)
class CommandSuggestion:
    """One popup row.

    Args:
        insert_text: Full-draft replacement text applied on accept (note the
            trailing space, which re-triggers arg-mode for ``/skills ``).
        label: Display label, e.g. ``"/prompt"`` (command mode) or the bare
            skill name (skills-arg mode).
        description: Short human-readable description; may be empty.
    """

    insert_text: str
    label: str
    description: str = ""


def suggestions_for_draft(
    draft_text: str,
    registry: ConsoleCommandRegistry,
    skill_candidates: tuple[SkillCommandCandidate, ...],
) -> list[CommandSuggestion] | None:
    """Compute popup suggestions for one composer draft.

    Returns ``None`` when the draft is in no completion context (caller hides
    the popup); otherwise a possibly-empty list (empty also hides the popup).
    Two contexts: command mode (``^/\\S*$`` — commands then skills, prefix-
    filtered) and skills-arg mode (``^/skills\\s+\\S*$`` — skill names for the
    first argument).
    """
    skills_arg_match = _SKILLS_ARG_MODE_PATTERN.match(draft_text)
    if skills_arg_match is not None:
        prefix = skills_arg_match.group(1).lower()
        return [
            CommandSuggestion(
                insert_text=f"{COMMAND_PREFIX}{SKILLS_COMMAND_NAME} {candidate.name} ",
                label=candidate.name,
                description=candidate.description,
            )
            for candidate in skill_candidates
            if candidate.name.lower().startswith(prefix)
        ]

    command_match = _COMMAND_MODE_PATTERN.match(draft_text)
    if command_match is None:
        return None

    prefix = command_match.group(1).lower()
    command_names = registry.available_names()
    suggestions = [
        CommandSuggestion(
            insert_text=f"{COMMAND_PREFIX}{name} ",
            label=f"{COMMAND_PREFIX}{name}",
            description=_COMMAND_DESCRIPTIONS.get(name.lower(), ""),
        )
        for name in command_names
        if name.lower().startswith(prefix)
    ]
    command_names_lower = {name.lower() for name in command_names}
    suggestions.extend(
        CommandSuggestion(
            insert_text=f"{COMMAND_PREFIX}{candidate.name} ",
            label=f"{COMMAND_PREFIX}{candidate.name}",
            description=candidate.description,
        )
        for candidate in skill_candidates
        if candidate.name.lower().startswith(prefix)
        and candidate.name.lower() not in command_names_lower
    )
    return suggestions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_command_suggestions.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_command_suggestions.py Tests/Chat/test_console_command_suggestions.py
git commit -m "feat: pure slash-command suggestion provider for console composer"
```

---

## Task 2: `ConsoleCommandPopup` widget + CSS

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_command_popup.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py` (add export)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (append styles)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_command_popup.py`

- [ ] **Step 1: Write the failing widget test**

Create `Tests/UI/test_console_command_popup.py`:

```python
"""ConsoleCommandPopup widget behavior; ChatScreen integration (Tasks 3-4)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_command_suggestions import CommandSuggestion
from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup

SUGGESTIONS = [
    CommandSuggestion(insert_text="/a ", label="/a", description="first"),
    CommandSuggestion(insert_text="/b ", label="/b", description="second"),
]


class _PopupApp(App):
    def compose(self) -> ComposeResult:
        # The popup repositions against whatever carries this id; a Static
        # suffices for widget-level tests.
        yield Static("anchor", id="console-native-composer")
        yield ConsoleCommandPopup()


@pytest.mark.asyncio
async def test_popup_show_highlight_accept_hide():
    app = _PopupApp()
    async with app.run_test(size=(80, 24)) as pilot:
        popup = app.screen.query_one(ConsoleCommandPopup)
        assert not popup.is_open

        popup.show_suggestions(SUGGESTIONS)
        await pilot.pause()
        assert popup.is_open
        assert popup.accept_selected().label == "/a"

        popup.move_highlight(1)
        assert popup.accept_selected().label == "/b"

        popup.move_highlight(1)  # wraps
        assert popup.accept_selected().label == "/a"

        popup.hide()
        await pilot.pause()
        assert not popup.is_open
        assert popup.accept_selected() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_command_popup.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.Console.console_command_popup'`

- [ ] **Step 3: Implement the widget**

Create `tldw_chatbook/Widgets/Console/console_command_popup.py`:

```python
"""Floating slash-command suggestion popup for the native Console composer.

Screen-owned overlay: the owning screen feeds it suggestions, routes
Up/Down/Enter/Tab/Escape to it while open, and it never takes focus. It
positions itself (``position: absolute`` + ``styles.offset``) so its bottom
edge sits just above the composer — the same anchored-overlay technique as
``Widgets/tooltip.py``.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from ...Chat.console_command_suggestions import CommandSuggestion

MAX_VISIBLE_ROWS = 8
MIN_WIDTH = 30


class _SuggestionOption(Option):
    """OptionList row carrying its originating `CommandSuggestion`."""

    def __init__(self, suggestion: CommandSuggestion) -> None:
        prompt = Text(suggestion.label, style="bold")
        if suggestion.description:
            prompt.append("  ")
            prompt.append(suggestion.description, style="dim")
        super().__init__(prompt)
        self.suggestion = suggestion


class ConsoleCommandPopup(Widget):
    """Non-focusable overlay listing slash-command completions."""

    can_focus = False

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "console-command-popup")
        super().__init__(**kwargs)
        self._suggestions: list[CommandSuggestion] = []
        self._desired_height = 0
        # Hidden by default in code (not just TCSS) so the widget is correct
        # even where the bundled stylesheet is not loaded (bare test apps).
        self.display = False

    def compose(self) -> ComposeResult:
        option_list = OptionList(id="console-command-popup-options")
        option_list.can_focus = False
        yield option_list

    @property
    def is_open(self) -> bool:
        """Return whether the popup is currently displayed."""
        return self.display

    def show_suggestions(self, suggestions: list[CommandSuggestion]) -> None:
        """Replace rows, reset the highlight, reposition, and show."""
        self._suggestions = list(suggestions)
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(
            [_SuggestionOption(suggestion) for suggestion in self._suggestions]
        )
        self._desired_height = min(len(self._suggestions), MAX_VISIBLE_ROWS)
        self.styles.height = self._desired_height
        option_list.highlighted = 0
        self.reposition()
        self.display = True

    def hide(self) -> None:
        """Hide the popup and drop its rows."""
        self.display = False
        self._suggestions = []

    def move_highlight(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        count = len(self._suggestions)
        if count == 0:
            return
        option_list = self.query_one(OptionList)
        current = option_list.highlighted or 0
        option_list.highlighted = (current + delta) % count

    def accept_selected(self) -> CommandSuggestion | None:
        """Return the highlighted suggestion, or ``None`` when unavailable."""
        highlighted = self.query_one(OptionList).highlighted
        if highlighted is None or not (0 <= highlighted < len(self._suggestions)):
            return None
        return self._suggestions[highlighted]

    def reposition(self) -> None:
        """Anchor the popup's bottom edge just above the composer.

        ``composer.region`` is Screen-relative and ``content_region`` is
        absolute, so the offset math works regardless of which container the
        popup is mounted in.
        """
        if self.parent is None:
            return
        try:
            composer = self.screen.query_one("#console-native-composer")
        except Exception:
            return
        anchor = composer.region
        origin = self.parent.content_region
        x = anchor.x - origin.x
        y = anchor.y - origin.y - self._desired_height
        self.styles.offset = (max(x, 0), max(y, 0))
        self.styles.width = max(anchor.width, MIN_WIDTH)
```

- [ ] **Step 4: Export the widget**

In `tldw_chatbook/Widgets/Console/__init__.py`, add the import next to the existing `console_composer_bar` import (line 4) and add `"ConsoleCommandPopup"` to `__all__` (near line 23):

```python
from .console_command_popup import ConsoleCommandPopup
```

- [ ] **Step 5: Add the CSS and rebuild**

Append to `tldw_chatbook/css/components/_agentic_terminal.tcss`:

```tcss
/* Slash-command completion popup: floats above the composer. Position,
 * offset, width, and height are set imperatively in
 * ConsoleCommandPopup.show_suggestions/reposition. */
#console-command-popup {
    position: absolute;
    display: none;
    background: $ds-surface-panel;
    padding: 0 1;
}
```

Then regenerate the bundle:

Run: `python3 tldw_chatbook/css/build_css.py`
Expected: prints "CSS build complete"; `git status` shows `tldw_chatbook/css/tldw_cli_modular.tcss` modified.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_command_popup.py -q`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_command_popup.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_command_popup.py
git commit -m "feat: floating ConsoleCommandPopup overlay widget"
```

---

## Task 3: ChatScreen wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

All edits are in `ChatScreen`. No new tests in this task; the existing suite must stay green (Task 4 adds the integration tests).

- [ ] **Step 1: Add imports**

Add near the other `...Chat` / `...Widgets.Console` imports (chat_screen.py:~215-256):

```python
from ...Chat.console_command_suggestions import suggestions_for_draft
from ...Widgets.Console.console_command_popup import ConsoleCommandPopup
```

- [ ] **Step 2: Mount the popup in compose()**

In `compose()`, immediately after `yield self._frame_console_region(composer)` (chat_screen.py:7599), at the same indent level, add:

```python
            yield ConsoleCommandPopup()
```

- [ ] **Step 3: Add the popup helper methods**

Add these methods next to `_sync_console_workbench_actions_from_draft` (chat_screen.py:11142):

```python
    def _console_command_popup_or_none(self) -> ConsoleCommandPopup | None:
        try:
            return self.query_one("#console-command-popup", ConsoleCommandPopup)
        except QueryError:
            return None

    def _sync_console_command_popup(self) -> None:
        """Show/hide the slash-command popup from the current composer draft."""
        popup = self._console_command_popup_or_none()
        if popup is None:
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        if composer.has_paste_segments():
            popup.hide()
            return
        suggestions = suggestions_for_draft(
            composer.draft_text(),
            self._console_command_registry,
            self._console_skill_candidates,
        )
        if not suggestions:
            popup.hide()
            return
        popup.show_suggestions(suggestions)

    def _dismiss_console_command_popup(self) -> bool:
        """Hide the popup if open. Returns True when it was open."""
        popup = self._console_command_popup_or_none()
        if popup is None or not popup.is_open:
            return False
        popup.hide()
        return True

    def _accept_console_command_popup(self) -> bool:
        """Insert the highlighted suggestion into the draft. True when accepted."""
        popup = self._console_command_popup_or_none()
        if popup is None or not popup.is_open:
            return False
        suggestion = popup.accept_selected()
        if suggestion is None:
            return False
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return False
        composer.load_draft(suggestion.insert_text)
        self._sync_console_workbench_actions_from_draft()
        return True
```

- [ ] **Step 4: Hook the draft-mutation funnel**

At the end of `_sync_console_workbench_actions_from_draft` (chat_screen.py:11142), add:

```python
        self._sync_console_command_popup()
```

- [ ] **Step 5: Route Up/Down/Enter in on_key**

In `on_key` (chat_screen.py:11307), immediately after the `if not self._should_capture_console_input(composer): return` guard, insert:

```python
        popup = self._console_command_popup_or_none()
        if popup is not None and popup.is_open:
            if event.key == "up":
                popup.move_highlight(-1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "down":
                popup.move_highlight(1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                self._accept_console_command_popup()
                event.stop()
                event.prevent_default()
                return
```

(Enter acceptance deliberately runs before the existing `enter` branch at ~11379, so an open popup never triggers the paste-token/send path.)

- [ ] **Step 6: Route Tab via action_focus_next**

At the top of `action_focus_next` (chat_screen.py:630), before the setup-modal line, add:

```python
        if self._accept_console_command_popup():
            return
```

- [ ] **Step 7: Route Escape via both escape actions**

At the top of `action_expand_collapsed_console_composer` (chat_screen.py:624), after the setup-modal guard, add:

```python
        if self._dismiss_console_command_popup():
            return
```

At the top of `action_focus_console_composer_home` (chat_screen.py:1198), after the setup-modal guard, add the same two lines.

- [ ] **Step 8: Reposition on screen resize**

Add a new handler (the class has no existing `on_resize`):

```python
    def on_resize(self) -> None:
        """Keep an open command popup anchored above the composer."""
        popup = self._console_command_popup_or_none()
        if popup is not None and popup.is_open:
            popup.reposition()
```

- [ ] **Step 9: Sync the popup after programmatic draft loads/clears**

The funnel (Step 4) covers keystrokes and paste, but these sites mutate the draft programmatically: chat_screen.py:3084, 3086, 8564, 8900, 8945, 9310, 9434, 11656, 11675 (`composer.load_draft(...)` / `composer.clear_draft()`). After each of those calls, add `self._sync_console_command_popup()` **unless** `_sync_console_workbench_actions_from_draft()` is already called within the same code block (it now syncs the popup too). Do not touch line 7596 (inside `compose()`; the popup is not yet mounted and the first keystroke syncs anyway).

- [ ] **Step 10: Run the existing console test suite**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_command_composer.py Tests/UI/test_console_composer_collapse.py Tests/UI/test_console_composer_cursor.py Tests/Chat/ -q`
Expected: all pass (no regressions; popup tests still pending Task 4)

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "feat: wire slash-command popup into ChatScreen key routing and draft sync"
```

---

## Task 4: Screen-level integration tests

**Files:**
- Modify: `Tests/UI/test_console_command_popup.py` (append)

Use the harness imports from `Tests/UI/test_console_command_composer.py:9-23` (`_build_test_app`, `_configure_native_ready_console`, `_wait_for_selector`, `ConsoleHarness`).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_command_popup.py`:

```python
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_skill_resolver import SkillCommandCandidate
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def _popup_labels(popup) -> list[str]:
    return [s.label for s in popup._suggestions]


@pytest.mark.asyncio
async def test_slash_opens_popup_and_typing_filters():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        assert not popup.is_open

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open
        assert _popup_labels(popup) == ["/prompt", "/system", "/skills"]

        await pilot.press("s", "y", "s")
        await pilot.pause()
        assert _popup_labels(popup) == ["/system"]


@pytest.mark.asyncio
async def test_enter_accepts_and_inserts_without_sending():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/", "s", "k")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/skills "
        # No skill candidates configured -> arg-mode list is empty -> hidden.
        assert not popup.is_open


@pytest.mark.asyncio
async def test_down_up_navigates_and_tab_accepts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.press("/")
        await pilot.pause()
        await pilot.press("down")  # highlight "/system"
        await pilot.press("tab")
        await pilot.pause()
        assert composer.draft_text() == "/system "


@pytest.mark.asyncio
async def test_escape_closes_popup_and_keeps_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open
        await pilot.press("escape")
        await pilot.pause()
        assert not popup.is_open
        assert composer.draft_text() == "/"


@pytest.mark.asyncio
async def test_skill_entries_and_skills_arg_mode():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        console._console_skill_candidates = (
            SkillCommandCandidate(name="web-search", description="Search the web"),
        )

        await pilot.press("/", "w")
        await pilot.pause()
        assert _popup_labels(popup) == ["/web-search"]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/web-search "

        composer.load_draft("/skills w")
        console._sync_console_command_popup()
        await pilot.pause()
        assert popup.is_open
        assert _popup_labels(popup) == ["web-search"]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/skills web-search "
```

- [ ] **Step 2: Run tests, fix any failures**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_command_popup.py -q`
Expected: 6 passed. If positioning/visibility asserts fail, debug the widget's `display`/`height` handling — do not weaken the assertions.

- [ ] **Step 3: Commit**

```bash
git add Tests/UI/test_console_command_popup.py
git commit -m "test: ChatScreen integration tests for slash-command popup"
```

---

## Task 5: Verification sweep

- [ ] **Step 1: Run the full affected suites**

Run: `.venv/bin/python -m pytest Tests/UI/ Tests/Chat/ -q`
Expected: all pass

- [ ] **Step 2: Lint the touched files**

Run: `.venv/bin/python -m ruff check tldw_chatbook/Chat/console_command_suggestions.py tldw_chatbook/Widgets/Console/console_command_popup.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_command_suggestions.py Tests/UI/test_console_command_popup.py`
Expected: no errors

- [ ] **Step 3: Update the spec's status line**

In `Docs/superpowers/specs/2026-08-03-console-slash-command-popup-design.md`, change `Status: Approved (design)` to `Status: Implemented`.

- [ ] **Step 4: Final commit**

```bash
git add Docs/superpowers/specs/2026-08-03-console-slash-command-popup-design.md
git commit -m "docs: mark slash-command popup spec implemented"
```
