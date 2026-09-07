# TASK-436 Characters empty-state guidance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the Roleplay/Personas workbench is in Characters mode with no character selected, the center pane shows onboarding guidance naming the three next actions (select a row, New, Import) instead of a blank pane; the guidance disappears the moment a selection exists.

**Architecture:** Add one dedicated `Static` (`#personas-characters-empty`) to the center detail stack and register it in `_CENTER_VIEW_IDS`. Resolve the empty state once, at the top of `_show_center`: when `visible_id is None` and Characters mode has no selection, show the guidance widget instead of blanking. Every no-selection path (mode-enter, first mount, delete, cancel-New, restore-failure) already funnels through `_show_center(None)`, so this single choke point covers them all; explicit-id calls (card/editor) hide the guidance automatically (AC#2).

**Tech Stack:** Python 3.11+, Textual, pytest.

## Global Constraints

- Guidance is **Characters-mode only** — `active_mode == "characters"` and `state.selected_entity_id` empty. Personas/Dictionaries/Lore `_show_center(None)` calls must still blank the center.
- **Do NOT touch `#personas-mode-placeholder`** or `_mode_placeholder_text`/`_MODE_PLACEHOLDER_BODY` — that widget is the retired-"prompts" mode fallback, pinned by `test_personas_workbench.py:434` and `:828`. The new guidance is a separate widget.
- Guidance copy is a **single, non-adaptive** module constant (it renders at `on_mount:829` before `_character_total` is loaded, so a count-adaptive copy would be wrong on first mount). It must name all three actions: select a row, **New**, **Import**.
- CSS goes in the screen's inline `DEFAULT_CSS` (`personas_screen.py:349`), **not** the generated `tldw_cli_modular.tcss` bundle.
- Guidance body is static app copy only (`markup=True` emphasises literal action words; no user/entity interpolation).
- AC#2 needs **no** extra wiring — it relies on the exclusive `_show_center` (selecting → `#ccp-character-card-view`, New/Import → `#ccp-character-editor-view`).

---

### Task 1: Characters-mode empty-state guidance

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (constant near `:191`; `_CENTER_VIEW_IDS` `:258-267`; `compose_content` Static after `:683`; two helper methods near `_show_center`; the `_show_center` branch `:5756`; `DEFAULT_CSS` after the `#personas-detail-stack` rule `:441`)
- Test: `Tests/UI/test_personas_workbench.py` (new test class/methods mirroring the existing `PersonasTestApp`/`_mounted`/`_select_first_character` harness)

**Interfaces:**
- Produces: a `#personas-characters-empty` `Static` that is `display=True` exactly when `active_mode == "characters"` and no selection, else `display=False`. New methods `_should_show_characters_empty_guidance() -> bool` and `_characters_empty_guidance_text() -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/UI/test_personas_workbench.py`. Mirror the file's harness: `PersonasTestApp(mock_app_instance)`, `async with app.run_test(size=(160, 50)) as pilot`, the module-level `_mounted(pilot)` helper, the `stub_characters` fixture, and the existing `_select_first_character(pilot)` selection helper (used by `TestCharacterSelectionAndEdit`; call it the same way those tests do, or inline a `PersonaEntitySelected`/row click as the neighbouring tests do). `Static` is already imported.

```python
class TestCharactersEmptyStateGuidance:
    """task-436: Characters mode shows onboarding guidance when nothing is selected."""

    async def test_guidance_shown_when_no_selection(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            assert screen.state.active_mode == "characters"
            assert not screen.state.selected_entity_id
            guidance = screen.query_one("#personas-characters-empty", Static)
            assert guidance.display is True
            body = str(guidance.renderable)
            assert "New" in body and "Import" in body
            assert screen.query_one("#ccp-character-card-view").display is False

    async def test_guidance_hidden_after_selection(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self_select_first_character(pilot)  # see note below
            assert screen.state.selected_entity_id
            assert screen.query_one("#personas-characters-empty", Static).display is False
            assert screen.query_one("#ccp-character-card-view").display is True

    async def test_guidance_hidden_in_other_modes(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await screen._apply_mode("lore")
            await pilot.pause()
            assert screen.query_one("#personas-characters-empty", Static).display is False
            await screen._apply_mode("characters")
            await pilot.pause()
            assert screen.query_one("#personas-characters-empty", Static).display is True

    async def test_guidance_returns_after_delete(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self_select_first_character(pilot)
            await screen._delete_entity()  # match the real delete entry point (see note)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.state.selected_entity_id
            assert screen.query_one("#personas-characters-empty", Static).display is True
```

> **Implementer notes for Step 1 (resolve before writing):**
> - Replace `self_select_first_character(pilot)` with however this file selects the first character — the `TestCharacterSelectionAndEdit` class has a `_select_first_character` helper; either subclass/reuse it, make these methods part of that class, or copy its selection idiom (it posts the library row's `PersonaEntitySelected` / clicks the first `.personas-library-row`). Use the exact working idiom already in the file.
> - For the delete test, drive deletion the same way an existing delete test does (find it via `grep -n "delete" Tests/UI/test_personas_workbench.py`) rather than guessing `_delete_entity`'s signature — the point is: after a real delete of the selected character, guidance returns.
> - Add a `test_guidance_returns_after_cancel_new` only if the file already has a New-then-cancel idiom to copy; otherwise omit it (the mode-switch + delete tests already exercise the re-show paths).

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python -m pytest Tests/UI/test_personas_workbench.py -k CharactersEmptyState -q`
Expected: FAIL — `#personas-characters-empty` does not exist yet (`NoMatches`/`QueryError`).

- [ ] **Step 3: Add the guidance constant + widget + `_CENTER_VIEW_IDS` entry**

In `personas_screen.py`, after `_PLACEHOLDER_FALLBACK = "This mode is coming soon."` (`:191`):
```python
#: Onboarding guidance shown in the Characters center pane when nothing is
#: selected (task-436). Non-adaptive: rendered at on_mount before the character
#: count loads, so it must read correctly whether or not characters exist.
_CHARACTERS_EMPTY_GUIDANCE = (
    "No character selected. Pick one from the list on the left, or use "
    "[b]New[/b] or [b]Import[/b] to add a character."
)
```

Add to `_CENTER_VIEW_IDS` (`:258-267`), after `"#personas-mode-placeholder",`:
```python
    "#personas-mode-placeholder",
    "#personas-characters-empty",
)
```

In `compose_content`, immediately after the `#personas-mode-placeholder` Static (`:681-683`), still inside the `#personas-detail-stack` container:
```python
                        yield Static(
                            self._mode_placeholder_text("prompts"),
                            id="personas-mode-placeholder",
                        )
                        yield Static(
                            _CHARACTERS_EMPTY_GUIDANCE,
                            id="personas-characters-empty",
                            markup=True,
                        )
```

- [ ] **Step 4: Add the two helper methods**

Near `_show_center` (e.g. just above it, around `:5755`), add:
```python
    def _should_show_characters_empty_guidance(self) -> bool:
        """True when the Characters center should show onboarding guidance.

        Returns:
            True in Characters mode with no selection (empty center), else False.
        """
        return (
            self.state.active_mode == "characters"
            and not self.state.selected_entity_id
        )

    def _characters_empty_guidance_text(self) -> str:
        """Return the onboarding guidance for the empty Characters center pane.

        Returns:
            Static guidance copy naming the three next actions.
        """
        return _CHARACTERS_EMPTY_GUIDANCE
```

- [ ] **Step 5: Resolve the guidance inside `_show_center`**

At the very top of `_show_center` (`:5756`), before the loop over `_CENTER_VIEW_IDS`:
```python
    def _show_center(self, visible_id: str | None) -> None:
        # Characters mode with nothing selected shows onboarding guidance, not a
        # blank pane (task-436). Every no-selection path funnels through here, so
        # resolving it once keeps mode-enter, first mount, delete, cancel-New and
        # restore-failure consistent; non-character modes and explicit-id calls
        # (card / editor) are unaffected, which makes AC#2 automatic.
        if visible_id is None and self._should_show_characters_empty_guidance():
            try:
                self.query_one("#personas-characters-empty", Static).update(
                    self._characters_empty_guidance_text()
                )
            except QueryError:
                pass
            else:
                visible_id = "#personas-characters-empty"
        # ... existing body unchanged ...
```
(The `try/except QueryError` mirrors `_show_center`'s existing tolerance for widgets not yet mounted; only set `visible_id` if the update succeeded. `QueryError` is already imported in this module.)

- [ ] **Step 6: Add the CSS rule to `DEFAULT_CSS`**

In `DEFAULT_CSS`, right after the `#personas-detail-stack { ... }` rule (`:437-441`):
```css
    #personas-characters-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        text-align: center;
        padding: 2 4;
        color: $text-muted;
    }
```

- [ ] **Step 7: Run the tests to confirm they pass**

Run: `python -m pytest Tests/UI/test_personas_workbench.py -k CharactersEmptyState -q`
Expected: PASS.

- [ ] **Step 8: Regression — personas suite + any blank-center assumption**

Run: `python -m pytest Tests/UI/test_personas_workbench.py Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_lore.py -q`
Expected: PASS. If any pre-existing test assumed the Characters center is fully blank with no selection (e.g. asserting some detail widget hidden as a proxy for "empty"), update it to reflect that `#personas-characters-empty` is now the visible empty state — do NOT weaken the new-guidance assertions. The retired-"prompts" placeholder tests (`:434`,`:828`) and existing `#ccp-character-card-view` display assertions must still pass unchanged (verify).

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): Characters center empty-state guidance (task-436)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 (guidance names select/New/Import when no selection) → Steps 3-6 + tests `test_guidance_shown_when_no_selection`, `test_guidance_hidden_in_other_modes` (switch-back). AC#2 (guidance disappears once a selection exists) → the exclusive `_show_center` + `test_guidance_hidden_after_selection`; re-show after deselect → `test_guidance_returns_after_delete`. Both covered.
- **Placeholder scan:** the only judgement calls are the test-harness selection/delete idioms, explicitly delegated to the implementer with a concrete way to resolve each (reuse the file's existing helpers) — no `TODO`/`TBD` in the shipped code.
- **Consistency:** `#personas-characters-empty`, `_should_show_characters_empty_guidance`, `_characters_empty_guidance_text`, `_CHARACTERS_EMPTY_GUIDANCE` used identically across compose, `_CENTER_VIEW_IDS`, helpers, `_show_center`, CSS, and tests. `#personas-mode-placeholder` is left untouched.
