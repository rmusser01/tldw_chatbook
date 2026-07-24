# TASK-438 Alternate greeting selector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a character has alternate greetings, the Roleplay preview shows a dropdown to pick which greeting seeds the conversation, and Reset returns to the chosen greeting (not the primary).

**Architecture:** A `Select` in the preview pane (shown only when alternates exist). The controller owns the placeholder-processed greetings list `[first_message, *alternate_greetings]` and a `_current_greeting_index`; picking an option re-seeds via `pane.seed_greeting(chosen)`, which sets `pane._greeting` so the existing Reset (`_render_seed_lines` from `_greeting`) returns to the chosen greeting with no new Reset logic.

**Tech Stack:** Python 3.11+, Textual, pytest.

## Global Constraints

- The selector lives in the **preview pane** only (not the card view / Console). It is shown **only when `len(greetings) > 1`** (≥1 alternate); a character with no alternates shows no selector (unchanged behavior).
- `alternate_greetings` is a top-level `list[str]` on the record, available reliably only from the **async `handle_character_loaded`** path (and synchronously on a cached re-selection). Do not attempt to populate synchronously on a cold first selection.
- Re-seed is guarded by the controller's `_current_greeting_index` (re-selecting the current index is a no-op) — this absorbs the **asynchronous `Select.Changed`** fired by the programmatic `set_options`/`value=0`, so no suppression flag is used.
- Reset uses the existing `_greeting` re-render — do NOT add greeting logic to the Reset path. Picking a greeting must go through `seed_greeting` (which sets `_greeting`).
- The selector is cleared (`set_greetings([])` → row hidden) whenever the preview loses its character, i.e. `PersonasPreviewController.reset(...)` is called with `seeded_for is None` (all persona/mode-switch/delete blank paths).
- Placeholder processing (`replace_placeholders(g, name, "User")`) is applied uniformly to every greeting, matching the existing `first_message` handling.

---

### Task 1: Alternate greeting selector in the preview

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` (new `PreviewGreetingSelected`)
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` (import `Select` + `PreviewGreetingSelected`; compose row ~:123; `set_greetings`; `_greeting_option_label`; `@on(Select.Changed)`)
- Modify: `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` (`__init__` :41; `reset` :63; `reset_for_character` :94-100; `handle_character_loaded` :244-261; new `_load_greetings` + `handle_greeting_selected`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (import `PreviewGreetingSelected`; `@on` handler near :3592-3614)
- Test: `Tests/UI/test_personas_preview.py`, `Tests/UI/test_personas_workbench.py`

**Interfaces:**
- Produces: `PreviewGreetingSelected(index: int)`; `PersonasPreviewPane.set_greetings(greetings: list[str])`; `PersonasPreviewController.handle_greeting_selected(index: int)`.

- [ ] **Step 1: Write the failing tests**

**Pane tests** (`Tests/UI/test_personas_preview.py`, reuse the `PreviewApp` harness; import `Select` from `textual.widgets` and `PreviewGreetingSelected` from `tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages`):
```python
async def test_greeting_selector_hidden_without_alternates():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_greetings(["Only greeting."])
        await pilot.pause()
        assert app.query_one("#personas-preview-greeting-row").display is False


async def test_greeting_selector_shown_with_alternates():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_greetings(["Primary.", "Alt one.", "Alt two."])
        await pilot.pause()
        assert app.query_one("#personas-preview-greeting-row").display is True
        from textual.widgets import Select
        select = app.query_one("#personas-preview-greeting-select", Select)
        assert len(list(select._options)) == 3  # 3 greetings


async def test_choosing_greeting_posts_message():
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        PreviewGreetingSelected,
    )
    posted = []
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)

        def _capture(message):
            if isinstance(message, PreviewGreetingSelected):
                posted.append(message.index)
        # drive the pane handler directly with a Select.Changed for index 1
        from textual.widgets import Select
        pane.set_greetings(["Primary.", "Alt one."])
        await pilot.pause()
        select = app.query_one("#personas-preview-greeting-select", Select)
        select.value = 1
        await pilot.pause()
        # the pane re-posts PreviewGreetingSelected to its screen; assert via
        # a spy on post_message OR assert the Changed handler ran (adapt to the
        # harness — simplest: check the pane's handler posts by monkeypatching
        # pane.post_message to record PreviewGreetingSelected indices).
```
> Implementer note for Step 1: `Select._options` is private; if it is not stable in this Textual version, assert option count another way (e.g. select the value and confirm it takes, or query the rendered options). For `test_choosing_greeting_posts_message`, the robust approach is to monkeypatch `pane.post_message` to capture `PreviewGreetingSelected` instances, then set `select.value = 1` and assert index 1 was posted (the programmatic `value=0` from `set_greetings` may also post index 0 — assert `1 in captured`, not exact equality).

**Integration tests** (`Tests/UI/test_personas_workbench.py`): add an `alternate_greetings` entry to the `CHARACTERS` fixture (`:43-57`) — give character id 1 ("Detective Sam") `"alternate_greetings": ["An alternate opener.", "A third opener."]`. Then:
```python
    async def test_alternate_greeting_selector_seeds_and_reset_returns_to_choice(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            assert screen.query_one("#personas-preview-greeting-row").display is True
            # choose alternate index 1
            await screen.preview.handle_greeting_selected(1)
            await pilot.pause()
            assert "An alternate opener." in pane.transcript_text()
            # send a turn, then Reset -> returns to the CHOSEN alternate, not primary
            pane.append_user("hi")
            pane.append_reply("hello")
            await pilot.pause()
            await pane.reset()
            await pilot.pause()
            assert "An alternate opener." in pane.transcript_text()
            assert "hi" not in pane.transcript_text()
```
And a no-alternates case (mirror an existing character without `alternate_greetings`) asserting `#personas-preview-greeting-row` `display is False`.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python -m pytest Tests/UI/test_personas_preview.py -k greeting_selector Tests/UI/test_personas_workbench.py -k alternate_greeting -q`
Expected: FAIL — `#personas-preview-greeting-row`/`set_greetings`/`handle_greeting_selected` do not exist.

- [ ] **Step 3: Add the `PreviewGreetingSelected` message**

In `personas_pane_messages.py`, alongside the other preview messages:
```python
class PreviewGreetingSelected(Message):
    """The user picked a greeting (index into the greetings list) to seed from."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
```

- [ ] **Step 4: Pane — Select import, compose row, `set_greetings`, handler**

`personas_preview_pane.py`:
- Import: `from textual.widgets import Button, Input, Select, Static` (add `Select`); add `PreviewGreetingSelected` to the `.personas_pane_messages` import block.
- In `compose()`, right after the status Static (`:123`, before the `Input` at `:124`):
```python
            yield Static("", id="personas-preview-status")
            with Horizontal(id="personas-preview-greeting-row", classes="ds-toolbar"):
                yield Static("Greeting:", classes="personas-preview-greeting-label")
                yield Select(
                    [],
                    id="personas-preview-greeting-select",
                    classes="form-select",
                    allow_blank=False,
                    prompt="Greeting",
                )
            yield Input(placeholder="Test message...", id="personas-preview-input")
```
- Hide the row initially — in `__init__` after mount is not available, so set it in `on_mount` (the pane already has one; if not, do it at the top of `set_greetings` default). Simplest: in `compose` the row renders; add `self.query_one("#personas-preview-greeting-row").display = False` to the pane's existing `on_mount`, or rely on `set_greetings([])` being called on first clear. To be safe, set `display = False` in the compose by constructing the `Horizontal` and setting `.display` is not possible inline; instead add to `on_mount`:
```python
    def on_mount(self) -> None:
        self.query_one("#personas-preview-greeting-row").display = False
```
(If `on_mount` already exists in the pane, add this line to it.)
- Add the methods (near `set_speakers`/`reset_speakers`):
```python
    def set_greetings(self, greetings: list[str]) -> None:
        """Populate the greeting selector; show it only when alternates exist.

        Args:
            greetings: Processed greetings, ``greetings[0]`` the primary
                ``first_message`` and the rest alternates.
        """
        row = self.query_one("#personas-preview-greeting-row")
        if len(greetings) > 1:
            select = self.query_one("#personas-preview-greeting-select", Select)
            select.set_options(
                [(self._greeting_option_label(i, g), i) for i, g in enumerate(greetings)]
            )
            select.value = 0
            row.display = True
        else:
            row.display = False

    def _greeting_option_label(self, index: int, text: str) -> str:
        """Dropdown label: 'Greeting N (default): <~40-char preview>'."""
        preview = " ".join(str(text).split())
        if len(preview) > 40:
            preview = preview[:39] + "…"
        tag = " (default)" if index == 0 else ""
        base = f"Greeting {index + 1}{tag}"
        return f"{base}: {preview}" if preview else base

    @on(Select.Changed, "#personas-preview-greeting-select")
    def _handle_greeting_selected(self, event: Select.Changed) -> None:
        if isinstance(event.value, int):
            self.post_message(PreviewGreetingSelected(event.value))
```

- [ ] **Step 5: Controller — greetings list, index, re-seed, clear**

`personas_preview_controller.py`:
- `__init__` (after `self._readout_nav_provider`, `:55`):
```python
        self._greetings: list[str] = []
        self._current_greeting_index: int = 0
```
- `reset` (`:63-78`) — clear the selector on the no-character (blank) path. After `self.seeded_for = seeded_for` (`:77`), before `refresh_provider_readout`:
```python
        if seeded_for is None:
            self._greetings = []
            try:
                self.screen.query_one(PersonasPreviewPane).set_greetings([])
            except QueryError:
                pass
```
- New helper (place above `reset_for_character`):
```python
    def _load_greetings(self, record: dict[str, Any], name: str) -> str:
        """Store the processed greeting list, populate the selector, return the primary."""
        raw = [str(record.get("first_message") or "")]
        raw += [
            str(g)
            for g in (record.get("alternate_greetings") or [])
            if isinstance(g, str)
        ]
        self._greetings = [replace_placeholders(g, name, "User") for g in raw]
        self._current_greeting_index = 0
        try:
            self.screen.query_one(PersonasPreviewPane).set_greetings(self._greetings)
        except QueryError:
            pass
        return self._greetings[0] if self._greetings else ""
```
- `reset_for_character` (`:97-100`): replace the inline greeting build with:
```python
        greeting = self._load_greetings(record, character_name)
        await self.reset(greeting, seeded_for=character_id)
```
- `handle_character_loaded` (`:246-248`): replace
```python
        greeting = replace_placeholders(
            str(record.get("first_message") or ""), name, "User"
        )
```
  with `greeting = self._load_greetings(record, name)`.
- New selection handler (place after `handle_reset`):
```python
    async def handle_greeting_selected(self, index: int) -> None:
        """Re-seed the preview from the chosen greeting (AC#1); Reset then returns to it (AC#2)."""
        if index == self._current_greeting_index or not (
            0 <= index < len(self._greetings)
        ):
            return
        self._current_greeting_index = index
        self.invalidate()
        try:
            await self.screen.query_one(PersonasPreviewPane).seed_greeting(
                self._greetings[index]
            )
        except QueryError:
            pass
```

- [ ] **Step 6: Screen wiring**

`personas_screen.py`:
- Add `PreviewGreetingSelected` to the import from `personas_pane_messages`.
- Add the handler alongside the other preview `@on` handlers (`:3592-3614`):
```python
    @on(PreviewGreetingSelected)
    async def _handle_preview_greeting_selected(
        self, message: PreviewGreetingSelected
    ) -> None:
        message.stop()
        await self.preview.handle_greeting_selected(message.index)
```

- [ ] **Step 7: Run the tests to confirm they pass**

Run: `python -m pytest Tests/UI/test_personas_preview.py -k greeting Tests/UI/test_personas_workbench.py -k alternate_greeting -q`
Expected: PASS.

- [ ] **Step 8: Regression**

Run:
```bash
python -m pytest Tests/UI/test_personas_preview.py Tests/UI/test_personas_preview_restore.py -q
python -m pytest Tests/UI/test_personas_workbench.py -q
```
Expected: PASS. The added `alternate_greetings` fixture field must not break existing character tests (they ignore it). `test_character_book_errors_render_in_editor_footer` is a known pre-existing test-order isolation flake (passes alone) — ignore it.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py \
        tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py \
        tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py \
        tldw_chatbook/UI/Screens/personas_screen.py \
        Tests/UI/test_personas_preview.py Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): alternate greeting selector in the preview (task-438)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 (pick which greeting seeds) → Steps 4-6 + `test_greeting_selector_shown_with_alternates`/`test_choosing_greeting_posts_message`/the integration seed assertion. AC#2 (Reset returns to chosen) → `handle_greeting_selected` → `seed_greeting` sets `_greeting` → existing Reset re-render; pinned by `test_alternate_greeting_selector_seeds_and_reset_returns_to_choice`. Both covered.
- **Placeholder scan:** the only implementer judgement is the Textual/`Select` test-introspection API (`_options`/capturing `post_message`), flagged with a robust fallback; all shipped code is concrete.
- **Type/name consistency:** `set_greetings`, `_greeting_option_label`, `PreviewGreetingSelected(index)`, `handle_greeting_selected`, `_load_greetings`, `_greetings`, `_current_greeting_index` used identically across pane/controller/screen/tests. `_greeting` (singular, the seeded text for Reset) is unchanged from TASK-437/434.
- **Idempotence:** the programmatic `set_options`/`value=0` fires `Select.Changed(0)` → pane posts `PreviewGreetingSelected(0)` → `handle_greeting_selected(0)` no-ops because `0 == _current_greeting_index` — no double-seed.
