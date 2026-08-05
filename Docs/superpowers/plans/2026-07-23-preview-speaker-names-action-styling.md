# TASK-437 Preview speaker names + action styling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Roleplay preview transcript labels replies with the character's real name (user stays "you" until TASK-442) and renders single-asterisk `*action*` spans as italics instead of literal asterisks.

**Architecture:** Add speaker-label fields + a `set_speakers` setter to `PersonasPreviewPane`; the five hardcoded `"character:"`/`"you:"` label sites read those fields; the screen/controller call `set_speakers(character=<name>)` when a character is seeded. For styling, render each line as a Rich `Text` built by `Text.from_markup(escape(line) → *italic*)` — keeping `self._lines` plain (so `transcript_text()` stays raw) and keeping the `renderable` shim returning plain text.

**Tech Stack:** Python 3.11+, Textual, Rich, pytest.

## Global Constraints

- Speaker labels are **display-only** (verified: `transcript_text()` is only boolean-guarded and staged as a handoff text-blob; the LLM payload uses `self.history` role-dicts). Do not change `self._lines`/`transcript_text()` plumbing or the `role_class` selectors (`personas-preview-line-you`/`-character`).
- User label stays `"you"` (TASK-442 owns the persona/user name). Empty/unknown character name ⇒ label stays `"character"`.
- The styling helper MUST NOT be named `_render_markup` (that shadows a Textual `Widget` instance attribute → `TypeError`). Use `_styled_line`.
- Style via a Rich **`Text` object** (`Text.from_markup`), NOT a markup string: the repo's `Static.renderable` shim (`tldw_chatbook/__init__.py:30`) returns `content`, so a markup string would surface raw `[i]` tags / escape backslashes in `str(renderable)` and break the literal-rendering regression. `escape()` first, then only balanced `[i]…[/i]` — `MarkupError` is impossible.
- Only single-asterisk italic (AC#2) — no bold/lists/headings, no Markdown widget (it over-interprets chat text).

---

### Task 1: Real speaker names + `*action*` italic in the preview transcript

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` (imports; `__init__` ~:84; new `set_speakers` + `_styled_line`; five label sites :173/:177/:187/:205/:264; four render sites :188-194/:208/:269-273/:282-284)
- Modify: `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` (`handle_character_loaded` ~:245-256)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_select_character` ~:1786)
- Test: `Tests/UI/test_personas_preview.py` (new name + styling tests; existing plain-label tests are unaffected)

**Interfaces:**
- Produces: `PersonasPreviewPane.set_speakers(*, character=None, user=None)`; `PersonasPreviewPane._styled_line(line: str) -> rich.text.Text`. After `set_speakers(character="X")`, reply/greeting lines label `"X: …"`; a `*a*` span renders italic with the asterisks consumed.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/UI/test_personas_preview.py` (reuses the file's `PreviewApp` harness, `_line_texts` at :41-42, and `pane` access). `Text` styling is asserted via the rendered line's span list.

```python
async def test_speaker_labels_use_character_name():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="Sherlock Holmes")
        pane.seed_greeting("Greetings.")
        await pilot.pause()
        pane.append_user("Hi")
        pane.append_reply("Elementary.")
        await pilot.pause()
        assert _line_texts(app) == [
            "Sherlock Holmes: Greetings.",
            "you: Hi",
            "Sherlock Holmes: Elementary.",
        ]
        assert pane.transcript_text() == (
            "Sherlock Holmes: Greetings.\nyou: Hi\nSherlock Holmes: Elementary."
        )


async def test_speaker_labels_default_without_name():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.append_user("Hi")
        pane.append_reply("Hello.")
        await pilot.pause()
        assert _line_texts(app) == ["you: Hi", "character: Hello."]


async def test_set_speakers_ignores_empty_name():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.set_speakers(character="")
        pane.append_reply("Hi.")
        await pilot.pause()
        assert _line_texts(app) == ["character: Hi."]


def test_styled_line_italicizes_action_and_escapes_markup():
    # Pure-helper unit test — no app needed for the transform assertions.
    app = PreviewApp()

    async def run():
        async with app.run_test() as pilot:
            pane = app.query_one(PersonasPreviewPane)
            waves = pane._styled_line("*waves*")
            assert str(waves) == "waves"
            assert any("italic" in str(span.style) for span in waves.spans)
            assert str(pane._styled_line("[/oops]")) == "[/oops]"
            assert str(pane._styled_line("you: 5 * 3")) == "you: 5 * 3"
    import asyncio
    asyncio.run(run())


async def test_action_span_renders_italic_not_literal_asterisks():
    app = PreviewApp()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasPreviewPane)
        pane.append_reply("*smiles warmly*")
        await pilot.pause()
        line = app.query(".personas-preview-line").last()
        assert "*" not in str(line.renderable)
        assert "smiles warmly" in str(line.renderable)
        assert any("italic" in str(s.style) for s in line.renderable.spans)
```
> Implementer note: `PersonasPreviewPane` is imported in this test file already (or import it from `tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane`). Adapt `.last()`/span access to the Textual/Rich versions in `.venv` if the API differs; the span check may also be written as `waves.spans and waves.spans[0].style`.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python -m pytest Tests/UI/test_personas_preview.py -k "speaker or styled or action_span or set_speakers" -q`
Expected: FAIL — `set_speakers`/`_styled_line` do not exist; labels are hardcoded; asterisks literal.

- [ ] **Step 3: Add imports + label fields + `set_speakers` + `_styled_line`**

In `personas_preview_pane.py`, add module imports (after the existing `from textual.widgets import ...`, :14):
```python
import re

from rich.markup import escape
from rich.text import Text
```
In `__init__` (after `self._partial_text: str = ""`, ~:94):
```python
        self._character_label = "character"
        self._user_label = "you"
```
Add the public setter (near `append_user`, ~:170):
```python
    def set_speakers(self, *, character: str | None = None, user: str | None = None) -> None:
        """Set transcript speaker labels; empty/None keeps the current label.

        Args:
            character: Display name for character/greeting lines (e.g. the card name).
            user: Display name for the user's lines (unused until TASK-442).
        """
        if character:
            self._character_label = character
        if user:
            self._user_label = user
```
Add the styling helper + pattern (in the "Internals" section, near `_append_line`, ~:277):
```python
    _ACTION_SPAN = re.compile(r"\*([^*\n]+)\*")

    def _styled_line(self, line: str) -> Text:
        """Render a transcript line: escape Rich markup, italicize *action* spans.

        Args:
            line: Plain transcript line (``"label: text"``).

        Returns:
            A Rich ``Text`` whose plain string equals the line with matched
            ``*...*`` asterisks removed, and with italic spans over those runs.
        """
        return Text.from_markup(self._ACTION_SPAN.sub(r"[i]\1[/i]", escape(line)))
```

- [ ] **Step 4: Read the label fields at the five label sites**

- `append_user` (:173): `f"you: {text}"` → `f"{self._user_label}: {text}"`
- `append_reply` (:177): `f"character: {text}"` → `f"{self._character_label}: {text}"`
- `begin_reply` (:187): `line = "character:"` → `line = f"{self._character_label}:"`
- `append_reply_chunk` (:205): `line = f"character: {self._partial_text}"` → `line = f"{self._character_label}: {self._partial_text}"`
- `_render_seed_lines` (:264): `line = f"character: {self._greeting}"` → `line = f"{self._character_label}: {self._greeting}"`

- [ ] **Step 5: Render each line via `_styled_line` at the four mount/update sites**

`self._lines` still stores the **plain** `line`; only the `Static` content becomes the styled `Text`. Update the `markup=False` comments to note the content is a pre-styled `Text`.

- `_append_line` (:282-284):
```python
        self.query_one("#personas-preview-transcript", VerticalScroll).mount(
            Static(self._styled_line(line),
                   classes=f"personas-preview-line {role_class}", markup=False)
        )
```
- `begin_reply` (:188-194): `Static(self._styled_line(line), classes="personas-preview-line personas-preview-line-character", markup=False)`
- `append_reply_chunk` (:208): `self._partial_widget.update(self._styled_line(line))`
- `_render_seed_lines` (:269-273): `Static(self._styled_line(line), classes="personas-preview-line personas-preview-line-character", markup=False)`

- [ ] **Step 6: Wire `set_speakers` at the two character-seeding points**

Screen `_select_character` (`personas_screen.py`), right after `self.state.select_entity(...)` (~:1786), before the reset/restore dispatch:
```python
        self.query_one(PersonasPreviewPane).set_speakers(character=entity_name)
```
Controller `handle_character_loaded` (`personas_preview_controller.py`), after `name = ...` (:245) and before the `if self.seeded_for ...` branch (:251):
```python
        pane.set_speakers(character=name)
```

- [ ] **Step 7: Run the new tests to confirm they pass**

Run: `python -m pytest Tests/UI/test_personas_preview.py -k "speaker or styled or action_span or set_speakers" -q`
Expected: PASS.

- [ ] **Step 8: Regression — preview + controller-restore + personas suites**

Run:
```bash
python -m pytest Tests/UI/test_personas_preview.py Tests/UI/test_personas_preview_restore.py -q
python -m pytest Tests/UI/test_personas_workbench.py -q
```
Expected: PASS. The existing preview tests seed no character name and use no asterisks, so their `_line_texts`/`transcript_text()` results are unchanged (plain lines round-trip through `_styled_line` identically; `[/oops]`/`[bold]unclosed` stay literal — `test_markup_like_transcript_content_renders_without_raising` still passes). If a controller-restore test asserts `set_speakers` interactions on the mock pane, add/adjust it to expect the call; do not weaken the new assertions.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py \
        tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py \
        tldw_chatbook/UI/Screens/personas_screen.py \
        Tests/UI/test_personas_preview.py
git commit -m "feat(roleplay): preview uses real speaker names + italic *action* text (task-437)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 (real character name; persona/user "once available") → Steps 3/4/6 + `test_speaker_labels_use_character_name`/`_default_without_name`/`test_set_speakers_ignores_empty_name`; user stays `"you"` (TASK-442). AC#2 (single-asterisk italic) → Steps 3/5 + `test_styled_line_*`/`test_action_span_renders_italic_not_literal_asterisks`. Both covered.
- **Placeholder scan:** the only judgement call is the Rich/Textual span-access API in the styling assertions, flagged for the implementer with a fallback form; all shipped code is concrete.
- **Consistency:** `_styled_line` (never `_render_markup`), `set_speakers`, `_character_label`/`_user_label`, `_ACTION_SPAN`, `Text.from_markup` used identically across the pane, and `set_speakers` is called with the same `character=` kwarg from both wiring sites. `self._lines`/`transcript_text()` stay plain throughout.
