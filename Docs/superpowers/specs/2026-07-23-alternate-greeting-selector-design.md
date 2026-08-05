# TASK-438 — Alternate greeting selector in the Roleplay preview

- **Date:** 2026-07-23
- **Task:** TASK-438 (RP/character-card UX review). The preview always seeds the primary `first_message`; alternate greetings are unreachable.
- **Branch base:** origin/dev (tip `c700081db`, includes TASK-437 preview speaker labels).

## Problem

A character card can carry `alternate_greetings` (the card view even shows "Alternate greetings: N"), but the Roleplay preview always seeds the primary `first_message` and there is no way to start from an alternate anywhere in the app.

- **AC#1:** when a character has alternate greetings, the preview offers a way to pick which greeting seeds the conversation.
- **AC#2:** Reset returns to the chosen greeting, not silently to the primary one.

## Key mechanics (verified)

- **`alternate_greetings` is a top-level `list[str]`** on the app record dict, alongside `first_message` (`Character_Chat_Lib.py:1310/1315`, placeholder-processed at load `:736-745`; DB round-trips it as JSON, `ChaChaNotes_DB.py:205/4217`).
- **The preview seeds only `first_message`** — `reset_for_character` (`personas_preview_controller.py:97-100`) and `handle_character_loaded` (`:246-261`) both `replace_placeholders(record.get("first_message"), name, "User")` → `pane.seed_greeting(...)`. `alternate_greetings` is read nowhere in the seeding path.
- **Reset is already a dumb re-render from `pane._greeting`** — `PersonasPreviewPane.reset()` → `_render_seed_lines()` rebuilds from `self._greeting` (`personas_preview_pane.py:334-357`); `seed_greeting`/`refresh_greeting_seed` set `self._greeting` (`:158-175`). **So AC#2 reduces to: write the chosen greeting into `_greeting` via `seed_greeting` — Reset needs no new logic.**
- **Availability is async.** On first selection the screen's record is `None` (`_full_character_record` returns None unless the handler already cached the full card, `personas_screen.py:4032-4042`); the greeting — and thus `alternate_greetings` — is only reliably present once `handle_character_loaded` runs with `card_data`. So the selector must populate from that async path (and synchronously on cached re-selection).
- **Toolbar + message pattern.** Preview controls compose in `personas_preview_pane.py:107-147` (`Input`, then a `Horizontal(classes="ds-toolbar")` of buttons). Pane→screen uses `post_message(PreviewXRequested())` → `@on(...)` in `personas_screen.py:3592-3614` → `self.preview.<method>()`. Existing messages in `personas_pane_messages.py:119-136`.
- **`Select` widget (Textual, verified):** `Select(options=[(label, value)], *, allow_blank, value=, id=, classes=)`; repopulate via `select.set_options([(label, value)])` (mirrors `Widgets/Evals/ab_test_dialog.py:446/458`); emits `Select.Changed(select, value)`; codebase style `classes="form-select"` + `@on(Select.Changed)`. **Setting `select.value`/`set_options` fires `Changed` asynchronously**, so a suppression flag is unreliable — the re-seed is guarded by a `_current_greeting_index` on the controller instead (idempotent).

## Design

### 1. Greeting row + `Select` in the preview pane

In `compose()` (just above the `Input` at `:124`, so it sits between the transcript and the toolbar), add a hidden row:
```python
with Horizontal(id="personas-preview-greeting-row", classes="ds-toolbar"):
    yield Static("Greeting:", classes="personas-preview-greeting-label")
    yield Select([], id="personas-preview-greeting-select",
                 classes="form-select", allow_blank=False,
                 prompt="Greeting")
# row starts hidden; shown by set_greetings() only when there are alternates
```
Set `self.query_one("#personas-preview-greeting-row").display = False` at mount (or via `set_greetings`).

New pane API:
```python
def set_greetings(self, greetings: list[str]) -> None:
    """Populate the greeting selector; show it only when there are alternates.

    greetings[0] is the primary first_message; the rest are alternates.
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
    return f"Greeting {index + 1}{tag}: {preview}" if preview else f"Greeting {index + 1}{tag}"
```

Handle user selection (mirrors the existing `@on` handlers), posting a new message:
```python
@on(Select.Changed, "#personas-preview-greeting-select")
def _handle_greeting_selected(self, event: Select.Changed) -> None:
    if isinstance(event.value, int):
        self.post_message(PreviewGreetingSelected(event.value))
```
(The pane posts on **every** `Changed`, including the programmatic one fired by `set_options`/`value=0`; the controller's index guard makes a re-select of the current index a no-op, so no fragile suppression flag is needed.)

### 2. New message

`personas_pane_messages.py` (alongside the other preview messages):
```python
class PreviewGreetingSelected(Message):
    """The user picked a greeting (index into the greetings list) to seed from."""
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
```

### 3. Controller: own the greetings list + index, re-seed on change

`PersonasPreviewController` gains `self._greetings: list[str] = []` and `self._current_greeting_index: int = 0`.

Add a helper that builds the placeholder-processed list and pushes it to the pane, used at both seed points:
```python
def _load_greetings(self, record: dict, name: str) -> str:
    """Store the processed greeting list, populate the selector, return the primary."""
    raw = [str(record.get("first_message") or "")]
    raw += [str(g) for g in (record.get("alternate_greetings") or []) if isinstance(g, str)]
    self._greetings = [replace_placeholders(g, name, "User") for g in raw]
    self._current_greeting_index = 0
    self.screen.query_one(PersonasPreviewPane).set_greetings(self._greetings)
    return self._greetings[0] if self._greetings else ""
```
- `reset_for_character` (`:97-100`): when `record` is present, `greeting = self._load_greetings(record, character_name)`; when `record is None`, clear (`self._greetings = []`; `pane.set_greetings([])`) then `await self.reset("")`.
- `handle_character_loaded` (`:246`): `greeting = self._load_greetings(record, name)` in place of the inline `replace_placeholders(first_message...)`. The rest (`refresh_greeting_seed` preserve branch / `seed_greeting`) is unchanged and now uses the primary from the list.

Handle the selection (screen forwards it):
```python
async def handle_greeting_selected(self, index: int) -> None:
    """Re-seed the preview from the chosen greeting (AC#1); Reset then returns to it (AC#2)."""
    if index == self._current_greeting_index or not (0 <= index < len(self._greetings)):
        return
    self._current_greeting_index = index
    self.invalidate()
    await self.screen.query_one(PersonasPreviewPane).seed_greeting(self._greetings[index])
```
`seed_greeting` sets `pane._greeting = chosen`, so **Reset (`_render_seed_lines` from `_greeting`) returns to the chosen greeting — AC#2 with no new Reset logic.** `invalidate()` drops the prior ephemeral turns (the user is choosing a fresh starting greeting).

### 4. Screen wiring

`personas_screen.py`, alongside the other preview `@on` handlers (`:3592-3614`):
```python
@on(PreviewGreetingSelected)
async def _handle_preview_greeting_selected(self, message: PreviewGreetingSelected) -> None:
    message.stop()
    await self.preview.handle_greeting_selected(message.index)
```

### Clearing on non-character context

The selector must hide when the preview loses its character (persona/dictionary/lore selection, mode switch). Those paths all call `self.preview.reset("")` — i.e. `PersonasPreviewController.reset(text, *, seeded_for=None)` with the default `seeded_for=None`, whereas a real character seed passes `seeded_for=<id>` (`reset_for_character:100`). So `reset` clears the selector exactly when `seeded_for is None`:
```python
async def reset(self, text="", *, seeded_for=None):
    ...
    if seeded_for is None:
        self._greetings = []
        try:
            self.screen.query_one(PersonasPreviewPane).set_greetings([])
        except QueryError:
            pass
    ...
```
This mirrors where TASK-437's `reset_speakers()` hangs off the same blank/mode-switch points.

## Testing

- **Pane (`test_personas_preview.py`):** `set_greetings([primary])` hides the row; `set_greetings([primary, alt])` shows it and populates two options; choosing an option posts `PreviewGreetingSelected(index)`.
- **Controller/integration (`test_personas_workbench.py`):** a new fixture character with `alternate_greetings` (the `CHARACTERS` fixture at `:43-57` needs an entry). After load, the selector is visible with N+1 options; selecting alternate index 1 re-seeds the transcript to that greeting; **pressing Reset returns to the chosen greeting, not the primary** (extends the existing `test_reset_after_character_reload_uses_updated_greeting` pattern at `:2980`); a character with no alternates shows no selector.
- **Idempotence:** the programmatic `set_options`/`value=0` on load does not re-seed or duplicate the greeting (the `_current_greeting_index` guard) — assert the transcript has exactly one greeting line after load.
- **Restore (`test_personas_preview_restore.py`):** unaffected — `restore_conversation` still seeds the saved `_greeting`; the chosen greeting survives navigation via TASK-434's `_greeting` capture.

## Risks / mitigations

- **Programmatic `Changed` re-entrancy:** guarded by the controller `_current_greeting_index` (a re-select of the current index is a no-op), robust to however many `Changed` events `set_options`/`value=` fire.
- **Async availability:** the selector populates from `handle_character_loaded` (and cached re-selection), exactly like the greeting itself — no attempt to populate synchronously on a cold first selection.
- **Placeholder double-processing:** `replace_placeholders` is idempotent on already-substituted text; applying it uniformly to every greeting matches the existing `first_message` handling.
- **Speaker labels (437):** `seed_greeting` renders with the current `_character_label`; choosing a greeting re-renders consistently.
- **Stale selector on non-character context:** cleared via `set_greetings([])` on the blank/mode-switch paths.

## Non-goals

- A greeting picker in the **card view** or in **Console/Start-Chat** (the AC scopes this to the preview; Console handoff seeds from whatever the preview shows).
- Persisting the *selected index* across navigation as a distinct field (the chosen greeting **text** already survives via `_greeting`; the dropdown re-defaults to index 0 on reload while the transcript shows the chosen text — a cosmetic edge, possible follow-up).
- Editing/adding alternate greetings (read-only selection only).
