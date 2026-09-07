# TASK-434 Personas workbench selection+preview persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Returning to the Personas screen within a session restores the previously selected item, mode, and center view (AC#1), and the preview conversation (greeting + turns) survives the round-trip (AC#2).

**Architecture:** The app stores `screen.save_state()` on navigate-away and calls `restore_state()` on a fresh screen when returning. `PersonasScreen` will override both to persist its `PersonasWorkbenchState` dataclass and the preview conversation, and re-apply the selection at the end of `on_mount`.

**Tech Stack:** Python 3.11+, Textual, pytest (+ pytest-asyncio).

## Global Constraints

- The preview is split: `PersonasPreviewController.history` holds only sent turns; the **greeting** is in `PersonasPreviewPane._greeting` (NOT in history). Capture and restore **both**.
- In `restore_conversation`, set `self.seeded_for` **before the first `await`** — `invalidate()` does NOT cancel the character-load worker, whose `handle_character_loaded` must hit the `:133` guard (`refresh_greeting_seed`, no erase).
- `restore_state` runs **before** mount; rebuild `PersonasWorkbenchState` filtered to known dataclass fields (version-drift tolerant), and only stash a pending restore when there is a real selection.
- Non-goals: restoring an unsaved editor, cross-restart persistence, changing the screen-caching model.
- Files: `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py`, `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py`, `tldw_chatbook/UI/Screens/personas_screen.py`, and tests.

---

### Task 1: Preview capture + restore primitives

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` (add a `greeting_text` property)
- Modify: `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` (add `restore_conversation`)
- Test: `Tests/UI/test_personas_preview.py` (pane property, via the existing pane harness) + `Tests/UI/test_personas_preview_restore.py` (new — controller `restore_conversation` with a mock screen/pane)

**Interfaces:**
- Produces: `PersonasPreviewPane.greeting_text -> str` (read-only property returning `self._greeting`).
- Produces: `PersonasPreviewController.restore_conversation(self, *, greeting: str, history: list[dict], seeded_for) -> None` (async).

- [ ] **Step 1: Write the failing pane-property test**

```python
# Tests/UI/test_personas_preview.py  (mirror the existing pane-harness tests, e.g. test_seed_append_reset_roundtrip)
@pytest.mark.asyncio
async def test_greeting_text_property_returns_seeded_greeting():
    app = PreviewApp()  # the pane harness this file already defines (composes PersonasPreviewPane)
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Hello, traveller.")
        assert pane.greeting_text == "Hello, traveller."
        pane.refresh_greeting_seed("Updated greeting.")
        assert pane.greeting_text == "Updated greeting."
```
(Use the exact harness/import the other tests in that file use — copy their setup.)

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest Tests/UI/test_personas_preview.py::test_greeting_text_property_returns_seeded_greeting -q`
Expected: FAIL (`AttributeError: 'PersonasPreviewPane' object has no attribute 'greeting_text'`).

- [ ] **Step 3: Add the `greeting_text` property** (`personas_preview_pane.py`, near the public API, after `refresh_greeting_seed`)

```python
    @property
    def greeting_text(self) -> str:
        """The greeting a Reset restores (transcript line 0), for state capture."""
        return self._greeting
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `python -m pytest Tests/UI/test_personas_preview.py::test_greeting_text_property_returns_seeded_greeting -q`
Expected: PASS.

- [ ] **Step 5: Write the failing `restore_conversation` test** (new file, mock screen/pane)

```python
# Tests/UI/test_personas_preview_restore.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
    PersonasPreviewController,
)


def _controller_with_mock_pane():
    ctrl = PersonasPreviewController.__new__(PersonasPreviewController)
    ctrl.history = [{"role": "assistant", "content": "old"}]
    ctrl.seeded_for = None
    ctrl.generation = 0
    ctrl.gateway = None
    events = []
    pane = MagicMock()
    pane.append_user = lambda t: events.append(("user", t))
    pane.append_reply = lambda t: events.append(("reply", t))

    async def _seed(text):
        # Ordering guard: seeded_for MUST already be set when the first await runs.
        events.append(("seed", text, ctrl.seeded_for))
    pane.seed_greeting = AsyncMock(side_effect=_seed)

    screen = MagicMock()
    screen.query_one.return_value = pane
    screen.workers.cancel_group = MagicMock()
    ctrl.screen = screen
    return ctrl, events


@pytest.mark.asyncio
async def test_restore_conversation_seeds_greeting_then_turns_and_sets_seeded_for_first():
    ctrl, events = _controller_with_mock_pane()
    history = [
        {"role": "assistant", "content": "Greetings."},   # note: greeting is NOT in history
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    await ctrl.restore_conversation(
        greeting="Greetings.", history=history, seeded_for="7"
    )
    # seeded_for was already "7" at the first await (the seed call)
    seed_events = [e for e in events if e[0] == "seed"]
    assert seed_events and seed_events[0][2] == "7"
    # greeting seeded, then each history turn rendered in order
    assert events[0] == ("seed", "Greetings.", "7")
    assert ("user", "hi") in events and ("reply", "hello") in events
    # controller state updated
    assert ctrl.seeded_for == "7"
    assert ctrl.history == history
```

- [ ] **Step 6: Run it to confirm it fails**

Run: `python -m pytest Tests/UI/test_personas_preview_restore.py -q`
Expected: FAIL (`AttributeError: ... has no attribute 'restore_conversation'`).

- [ ] **Step 7: Add `restore_conversation`** (`personas_preview_controller.py`, after `reset_for_character`)

```python
    async def restore_conversation(
        self, *, greeting: str, history: list[dict], seeded_for
    ) -> None:
        """Rebuild the preview (greeting + turns) from saved state (task-434).

        Sets ``seeded_for`` before the first ``await``: ``invalidate`` cancels
        only the preview worker group, so the character-load worker's
        ``handle_character_loaded`` may still fire -- and must hit the
        seeded-for guard (``:133``) rather than erase the restored turns.
        """
        self.invalidate()
        self.seeded_for = str(seeded_for) if seeded_for else None
        try:
            pane = self.screen.query_one(PersonasPreviewPane)
        except QueryError:
            return
        await pane.seed_greeting(greeting)
        for message in history:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                pane.append_user(content)
            elif role == "assistant":
                pane.append_reply(content)
        self.history = [dict(m) for m in history]
```
(`PersonasPreviewPane` and `QueryError` are already imported in this module — confirm; if not, add them.)

- [ ] **Step 8: Run it to confirm it passes**

Run: `python -m pytest Tests/UI/test_personas_preview_restore.py Tests/UI/test_personas_preview.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py Tests/UI/test_personas_preview.py Tests/UI/test_personas_preview_restore.py
git commit -m "feat(personas): preview greeting_text + restore_conversation primitives (task-434)"
```

---

### Task 2: PersonasScreen state persistence + deferred re-selection

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (add `import dataclasses`; override `save_state`/`restore_state`; add `_apply_pending_restore`; call it at the end of `on_mount`; add `restore_preview` param to `_select_character`)
- Test: `Tests/UI/test_personas_workbench_state.py` (new — full-screen round-trip; mirror the harness in `Tests/UI/test_personas_dictionaries.py`: a harness `App` that `push_screen(PersonasScreen(self))` + a `_mounted(pilot)` helper)

**Interfaces:**
- Consumes: `PersonasPreviewController.restore_conversation` + `PersonasPreviewPane.greeting_text` (Task 1); `PersonasWorkbenchState` (dataclass); the existing `_select_character/_select_profile/_select_dictionary/_select_lore_entry`.

- [ ] **Step 1: Write the failing AC#1 round-trip test**

```python
# Tests/UI/test_personas_workbench_state.py
# Harness: copy the PersonasScreen-mounting app + _mounted(pilot) from Tests/UI/test_personas_dictionaries.py.
@pytest.mark.asyncio
async def test_save_restore_preserves_character_selection_and_center():
    app = _PersonasHarness()  # push_screen(PersonasScreen(self))
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        # select a seeded character (use the same seeding the dictionaries/workbench tests use)
        await screen._select_character("char-1", "Elara")
        await pilot.pause()
        assert screen.state.selected_entity_id == "char-1"

        saved = screen.save_state()
        assert saved["personas_workbench"]["selected_entity_id"] == "char-1"

    # Fresh screen restores from saved state, then mounts
    app2 = _PersonasHarness()
    async with app2.run_test() as pilot2:
        screen2 = app2.query_one(PersonasScreen)  # or however the harness exposes it
        screen2.restore_state(saved)
        await pilot2.pause()
        # after mount + deferred restore
        assert screen2.state.selected_entity_id == "char-1"
        assert screen2.state.selected_entity_kind == "character"
        # center shows the character card view, not blank
        assert screen2.query_one("#ccp-character-card-view").display is True
```
Adjust the harness/seeding and the "fresh screen" mounting to whatever the dictionaries/workbench harness supports (it may require creating the screen, calling `restore_state`, then pushing it — mirror how the app does it: `restore_state` before `switch_screen`). If the harness can't cleanly re-create a second screen, drive `restore_state` on a freshly constructed `PersonasScreen(app)` and `push_screen` it, then assert after `pilot.pause()`.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest Tests/UI/test_personas_workbench_state.py::test_save_restore_preserves_character_selection_and_center -q`
Expected: FAIL — `save_state` returns the base state_data without `personas_workbench`, and the restored screen shows "Selected: none"/blank center.

- [ ] **Step 3: Add `import dataclasses`** at the top of `personas_screen.py` (with the other stdlib imports).

- [ ] **Step 4: Override `save_state` / `restore_state`** (methods on `PersonasScreen`)

```python
    def save_state(self) -> dict:
        state = dict(super().save_state() or {})
        state["personas_workbench"] = dataclasses.asdict(self.state)
        preview = getattr(self, "preview", None)
        if preview is not None:
            greeting = ""
            try:
                greeting = self.query_one(PersonasPreviewPane).greeting_text
            except QueryError:
                pass
            state["personas_preview"] = {
                "greeting": greeting,
                "history": [dict(m) for m in preview.history],
                "seeded_for": preview.seeded_for,
            }
        return state

    def restore_state(self, state: dict) -> None:
        super().restore_state(state)
        if not isinstance(state, dict):
            self._pending_restore = None
            return
        wb = state.get("personas_workbench")
        if isinstance(wb, dict):
            names = {f.name for f in dataclasses.fields(PersonasWorkbenchState)}
            self.state = PersonasWorkbenchState(
                **{k: v for k, v in wb.items() if k in names}
            )
        self._pending_restore = (
            {
                "kind": self.state.selected_entity_kind,
                "id": self.state.selected_entity_id,
                "name": self.state.selected_entity_name,
                "preview": state.get("personas_preview"),
            }
            if self.state.selected_entity_id
            else None
        )
```
Initialize `self._pending_restore = None` in `__init__` (next to `self.state = PersonasWorkbenchState()`).

- [ ] **Step 5: Add `_apply_pending_restore` and call it at the end of `on_mount`**

In `on_mount`, after `self._sync_title_and_console_actions()`:
```python
        await self._apply_pending_restore()
```
New method:
```python
    async def _apply_pending_restore(self) -> None:
        """Re-apply a selection saved before a navigation round-trip (task-434)."""
        pending = getattr(self, "_pending_restore", None)
        self._pending_restore = None
        if not pending or not pending.get("id"):
            return
        kind = pending.get("kind")
        entity_id = str(pending["id"])
        name = str(pending.get("name") or "")
        try:
            if kind == "character":
                await self._select_character(
                    entity_id, name, restore_preview=pending.get("preview")
                )
            elif kind == "persona_profile":
                await self._select_profile(entity_id, name)
            elif kind == "dictionary":
                await self._select_dictionary(entity_id, name)
            elif kind == "lore":
                await self._select_lore_entry(entity_id, name)
        except Exception:
            # A stale/deleted entity must degrade to the blank center, not crash.
            logger.opt(exception=True).warning(
                "Personas: could not restore selection %s/%s; showing blank center.",
                kind,
                entity_id,
            )
            self._show_center(None)
```
(Use whatever logger the module already binds.)

- [ ] **Step 6: Add the `restore_preview` branch to `_select_character`**

Change the signature to `async def _select_character(self, entity_id, entity_name, *, restore_preview=None)` and replace the `reset_for_character` tail:
```python
        if restore_preview is not None:
            await self.preview.restore_conversation(
                greeting=str(restore_preview.get("greeting") or ""),
                history=list(restore_preview.get("history") or []),
                seeded_for=entity_id,
            )
        else:
            record = self._full_character_record(entity_id)
            await self.preview.reset_for_character(
                character_id=entity_id, character_name=entity_name, record=record
            )
```

- [ ] **Step 7: Run the AC#1 test to confirm it passes**

Run: `python -m pytest Tests/UI/test_personas_workbench_state.py::test_save_restore_preserves_character_selection_and_center -q`
Expected: PASS.

- [ ] **Step 8: Write + pass the AC#2 preview-survival test**

```python
@pytest.mark.asyncio
async def test_save_restore_preserves_preview_greeting_and_turns():
    app = _PersonasHarness()
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._select_character("char-1", "Elara")
        await pilot.pause()
        # simulate a preview conversation: greeting seeded + a couple of turns
        pane = screen.query_one(PersonasPreviewPane)
        await pane.seed_greeting("Greetings, traveller.")
        pane.append_user("hi"); screen.preview.history.append({"role": "user", "content": "hi"})
        pane.append_reply("well met"); screen.preview.history.append({"role": "assistant", "content": "well met"})
        saved = screen.save_state()

    # restore into a fresh screen
    screen2 = PersonasScreen(app)
    screen2.restore_state(saved)
    await app.push_screen(screen2)
    await pilot.pause()  # (use the harness's pilot; or a fresh run_test as in Step 1)
    pane2 = screen2.query_one(PersonasPreviewPane)
    assert pane2.greeting_text == "Greetings, traveller."
    text = pane2.transcript_text()
    assert "Greetings, traveller." in text and "hi" in text and "well met" in text
    assert screen2.preview.history == saved["personas_preview"]["history"]
```
Then add a test that drives `handle_character_loaded` after restore and asserts the turns survive (the `:133` guard). Adjust the second-screen mounting mechanics to the harness (mirror Step 1's approach).

Run: `python -m pytest Tests/UI/test_personas_workbench_state.py -q`
Expected: PASS.

- [ ] **Step 9: Regression + commit**

Run:
```bash
python -c "import tldw_chatbook.UI.Screens.personas_screen"
python -m pytest Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_preview.py Tests/UI/test_personas_preview_restore.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_lore.py -q
```
Expected: PASS (existing personas tests unaffected).

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench_state.py
git commit -m "feat(personas): persist workbench selection + preview across navigation (task-434)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 → Task 2 (save/restore `PersonasWorkbenchState` + deferred `_select_*` re-selection). AC#2 → Task 1 primitives + Task 2's `restore_preview` branch. Both covered.
- **Placeholder scan:** the only prose-only bits are "use the harness the existing tests use" (Task 1 pane harness; Task 2 `test_personas_dictionaries.py` harness) and the exact character-seeding id — these are verification-against-real-fixtures, with the source files named. No TBD logic.
- **Type/name consistency:** `restore_conversation(greeting, history, seeded_for)` and `greeting_text` used identically in Task 1 (definition) and Task 2 (call sites); `_pending_restore` dict shape identical in `restore_state` and `_apply_pending_restore`; `personas_workbench`/`personas_preview` keys identical in `save_state`/`restore_state`.
- **Ordering:** `seeded_for` set before the first `await` in `restore_conversation` (Global Constraint) — pinned by Task 1 Step 5's test.
