# Roleplay P2d-1 — Lore matching-controls editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user edit a Lore entry's `selective`, `secondary_keys`, and `case_sensitive` matching fields from the `PersonasLoreDetailWidget` entry editor (they already work end-to-end in the schema, matcher, and manager — only the editor doesn't expose them).

**Architecture:** Pure UI surfacing. Task 1 adds three form controls to the detail widget and threads them through `entry_form_payload()` / `_fill_form_from_entry()`, plus a `_sync_secondary_keys_disabled()` visual hint. Task 2 threads the three fields through the screen's `_handle_lore_entry_add` (its explicit-kwarg call to `create_world_book_entry` silently drops anything not named — the exact P2c-priority bug); update already forwards `**payload`. Task 3 is one end-to-end integration test proving an editor-created selective entry gates matching in `WorldInfoProcessor`.

**Tech Stack:** Python ≥3.11, Textual, pytest + pytest-asyncio, SQLite (`CharactersRAGDB`).

## Global Constraints

- No schema migration — ChaChaNotes stays at `_CURRENT_SCHEMA_VERSION = 21`.
- No change to `WorldInfoProcessor` (matcher), `world_book_manager.py` (manager), or `ChaChaNotes_DB.py` (schema). `create_world_book_entry` already accepts `selective`/`secondary_keys`/`case_sensitive`; `update_world_book_entry(**kwargs)` already handles them; `get_world_book_entries` already returns them; `_entry_matches` already matches on them.
- `entry_form_payload()` must never raise (boolean switches + never-raising comma-split; empty → `[]`).
- `secondary_keys` is read/stored **regardless** of the Selective switch state (data fidelity — toggling Selective off must not erase typed secondary keys).
- No new `DataTable` column.
- No `rich.markup.escape()` here — the new `Static` labels are plain constant strings and the `Switch`/`Input` widgets carry no user markup.
- Implementers stage ONLY their task's files (`git add <explicit paths>`; never `git add -A`; never stage `.superpowers/`).
- **Test environment (run from the worktree root):**
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home \
  XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  The venv lives in the MAIN checkout, not the worktree; `import tldw_chatbook` resolves to the worktree source because the worktree root (cwd) is on `sys.path`.

## File Structure

- `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` — the detail widget: compose (`:110-143`), `on_mount` (`:145-153`), `entry_form_payload` (`:271-310`), `_fill_form_from_entry` (`:337-350`), event handlers (`:368-434`). All of Task 1's production code.
- `tldw_chatbook/UI/Screens/personas_screen.py` — `_handle_lore_entry_add` (the explicit-kwarg call to `create_world_book_entry`). Task 2's one-spot change.
- `Tests/UI/test_personas_lore.py` — all tests. Widget-only tests use the `_DetailHost` harness (`:21-30`); real-DB tests use `LorePersonasTestApp` (`:246-265`) + fixtures `lore_db` (`:215`), `seeded_lore_book` (`:222`), `stub_characters_lore` (`:237`), `mock_app_instance` (from `Tests/UI/conftest.py:97`), and helpers `_enter_lore` (`:273`) / `_select_first_lore` (`:282`).

---

### Task 1: Detail-widget matching controls

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` (compose `:114-124`, `on_mount` `:145-153`, `entry_form_payload` `:271-310`, `_fill_form_from_entry` `:337-350`, add a helper + a `Switch.Changed` handler near `:426`)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: nothing new. `Static`, `Switch`, `Input`, `Horizontal` are already imported (`:10-22`); `@on` from `textual` (`:8`).
- Produces (for Task 2): `entry_form_payload()` returns a dict that now also contains `"case_sensitive": bool`, `"selective": bool`, `"secondary_keys": list[str]`. New widget ids: `#personas-lore-entry-case-sensitive` (Switch), `#personas-lore-entry-selective` (Switch), `#personas-lore-entry-secondary-keys` (Input). New method `_sync_secondary_keys_disabled(self) -> None`.

- [ ] **Step 1: Write the failing widget tests**

Add these four tests to `Tests/UI/test_personas_lore.py`. First ensure `Switch` is imported — change the widget import line (`:10`) from
`from textual.widgets import Button, DataTable, Input, ListView, Static, TextArea`
to
`from textual.widgets import Button, DataTable, Input, ListView, Static, Switch, TextArea`.

Then append after `test_entry_priority_round_trips_through_form` (`:123`):

```python
@pytest.mark.asyncio
async def test_matching_controls_round_trip_through_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        app.query_one("#personas-lore-entry-selective", Switch).value = True
        app.query_one("#personas-lore-entry-secondary-keys", Input).value = " sword , shield ,"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        payload = app.posted[-1]
        assert payload["case_sensitive"] is True
        assert payload["selective"] is True
        assert payload["secondary_keys"] == ["sword", "shield"]  # trimmed, blank dropped


@pytest.mark.asyncio
async def test_blank_secondary_keys_is_empty_list():
    """A blank secondary-keys field yields [] (never raises)."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        # secondary-keys left blank
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted[-1]["secondary_keys"] == []


@pytest.mark.asyncio
async def test_secondary_keys_stored_even_when_not_selective():
    """Data fidelity: secondary keys are stored regardless of the Selective
    switch, so toggling Selective off does not erase them."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-selective", Switch).value = False
        app.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        payload = app.posted[-1]
        assert payload["selective"] is False
        assert payload["secondary_keys"] == ["sword"]


@pytest.mark.asyncio
async def test_fill_form_populates_matching_controls():
    """Selecting a row fills the three controls; an entry with selective=False
    but stored secondary keys keeps its keys (fidelity) and shows them disabled."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 7, "keys": ["Warden"], "content": "grim jailer",
             "position": "before_char", "enabled": True, "insertion_order": 0,
             "case_sensitive": True, "selective": False,
             "secondary_keys": ["alpha", "beta"]},
        ])
        await pilot.pause()
        table = app.query_one("#personas-lore-entries-table", DataTable)
        table.move_cursor(row=0)
        await pilot.pause()
        assert app.query_one("#personas-lore-entry-case-sensitive", Switch).value is True
        assert app.query_one("#personas-lore-entry-selective", Switch).value is False
        sec = app.query_one("#personas-lore-entry-secondary-keys", Input)
        assert sec.value == "alpha, beta"   # preserved even though selective is False
        assert sec.disabled is True          # selective off → disabled hint


@pytest.mark.asyncio
async def test_secondary_keys_disabled_hint_tracks_selective():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        await pilot.pause()
        sec = app.query_one("#personas-lore-entry-secondary-keys", Input)
        sel = app.query_one("#personas-lore-entry-selective", Switch)
        assert sec.disabled is True          # selective defaults off → disabled on mount
        sec.value = "kept"
        sel.value = True
        await pilot.pause()
        assert sec.disabled is False         # selective on → enabled
        assert sec.value == "kept"
        sel.value = False
        await pilot.pause()
        assert sec.disabled is True          # selective off → disabled again
        assert sec.value == "kept"           # value survives the toggle (fidelity)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "matching_controls or secondary_keys or fill_form_populates" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `NoMatches` querying `#personas-lore-entry-case-sensitive` (the controls don't exist yet).

- [ ] **Step 3: Add the three form controls to compose**

In `personas_lore_detail.py`, the entry-form block currently reads (`:114-124`):

```python
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Input(placeholder="Keys (comma-separated)", id="personas-lore-entry-keys")
                        yield Select(
                            [(label, value) for value, label in POSITIONS],
                            id="personas-lore-entry-position",
                            value="before_char",
                            allow_blank=False,
                        )
                        yield Input(placeholder="Priority", id="personas-lore-entry-priority", value="0")
                        yield Switch(value=True, id="personas-lore-entry-enabled", tooltip="Entry enabled")
                    yield TextArea(id="personas-lore-entry-content")
```

Insert a second form-row and the secondary-keys input between the first `Horizontal` and the content `TextArea`:

```python
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Input(placeholder="Keys (comma-separated)", id="personas-lore-entry-keys")
                        yield Select(
                            [(label, value) for value, label in POSITIONS],
                            id="personas-lore-entry-position",
                            value="before_char",
                            allow_blank=False,
                        )
                        yield Input(placeholder="Priority", id="personas-lore-entry-priority", value="0")
                        yield Switch(value=True, id="personas-lore-entry-enabled", tooltip="Entry enabled")
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Static("Case-sensitive", markup=False)
                        yield Switch(value=False, id="personas-lore-entry-case-sensitive")
                        yield Static("Selective", markup=False)
                        yield Switch(value=False, id="personas-lore-entry-selective")
                    yield Input(
                        placeholder="Secondary keys (comma-separated)",
                        id="personas-lore-entry-secondary-keys",
                    )
                    yield TextArea(id="personas-lore-entry-content")
```

- [ ] **Step 4: Add the `_sync_secondary_keys_disabled` helper and wire on_mount + a Switch.Changed handler**

The current `on_mount` (`:145-153`) is:

```python
    def on_mount(self) -> None:
        """Register the entries table's columns once the widget is mounted.

        Returns:
            None.
        """
        table = self.query_one("#personas-lore-entries-table", DataTable)
        table.add_columns("keys", "content", "position", "priority", "enabled")
```

Append the initial sync call (leave the docstring/columns as-is, add the last line):

```python
        table = self.query_one("#personas-lore-entries-table", DataTable)
        table.add_columns("keys", "content", "position", "priority", "enabled")
        self._sync_secondary_keys_disabled()
```

Add the helper method (place it right after `on_mount`, before `# ----- public API -----` at `:155`):

```python
    def _sync_secondary_keys_disabled(self) -> None:
        """Grey the secondary-keys input when Selective is off (pure visual hint;
        the stored value is preserved either way).

        Called from on_mount, from _fill_form_from_entry (Textual does NOT fire
        Switch.Changed when a switch is set to the value it already holds — the
        same hazard _set_enabled_switch guards against), and from the Selective
        switch's Changed handler.
        """
        selective = self.query_one("#personas-lore-entry-selective", Switch).value
        self.query_one("#personas-lore-entry-secondary-keys", Input).disabled = not selective
```

Add the live-toggle handler next to the existing `_enabled_changed` handler (`:426-434`):

```python
    @on(Switch.Changed, "#personas-lore-entry-selective")
    def _selective_changed(self, event: Switch.Changed) -> None:
        event.stop()
        self._sync_secondary_keys_disabled()
```

- [ ] **Step 5: Add the three fields to `entry_form_payload`**

The current method (`:271-310`) computes `enabled` at `:291` then builds the return dict at `:303-310`. Add the three reads after `enabled = ...` (`:291`):

```python
        enabled = bool(self.query_one("#personas-lore-entry-enabled", Switch).value)
        case_sensitive = bool(self.query_one("#personas-lore-entry-case-sensitive", Switch).value)
        selective = bool(self.query_one("#personas-lore-entry-selective", Switch).value)
        raw_secondary = self.query_one("#personas-lore-entry-secondary-keys", Input).value
        secondary_keys = [k.strip() for k in raw_secondary.split(",") if k.strip()]
```

Add the three keys to the returned dict (`:303-310`):

```python
        return {
            "keys": keys,
            "content": content,
            "position": position,
            "priority": priority,
            "enabled": enabled,
            "insertion_order": insertion_order,
            "case_sensitive": case_sensitive,
            "selective": selective,
            "secondary_keys": secondary_keys,
        }
```

Update the method docstring's field list (`:274-278`) to end with `..., ``priority``, ``case_sensitive``, ``selective``, ``secondary_keys``), or ``None`` if keys or content are empty.`

- [ ] **Step 6: Populate the three controls in `_fill_form_from_entry`**

The current method ends (`:349-350`) with:

```python
        self.query_one("#personas-lore-entry-priority", Input).value = str(entry.get("priority") or 0)
        self.query_one("#personas-lore-entry-enabled", Switch).value = bool(entry.get("enabled", True))
```

Append:

```python
        self.query_one("#personas-lore-entry-case-sensitive", Switch).value = bool(entry.get("case_sensitive", False))
        self.query_one("#personas-lore-entry-selective", Switch).value = bool(entry.get("selective", False))
        self.query_one("#personas-lore-entry-secondary-keys", Input).value = ", ".join(
            str(k) for k in (entry.get("secondary_keys") or [])
        )
        self._sync_secondary_keys_disabled()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "matching_controls or secondary_keys or fill_form_populates" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (4 tests). Then run the whole file to confirm no regressions:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): edit selective/secondary_keys/case_sensitive in the entry editor"
```

---

### Task 2: Screen add-handler threads the three fields

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_handle_lore_entry_add` — the `create_world_book_entry` call)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `entry_form_payload()` now carries `case_sensitive`/`selective`/`secondary_keys` (Task 1). `create_world_book_entry(..., selective=False, secondary_keys=None, case_sensitive=False, ...)` (existing manager signature). `update_world_book_entry(entry_id, **kwargs)` already handles the three fields (no change).
- Produces: entries created through the real screen add-handler persist all three fields.

- [ ] **Step 1: Write the failing real-handler tests**

Append to `Tests/UI/test_personas_lore.py` (after `test_add_entry_persists_priority_through_real_screen_handler`, `:529`). These use the real `LorePersonasTestApp` + real DB. `Switch` is already imported after Task 1.

```python
@pytest.mark.asyncio
async def test_add_entry_persists_matching_fields_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """The real _handle_lore_entry_add must forward selective/secondary_keys/
    case_sensitive to the DB — regression against the explicit-kwarg drop that
    bit `priority` in P2c."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "Ghost"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "a pale spirit"
        screen.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        ghost = next(e for e in entries if e["keys"] == ["Ghost"])
        assert ghost["case_sensitive"] is True
        assert ghost["selective"] is True
        assert ghost["secondary_keys"] == ["sword"]


@pytest.mark.asyncio
async def test_update_entry_persists_matching_fields_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """_handle_lore_entry_update forwards the three fields via **payload."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        table = screen.query_one("#personas-lore-entries-table", DataTable)
        table.move_cursor(row=0)          # select the seeded "Warden" entry → fills form
        await pilot.pause()
        screen.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "prison"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-update")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        warden = next(e for e in entries if e["keys"] == ["Warden"])
        assert warden["case_sensitive"] is True
        assert warden["selective"] is True
        assert warden["secondary_keys"] == ["prison"]
```

- [ ] **Step 2: Run the tests to verify the add test fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "persists_matching_fields" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: `test_add_entry_persists_matching_fields_through_real_screen_handler` FAILS — the persisted entry has `case_sensitive=False`/`selective=False`/`secondary_keys=[]` because the add-handler drops the fields. (`test_update_...` may already PASS, since update uses `**payload` — that is fine; it pins the update path.)

- [ ] **Step 3: Thread the three fields into `_handle_lore_entry_add`**

In `personas_screen.py`, `_handle_lore_entry_add` calls `create_world_book_entry` with an explicit kwarg list. Add the three fields:

```python
        await self._run_lore_entry_op(
            lambda manager: asyncio.to_thread(
                manager.create_world_book_entry,
                int(entity_id),
                keys=payload.get("keys", []),
                content=payload.get("content", ""),
                enabled=payload.get("enabled", True),
                position=payload.get("position", "before_char"),
                insertion_order=payload.get("insertion_order", 0),
                priority=payload.get("priority", 0),
                selective=payload.get("selective", False),
                secondary_keys=payload.get("secondary_keys", []),
                case_sensitive=payload.get("case_sensitive", False),
            ),
            "Could not add the entry",
        )
```

Leave `_handle_lore_entry_update` unchanged (it already forwards `**payload`).

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "persists_matching_fields" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "fix(lore): add-handler forwards selective/secondary_keys/case_sensitive to the DB"
```

---

### Task 3: End-to-end matching integration test

**Files:**
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: the real editor→add-handler path (Tasks 1-2) persisting a selective entry; `WorldBookManager.get_world_book` + `get_world_book_entries`; `WorldInfoProcessor(world_books=[...]).process_messages(...)`.
- Produces: nothing (leaf test).

This is one integration test that proves the surfaced config actually changes matching behavior end to end — not a broad matcher re-test (the matcher's selective/secondary logic is already covered by `Tests/Character_Chat/test_world_info.py`).

- [ ] **Step 1: Write the failing integration test**

Add the `WorldInfoProcessor` import next to the other Task-6 imports (near `:203`, after `from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager`):

```python
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor
```

Append the test to `Tests/UI/test_personas_lore.py`:

```python
@pytest.mark.asyncio
async def test_selective_entry_created_via_editor_gates_matching(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """A selective entry created through the real editor/add-handler path gates
    matching in WorldInfoProcessor: it fires only when a secondary key is present
    in the scan text. Proves editor config → DB → matcher end to end."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "hero"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "the brave hero"
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    manager = WorldBookManager(lore_db)
    book = manager.get_world_book(seeded_lore_book["book_id"])
    entries = manager.get_world_book_entries(seeded_lore_book["book_id"])
    world_book = {**book, "entries": entries}
    proc = WorldInfoProcessor(world_books=[world_book])

    # primary key "hero" present but secondary "sword" absent → selective entry does NOT fire
    r1 = proc.process_messages("the hero walks alone", [])
    assert all("brave hero" not in c for c in r1["injections"]["before_char"])
    # primary + secondary present → fires
    r2 = proc.process_messages("the hero draws a sword", [])
    assert any("brave hero" in c for c in r2["injections"]["before_char"])
```

- [ ] **Step 2: Run the test to verify it passes**

With Tasks 1-2 done this should PASS immediately (it depends on their code). Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py::test_selective_entry_created_via_editor_gates_matching \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS. If it fails at the `r1` assertion (entry fired without the secondary key), the selective/secondary_keys did not reach the DB — revisit Task 2. To prove the test has teeth, temporarily set the Selective switch line to `False` and confirm `r1` then fails (the entry fires on primary alone), then restore `True`.

- [ ] **Step 3: Run the full gate**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py Tests/Character_Chat/test_world_book_manager.py \
Tests/Character_Chat/test_world_info_diagnostics.py Tests/Character_Chat/test_world_info.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass. Then the import smoke:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: `APP IMPORT OK`.

- [ ] **Step 4: Commit**

```bash
git add Tests/UI/test_personas_lore.py
git commit -m "test(lore): editor-created selective entry gates matching end to end"
```

---

## Notes for the reviewer

- **No production code outside the two named files.** Any change to `world_info_processor.py`, `world_book_manager.py`, or `ChaChaNotes_DB.py` is out of scope for P2d-1 and should be rejected.
- **The add-handler regression (Task 2 Step 2) must be seen RED before the fix.** If it passed before Step 3, the test isn't exercising the real handler.
- **`secondary_keys` fidelity:** the value must be stored regardless of the Selective switch. A design that clears `secondary_keys` when Selective is off is a defect against the spec.
- **Disabled hint via helper, not the event alone:** `_sync_secondary_keys_disabled()` is called from `on_mount`, `_fill_form_from_entry`, AND the Changed handler — because Textual does not fire `Switch.Changed` on a no-op set (the `_set_enabled_switch` comment at `personas_lore_detail.py:225-235` documents this exact hazard). Relying only on the Changed handler would leave the hint stale after selecting a row.
