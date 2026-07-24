# Roleplay P1d — Portability + History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dictionaries can leave and enter the app (JSON lossless / markdown lossy-with-warning), their edit history is visible and revertible (Versions tab), a Stats tab summarizes them, and the two P1c-carried hardening riders land.

**Architecture:** Two seam lines fix the strategy-loss trap in the service; the Detail widget gains Stats + Versions TabPanes and Export buttons (I/O-free — new intent messages; the screen owns every file read/write and service call, reusing the `EnhancedFileOpen` + `_io_dialog_active` worker idiom and the `ConfirmationDialog` patterns).

**Tech Stack:** Python ≥3.11, Textual (TabPane/DataTable/ConfirmationDialog/EnhancedFileOpen), pytest (+asyncio UI harness), real `CharactersRAGDB` for service tests.

**Spec:** `Docs/superpowers/specs/2026-07-15-roleplay-p1d-portability-design.md` (committed `41fcea3b`) — its "Ground truths" section pins the verified traps; read once before Task 1.

## Global Constraints

- **The rename-retry traps (spec Ground truths):** `import_json`'s name is `data.get("name") or payload.get("name") or "Imported Dictionary"` — **`data.name` wins**, so the JSON conflict retry mutates `data["name"]`. `import_markdown` **requires** `payload["name"]` (KeyError otherwise) — the markdown path always sends the filename stem; its retry mutates that payload name.
- **JSON round-trip is lossless after Task 1** (strategy included); markdown loses everything but key+content — the lossy warning names the dropped fields verbatim: regex/type, probability, group, max replacements, timed effects, enabled, case-sensitivity, priority.
- **The widget performs no I/O** — Export/Versions actions are intent messages; the screen does files + services in `personas-io` workers with the `_io_dialog_active` refuse-reentry guard where a dialog is involved.
- Exports go to `get_user_data_dir()/exports/` (import from `..Utils.paths` — NOT config.py) as `{slug}-{YYYYMMDD-HHMMSS}.{json,md}`, written temp-then-atomic-rename; failures notify and leave no partial file.
- **The test fake mirrors REAL shapes exactly** — including `ConflictError` on duplicate names, the `data.name`-wins precedence (a payload-name-first fake would green-light the retry bug), version summaries WITHOUT snapshot, the ever-present `"baseline"` row, and `get_statistics`'s thin `{dictionary_id, entry_count, enabled, source}`.
- Only Personas-owned + Character_Chat dictionary files change. Google docstrings on new/modified public callables. Widget CSS structure-only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (isolated HOME; from this worktree use the main checkout's absolute `.venv/bin/python`):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```
- UI tests that click inside the detail widget use `app.run_test(size=(200, 60))` (established clipping precedent).

---

### Task 1: Service seam — strategy round-trips

**Files:**
- Modify: `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py` (`export_json` ~:540-554, `import_json` ~:521-538)
- Test: `Tests/Character_Chat/test_local_chat_dictionary_service.py`

**Interfaces:**
- Produces: `export_json`'s `data` block gains `"strategy": record.get("strategy")`; `import_json` passes `strategy=str(data.get("strategy") or "sorted_evenly")` into `cdl.save_chat_dictionary` (which already accepts the kwarg — verified). Task 5/6 rely on these for lossless round-trips.

- [ ] **Step 1: Write the failing test** (append):

```python
def test_json_export_import_roundtrips_every_field(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = service.create_dictionary(
        {
            "name": "Round Trip",
            "description": "all fields",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure", "probability": 0.85,
                 "group": "med", "timed_effects": {"sticky": 0, "cooldown": 5, "delay": 0},
                 "max_replacements": 3, "type": "literal",
                 "enabled": False, "case_sensitive": True, "priority": 7},
            ],
            "max_tokens": 750,
        }
    )
    # strategy is settable only via update (create ignores it — P1a ground truth).
    service.update_dictionary(created["id"], {"strategy": "character_lore_first"})

    exported = service.export_json(created["id"])
    assert exported["data"]["strategy"] == "character_lore_first"  # the fixed seam

    exported["data"]["name"] = "Round Trip (imported)"  # data.name WINS — mutate it
    imported = service.import_json({"data": exported["data"]})
    record = service.get_dictionary(imported["dictionary_id"])

    assert record["name"] == "Round Trip (imported)"
    assert record["description"] == "all fields"
    assert record["strategy"] == "character_lore_first"
    assert record["max_tokens"] == 750
    src_entry = service.get_dictionary(created["id"])["entries"][0]
    dup_entry = record["entries"][0]
    for field in ("pattern", "replacement", "probability", "group", "timed_effects",
                  "max_replacements", "type", "enabled", "case_sensitive", "priority"):
        assert dup_entry.get(field) == src_entry.get(field), field
```

- [ ] **Step 2: Run — expect FAIL** (`KeyError: 'strategy'` on the export assert).

- [ ] **Step 3: Implement.** In `export_json`, add `"strategy": record.get("strategy"),` to the `data` dict (next to `max_tokens`). In `import_json`, add `strategy=str(data.get("strategy") or "sorted_evenly"),` to the `save_chat_dictionary` call. Update both docstrings (one line each noting the round-tripped field).

- [ ] **Step 4: Run the service file — all PASS.**

- [ ] **Step 5: Commit** — `fix(chat-dictionaries): JSON export/import round-trips strategy` + trailer.

---

### Task 2: Carried riders — `malformed_probability` + panel clear symmetry

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_validation.py`, `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py` (`clear()`)
- Test: `Tests/UI/test_personas_dictionary_validation.py`, `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Produces: validation code `malformed_probability` (field `probability`, message `"Probability is not a number; the editor will display 100% until it is fixed."`) emitted when the value is present but non-coercible (replacing the silent `None` swallow); `clear()` also empties `#personas-dict-validation`'s options and `self._validation_findings`.

- [ ] **Step 1: Failing tests.** Module (append to `test_personas_dictionary_validation.py`):

```python
def test_malformed_probability_yields_advisory():
    findings = validate_entries([_entry("BP", probability="garbage")])
    assert [f.code for f in findings] == ["malformed_probability"]
    assert findings[0].field == "probability"
```

(Adjust the existing `test_probability_garbage_does_not_crash_and_yields_no_finding` to the new expectation — rename it `test_probability_garbage_yields_advisory_not_crash` and assert the finding instead of `== []`.)

UI (append to `TestDictionaryValidationPanel`):

```python
    async def test_clear_empties_validation_panel(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import OptionList

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "BP", "replacement": "dup", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 1
            detail = screen.query_one("#personas-dictionary-detail")
            detail.clear()
            await pilot.pause()
            assert panel.option_count == 0
            assert detail._validation_findings == []
```

- [ ] **Step 2: Run — FAIL** (empty findings list; panel keeps its option after clear).

- [ ] **Step 3: Implement.** In `validate_entries`, the probability block becomes:

```python
        probability = entry.get("probability")
        probability_value: float | None = None
        if probability is not None:
            try:
                probability_value = float(probability)
            except (TypeError, ValueError):
                findings.append(ValidationFinding(
                    code="malformed_probability", field="probability", entry_id=entry_id,
                    message="Probability is not a number; the editor will display 100% until it is fixed.",
                ))
        if probability_value is not None and probability_value == 0.0:
```

In the widget's `clear()`, append:

```python
        self._validation_findings = []
        try:
            panel = self.query_one("#personas-dict-validation", OptionList)
        except Exception:
            return
        panel.clear_options()
```

- [ ] **Step 4: Run both files — PASS** (the module file + the whole UI dictionaries file).

- [ ] **Step 5: Commit** — `fix(personas): malformed-probability advisory + validation-panel clear symmetry (P1c riders)` + trailer.

---

### Task 3: Stats tab

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`, `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `get_statistics(dictionary_id, mode="local") -> {"dictionary_id", "entry_count", "enabled", "source"}` (thin — verified).
- Produces: TabPane `#personas-dict-tab-stats` with `#personas-dict-stats-body` (Static); widget API `load_statistics(stats: dict | None, entries: list[dict]) -> None` (stats may be None when the fetch failed → render from entries alone with a dim note); screen fetches in `_select_dictionary` and re-feeds inside `_reload_selected_dictionary_entries`. Fake gains `get_statistics` mirroring the thin real shape.

- [ ] **Step 1: Failing test** (append):

```python
class TestDictionaryStatsTab:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_stats_render_service_plus_client_enrichment(self, mock_app_instance, stub_characters, fake_dict_service):
        fake_dict_service.records[1]["entries"][1].update({"type": "regex", "pattern": "/hr/i", "enabled": False, "priority": 5})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            body = str(screen.query_one("#personas-dict-stats-body", Static).renderable)
            assert "Entries: 2" in body
            assert "literal: 1" in body and "regex: 1" in body
            assert "disabled: 1" in body
            assert "Priority: 0..5" in body
            assert "tokens" in body  # approximate token total line
```

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-stats-body`).

- [ ] **Step 3: Implement.**

(a) Widget compose — after the Settings TabPane:

```python
            with TabPane("Stats", id="personas-dict-tab-stats"):
                yield Static("", id="personas-dict-stats-body", markup=False)
```

CSS: `#personas-dict-stats-body { height: auto; }`.

(b) Widget API:

```python
    def load_statistics(self, stats: dict | None, entries: list[dict]) -> None:
        """Render the Stats tab from service stats plus client-side enrichment.

        Args:
            stats: The service's get_statistics payload, or None when the
                fetch failed (entries-only rendering with a note).
            entries: The currently loaded API-named entry dicts.
        """
        literal = sum(1 for e in entries if (e.get("type") or "literal") != "regex")
        regex = len(entries) - literal
        disabled = sum(1 for e in entries if not e.get("enabled", True))
        tokens = sum(len(str(e.get("replacement") or "").split()) for e in entries)
        priorities = [int(e.get("priority") or 0) for e in entries] or [0]
        lines = [
            f"Entries: {stats.get('entry_count', len(entries)) if stats else len(entries)}"
            + ("" if stats else "  (service stats unavailable)"),
            f"Types — literal: {literal} · regex: {regex} · disabled: {disabled}",
            f"Approx. replacement tokens: {tokens}",
        ]
        if any(priorities):
            lines.append(f"Priority: {min(priorities)}..{max(priorities)}")
        if stats is not None:
            lines.append(f"Dictionary enabled: {'yes' if stats.get('enabled', True) else 'no'}")
        self.query_one("#personas-dict-stats-body", Static).update("\n".join(lines))
```

(c) Screen — in `_select_dictionary` (after `detail.load_dictionary(record)`), and at the end of `_reload_selected_dictionary_entries` (after `update_entries`):

```python
        stats = None
        try:
            stats = await service.get_statistics(int(entity_id), mode="local")
        except Exception:
            logger.opt(exception=True).warning(f"Could not load dictionary {entity_id} statistics.")
        detail.load_statistics(stats, list(record.get("entries") or []))
```

(in the reload variant, `entries` come from the freshly fetched record; keep variable names local to each site).

(d) Fake:

```python
    async def get_statistics(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        return {
            "dictionary_id": int(dictionary_id),
            "entry_count": len(record["entries"]),
            "enabled": bool(record.get("enabled", True)),
            "source": "local",
        }
```

- [ ] **Step 4: Run the UI file — PASS.**

- [ ] **Step 5: Commit** — `feat(personas): dictionary Stats tab (thin service stats + client enrichment)` + trailer.

---

### Task 4: Versions tab (list · view · revert)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`, `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `list_versions(id, mode) -> {"versions": [{revision, action, name, created_at}...], ...}` (no snapshot; always ≥1 `"baseline"` row); `get_version(id, revision) -> {..., "snapshot": {description, strategy, max_tokens, enabled, entries: [...]}}`; `revert_version(id, revision)` (internally optimistic-locked; may raise `ConflictError`).
- Produces: TabPane `#personas-dict-tab-versions` with DataTable `#personas-dict-versions-table` (columns `rev · action · name · created`; row key = str(revision)), Buttons `#personas-dict-version-view` / `#personas-dict-version-revert`, detail Static `#personas-dict-version-snapshot`; messages `DictionaryVersionViewRequested(revision: int)` / `DictionaryVersionRevertRequested(revision: int)`; widget APIs `load_versions(summaries: list[dict])`, `show_version_snapshot(record: dict)`. Fake gains an in-memory history keyed by revision with a baseline row, mirroring real shapes.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryVersionsTab:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_versions_listed_with_baseline(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            assert table.row_count >= 1
            assert str(table.get_cell_at((0, 1))) == "baseline"

    async def test_view_shows_snapshot_summary(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-versions-table", DataTable).move_cursor(row=0)
            await pilot.click("#personas-dict-version-view")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            snapshot = str(screen.query_one("#personas-dict-version-snapshot", Static).renderable)
            assert "rev 1" in snapshot and "entries: 2" in snapshot and "sorted_evenly" in snapshot

    async def test_revert_confirms_then_restores(self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch):
        from textual.widgets import DataTable

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            # Mutate: rename via settings save to create revision 2.
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            screen.query_one("#personas-dict-name", Input).value = "Renamed"
            await pilot.click("#personas-dict-settings-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Renamed"

            async def _yes(name):
                return True

            monkeypatch.setattr(screen, "_confirm_dictionary_revert", _yes)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-versions"
            await pilot.pause()
            table = screen.query_one("#personas-dict-versions-table", DataTable)
            table.move_cursor(row=0)  # revision 1 (baseline, pre-rename)
            await pilot.click("#personas-dict-version-revert")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.records[1]["name"] == "Medical Abbrev"  # restored
            assert screen.query_one("#personas-dict-name", Input).value == "Medical Abbrev"  # detail reloaded
```

(Import `TabbedContent` from `textual.widgets` at the top of the class's tests if not already file-level.)

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-versions-table`).

- [ ] **Step 3: Implement.**

(a) Widget messages:

```python
class DictionaryVersionViewRequested(Message):
    def __init__(self, revision: int) -> None:
        super().__init__()
        self.revision = revision


class DictionaryVersionRevertRequested(Message):
    def __init__(self, revision: int) -> None:
        super().__init__()
        self.revision = revision
```

(b) Compose — after the Stats TabPane:

```python
            with TabPane("Versions", id="personas-dict-tab-versions"):
                yield DataTable(id="personas-dict-versions-table", cursor_type="row")
                with Horizontal(classes="personas-dict-form-row"):
                    yield Button("View", id="personas-dict-version-view", classes="console-action-secondary")
                    yield Button("Revert…", id="personas-dict-version-revert", classes="console-action-secondary")
                yield Static("", id="personas-dict-version-snapshot", markup=False)
```

`on_mount` adds its columns: `versions_table.add_columns("rev", "action", "name", "created")`. CSS: table `height: auto; max-height: 8;`, snapshot `height: auto;`.

(c) Widget APIs + handlers:

```python
    def load_versions(self, summaries: list[dict]) -> None:
        """Render version summaries (newest first, as the service returns them).

        Args:
            summaries: list_versions()["versions"] records (no snapshots).
        """
        table = self.query_one("#personas-dict-versions-table", DataTable)
        table.clear()
        for record in summaries:
            table.add_row(
                str(record.get("revision")),
                str(record.get("action") or ""),
                str(record.get("name") or ""),
                str(record.get("created_at") or ""),
                key=str(record.get("revision")),
            )
        self.query_one("#personas-dict-version-snapshot", Static).update("")

    def show_version_snapshot(self, record: dict) -> None:
        """Render a get_version() record as a summary line block.

        Args:
            record: The full version record including ``snapshot``.
        """
        snapshot = record.get("snapshot") or {}
        entries = snapshot.get("entries") or []
        lines = [
            f"rev {record.get('revision')} · {record.get('action')} · {record.get('created_at')}",
            f"name: {snapshot.get('name')} · entries: {len(entries)}",
            f"strategy: {snapshot.get('strategy')} · budget: {snapshot.get('max_tokens')}"
            f" · enabled: {'yes' if snapshot.get('enabled', True) else 'no'}",
        ]
        if snapshot.get("description"):
            lines.append(f"description: {str(snapshot.get('description'))[:80]}")
        self.query_one("#personas-dict-version-snapshot", Static).update("\n".join(lines))

    def _selected_revision(self) -> int | None:
        table = self.query_one("#personas-dict-versions-table", DataTable)
        if table.cursor_row is None or table.cursor_row < 0 or table.row_count == 0:
            return None
        try:
            key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
            return int(str(key.value))
        except Exception:
            return None

    @on(Button.Pressed, "#personas-dict-version-view")
    def _version_view_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        revision = self._selected_revision()
        if revision is not None:
            self.post_message(DictionaryVersionViewRequested(revision))

    @on(Button.Pressed, "#personas-dict-version-revert")
    def _version_revert_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        revision = self._selected_revision()
        if revision is not None:
            self.post_message(DictionaryVersionRevertRequested(revision))
```

(d) Screen — import the two messages; fetch versions in `_select_dictionary` and after mutations (`_reload_selected_dictionary_entries` end) via a shared helper; handlers:

```python
    async def _refresh_dictionary_versions(self) -> None:
        """Feed the Versions tab for the selected dictionary (best-effort)."""
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            response = await service.list_versions(int(entity_id), mode="local")
        except Exception:
            logger.opt(exception=True).warning(f"Could not list dictionary {entity_id} versions.")
            detail.load_versions([])
            return
        detail.load_versions(list(response.get("versions") or []))

    @on(DictionaryVersionViewRequested)
    async def _handle_dictionary_version_view(self, message: DictionaryVersionViewRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            record = await service.get_version(int(entity_id), message.revision, mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not load version {message.revision}.")
            detail.set_status(f"Could not load version: {exc}")
            return
        detail.show_version_snapshot(record)

    @on(DictionaryVersionRevertRequested)
    async def _handle_dictionary_version_revert(self, message: DictionaryVersionRevertRequested) -> None:
        message.stop()
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._dictionary_revert_worker(message.revision), group="personas-io")

    async def _confirm_dictionary_revert(self, revision: int) -> bool:
        """True when the user confirmed the revert (worker context required)."""
        dialog = ConfirmationDialog(
            title="Revert",
            message=f"Revert to revision {revision}? Current settings and entries are replaced.",
            confirm_label="Revert",
            cancel_label="Cancel",
        )
        try:
            return bool(await self.app.push_screen_wait(dialog))
        except Exception:
            logger.opt(exception=True).warning("Could not show the revert confirmation dialog.")
            return False

    async def _dictionary_revert_worker(self, revision: int) -> None:
        try:
            if not await self._confirm_dictionary_revert(revision):
                return
            entity_id = self.state.selected_entity_id
            service = self._dictionary_scope_service()
            if service is None or not entity_id:
                return
            detail = self.query_one(PersonasDictionaryDetailWidget)
            try:
                record = await service.revert_version(int(entity_id), revision, mode="local")
            except ConflictError:
                detail.set_status(
                    "Revert failed: the dictionary changed since it was loaded. Reselect and try again."
                )
                return
            except Exception as exc:
                logger.opt(exception=True).warning(f"Could not revert to revision {revision}.")
                detail.set_status(f"Revert failed: {exc}")
                return
            raw_version = record.get("version")
            self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
            self.state.selected_entity_name = str(record.get("name") or "")
            detail.load_dictionary(record)
            detail.set_status(f"Reverted to revision {revision}.")
            await self._render_dictionary_rows(query=self.state.search_query)
            self.query_one(PersonasLibraryPane).mark_active_row("dictionary", entity_id)
            await self._refresh_dictionary_versions()
        finally:
            self._io_dialog_active = False
```

Call `await self._refresh_dictionary_versions()` at the end of `_select_dictionary` and `_reload_selected_dictionary_entries`, and after settings save.

(e) Fake — in-memory history mirroring real shapes:

```python
    def _record_version(self, record: dict, action: str) -> None:
        history = self.history.setdefault(int(record["id"]), [])
        entry = {
            "dictionary_id": int(record["id"]),
            "revision": int(record["version"]),
            "action": action,
            "name": record["name"],
            "created_at": "2026-07-15T00:00:00Z",
            "snapshot": copy.deepcopy(
                {k: record[k] for k in ("id", "name", "description", "strategy",
                                         "max_tokens", "enabled", "version", "entries")}
            ),
        }
        history[:] = [h for h in history if h["revision"] != entry["revision"]]
        history.append(entry)

    async def list_versions(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> dict:
        self._ensure_baseline(int(dictionary_id))
        versions = sorted(self.history.get(int(dictionary_id), []), key=lambda h: -h["revision"])
        return {
            "dictionary_id": int(dictionary_id),
            "versions": [{k: h[k] for k in ("dictionary_id", "revision", "action", "name", "created_at")}
                          for h in versions],
            "total": len(versions), "limit": 20, "offset": 0, "source": "local",
        }

    async def get_version(self, dictionary_id: int, revision: int, mode: str = "local") -> dict:
        self._ensure_baseline(int(dictionary_id))
        for h in self.history.get(int(dictionary_id), []):
            if h["revision"] == int(revision):
                return {**h, "source": "local"}
        raise ValueError(f"local_chat_dictionary_version_not_found:{revision}")

    async def revert_version(self, dictionary_id: int, revision: int, mode: str = "local") -> dict:
        target = await self.get_version(dictionary_id, revision)
        record = self.records[int(dictionary_id)]
        snapshot = target["snapshot"]
        for k in ("name", "description", "strategy", "max_tokens", "enabled"):
            record[k] = snapshot[k]
        record["entries"] = copy.deepcopy(snapshot["entries"])
        record["version"] += 1
        self._record_version(record, "revert")
        return {**self._summary(record), "reverted_to_revision": int(revision)}

    def _ensure_baseline(self, dictionary_id: int) -> None:
        if not self.history.get(dictionary_id):
            self._record_version(self.records[dictionary_id], "baseline")
```

`self.history: dict[int, list[dict]] = {}` in `__init__`; `update_dictionary` calls `self._record_version(record, "update")` after bumping the version (so settings-save creates revision 2 in the revert test).

- [ ] **Step 4: Run the UI file — PASS** (three new + all prior).

- [ ] **Step 5: Commit** — `feat(personas): dictionary Versions tab — list, snapshot view, confirmed revert` + trailer.

---

### Task 5: Export flow (JSON + lossy-markdown confirm)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py`, `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `export_json(id, mode)` / `export_markdown(id, mode)` (shapes per spec Ground truths). `get_user_data_dir` from `tldw_chatbook.Utils.paths` (the screen imports it; tests monkeypatch it to `tmp_path`).
- Produces: Settings-tab Buttons `#personas-dict-export-json` / `#personas-dict-export-md` (+ a dim note Static: `"Exports read the last saved state."`); message `DictionaryExportRequested(fmt: str)` (`"json"`|`"markdown"`); screen `_handle_dictionary_export` → markdown first gates on `_confirm_lossy_markdown_export()` (ConfirmationDialog naming the dropped fields) → `_dictionary_export_worker(fmt)` writes `{slug}-{timestamp}.{ext}` temp-then-rename under `get_user_data_dir()/exports/`, then `detail.set_status(f"Exported to {path}")`.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryExport:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_export_json_writes_file_and_reports_path(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path, monkeypatch):
        import tldw_chatbook.UI.Screens.personas_screen as screen_module

        monkeypatch.setattr(screen_module, "get_user_data_dir", lambda: tmp_path)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            await pilot.click("#personas-dict-export-json")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            files = list((tmp_path / "exports").glob("medical-abbrev-*.json"))
            assert len(files) == 1
            import json as jsonlib
            payload = jsonlib.loads(files[0].read_text())
            assert payload["data"]["name"] == "Medical Abbrev"
            assert payload["data"]["strategy"] == "sorted_evenly"
            status = str(screen.query_one("#personas-dict-status", Static).renderable)
            assert "Exported to" in status and str(files[0]) in status

    async def test_markdown_export_gates_on_lossy_confirm(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path, monkeypatch):
        import tldw_chatbook.UI.Screens.personas_screen as screen_module

        monkeypatch.setattr(screen_module, "get_user_data_dir", lambda: tmp_path)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)

            async def _declined():
                return False

            monkeypatch.setattr(screen, "_confirm_lossy_markdown_export", _declined)
            screen.query_one("#personas-dict-tabs", TabbedContent).active = "personas-dict-tab-settings"
            await pilot.pause()
            await pilot.click("#personas-dict-export-md")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert list((tmp_path / "exports").glob("*.md")) == []  # declined -> no file

            async def _accepted():
                return True

            monkeypatch.setattr(screen, "_confirm_lossy_markdown_export", _accepted)
            await pilot.click("#personas-dict-export-md")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            files = list((tmp_path / "exports").glob("medical-abbrev-*.md"))
            assert len(files) == 1
            assert "BP: blood pressure" in files[0].read_text()
```

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-export-json`).

- [ ] **Step 3: Implement.**

(a) Widget — message + Settings-tab additions after the Save button:

```python
class DictionaryExportRequested(Message):
    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.fmt = fmt
```

```python
                with Horizontal(classes="personas-dict-form-row"):
                    yield Button("Export JSON", id="personas-dict-export-json", classes="console-action-secondary")
                    yield Button("Export Markdown", id="personas-dict-export-md", classes="console-action-secondary")
                yield Static("Exports read the last saved state.", id="personas-dict-export-note", markup=False, classes="destination-purpose")
```

Handlers post `DictionaryExportRequested("json")` / `("markdown")` with `event.stop()`.

(b) Screen — import `get_user_data_dir` from `...Utils.paths` and the message; handler + worker:

```python
    _MARKDOWN_LOSSY_FIELDS = (
        "regex/type, probability, group, max replacements, timed effects, "
        "enabled, case-sensitivity, priority"
    )

    @on(DictionaryExportRequested)
    async def _handle_dictionary_export(self, message: DictionaryExportRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary" or not self.state.selected_entity_id:
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._dictionary_export_worker(message.fmt), group="personas-io")

    async def _confirm_lossy_markdown_export(self) -> bool:
        """True when the user accepted the lossy-markdown warning."""
        dialog = ConfirmationDialog(
            title="Export Markdown",
            message=(
                "Markdown keeps only pattern and replacement text. These fields "
                f"are DROPPED: {self._MARKDOWN_LOSSY_FIELDS}. Use JSON for a full backup."
            ),
            confirm_label="Export anyway",
            cancel_label="Cancel",
        )
        try:
            return bool(await self.app.push_screen_wait(dialog))
        except Exception:
            logger.opt(exception=True).warning("Could not show the lossy-export dialog.")
            return False

    async def _dictionary_export_worker(self, fmt: str) -> None:
        try:
            if fmt == "markdown" and not await self._confirm_lossy_markdown_export():
                return
            entity_id = self.state.selected_entity_id
            service = self._dictionary_scope_service()
            if service is None or not entity_id:
                return
            detail = self.query_one(PersonasDictionaryDetailWidget)
            try:
                if fmt == "json":
                    response = await service.export_json(int(entity_id), mode="local")
                    body = json.dumps(response, indent=2, ensure_ascii=False)
                    extension = "json"
                else:
                    response = await service.export_markdown(int(entity_id), mode="local")
                    body = str(response.get("content") or "")
                    extension = "md"
            except Exception as exc:
                logger.opt(exception=True).warning(f"Could not export dictionary {entity_id}.")
                detail.set_status(f"Export failed: {exc}")
                return
            name = str(self.state.selected_entity_name or "dictionary")
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "dictionary"
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            exports_dir = get_user_data_dir() / "exports"
            try:
                exports_dir.mkdir(parents=True, exist_ok=True)
                target = exports_dir / f"{slug}-{stamp}.{extension}"
                temp = exports_dir / f".{slug}-{stamp}.{extension}.tmp"
                temp.write_text(body, encoding="utf-8")
                temp.replace(target)
            except OSError as exc:
                logger.opt(exception=True).warning("Could not write the export file.")
                detail.set_status(f"Export failed: {exc}")
                return
            detail.set_status(f"Exported to {target}")
        finally:
            self._io_dialog_active = False
```

(`json`, `re`, `datetime` — verify existing imports in the screen; add what's missing.)

(c) Fake:

```python
    async def export_json(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        return {
            "dictionary_id": int(dictionary_id),
            "data": {
                "name": record["name"], "description": record.get("description") or "",
                "content": None,
                "entries": copy.deepcopy(record["entries"]),
                "strategy": record.get("strategy") or "sorted_evenly",
                "max_tokens": record.get("max_tokens") or 1000,
                "enabled": bool(record.get("enabled", True)),
                "version": record.get("version", 1),
            },
            "source": "local",
        }

    async def export_markdown(self, dictionary_id: int, mode: str = "local") -> dict:
        record = self.records[int(dictionary_id)]
        lines = [f"{e.get('pattern')}: {e.get('replacement')}" for e in record["entries"]]
        return {"dictionary_id": int(dictionary_id), "name": record["name"],
                "content": "\n".join(lines) + ("\n" if lines else ""), "source": "local"}
```

- [ ] **Step 4: Run the UI file — PASS.**

- [ ] **Step 5: Commit** — `feat(personas): dictionary export — JSON plus lossy-gated markdown, atomic file writes` + trailer.

---

### Task 6: Import flow (picker, dispatch, conflict-rename loop)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` (`set_mode`), `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: `import_json({"data": ...})` (data.name WINS — retry mutates `data["name"]`; `ConflictError` on duplicate); `import_markdown({"name": ..., "content": ...})` (`name` REQUIRED); `_unique_dictionary_name` naming probe (exists from P1a — reuse its pattern with an `(imported)` base); the `EnhancedFileOpen` + `_io_dialog_active` idiom (clone, don't share, the character worker).
- Produces: rail Import visible in dictionaries mode with tooltip `"Import a dictionary (JSON or Markdown)."` (characters keep their tooltip); screen `_open_dictionary_import_dialog` → `_dictionary_import_worker` → `_import_dictionary_from_path(path)`; conflict path loops through `_prompt_import_rename(suggested) -> str | None` (an Input dialog — reuse/instantiate the project's text-input dialog; if none exists, auto-rename WITHOUT a dialog using the disambiguation probe and notify the chosen name — implementer checks for an existing input-dialog widget first and reports which route was taken).

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryImport:
    async def test_import_button_visible_in_dictionaries_mode(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _enter_dictionaries(pilot)
            btn = screen.query_one("#personas-library-import", Button)
            assert btn.display is True
            assert "dictionary" in str(btn.tooltip).lower()

    async def test_import_json_file_selects_new_dictionary(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        import json as jsonlib

        payload = {"data": {"name": "Shipped In", "description": "", "content": None,
                             "entries": [{"pattern": "RR", "replacement": "resp rate",
                                          "probability": 1.0, "group": None, "timed_effects": None,
                                          "max_replacements": 1, "type": "literal",
                                          "enabled": True, "case_sensitive": False, "priority": 0}],
                             "strategy": "character_lore_first", "max_tokens": 500,
                             "enabled": True, "version": 3}}
        source = tmp_path / "shipped.json"
        source.write_text(jsonlib.dumps(payload))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Shipped In" in names
            assert screen.state.selected_entity_name == "Shipped In"
            imported = next(r for r in fake_dict_service.records.values() if r["name"] == "Shipped In")
            assert imported["strategy"] == "character_lore_first"

    async def test_import_conflict_renames_and_succeeds(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        import json as jsonlib

        payload = {"data": {"name": "Medical Abbrev", "description": "", "content": None,
                             "entries": [], "strategy": "sorted_evenly", "max_tokens": 1000,
                             "enabled": True, "version": 1}}
        source = tmp_path / "clash.json"
        source.write_text(jsonlib.dumps(payload))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "Medical Abbrev (imported)" in names  # data.name mutated, retried

    async def test_import_markdown_uses_filename_stem(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        source = tmp_path / "field-notes.md"
        source.write_text("RR: resp rate\n")
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            names = [r["name"] for r in fake_dict_service.records.values()]
            assert "field-notes" in names

    async def test_import_bad_json_creates_nothing(self, mock_app_instance, stub_characters, fake_dict_service, tmp_path):
        source = tmp_path / "broken.json"
        source.write_text("{not json")
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            before = len(fake_dict_service.records)
            await screen._import_dictionary_from_path(str(source))
            await pilot.pause()
            assert len(fake_dict_service.records) == before
```

Also fake imports (mirroring the real precedence):

```python
    async def import_json(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        data = dict(payload.get("data") or {})
        name = data.get("name") or payload.get("name") or "Imported Dictionary"  # data.name WINS
        created = await self.create_dictionary(
            {"name": name, "description": data.get("description") or "",
             "max_tokens": data.get("max_tokens") or 1000,
             "enabled": bool(data.get("enabled", True)),
             "entries": data.get("entries") or []}
        )
        if (data.get("strategy") or "sorted_evenly") != "sorted_evenly":
            await self.update_dictionary(created["id"], {"strategy": data["strategy"]})
        self.calls.append(("import_json", name))
        return {"dictionary_id": created["id"], "source": "local"}

    async def import_markdown(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        name = payload["name"]  # REQUIRED, like the real service
        entries = []
        for line in str(payload.get("content") or "").splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                entries.append({"pattern": key.strip(), "replacement": value.strip(),
                                "probability": 1.0, "group": None, "timed_effects": None,
                                "max_replacements": 1, "type": "literal",
                                "enabled": True, "case_sensitive": False, "priority": 0})
        created = await self.create_dictionary({"name": name, "entries": entries})
        self.calls.append(("import_markdown", name))
        return {"dictionary_id": created["id"], "source": "local"}
```

(The fake's `create_dictionary` already raises `ConflictError` on duplicate names and ignores strategy — precedence preserved.)

- [ ] **Step 2: Run — FAIL** (`AttributeError: _import_dictionary_from_path`; Import hidden).

- [ ] **Step 3: Implement.**

(a) Pane `set_mode`:

```python
    def set_mode(self, mode: str) -> None:
        """Gate the toolbar per mode: Import for characters+dictionaries, Duplicate for dictionaries."""
        self._import_visible = mode in ("characters", "dictionaries")
        import_button = self.query_one("#personas-library-import", Button)
        import_button.display = self._import_visible
        import_button.tooltip = (
            "Import a dictionary (JSON or Markdown)."
            if mode == "dictionaries"
            else "Import a character card (PNG or JSON)."
        )
        self.query_one("#personas-library-duplicate", Button).display = mode == "dictionaries"
```

(Note: `_import_visible` feeds the empty-state hint copy — dictionaries mode now honestly says "use New or Import".)

(b) Screen — the import action branch (`_handle_action_requested`) replaces its characters-only early return:

```python
        elif message.action == "import":
            if self.state.active_mode == "characters":
                await self._run_guarded(self._open_import_dialog)
            elif self.state.active_mode == "dictionaries":
                await self._run_guarded(self._open_dictionary_import_dialog)
```

(c) Dialog + worker + import core (clone of the character idiom):

```python
    async def _open_dictionary_import_dialog(self) -> None:
        """Continuation for the guarded dictionaries-import action."""
        if self._io_dialog_active:
            logger.debug("Import/export dialog already active; ignoring import request.")
            return
        self._io_dialog_active = True
        self.run_worker(self._dictionary_import_dialog_worker(), group="personas-io")

    async def _dictionary_import_dialog_worker(self) -> None:
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        try:
            picker = EnhancedFileOpen(
                title="Import Dictionary",
                filters=Filters(
                    ("Dictionaries", lambda p: p.suffix.lower() in (".json", ".md", ".markdown")),
                    ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                    ("Markdown Files", lambda p: p.suffix.lower() in (".md", ".markdown")),
                    ("All Files", lambda p: True),
                ),
                context="dictionary_import",
            )
            try:
                file_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the dictionary import dialog.")
                return
            if file_path:
                await self._import_dictionary_from_path(str(file_path))
        finally:
            self._io_dialog_active = False

    async def _import_dictionary_from_path(self, path: str) -> None:
        """Import a dictionary file; on a name conflict, auto-rename and retry."""
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        source = Path(path)
        try:
            text = await asyncio.to_thread(source.read_text, "utf-8")
        except OSError as exc:
            logger.opt(exception=True).warning(f"Could not read import file {path}.")
            self._notify(f"Import failed: {exc}", "error")
            return
        suffix = source.suffix.lower()
        if suffix == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                self._notify(f"Import failed: not valid JSON ({exc})", "error")
                return
            data = dict(parsed.get("data") if isinstance(parsed, dict) and "data" in parsed else parsed)
            request = {"data": data}
            def _rename(new_name: str) -> None:
                data["name"] = new_name          # data.name WINS - mutate it
            base_name = str(data.get("name") or "Imported Dictionary")
            importer = service.import_json
        elif suffix in (".md", ".markdown"):
            request = {"name": source.stem, "content": text}  # name REQUIRED
            def _rename(new_name: str) -> None:
                request["name"] = new_name
            base_name = source.stem
            importer = service.import_markdown
        else:
            self._notify("Import supports .json and .md files.", "warning")
            return
        try:
            result = await importer(request, mode="local")
        except ConflictError:
            renamed = self._unique_dictionary_name(f"{base_name} (imported)")
            _rename(renamed)
            try:
                result = await importer(request, mode="local")
            except Exception as exc:
                logger.opt(exception=True).warning("Dictionary import retry failed.")
                self._notify(f"Import failed: {exc}", "error")
                return
            self._notify(f"Name in use - imported as '{renamed}'.", "information")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Dictionary import failed for {path}.")
            self._notify(f"Import failed: {exc}", "error")
            return
        await self._render_dictionary_rows(query="")
        record = None
        try:
            record = await service.get_dictionary(int(result["dictionary_id"]), mode="local")
        except Exception:
            logger.opt(exception=True).warning("Imported dictionary could not be reloaded.")
        if record is not None:
            await self._select_dictionary(str(record.get("id")), str(record.get("name") or ""))
```

**Design resolution baked in:** the conflict flow auto-renames with the P1a disambiguation probe (`_unique_dictionary_name(f"{base} (imported)")`) and notifies the chosen name, rather than pushing an input dialog — the spec's "pre-filled rename dialog" collapses to this when no reusable text-input dialog exists in the project (implementer: check for one first — `grep -rn "class.*InputDialog\|TextInputDialog" tldw_chatbook/Widgets/` — and if a suitable one exists, use it with the same pre-fill and keep the same tests passing plus one dialog-cancel test; report which route was taken). `Path` and `asyncio` are already imported in the screen; verify `json`/`ConflictError`.

- [ ] **Step 4: Run the whole UI file — PASS** (five new + all prior).

- [ ] **Step 5: Commit** — `feat(personas): dictionary import — file picker, extension dispatch, conflict auto-rename` + trailer.

---

### Task 7: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-15-roleplay-p1d-portability-design.md` (status line; if Task 6 took the input-dialog route, also amend AC2's rename wording to match reality)

- [ ] **Step 1: Full gate**

```
HOME=... .venv/bin/python -m pytest \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_dictionary_validation.py \
  Tests/UI/test_personas_workbench.py Tests/Character_Chat/ Tests/Chat/test_chat_functions.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (exact counts in the report). Import smoke on the five touched modules.

- [ ] **Step 2: Spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1d).` (+ the AC2 wording amendment if applicable).

- [ ] **Step 3: Commit** — `docs(roleplay): mark P1d portability spec implemented` + trailer.
