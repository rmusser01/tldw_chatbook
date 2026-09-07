# TASK-434 — Personas workbench preserves selection + preview across navigation

- **Date:** 2026-07-23
- **Task:** TASK-434 (RP/character-card UX review). Personas ▸ Console ▸ back loses the working context.
- **Branch base:** origin/dev (tip `f08343b67`).
- **Scope:** Full — AC#1 (selection/mode/center) **and** AC#2 (preview transcript survives).

## Problem

Navigating Personas → Console → back resets the workbench to "Selected: none", a blank center, "Console blocked: select an item", and a collapsed preview. The design encourages the workbench↔Console loop, but every round-trip drops the working context.

**Root cause:** the app never caches screen instances (`app.py:5550` "Screens must never be cached and re-mounted"). Instead, on navigate-away it stores `screen.save_state()` in `app._screen_states[screen_name]`; on return it builds a *fresh* screen and calls `restore_state()` (`app.py:5647-5703`). `PersonasScreen` overrides neither, so the base no-ops run and the workbench `self.state` (a `PersonasWorkbenchState` dataclass) and the preview are lost. `on_mount` then blanks the center via `_show_center(None)` (`personas_screen.py:714`).

## Key mechanics (verified)

- `restore_state()` runs **before** the new screen mounts (`app.py:5682` precedes `switch_screen` at `:5703`). So it can seed `self.state` (which `on_mount`'s `set_mode(self.state.active_mode)` then honors) and stash a "pending restore" for `on_mount` to finish.
- `PersonasWorkbenchState` (`personas_state.py:36`) is a flat `@dataclass` of primitives (`active_mode`, `sort_key`, `tag_filter`, `page_offset`, `selected_entity_kind/id/name`, `search_query`, `filter_text`, `has_unsaved_changes`, `status_message`, ...) — cleanly `asdict()`-serializable.
- The preview conversation is split across two objects: `PersonasPreviewController.history` (`list[{"role","content"}]`) holds only the **sent turns** (user/assistant), while the **greeting is not in `history`** — it lives in `PersonasPreviewPane._greeting` (a string set by `seed_greeting`, rendered as transcript line 0). `seeded_for` (controller) is the character id the greeting seeded. So a faithful capture needs **both** `pane._greeting` and `controller.history`.
- Selecting a character runs `_select_character` → sets state, marks the row, loads the record (thread worker), `_show_center("#ccp-character-card-view")`, updates the inspector + conversations, and `reset_for_character` → `reset` → `invalidate()` which **unconditionally clears** `history`. The async load worker later calls `handle_character_loaded`, which **preserves** an in-progress conversation when `seeded_for == character_id and pane.transcript_text()` (`personas_preview_controller.py:133`). This guard is what a restore leverages.

## Design

### 1. `save_state()` override (`PersonasScreen`)
```python
def save_state(self) -> dict:
    state = dict(super().save_state() or {})
    state["personas_workbench"] = asdict(self.state)
    preview = getattr(self, "preview", None)
    if preview is not None:
        greeting = ""
        try:
            greeting = self.query_one(PersonasPreviewPane).greeting_text
        except QueryError:
            pass  # pane torn down / not mounted
        state["personas_preview"] = {
            "greeting": greeting,
            "history": [dict(m) for m in preview.history],
            "seeded_for": preview.seeded_for,
        }
    return state
```
Captures the whole workbench dataclass + **both** halves of the preview (greeting **and** turns). Add a small `PersonasPreviewPane.greeting_text` read-only property returning `self._greeting` (avoids reaching into a private attr from the screen). History is character-scoped; non-character selections restore an empty preview.

### 2. `restore_state()` override (`PersonasScreen`) — runs pre-mount
```python
def restore_state(self, state: dict) -> None:
    super().restore_state(state)
    wb = state.get("personas_workbench")
    if isinstance(wb, dict):
        fields = {f.name for f in dataclasses.fields(PersonasWorkbenchState)}
        self.state = PersonasWorkbenchState(**{k: v for k, v in wb.items() if k in fields})
    self._pending_restore = {
        "kind": self.state.selected_entity_kind,
        "id": self.state.selected_entity_id,
        "name": self.state.selected_entity_name,
        "preview": state.get("personas_preview"),
    } if self.state.selected_entity_id else None
```
Rebuilding the dataclass (filtered to known fields — tolerant of version drift) sets `active_mode` before mount, so `on_mount`'s `set_mode` and list refresh land in the right mode. Only a real selection schedules a pending restore.

### 3. Deferred re-selection at the end of `on_mount`
After the existing `on_mount` body (`refresh_character_list()` etc.), apply the pending restore instead of leaving the center blank:
```python
    ...
    self._sync_title_and_console_actions()
    await self._apply_pending_restore()

async def _apply_pending_restore(self) -> None:
    pending = getattr(self, "_pending_restore", None)
    self._pending_restore = None
    if not pending:
        return
    kind, entity_id, name = pending["kind"], pending["id"], pending["name"]
    preview = pending.get("preview")
    if kind == "character":
        await self._select_character(entity_id, name, restore_preview=preview)
    elif kind == "persona_profile":
        await self._select_profile(entity_id, name)
    elif kind == "dictionary":
        await self._select_dictionary(entity_id, name)
    elif kind == "lore":
        await self._select_lore_entry(entity_id, name)
```
The `_select_*` calls re-establish the selection, center pane, inspector, and conversations — AC#1. (Guard each in try/except-tolerant restore: a stale id whose entity was deleted must degrade to blank center, not crash — reuse the existing `_run_guarded`/selection error handling.)

### 4. Preview restore (AC#2) — new branch on `_select_character`
Add an optional `restore_preview` parameter so restore rebuilds the transcript itself instead of going through `reset_for_character` (which `invalidate()`s the turns and re-seeds the greeting):
```python
async def _select_character(self, entity_id, entity_name, *, restore_preview=None):
    ...  # state, mark row, load_character (schedules the char-load worker), show center, inspector, conversations
    if restore_preview is not None:
        await self.preview.restore_conversation(
            greeting=restore_preview.get("greeting", ""),
            history=restore_preview.get("history", []),
            seeded_for=entity_id,
        )
    else:
        record = self._full_character_record(entity_id)
        await self.preview.reset_for_character(
            character_id=entity_id, character_name=entity_name, record=record
        )
```
New `PersonasPreviewController.restore_conversation` — rebuilds greeting **and** turns, and sets `seeded_for` **before** any `await` so the still-pending character-load worker's `handle_character_loaded` hits the `:133` guard (`refresh_greeting_seed`, no erase) even if it interleaves:
```python
async def restore_conversation(self, *, greeting, history, seeded_for) -> None:
    self.invalidate()                       # clears history, cancels only preview workers
    self.seeded_for = str(seeded_for)       # set FIRST: invalidate does NOT cancel the char-load worker
    try:
        pane = self.screen.query_one(PersonasPreviewPane)
    except QueryError:
        return
    await pane.seed_greeting(greeting)       # renders + stores pane._greeting (transcript now non-empty)
    for m in history:
        role, content = m.get("role"), str(m.get("content") or "")
        if role == "user":
            pane.append_user(content)
        elif role == "assistant":
            pane.append_reply(content)
    self.history = [dict(m) for m in history]
```
The greeting comes from the saved `personas_preview["greeting"]`, so restore does not depend on the async record load; when the char-load worker later delivers, the `:133` guard (`seeded_for == id and transcript_text()`) just `refresh_greeting_seed`s the stored greeting (typically identical) without erasing the turns.

### Non-character selections and empty preview
Persona/dictionary/lore restore only AC#1 (they have no preview transcript). A character with an empty saved history falls through to the normal `reset_for_character` greeting seed.

## Testing

Mirror `Tests/UI/test_chat_screen_state.py` / `test_chat_screen_suspend.py` (bare-ish screen or pilot harness for `save_state`/`restore_state`), plus the existing Personas pilot harness (`Tests/UI/test_personas_*`):
- **save→restore round-trip (AC#1):** build a `PersonasScreen`, select a character, `state = screen.save_state()`; a fresh screen `restore_state(state)` then mounts → the restored screen shows the same selected id/kind/name, `active_mode`, and center pane (`#ccp-character-card-view`), not "Selected: none".
- **preview survives incl. greeting (AC#2):** seed a preview with a greeting **and** a multi-turn `history`, save, restore → `controller.history` equals the saved turns, `pane.greeting_text` equals the saved greeting, `pane.transcript_text()` contains the greeting **and** the turns, and `seeded_for` is restored.
- **restore survives the async char-load worker:** after restore, drive `handle_character_loaded(character_id=<id>, card_data=<record>)` and assert the transcript still has the turns (the `:133` guard `refresh_greeting_seed`s, does not `invalidate`) — this pins the "set `seeded_for` first" ordering.
- **non-character selection restores center only:** a saved persona/dictionary/lore selection restores its center pane; no preview crash.
- **stale/deleted entity:** a saved selection whose id no longer exists degrades to blank center without raising.
- **no selection:** empty pending restore leaves today's blank-center behavior unchanged.
- **Regression:** existing `test_personas_*` and `test_screen_navigation` stay green; `save_state` returning the enriched dict doesn't break the app's `add_runtime_policy_snapshot`/`reconcile_saved_screen_state` round-trip (it wraps a dict — our extra keys are preserved).

## Risks / mitigations

- **Async char-load worker race (AC#2):** `_select_character` schedules the char-load worker (which `invalidate()` does **not** cancel — that only cancels the `personas-preview` group). `restore_conversation` therefore sets `self.seeded_for` **before its first `await`**, and the greeting render makes the transcript non-empty, so even if `handle_character_loaded` interleaves it hits the `:133` guard and preserves the turns. Directly tested by driving the loaded-callback after restore.
- **Greeting not in `history`:** the greeting is captured separately (`pane.greeting_text`) and restored via `seed_greeting`, so it is not lost (a plain `history` capture would drop it).
- **Version drift in saved state:** `restore_state` filters to known dataclass fields and guards non-dict payloads, so an old/foreign saved blob can't crash construction.
- **Stale selection:** deleted-entity restore is guarded to fall back to blank center.
- **`reconcile_saved_screen_state`:** our keys ride inside the same state dict the app already snapshots/reconciles; no change to that flow.

## Non-goals

- Restoring an **in-progress editor** with unsaved changes (leaving with unsaved edits is already gated by the unsaved-changes dialog; restore targets the view/selection).
- Persisting the workbench across **app restarts** (`_screen_states` is in-memory session state, matching every other screen).
- Changing the navigation/screen-caching model.
- Preserving library scroll position beyond `page_offset` (already in the dataclass).
