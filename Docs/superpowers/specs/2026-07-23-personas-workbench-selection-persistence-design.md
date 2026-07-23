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
- The preview conversation lives in `PersonasPreviewController.history` (`list[{"role","content"}]`) with `seeded_for` (the character id the greeting seeded). The pane renders via `append_user`/`append_reply`/`seed_greeting`; `transcript_text()` reads it back.
- Selecting a character runs `_select_character` → sets state, marks the row, loads the record (thread worker), `_show_center("#ccp-character-card-view")`, updates the inspector + conversations, and `reset_for_character` → `reset` → `invalidate()` which **unconditionally clears** `history`. The async load worker later calls `handle_character_loaded`, which **preserves** an in-progress conversation when `seeded_for == character_id and pane.transcript_text()` (`personas_preview_controller.py:133`). This guard is what a restore leverages.

## Design

### 1. `save_state()` override (`PersonasScreen`)
```python
def save_state(self) -> dict:
    state = dict(super().save_state() or {})
    state["personas_workbench"] = asdict(self.state)
    state["personas_preview"] = {
        "history": [dict(m) for m in self.preview.history],
        "seeded_for": self.preview.seeded_for,
    }
    return state
```
Captures the whole workbench dataclass + the preview conversation. (History is character-scoped; for non-character selections it is empty/irrelevant and simply restores as empty.)

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
Add an optional `restore_preview` parameter so restore does not lose the transcript to `reset_for_character`'s `invalidate()`:
```python
async def _select_character(self, entity_id, entity_name, *, restore_preview=None):
    ...  # state, mark row, load_character (schedules worker), show center, inspector, conversations
    if restore_preview and restore_preview.get("history"):
        await self.preview.restore_history(
            restore_preview["history"], seeded_for=entity_id
        )
    else:
        record = self._full_character_record(entity_id)
        await self.preview.reset_for_character(
            character_id=entity_id, character_name=entity_name, record=record
        )
```
New `PersonasPreviewController.restore_history`:
```python
async def restore_history(self, history, *, seeded_for) -> None:
    self.invalidate()                    # cancel workers + clear
    try:
        pane = self.screen.query_one(PersonasPreviewPane)
    except QueryError:
        return
    await pane.restore_transcript(history)   # new pane method: reset + re-render each entry
    self.history = [dict(m) for m in history]
    self.seeded_for = str(seeded_for)
```
New `PersonasPreviewPane.restore_transcript(history)`: `await self.reset()` then render each entry — the first `assistant` entry via `seed_greeting` (preserves greeting styling), the rest via `append_user`/`append_reply`. Setting `seeded_for` + a non-empty transcript means the later `handle_character_loaded` worker hits the `:133` guard (`refresh_greeting_seed`, no erase) — the ordering holds because `_select_character` sets these synchronously within the coroutine before the thread worker's callback can run.

### Non-character selections and empty preview
Persona/dictionary/lore restore only AC#1 (they have no preview transcript). A character with an empty saved history falls through to the normal `reset_for_character` greeting seed.

## Testing

Mirror `Tests/UI/test_chat_screen_state.py` / `test_chat_screen_suspend.py` (bare-ish screen or pilot harness for `save_state`/`restore_state`), plus the existing Personas pilot harness (`Tests/UI/test_personas_*`):
- **save→restore round-trip (AC#1):** build a `PersonasScreen`, select a character, `state = screen.save_state()`; a fresh screen `restore_state(state)` then mounts → the restored screen shows the same selected id/kind/name, `active_mode`, and center pane (`#ccp-character-card-view`), not "Selected: none".
- **preview survives (AC#2):** seed a preview with a multi-message `history`, save, restore → the restored preview's `controller.history` equals the saved list and `pane.transcript_text()` contains the earlier turns (not just the greeting); `seeded_for` restored.
- **restore skips the greeting-reset:** assert `_select_character(..., restore_preview=...)` does NOT call `reset_for_character` (or that the restored transcript is intact after the character-load worker fires — drive the `handle_character_loaded` guard).
- **non-character selection restores center only:** a saved persona/dictionary/lore selection restores its center pane; no preview crash.
- **stale/deleted entity:** a saved selection whose id no longer exists degrades to blank center without raising.
- **no selection:** empty pending restore leaves today's blank-center behavior unchanged.
- **Regression:** existing `test_personas_*` and `test_screen_navigation` stay green; `save_state` returning the enriched dict doesn't break the app's `add_runtime_policy_snapshot`/`reconcile_saved_screen_state` round-trip (it wraps a dict — our extra keys are preserved).

## Risks / mitigations

- **Async greeting-seed race (AC#2):** the restore sets `seeded_for` + renders the transcript synchronously within `_select_character` before the thread worker's `handle_character_loaded` callback can run; the `:133` guard then preserves it. Tested by driving the loaded-callback after restore.
- **Version drift in saved state:** `restore_state` filters to known dataclass fields and guards non-dict payloads, so an old/foreign saved blob can't crash construction.
- **Stale selection:** deleted-entity restore is guarded to fall back to blank center.
- **`reconcile_saved_screen_state`:** our keys ride inside the same state dict the app already snapshots/reconciles; no change to that flow.

## Non-goals

- Restoring an **in-progress editor** with unsaved changes (leaving with unsaved edits is already gated by the unsaved-changes dialog; restore targets the view/selection).
- Persisting the workbench across **app restarts** (`_screen_states` is in-memory session state, matching every other screen).
- Changing the navigation/screen-caching model.
- Preserving library scroll position beyond `page_offset` (already in the dataclass).
