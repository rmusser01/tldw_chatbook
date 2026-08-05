# TASK-436 — Characters mode center-pane empty-state guidance

- **Date:** 2026-07-23
- **Task:** TASK-436 (RP/character-card UX review). The Characters center pane is blank with no selection → reads as broken.
- **Branch base:** origin/dev (tip `4d6bb49f3`).

## Problem

On first visit to the Roleplay/Personas workbench (Characters mode is the default), the center pane is a large blank area with no copy until a character is selected. Combined with the inspector's "Console blocked: select an item", the first impression reads as **broken** rather than "select or create a character".

**Root cause:** the Characters branch of `_apply_mode` (`personas_screen.py:1644-1646`), plus `on_mount` (`:829`) and every deselect path, call `self._show_center(None)`. `_show_center(None)` matches none of the exclusive `_CENTER_VIEW_IDS` (`:257-267`), so it sets `display=False` on **every** center view → `#personas-detail-stack` renders nothing (`_show_center`, `:5756-5811`).

The existing library-rail empty row ("No characters yet – use New or Import", `personas_library_pane.py:265-276`) only fires when the **library itself is empty**. In the observed scenario the seeded Default Assistant character exists, so the rail shows rows while the center stays blank — exactly the gap this task closes.

## Key mechanics (verified)

- **The center is a single swappable stack.** `_show_center(visible_id)` iterates `_CENTER_VIEW_IDS` and sets each widget's `display = (selector == visible_id)`; `None` hides all. It also gates `#personas-character-attachments` / the dict panel to only the two character view ids (`#ccp-character-card-view`, `#ccp-character-editor-view`) — so a non-character `visible_id` (including a new guidance id) hides the attachment widgets automatically.
- **`#personas-mode-placeholder` is NOT reusable.** It is the fallback for retired/un-built modes only — the `else` branch of `_apply_mode` (`:1658-1665`) shows it via `_mode_placeholder_text(mode)` → `_MODE_PLACEHOLDER_BODY`/`_PLACEHOLDER_FALLBACK` (`:188-192`). Its "prompts" behavior ("moving to the Library") is pinned by `test_personas_workbench.py:434` and `:828`. Overloading it for "characters, no selection" would be semantically wrong and break those tests. → use a **dedicated widget**.
- **All no-selection paths funnel through `_show_center(None)`:** mode-enter (`:1646`), first mount (`:829`), restore-failure (`:816`, task-434), delete (`_delete_entity` `:5292`/`:5333`), and cancel-New (`_finish_cancel_edit` `:5726`). `_finish_cancel_edit` sets `_edit_mode="view"` **before** its `_show_center(None)` (`:5716`,`:5726`), so cancelling New with no prior selection lands on the empty state, not a stale editor.
- **Selection signal.** `state.selected_entity_id` (`personas_state.py:49-51`) is empty when nothing is selected. Task-434's `restore_state` populates it **pre-mount** from the saved dataclass, so it is reliable at `on_mount:829`. Selecting a character runs `_show_center("#ccp-character-card-view")` (`_select_character`); New/Import open the editor via `_show_center("#ccp-character-editor-view")` — both explicit ids, so they hide any guidance automatically.
- **Copy idiom.** Empty-state / descriptor copy on this screen lives in module constants (`_MODE_DESCRIPTORS:174`, `_MODE_PLACEHOLDER_BODY:188`, `_PLACEHOLDER_FALLBACK:191`) surfaced by small `_..._text(mode)` helpers. The library-rail empty row adapts on `_import_visible` (populated vs empty).

## Design

### 1. A dedicated guidance widget in the center stack

Add a `Static` to `#personas-detail-stack` (in `compose_content`, alongside the other detail views) and register it in `_CENTER_VIEW_IDS`:

```python
yield Static("", id="personas-characters-empty", markup=True)
```
```python
_CENTER_VIEW_IDS = ( ... existing ..., "#personas-characters-empty", )
```

`markup=True` so the three action words can be emphasised; the body is **static app copy only** (no user/entity interpolation), so there is no markup-injection surface.

### 2. Resolve the empty state once, inside `_show_center`

Rather than editing every `_show_center(None)` call site (mode-enter, mount, delete, cancel-New, restore-failure — and any future path), resolve the guidance at the single choke point. At the top of `_show_center`:

```python
def _show_center(self, visible_id: str | None) -> None:
    # Characters mode with nothing selected shows onboarding guidance, not a
    # blank pane (task-436). Every no-selection path funnels through here, so
    # resolving it once keeps mode-enter, first mount, delete, cancel-New and
    # restore-failure consistent; non-character modes and explicit-id calls
    # (card / editor) are unaffected, which makes AC#2 automatic.
    if visible_id is None and self._should_show_characters_empty_guidance():
        self.query_one("#personas-characters-empty", Static).update(
            self._characters_empty_guidance_text()
        )
        visible_id = "#personas-characters-empty"
    # ... existing loop over _CENTER_VIEW_IDS + attachment/dict-panel gates ...
```

with:

```python
def _should_show_characters_empty_guidance(self) -> bool:
    """True when the Characters center should show onboarding guidance."""
    return (
        self.state.active_mode == "characters"
        and not self.state.selected_entity_id
    )
```

Because `_should_show_characters_empty_guidance()` gates on `active_mode == "characters"`, the personas/dictionaries/lore `_show_center(None)` calls still blank the center (guidance is Characters-only per AC). Explicit-id calls never enter the branch, so a selected character card, the editor, and the unbuilt-mode placeholder are all unchanged.

### 3. Guidance copy (AC#1 — names the three next actions)

A single module constant + helper mirroring `_mode_placeholder_text`. **Non-adaptive on purpose:** `on_mount` shows the center (`:829`) *before* `refresh_character_list()` (`:830`) populates `self._character_total`, and nothing re-renders the center after the load — so a count-adaptive copy would render its "empty library" variant on first mount even when characters exist. One copy that reads correctly whether or not characters exist avoids that timing trap and still satisfies AC#1 in both states:

```python
_CHARACTERS_EMPTY_GUIDANCE = (
    "No character selected. Pick one from the list on the left, or use "
    "[b]New[/b] or [b]Import[/b] to add a character."
)
```
```python
def _characters_empty_guidance_text(self) -> str:
    """Onboarding guidance for the empty Characters center pane."""
    return _CHARACTERS_EMPTY_GUIDANCE
```

The copy names all three next actions — select a row, **New**, **Import**. When the library is empty ("Pick one from the list" is momentarily moot), the rail's own empty row ("No characters yet – use New or Import") supplies the same New/Import cue, so the two never conflict. A helper (rather than an inline literal) keeps the copy in one place and leaves room to make it count-adaptive later behind a proper post-load refresh, without changing `_show_center`'s call shape.

### 4. CSS — read as intentional guidance

Add a rule (in the screen's tcss / the same file that styles `#personas-detail-stack`) so the guidance is centered, muted, and padded rather than a stray top-left line:

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

### AC#2 — guidance disappears once a selection exists

No extra wiring: `_select_character` → `_show_center("#ccp-character-card-view")` and New/Import → `_show_center("#ccp-character-editor-view")` are explicit ids, so `_show_center` sets `#personas-characters-empty` `display=False`. Deleting the selected character (`_delete_entity`) clears the selection and calls `_show_center(None)` → guidance re-appears (still Characters mode, now no selection).

## Testing

Mirror the existing personas pilot harness (`Tests/UI/test_personas_workbench.py` / `_workbench_state.py`):

- **AC#1 populated:** mount in Characters mode with ≥1 character, no selection → `#personas-characters-empty` `display is True` and its renderable names "New" and "Import" (and references selecting from the list).
- **AC#1 empty library:** no characters → the same guidance still shows and names New/Import (no stale-variant / crash).
- **first-mount timing:** the guidance rendered at `on_mount` (before `refresh_character_list`) names New/Import correctly even though `_character_total` is not yet loaded (pins the reason the copy is non-adaptive).
- **AC#2 select hides:** select a character → `#personas-characters-empty` `display is False` and `#ccp-character-card-view` `display is True`.
- **mode isolation:** switch to Lore/Personas (no selection) → `#personas-characters-empty` `display is False` (guidance is Characters-only); switch back to Characters → it shows again.
- **delete re-shows:** with a selected character, delete it → guidance re-appears.
- **cancel-New re-shows:** open New (no prior selection), cancel the editor → guidance shows (not blank), `#ccp-character-editor-view` hidden.
- **regression:** the retired-"prompts" placeholder tests (`test_personas_workbench.py:434/828`) still pass (that widget is untouched); existing `#ccp-character-card-view` display assertions still hold. Run the personas suite and update any test that assumed a fully-blank Characters-empty center.

## Risks / mitigations

- **Overloading `_show_center` with state:** it already reads state to gate attachments; the added branch is guarded to `visible_id is None` and Characters-mode-no-selection only, and explicit-id calls (the common case) are untouched. A single choke point is safer than editing ~6 call sites and missing a future one.
- **First-mount count timing:** guidance renders at `on_mount:829` before the character list loads (`:830`); the copy is deliberately non-adaptive so it is correct regardless of the not-yet-known count (see §3).
- **task-434 restore:** the no-selection restore path (`:816`) already calls `_show_center(None)`; it now yields guidance in Characters mode — the intended behavior. Restore **with** a selection uses `_select_character` (explicit id) → unaffected.
- **markup:** body is static app copy; `markup=True` only emphasises literal action words, no injection.

## Non-goals

- Guidance for Personas/Dictionaries/Lore empty centers (Dictionaries/Lore already have try-it guidance; extending center guidance to all modes is a separate follow-up if wanted).
- Changing the inspector "Console blocked: select an item" copy (TASK-443's remit).
- Any change to selection, mode-switch, or restore semantics.
