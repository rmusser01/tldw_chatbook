# Roleplay P3c — Console Character Avatar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the active character's avatar + name persistently in a new collapsible "Character" section of the native Console left rail, following the session and reusing the existing image renderer.

**Architecture:** Read the active character off the live `ConsoleChatSession` (#754); render its `image` BLOB via the chat screen's existing `ConsoleImageRenderCache` + `resolve_default_mode` + the shared `fit_image_cell_size` (#775), off-thread; cache the decoded spec + refresh it only when `(character_id,)` changes, hooked into `_sync_native_console_chat_ui` (mirroring the dictionary "what's in play" pattern); a config-gated 5th rail section mounts it.

**Tech Stack:** Python 3.11+, Textual, PIL/rich-pixels/textual-image, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-22-roleplay-p3c-console-character-avatar-design.md` (committed `2fb0d0d81`).

## Global Constraints

- **NO ChaChaNotes migration** (v22) — reads existing `character_cards.image` + `conversations.character_id` (already written by #754).
- **Resolve the active character ONLY off the live `ConsoleChatSession`** (`_active_native_console_session().character_id`/`.character_name`); never read legacy `app.current_chat_*` reactives (split-brain).
- **Reuse `self._console_image_cache`** (via `_ensure_console_image_view()` — do NOT create a second cache), `resolve_default_mode`, and `fit_image_cell_size`; avatar decode runs **off-thread** (`asyncio.to_thread`).
- **Zero-DB-on-recompose:** the DB fetch/decode happens only in the scope-changed refresh; compose reads only the cache; **the cache holds a SPEC (data), not a live widget** (each mount builds a fresh widget). Re-check the `(character_id,)` scope AFTER the await to drop a stale render.
- **The refresh must NEVER raise into `_sync_native_console_chat_ui`** — wrap; failure → text/empty fallback.
- **Config-gated** on `[chat.images].show_character_avatar` (default **True**); the section is not composed when off.
- **Characters-only** (no persona image).
- **DO NOT modify `_active_console_dictionary_scope_ids`** — it is NOT inert (feeds the dictionary/world-book "what's in play" summarize via `character_id`); leave it `(conversation_id, None)`. Add a pin-test.
- **Rail adds ONLY** `character_open` (to `ConsoleRailPreferences` + `ConsoleRailState` dataclasses AND both construction sites in `console_rail_state.py`) + one compose block; the section toggle + requery are generic and auto-handle it.
- **Narrow-rail avatar box** sized to the rail (`CHARACTER_AVATAR_COLS = 16`, `CHARACTER_AVATAR_LINES = 8` — adjust at implementation if it overflows the rail); pixels is the safe fallback; don't rebuild the graphics widget every sync tick.
- **Branch off the LATEST `origin/dev`** (`dc21e3f04` at plan time — has #754 + #775; re-verify at merge). **Concurrent-session hazard:** `chat_screen.py` is heavily edited by other sessions — keep changes localized to the new accessor/section/refresh; expect a non-trivial rebase.
- **Implementers stage ONLY their task's files** — never `git add -A`, never `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** — scope to this worktree.
- **`Tests/UI/pytest.ini` sets `asyncio_mode = auto`** → keep async tests in `Tests/UI/` and do NOT mix `Tests/UI/` with another directory in ONE pytest invocation (rootdir shift drops auto-mode), OR add explicit `@pytest.mark.asyncio` per async test.
- **Test command** (prefix every run; run ONLY your task's files):
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

## File Structure

**Modified:**
- `tldw_chatbook/Chat/console_image_view.py` — add pure `resolve_show_character_avatar` (Task 1).
- `tldw_chatbook/Config.py` — document `show_character_avatar` in the `[chat.images]` block (Task 1).
- `tldw_chatbook/UI/Screens/chat_screen.py` — accessors (T1), the "Character" compose block (T2), the avatar cache + refresh + build (T3), the sync-tick wire (T4).
- `tldw_chatbook/Chat/console_rail_state.py` — `character_open` field (2 dataclasses + 2 construction sites) (Task 2).

**Tests (`Tests/UI/` — asyncio auto-mode):** grep `Tests/UI/` for the existing Console chat-screen harness (e.g. `ConsoleHarness` in `test_product_maturity_gate1_core_loop_screen_adaptation.py`, and the console-left-rail section tests) and mirror them for real-screen tests. Pure helpers can be tested under `Tests/Chat/`.

---

## Task 1: Config helper + resolution accessors

**Files:**
- Modify: `tldw_chatbook/Chat/console_image_view.py` (near `resolve_default_mode`)
- Modify: `tldw_chatbook/Config.py` (`[chat.images]` block, ~:2740-2754 — comment/doc only)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (near `_current_console_rail_conversation_id`, ~:3866)
- Test: `Tests/Chat/test_console_image_view.py` (config helper), `Tests/UI/test_console_character_avatar.py` (accessors)

**Interfaces:**
- Produces: `resolve_show_character_avatar(app_config) -> bool` (console_image_view.py); `PersonasScreen`… no — `ChatScreen._current_console_rail_character_id() -> int | None`, `ChatScreen._current_console_rail_character_name() -> str | None`.

- [ ] **Step 1: Write the failing config-helper test**

Append to `Tests/Chat/test_console_image_view.py` (import `resolve_show_character_avatar`):

```python
def test_resolve_show_character_avatar_defaults_true():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar
    assert resolve_show_character_avatar({}) is True
    assert resolve_show_character_avatar({"chat": {"images": {}}}) is True


def test_resolve_show_character_avatar_explicit_false():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar
    assert resolve_show_character_avatar(
        {"chat": {"images": {"show_character_avatar": False}}}
    ) is False


def test_resolve_show_character_avatar_live_shape():
    from tldw_chatbook.Chat.console_image_view import resolve_show_character_avatar
    assert resolve_show_character_avatar(
        {"COMPREHENSIVE_CONFIG_RAW": {"chat": {"images": {"show_character_avatar": False}}}}
    ) is False
```

- [ ] **Step 2: Run — verify it fails**

Run `Tests/Chat/test_console_image_view.py -k resolve_show_character_avatar`. Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the config helper**

In `console_image_view.py`, right after `resolve_default_mode` (or near `_chat_images_config`):

```python
def resolve_show_character_avatar(app_config: Mapping[str, Any]) -> bool:
    """Whether the Console shows the active character's avatar (default True).

    Reads ``[chat.images].show_character_avatar`` via the same both-shapes
    accessor as ``resolve_default_mode`` (raw TOML or the live
    ``COMPREHENSIVE_CONFIG_RAW`` nesting).
    """
    value = _chat_images_config(app_config).get("show_character_avatar", True)
    return bool(value)
```

In `Config.py`'s `[chat.images]` documentation block (~:2740-2754), add a `# show_character_avatar = true  # show the active character's avatar in the Console left rail` comment alongside the other keys (documentation only — the default lives in the resolver).

- [ ] **Step 4: Run — verify config-helper tests pass**

Run `Tests/Chat/test_console_image_view.py -k resolve_show_character_avatar`. Expected: PASS (3).

- [ ] **Step 5: Write the failing accessor test**

Create `Tests/UI/test_console_character_avatar.py`. Mirror the existing Console chat-screen harness (grep `Tests/UI` for `ConsoleHarness`; it builds a `ChatScreen` with a real store). Sketch:

```python
import pytest
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.asyncio


async def test_current_console_rail_character_id_reads_active_session(console_screen_with_character):
    # console_screen_with_character: a ChatScreen whose active native session has
    # character_id=7, character_name="Ada" (shape it to the real harness).
    screen = console_screen_with_character
    assert screen._current_console_rail_character_id() == 7
    assert screen._current_console_rail_character_name() == "Ada"


async def test_current_console_rail_character_id_none_for_generic_session(console_screen_generic):
    screen = console_screen_generic
    assert screen._current_console_rail_character_id() is None
    assert screen._current_console_rail_character_name() is None


async def test_p3c_leaves_dictionary_scope_ids_unchanged(console_screen_with_character):
    # Pin: P3c must NOT make _active_console_dictionary_scope_ids character-aware
    # (that would change the dictionary/world-book "what's in play" content).
    screen = console_screen_with_character
    conv_id, char_id = screen._active_console_dictionary_scope_ids()
    assert char_id is None
```

(Shape the two fixtures to the real harness: an active `ConsoleChatSession` with `character_id`/`character_name` set vs a generic one. If the harness makes constructing an active session awkward, set the fields on the store's active session directly.)

- [ ] **Step 6: Run — verify it fails**

Run `Tests/UI/test_console_character_avatar.py`. Expected: FAIL — `_current_console_rail_character_id` missing.

- [ ] **Step 7: Implement the accessors**

In `chat_screen.py`, right after `_current_console_rail_conversation_id` (~:3876), add (NO legacy fallback — resolve only off the live session):

```python
    def _current_console_rail_character_id(self) -> Optional[int]:
        """Active native Console session's character id (int), or None.

        Resolved ONLY off the live session (#754 sets it at Start-Chat, on
        DB-resume, and on screen-state restore); never from legacy
        ``app.current_chat_*`` reactives. None for a generic session.
        """
        native_session = self._active_native_console_session()
        if native_session is None:
            return None
        character_id = getattr(native_session, "character_id", None)
        try:
            return int(character_id) if character_id is not None else None
        except (TypeError, ValueError):
            return None

    def _current_console_rail_character_name(self) -> Optional[str]:
        """Active native Console session's character name, or None."""
        native_session = self._active_native_console_session()
        if native_session is None:
            return None
        name = getattr(native_session, "character_name", None)
        return str(name) if name else None
```

- [ ] **Step 8: Run — verify accessor tests pass**

Run `Tests/UI/test_console_character_avatar.py`. Expected: PASS (3, incl. the dictionary-scope pin).

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Chat/console_image_view.py tldw_chatbook/Config.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_image_view.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3c Task 1 — show_character_avatar config + active-character rail accessors"
```

---

## Task 2: The "Character" collapsible rail section

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py` (`ConsoleRailPreferences` ~:76, `ConsoleRailState` ~:106, raw-dict construction ~:244-247, preferences construction ~:519-522)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (compose_content left-rail body, after the Details section ~:8658)
- Test: `Tests/UI/test_console_character_avatar.py` (extend)

**Interfaces:**
- Consumes: `resolve_show_character_avatar` (T1), `_current_console_rail_character_id`/`_name` (T1), and (T3) `self._active_character_avatar` (spec | None) + `self._active_character_avatar_name` + `_build_character_avatar_widget(spec)`.
- Produces: the `#console-rail-section-body-character` section + `#console-character-avatar` container + `#console-character-name` Static; `ConsoleRailState.character_open`.

**Note:** the section is composed reading only the cache (`self._active_character_avatar`); T3 fills that cache. For this task, initialize `self._active_character_avatar = None` and `self._active_character_avatar_name = None` in `__init__` so the section composes an empty state, and add a temporary `_build_character_avatar_widget` stub that renders the empty/text state (T3 extends it to real images).

- [ ] **Step 1: Add `character_open` to the rail-state dataclasses + both construction sites**

In `console_rail_state.py`:
- `ConsoleRailPreferences` (after `agent_open` ~:76): `character_open: bool = True`
- `ConsoleRailState` (after `agent_open` ~:106): `character_open: bool = True`
- Raw-dict construction (~:244-247, after the `agent_open=` line): `character_open=_coerce_bool(raw.get("character_open"), defaults.character_open),`
- Preferences construction (~:519-522, after the `agent_open=` line): `character_open=preferences.character_open,`

- [ ] **Step 2: Write the failing rail-section test**

Extend `Tests/UI/test_console_character_avatar.py`:

```python
async def test_character_section_composes_when_config_on(console_screen_with_character):
    screen = console_screen_with_character  # config default → show_character_avatar True
    assert screen.query("#console-rail-section-body-character")   # section present
    assert screen.query("#console-character-name")


async def test_character_section_absent_when_config_off(console_screen_avatar_off):
    # console_screen_avatar_off: app_config has chat.images.show_character_avatar = False
    screen = console_screen_avatar_off
    assert not screen.query("#console-rail-section-body-character")


async def test_character_section_empty_state_for_generic_session(console_screen_generic):
    screen = console_screen_generic
    name = screen.query_one("#console-character-name")
    assert "No character" in str(name.renderable)  # empty-state copy
```

- [ ] **Step 3: Run — verify it fails**

Run `Tests/UI/test_console_character_avatar.py`. Expected: FAIL — no `#console-rail-section-body-character`.

- [ ] **Step 4: Add the cache fields + a build stub + the compose block**

In `chat_screen.py` `__init__` (near the other `_active_*_summary` fields, ~:1911): `self._active_character_avatar = None` (spec | None), `self._active_character_avatar_name = None`, `self._last_console_avatar_scope = None`.

Add the build helper (T3 will extend the image branches; for now it renders the text/empty state):

```python
    def _build_character_avatar_widget(self, spec) -> Widget:
        """Build a fresh avatar widget from the cached spec (data, not a widget).

        T3 fills `spec` with {character_id, name, mode, pil, pixels}. With no
        spec / no image, render a compact text placeholder.
        """
        from textual.widgets import Static as _S
        if not spec or (spec.get("pil") is None and spec.get("pixels") is None):
            hint = "no avatar" if spec else "No character in this chat"
            return _S(hint, id="console-character-avatar-empty")
        # T3 extends: graphics -> textual_image Image(pil) with fit dims;
        # pixels -> Static(Pixels). Placeholder until then:
        return _S("", id="console-character-avatar-empty")
```

In `compose_content`, after the Details section (~:8658), gated on config:

```python
                        # Section 5: Character (avatar of the active character).
                        if resolve_show_character_avatar(
                            getattr(getattr(self, "app_instance", None), "app_config", {}) or {}
                        ):
                            yield ConsoleRailSectionHeader(
                                "Character",
                                section_id="character",
                                open=rail_state.character_open,
                                id="console-rail-section-header-character",
                            )
                            character_body = Vertical(
                                id="console-rail-section-body-character",
                                classes="console-rail-section-body",
                            )
                            character_body.styles.height = "auto"
                            if not rail_state.character_open:
                                character_body.styles.display = "none"
                            with character_body:
                                avatar_holder = Container(id="console-character-avatar")
                                with avatar_holder:
                                    yield self._build_character_avatar_widget(
                                        self._active_character_avatar
                                    )
                                yield Static(
                                    self._active_character_avatar_name or "No character in this chat",
                                    id="console-character-name",
                                )
```

(Import `resolve_show_character_avatar` at the top of `chat_screen.py`; `Container`/`Static`/`Vertical`/`ConsoleRailSectionHeader` are already imported.)

- [ ] **Step 5: Run — verify it passes**

Run `Tests/UI/test_console_character_avatar.py`. Expected: PASS. Confirm the generic empty-state copy and the config-off absence. Also toggle the section via the existing rail toggle in a test if the harness supports it (the generic toggle handles `character` automatically).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_rail_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3c Task 2 — Character rail section (config-gated, empty state)"
```

---

## Task 3: Avatar cache + scope-guarded off-thread refresh + render

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (extend `_build_character_avatar_widget`; add `_refresh_active_character_avatar_if_scope_changed` + `_render_character_avatar_into_section`; constants)
- Test: `Tests/UI/test_console_character_avatar.py` (extend, real-screen + real-DB)

**Interfaces:**
- Consumes: `_current_console_rail_character_id` (T1); `_ensure_console_image_view()` → `(_, cache)`; `self._console_image_default_mode`; `fit_image_cell_size`; `get_character_card_by_id`.
- Produces: `_refresh_active_character_avatar_if_scope_changed()` (async), consumed by T4; the filled `self._active_character_avatar` spec.

- [ ] **Step 1: Write the failing refresh test (real screen + real DB)**

Extend `Tests/UI/test_console_character_avatar.py`. Seed a character WITH an image in the screen's real `CharactersRAGDB`, set the active session's `character_id` to it:

```python
async def test_refresh_populates_avatar_cache_and_mounts(console_screen_with_db):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")   # harness helper

    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar.get("character_id") == char_id
    assert screen._active_character_avatar.get("pil") is not None or \
           screen._active_character_avatar.get("pixels") is not None

    # unchanged scope → no re-fetch (spy the DB fetch)
    calls = []
    orig = screen._fetch_character_card_for_avatar   # the off-thread fetch wrapper
    screen._fetch_character_card_for_avatar = lambda cid: (calls.append(cid), orig(cid))[1]
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert calls == []   # scope guard short-circuits before any fetch


async def test_refresh_clears_avatar_for_generic_session(console_screen_with_db):
    app, screen, db = console_screen_with_db
    _set_active_console_character(screen, None, None)
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is None


async def test_refresh_never_raises_on_bad_image(console_screen_with_db):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad", "image": b"not-an-image"})
    _set_active_console_character(screen, char_id, "Bad")
    await screen._refresh_active_character_avatar_if_scope_changed()  # must not raise
    # decode failed → empty/text spec, name still set
    assert screen._active_character_avatar_name == "Bad"
```

- [ ] **Step 2: Run — verify it fails**

Expected: FAIL — `_refresh_active_character_avatar_if_scope_changed` missing.

- [ ] **Step 3: Add constants + the fetch wrapper + the refresh + extend the build**

Constants (module level, near the other Console constants):

```python
CHARACTER_AVATAR_COLS = 16
CHARACTER_AVATAR_LINES = 8
```

Fetch wrapper (synchronous, run off-thread; a seam the test can spy):

```python
    def _fetch_character_card_for_avatar(self, character_id: int):
        # Canonical DB accessor used throughout chat_screen.py (e.g. the resume
        # path `_resolve_resumed_character_name`); there is no `self.chachanotes_db`.
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return None
        try:
            return db.get_character_card_by_id(int(character_id))
        except Exception:
            logger.opt(exception=True).debug("avatar: character fetch failed")
            return None
```

(Verified: `getattr(self.app_instance, "chachanotes_db", None)` is the accessor used everywhere in `chat_screen.py`; the resume path mirrors this exact off-thread fetch.)

The refresh (mirror `_refresh_active_dictionaries_summary_if_scope_changed`, but decode off-thread and build a spec):

```python
    async def _refresh_active_character_avatar_if_scope_changed(self) -> None:
        """Refresh the cached character avatar only when the active character changed."""
        character_id = self._current_console_rail_character_id()
        scope = (character_id,)
        if scope == self._last_console_avatar_scope:
            return
        self._last_console_avatar_scope = scope
        name = self._current_console_rail_character_name()
        self._active_character_avatar_name = name
        if character_id is None:
            self._active_character_avatar = None
            await self._render_character_avatar_into_section()
            return
        _, cache = self._ensure_console_image_view()
        mode = getattr(self, "_console_image_default_mode", "pixels")
        key = f"character:{character_id}"
        spec = {"character_id": character_id, "name": name, "mode": mode, "pil": None, "pixels": None}
        try:
            card = await asyncio.to_thread(self._fetch_character_card_for_avatar, character_id)
            image = (card or {}).get("image")
            if isinstance(image, (bytes, bytearray)) and image:
                ok = await asyncio.to_thread(cache.prepare, key, bytes(image))
                if ok:
                    if mode == "graphics":
                        spec["pil"] = cache.get_pil(key)
                    else:
                        spec["pixels"] = cache.get_pixels(key)
        except Exception:
            logger.opt(exception=True).debug("avatar: refresh failed")
        # Drop a stale render if the active character changed during decode.
        if (self._current_console_rail_character_id(),) != scope or not self.is_mounted:
            return
        self._active_character_avatar = spec
        await self._render_character_avatar_into_section()

    async def _render_character_avatar_into_section(self) -> None:
        """Re-mount the avatar widget + name into the (already-composed) section.

        Async because Textual `Widget.mount()` returns an `AwaitMount` that
        must be awaited so the widget is present before the caller returns
        (the integration test asserts the mounted state right after the tick).
        """
        try:
            holder = self.query_one("#console-character-avatar", Container)
        except QueryError:
            return  # section not composed (config off / not mounted)
        await holder.remove_children()
        await holder.mount(self._build_character_avatar_widget(self._active_character_avatar))
        try:
            self.query_one("#console-character-name", Static).update(
                self._active_character_avatar_name or "No character in this chat"
            )
        except QueryError:
            pass
```

Extend `_build_character_avatar_widget` to mount real images (mirror `console_transcript._image_row_widget`):

```python
    def _build_character_avatar_widget(self, spec) -> Widget:
        if not spec or (spec.get("pil") is None and spec.get("pixels") is None):
            hint = "no avatar" if (spec and spec.get("character_id") is not None) else "No character in this chat"
            return Static(hint, id="console-character-avatar-empty")
        if spec.get("mode") == "graphics" and spec.get("pil") is not None:
            try:
                from textual_image.widget import Image as _GraphicsImage
                widget = _GraphicsImage(spec["pil"], id="console-character-avatar-image")
                w, h = fit_image_cell_size(spec["pil"].width, spec["pil"].height,
                                           CHARACTER_AVATAR_COLS, CHARACTER_AVATAR_LINES)
                widget.styles.width = w
                widget.styles.height = h
                return widget
            except Exception:
                logger.opt(exception=True).debug("avatar: graphics mount failed")
        pixels = spec.get("pixels")
        if pixels is None and spec.get("pil") is not None:
            scaled = spec["pil"].copy()
            scaled.thumbnail((CHARACTER_AVATAR_COLS, CHARACTER_AVATAR_LINES * 2))
            from rich_pixels import Pixels
            pixels = Pixels.from_image(scaled)
        widget = Static(pixels if pixels is not None else "", id="console-character-avatar-image")
        widget.styles.max_width = CHARACTER_AVATAR_COLS
        widget.styles.max_height = CHARACTER_AVATAR_LINES
        return widget
```

(Import `fit_image_cell_size` at the top; `Static`/`Container`/`QueryError`/`asyncio`/`logger` already imported. Confirm imports.)

- [ ] **Step 4: Run — verify it passes**

Run `Tests/UI/test_console_character_avatar.py`. Expected: PASS (incl. scope-guard no-refetch, generic-clear, bad-image never-raises).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3c Task 3 — avatar cache + scope-guarded off-thread refresh + render"
```

---

## Task 4: Wire into the sync tick + integration

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_sync_native_console_chat_ui` ~:10129)
- Test: `Tests/UI/test_console_character_avatar.py` (integration)

**Interfaces:**
- Consumes: `_refresh_active_character_avatar_if_scope_changed` (T3).

- [ ] **Step 1: Write the failing integration test**

```python
async def test_sync_tick_refreshes_avatar(console_screen_with_db):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    await screen._sync_native_console_chat_ui()   # the central sync entrypoint

    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar_name == "Ada"
    name = screen.query_one("#console-character-name")
    assert "Ada" in str(name.renderable)
```

- [ ] **Step 2: Run — verify it fails**

Expected: FAIL — the sync tick doesn't refresh the avatar yet.

- [ ] **Step 3: Wire the refresh into the sync tick**

In `_sync_native_console_chat_ui` (~:10129), after the dictionary/world-book refreshes, add:

```python
            await self._refresh_active_character_avatar_if_scope_changed()
```

(Match the surrounding `try`/guard structure — the refresh already never raises, but keep it inside whatever guard wraps the dictionary refreshes.)

- [ ] **Step 4: Run — verify it passes**

Run `Tests/UI/test_console_character_avatar.py`. Expected: PASS.

- [ ] **Step 5: Run the focused suite + app import**

Run `Tests/UI/test_console_character_avatar.py` + the existing Console rail/chat-screen test modules you touched or mirrored (name them explicitly — grep `Tests/UI` for console-left-rail / `ConsoleHarness` tests). Also `python -c "import tldw_chatbook.app"` (env prefix). Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_character_avatar.py
git commit -m "feat(console): P3c Task 4 — refresh avatar from the Console sync tick + integration"
```

---

## Self-Review Notes (author)

- **Spec coverage:** config helper (T1), accessors (T1), the dictionary-scope pin (T1), the rail section + `character_open` + empty state + config gate (T2), the cache-as-spec + scope-guarded off-thread refresh + never-raise + post-await re-check + narrow-rail fit (T3), the sync-tick wire + integration (T4) — all mapped. No migration; `_active_console_dictionary_scope_ids` untouched (pinned).
- **Type consistency:** `resolve_show_character_avatar` (T1) → used in T2 compose; `_current_console_rail_character_id`/`_name` (T1) → used in T2/T3; `self._active_character_avatar` (spec dict) + `_active_character_avatar_name` + `_last_console_avatar_scope` + `_build_character_avatar_widget` (T2 stub → T3 real) + `_refresh_active_character_avatar_if_scope_changed` + `_render_character_avatar_into_section` + `_fetch_character_card_for_avatar` (T3) → consumed by T4. `CHARACTER_AVATAR_COLS/LINES`, cache key `f"character:{id}"` consistent.
- **Plan-time confirmations (read fresh):** the Console DB accessor (`self.chachanotes_db` vs `self.app_instance.chachanotes_db`); the real Console chat-screen test harness + how to set an active session's `character_id` (`_set_active_console_character` helper); the exact `try`/guard wrapping the dictionary refreshes in `_sync_native_console_chat_ui`; whether the graphics avatar renders cleanly in the rail `VerticalScroll` (fall back to pixels if not); the narrow-rail box constants against the real rail width.
