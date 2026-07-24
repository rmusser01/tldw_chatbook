"""Screen-level tests for the P2a generation-variant actions (Task 8).

Covers the three new ``ChatScreen`` helpers browse/keep/regenerate-append
wire into (`_select_console_generation_variant`, `_keep_console_generation_
variant`, `_regenerate_console_generation_variant`) plus the button-id
dispatch routing in `handle_console_message_action` that picks the
generation-message branch over the text-sibling one.

Follows ``Tests/UI/test_console_native_chat_flow.py``'s ``_bare_console_
screen`` pattern (``ChatScreen.__new__(ChatScreen)``, bypassing ``__init__``)
plus ``Tests/Chat/test_console_generation_store.py``'s plain in-memory
``ConsoleChatStore`` fixture style -- no mounted Textual app is needed since
none of the exercised logic touches widgets directly; `_sync_native_console_
chat_ui`/`app_instance.notify` are stubbed, matching the brief's "mock store
+ fake generate" guidance.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImage
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_generate_image import BatchResult
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _meta(*, prompt: str = "a red dragon", backend: str = "swarmui", seed=42):
    return GenerationVariantMeta(
        prompt=prompt,
        negative_prompt="blurry",
        backend=backend,
        model=None,
        seed=seed,
        style=None,
        params={},
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    """Real (decodable) PNG bytes, distinct by solid color per variant."""
    buffer = BytesIO()
    PILImage.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _seed_generation_message(store: ConsoleChatStore, *, variant_count: int = 1):
    """Seed a generation message with ``variant_count`` distinct-bytes variants."""
    session = store.ensure_session(title="Chat 1")
    variants = [
        (f"img{index}".encode(), "image/png", _meta(seed=42 if index == 0 else -1))
        for index in range(variant_count)
    ]
    message = store.append_generation_message(
        session.id,
        content="[image] a red dragon",
        variants=variants,
        persist=False,
    )
    return session, message


def _bare_generation_screen(store: ConsoleChatStore) -> ChatScreen:
    """Build a ``ChatScreen`` shell wired for direct action-handler calls.

    Bypasses ``ChatScreen.__init__`` (no mounted Textual app needed) and
    stubs the two seams the new handlers touch that WOULD need one:
    ``app_instance.notify`` (recorded, never raises) and
    ``_sync_native_console_chat_ui`` (an ``AsyncMock`` no-op -- the real
    method walks the live render/inspector pipeline, irrelevant to the pure
    store-mutation logic under test here).
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_chat_store = store
    screen._console_message_action_service = ConsoleMessageActionService()
    screen._pending_console_delete_message_id = None
    screen.app_instance = SimpleNamespace(notify=lambda *a, **k: None)
    screen._sync_native_console_chat_ui = AsyncMock()
    return screen


def _fake_batch(*, calls: list, data: bytes = b"newimg") -> callable:
    """Return a fake ``run_generation_batch`` recording every call's kwargs."""

    def _run(*, backend, prompt, negative_prompt, seed, count, **_ignored):
        calls.append(
            {
                "backend": backend,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "count": count,
            }
        )
        meta = GenerationVariantMeta(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            backend=backend,
            model=None,
            seed=seed,
            style=None,
            params={},
        )
        return BatchResult(successes=[(data, "image/png", meta)], errors=[])

    return _run


def _failing_batch(*, calls: list) -> callable:
    def _run(*, backend, prompt, negative_prompt, seed, count, **_ignored):
        calls.append(True)
        return BatchResult(successes=[], errors=["backend unreachable"])

    return _run


# --- Browse: ephemeral, clamped, no store call --------------------------------


def test_browse_next_then_previous_mutates_screen_state_only():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    before_attachment_bytes = [a.data for a in store.get_message(message.id).attachments]

    screen._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1
    screen._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 2

    screen._select_console_generation_variant(message, direction="variant-previous")
    assert screen._generation_browse[message.id] == 1

    # Ephemeral: attachments (and their byte order) are untouched by browsing.
    after_attachment_bytes = [a.data for a in store.get_message(message.id).attachments]
    assert after_attachment_bytes == before_attachment_bytes


def test_browse_clamps_at_boundaries():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=2)
    screen = _bare_generation_screen(store)

    # Already at 0 -- "previous" is a no-op.
    screen._select_console_generation_variant(message, direction="variant-previous")
    assert screen._generation_browse.get(message.id, 0) == 0

    screen._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1
    # Already at the last index -- "next" is a no-op.
    screen._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1


def test_browse_noop_for_single_variant_message():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)

    screen._select_console_generation_variant(message, direction="variant-next")

    assert screen._console_generation_browse().get(message.id, 0) == 0


# --- Keep: durable reorder + browse reset --------------------------------------


def test_keep_reorders_store_and_resets_browse():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    screen._console_generation_browse()[message.id] = 2

    screen._keep_console_generation_variant(message)

    kept = store.get_message(message.id)
    assert kept.attachments[0].data == b"img2"
    assert screen._generation_browse[message.id] == 0


def test_keep_noop_when_browsed_index_is_zero():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    # Nothing set self._generation_browse[message.id] -- defaults to 0.

    screen._keep_console_generation_variant(message)

    untouched = store.get_message(message.id)
    assert untouched.attachments[0].data == b"img0"


def test_keep_evicts_stale_render_cache_entries_so_rebuild_shows_kept_variant():
    """Regression: keep swaps store bytes (position 0 <-> browsed position)
    but the render cache is keyed by composite ``f"{message_id}:{i}"`` and
    is never invalidated on its own -- and the prep path skips re-decoding
    whatever key is already cached. Repro this fixes: generate -> regenerate
    -> browse to variant 1 -> Keep -> the card kept showing the OLD
    canonical image (paired with the new details), self-healing only on
    reload or an unrelated LRU eviction.
    """
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    red = _png_bytes((200, 0, 0))
    green = _png_bytes((0, 200, 0))
    message = store.append_generation_message(
        session.id,
        content="[image] a red dragon",
        variants=[
            (red, "image/png", _meta(seed=42)),
            (green, "image/png", _meta(seed=-1)),
        ],
        persist=False,
    )
    screen = _bare_generation_screen(store)
    _state, cache = screen._ensure_console_image_view()
    # Simulate a reader who already browsed to variant 1 (and back), so BOTH
    # composite keys are decoded and cached before the keep happens.
    cache.prepare(f"{message.id}:0", red)
    cache.prepare(f"{message.id}:1", green)
    screen._console_generation_browse()[message.id] = 1

    screen._keep_console_generation_variant(message)

    # Store-level swap: position 0 is now the (formerly variant-1) green bytes.
    kept = store.get_message(message.id)
    assert kept.attachments[0].data == green
    assert screen._generation_browse[message.id] == 0

    # Render cache: neither composite key may still hand back a decoded
    # image -- the old cached PIL/pixels under BOTH keys must be gone,
    # otherwise the next spec build would resolve `f"{message.id}:0"` to the
    # stale (pre-keep) red canonical instead of re-decoding the swapped bytes.
    assert cache.get_pil(f"{message.id}:0") is None
    assert cache.get_pil(f"{message.id}:1") is None

    # The rebuilt card spec must not carry the stale pre-keep image: either
    # it's undecoded pending re-prep, or (once re-prepped below) it shows
    # the KEPT variant -- never the old red canonical.
    card_specs = screen._build_generation_card_specs([kept])
    spec = card_specs[message.id]
    assert spec.browsed_index == 0
    assert spec.pixels is None and spec.pil is None  # decoded=False, re-prep queued

    pending = screen._pending_console_generation_card_images([kept], card_specs)
    assert pending == [(f"{message.id}:0", green)]
    for cache_key, data in pending:
        cache.prepare(cache_key, data)

    rebuilt_specs = screen._build_generation_card_specs([kept])
    rebuilt = rebuilt_specs[message.id]
    assert rebuilt.pixels is not None or rebuilt.pil is not None  # decoded=True now
    # Pull the actual decoded PIL back out of the cache to inspect its color
    # (mode-agnostic -- works whether the session default is pixels/graphics).
    redecoded = cache.get_pil(f"{message.id}:0")
    assert redecoded is not None
    assert redecoded.getpixel((0, 0)) == (0, 200, 0)  # the KEPT (green) variant


# --- Regenerate: cap + in-flight refusal, failure/success paths ---------------


def test_regenerate_refused_at_cap(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=2)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        chat_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=2),
    )
    calls: list = []
    monkeypatch.setattr(
        chat_screen_module, "run_generation_batch", _fake_batch(calls=calls)
    )
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))

    import asyncio

    asyncio.run(screen._regenerate_console_generation_variant(message.id))

    assert calls == []  # generation never ran
    assert len(store.get_message(message.id).generation_metadata) == 2
    assert notifications and notifications[0][1].get("severity") == "warning"
    assert "maximum" in notifications[0][0].lower()


def test_regenerate_refused_while_inflight(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        chat_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        chat_screen_module, "run_generation_batch", _fake_batch(calls=calls)
    )
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))
    screen._console_imagegen_inflight_message_ids().add(message.id)

    import asyncio

    asyncio.run(screen._regenerate_console_generation_variant(message.id))

    assert calls == []
    assert len(store.get_message(message.id).generation_metadata) == 1
    assert notifications and notifications[0][1].get("severity") == "warning"
    assert "already running" in notifications[0][0].lower()


def test_regenerate_failure_leaves_message_untouched_and_reports_error(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        chat_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        chat_screen_module, "run_generation_batch", _failing_batch(calls=calls)
    )
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))

    import asyncio

    asyncio.run(screen._regenerate_console_generation_variant(message.id))

    assert len(calls) == 1
    untouched = store.get_message(message.id)
    assert len(untouched.generation_metadata) == 1
    assert untouched.attachments[0].data == b"img0"
    assert notifications and notifications[0][1].get("severity") == "error"
    assert message.id not in screen._console_imagegen_inflight_message_ids()


def test_regenerate_success_appends_variant_and_browses_to_new_index(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        chat_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        chat_screen_module,
        "run_generation_batch",
        _fake_batch(calls=calls, data=b"appended"),
    )

    import asyncio

    asyncio.run(screen._regenerate_console_generation_variant(message.id))

    appended = store.get_message(message.id)
    assert len(appended.generation_metadata) == 2
    assert appended.attachments[1].data == b"appended"
    assert screen._generation_browse[message.id] == 1
    assert message.id not in screen._console_imagegen_inflight_message_ids()
    screen._sync_native_console_chat_ui.assert_awaited()
    # Rebuilds the request from position 0's meta (same backend/prompt/
    # negative) but forces seed=-1 regardless of the canonical variant's
    # own seed (42) -- the identical-image guard from spec §4.
    assert len(calls) == 1
    assert calls[0]["backend"] == "swarmui"
    assert calls[0]["prompt"] == "a red dragon"
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["seed"] == -1
    assert calls[0]["count"] == 1


# --- Full dispatch routing through handle_console_message_action --------------


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_keep_button_for_generation_message():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    screen._generation_browse = {message.id: 2}
    button = Button("keep", id=f"console-message-action-keep-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    kept = store.get_message(message.id)
    assert kept.attachments[0].data == b"img2"
    assert screen._generation_browse[message.id] == 0


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_variant_next_for_generation_message():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=2)
    screen = _bare_generation_screen(store)
    button = Button("next", id=f"console-message-action-variant-next-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert screen._generation_browse[message.id] == 1
    # No store mutation from browsing.
    assert store.get_message(message.id).attachments[0].data == b"img0"


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_regenerate_for_generation_message(
    monkeypatch,
):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        chat_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    monkeypatch.setattr(
        chat_screen_module, "run_generation_batch", _fake_batch(calls=[])
    )
    button = Button("regen", id=f"console-message-action-regenerate-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    appended = store.get_message(message.id)
    assert len(appended.generation_metadata) == 2
