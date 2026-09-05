"""Controller-level tests for the P2a generation-variant actions (Task 8).

Covers the three ``ConsoleImageController`` browse/keep/regenerate-append
actions plus the button-id dispatch routing in
``handle_console_message_action`` that picks the generation-message branch
over the text-sibling one.

Follows ``Tests/UI/test_console_native_chat_flow.py``'s ``_bare_console_
screen`` pattern (``ChatScreen.__new__(ChatScreen)``, bypassing ``__init__``)
plus ``Tests/Chat/test_console_generation_store.py``'s plain in-memory
``ConsoleChatStore`` fixture style -- no mounted Textual app is needed since
none of the exercised logic touches widgets directly; `_sync_native_console_
chat_ui`/`app_instance.notify` are stubbed, matching the brief's "mock store
+ fake generate" guidance.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImage
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.Chat.console_generate_image import (
    BatchResult,
    LLMContextOptions,
    reset_llm_context_executor,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech import ConsoleSpeechSnapshotRejected
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSEventHandler,
    TTSMessageSpeechRequestEvent,
    TTSPlaybackEvent,
    TTSPlaybackLifecycle,
)
from tldw_chatbook.UI.Console_Modules import image as image_module
from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from Tests.UI.console_controller_stubs import (
    NO_APP,
    stub_fleet_controller,
    stub_library_activity_controller,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.fixture(autouse=True)
def _reset_llm_context_executor_state():
    """Same rationale as `test_console_generate_image.py`'s fixture of the
    same name: the shared single-worker LLM-context executor (Qodo PR #867
    fix) is process-wide module state -- reset around every test so this
    file's own slow/timeout fake calls can never bleed a spurious
    saturation failure into an unrelated later test (in this file, another
    test file in the same session, or vice versa)."""
    reset_llm_context_executor()
    yield
    reset_llm_context_executor()


def _meta(
    *,
    prompt: str = "a red dragon",
    backend: str = "swarmui",
    seed=42,
    style: str | None = None,
):
    return GenerationVariantMeta(
        prompt=prompt,
        negative_prompt="blurry",
        backend=backend,
        model=None,
        seed=seed,
        style=style,
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
    stubs the seams the new handlers touch that WOULD need one:
    ``app_instance.notify`` (recorded, never raises) and
    ``_sync_native_console_chat_ui`` (an ``AsyncMock`` no-op -- the real
    method walks the live render/inspector pipeline, irrelevant to the pure
    store-mutation logic under test here).

    ``handle_console_message_action`` moved to ``ConsoleMessageController``
    (wave-3 console decomposition, task 1); ``screen._message`` is built the
    same way ``screen._session`` already was here -- bypassing ITS
    ``__init__`` too, with only the constructor callables this file's
    generation-branch scenarios (regenerate/keep/variant/speak/speak-stop)
    can actually reach wired for real. The four callables reaching
    clusters these tests never touch (change-review, the transcript-sync
    timer, native message selection, the conversation-browser cache) are
    stubbed to raise -- a fail-loud guard if a future test in this file
    starts exercising a branch that needs one for real, instead of a
    silently-wrong no-op.
    """
    screen = ChatScreen.__new__(ChatScreen)
    from tldw_chatbook.UI.Console_Modules.wiring import (
        build_console_commands_controller,
    )

    build_console_commands_controller(screen)
    # Must precede the `_console_chat_store` assignment below: that is a
    # property whose setter reaches `ConsoleRuntime.attach_view` ->
    # `ChatScreen.console_view_hooks`, which reads
    # `self._fleet._console_wake_user_priority` (TASK-21381) and
    # `self._library_activity.build_provider` (TASK-23144). Without either
    # controller the shell dies during SETUP with an AttributeError naming an
    # attribute this file never mentions.
    stub_fleet_controller(screen, context="_bare_generation_screen")
    # These scenarios never send a raw-CLI action.  The wrapper still checks
    # that action family first, so make the deliberately absent controller
    # explicit for this detached shell.
    screen._raw_cli = None
    stub_library_activity_controller(
        screen,
        context="_bare_generation_screen",
        # `app_instance` is assigned two lines further down; the
        # generation branches touch no library-activity seam.
        app_instance=NO_APP,
    )
    screen._console_chat_store = store
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
    screen._session._chat_store_accessor = lambda: screen._console_chat_store
    screen._session._current_chat_store_accessor = lambda: screen._console_chat_store
    screen.app_instance = SimpleNamespace(notify=lambda *a, **k: None)
    screen._sync_native_console_chat_ui = AsyncMock()
    # `_clear_console_composer_draft` now also syncs the slash-command popup;
    # on a detached screen (no `_nodes`) that query dies with AttributeError,
    # not the guarded QueryError — same class of seam as the sync stub above.
    screen._sync_console_command_popup = lambda: None
    screen._image = ConsoleImageController(
        screen,
        app_instance=screen.app_instance,
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        recent_console_image_messages=(
            lambda messages: screen._recent_console_image_messages(messages)
        ),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        console_generation_browse=lambda: screen._console_generation_browse(),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        ensure_console_chat_store=lambda: screen._ensure_console_chat_store(),
        build_console_provider_selection=(
            lambda: screen._build_console_provider_selection()
        ),
        ensure_console_provider_gateway=(
            lambda: screen._ensure_console_provider_gateway()
        ),
        console_image_preparing=(
            lambda: getattr(screen, "_console_image_preparing", None)
        ),
        current_console_chat_store=lambda: screen._console_chat_store,
        console_composer_or_none=lambda: screen._console_composer_or_none(),
        console_visible_draft_session_id=(
            lambda: screen._console_visible_draft_session_id
        ),
        append_native_console_system_message=(
            lambda *args, **kwargs: screen._append_native_console_system_message(
                *args, **kwargs
            )
        ),
        request_console_control_bar_sync=(
            lambda: screen._request_console_control_bar_sync()
        ),
        default_console_session_settings=(
            lambda: screen._session._default_console_session_settings()
        ),
        clear_console_composer_draft=(lambda: screen._clear_console_composer_draft()),
    )

    def _unreached(*_args, **_kwargs):
        raise AssertionError(
            "_bare_generation_screen: this constructor callable is not "
            "wired for real -- the scenario reaching it needs its own stub."
        )

    screen._message = ConsoleMessageController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: screen._console_chat_store,
        current_chat_store_accessor=lambda: screen._console_chat_store,
        ensure_console_chat_controller=_unreached,
        current_chat_controller_accessor=lambda: None,
        sync_native_console_chat_ui=screen._sync_native_console_chat_ui,
        active_session_is_ephemeral=(
            lambda: screen._session._console_active_session_is_ephemeral()
        ),
        active_native_console_session=_unreached,
        current_console_conversation_id=_unreached,
        active_console_provider_model_display=_unreached,
        console_initial_session_title_for_workspace=lambda workspace_id: "",
        console_change_review_run_id=_unreached,
        open_change_review=_unreached,
        start_console_transcript_sync_timer=_unreached,
        clear_native_console_message_selection=_unreached,
        regenerate_console_generation_variant=(
            lambda message_id: screen._image._regenerate_console_generation_variant(
                message_id
            )
        ),
        select_console_generation_variant=(
            lambda message, direction: screen._image._select_console_generation_variant(
                message, direction=direction
            )
        ),
        keep_console_generation_variant=(
            lambda message: screen._image._keep_console_generation_variant(message)
        ),
        handle_console_toggle_image_view=_unreached,
        invalidate_console_persisted_rows_cache=_unreached,
    )
    return screen


def _fake_batch(*, calls: list, data: bytes = b"newimg") -> callable:
    """Return a fake ``run_generation_batch`` recording every call's kwargs.

    Mirrors the real function's meta construction closely enough for
    style-threading assertions: whatever ``style_name`` the caller passes
    in lands on the returned variant's ``GenerationVariantMeta.style``.
    """

    def _run(
        *, backend, prompt, negative_prompt, seed, count, style_name=None, **_ignored
    ):
        calls.append(
            {
                "backend": backend,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "count": count,
                "style_name": style_name,
            }
        )
        meta = GenerationVariantMeta(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            backend=backend,
            model=None,
            seed=seed,
            style=style_name,
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
    before_attachment_bytes = [
        a.data for a in store.get_message(message.id).attachments
    ]

    screen._image._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1
    screen._image._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 2

    screen._image._select_console_generation_variant(
        message, direction="variant-previous"
    )
    assert screen._generation_browse[message.id] == 1

    # Ephemeral: attachments (and their byte order) are untouched by browsing.
    after_attachment_bytes = [a.data for a in store.get_message(message.id).attachments]
    assert after_attachment_bytes == before_attachment_bytes


def test_browse_clamps_at_boundaries():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=2)
    screen = _bare_generation_screen(store)

    # Already at 0 -- "previous" is a no-op.
    screen._image._select_console_generation_variant(
        message, direction="variant-previous"
    )
    assert screen._generation_browse.get(message.id, 0) == 0

    screen._image._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1
    # Already at the last index -- "next" is a no-op.
    screen._image._select_console_generation_variant(message, direction="variant-next")
    assert screen._generation_browse[message.id] == 1


def test_browse_noop_for_single_variant_message():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)

    screen._image._select_console_generation_variant(message, direction="variant-next")

    assert screen._console_generation_browse().get(message.id, 0) == 0


# --- Keep: durable reorder + browse reset --------------------------------------


def test_keep_reorders_store_and_resets_browse():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    screen._console_generation_browse()[message.id] = 2

    screen._image._keep_console_generation_variant(message)

    kept = store.get_message(message.id)
    assert kept.attachments[0].data == b"img2"
    assert screen._generation_browse[message.id] == 0


def test_keep_noop_when_browsed_index_is_zero():
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=3)
    screen = _bare_generation_screen(store)
    # Nothing set self._generation_browse[message.id] -- defaults to 0.

    screen._image._keep_console_generation_variant(message)

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

    screen._image._keep_console_generation_variant(message)

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
    card_specs = screen._image._build_generation_card_specs([kept])
    spec = card_specs[message.id]
    assert spec.browsed_index == 0
    assert spec.pixels is None and spec.pil is None  # decoded=False, re-prep queued

    pending = screen._image._pending_console_generation_card_images([kept], card_specs)
    assert pending == [(f"{message.id}:0", green)]
    for cache_key, data in pending:
        cache.prepare(cache_key, data)

    rebuilt_specs = screen._image._build_generation_card_specs([kept])
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
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=2),
    )
    calls: list = []
    monkeypatch.setattr(image_module, "run_generation_batch", _fake_batch(calls=calls))
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))

    import asyncio

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    assert calls == []  # generation never ran
    assert len(store.get_message(message.id).generation_metadata) == 2
    assert notifications and notifications[0][1].get("severity") == "warning"
    assert "maximum" in notifications[0][0].lower()


def test_regenerate_refused_while_inflight(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(image_module, "run_generation_batch", _fake_batch(calls=calls))
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))
    screen._image._console_imagegen_inflight_message_ids().add(message.id)

    import asyncio

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    assert calls == []
    assert len(store.get_message(message.id).generation_metadata) == 1
    assert notifications and notifications[0][1].get("severity") == "warning"
    assert "already running" in notifications[0][0].lower()


def test_h3_edit_regenerate_refuses_before_capacity_or_inflight_checks(monkeypatch):
    store = ConsoleChatStore()
    session = store.ensure_session(title="H3 edit")
    message = store.append_generation_message(
        session.id,
        content="[image] edited result",
        variants=[
            (
                b"edited",
                "image/png",
                GenerationVariantMeta(
                    prompt="private edit instruction",
                    negative_prompt="",
                    backend="comfyui",
                    model=None,
                    seed=1,
                    style=None,
                    params={"operation": "edit"},
                ),
            )
        ],
    )
    screen = _bare_generation_screen(store)
    notices: list[tuple[str, str | None]] = []
    screen.app_instance.notify = lambda text, severity=None: notices.append(
        (text, severity)
    )
    screen._image._console_imagegen_inflight_message_ids = lambda: (
        _ for _ in ()
    ).throw(AssertionError("edit refusal must precede the in-flight gate"))
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("edit refusal must precede the capacity gate")
        ),
    )
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("edit Regenerate must not dispatch")
        ),
    )

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    assert notices == [
        (
            "Image edits cannot be regenerated. Restage the source image and run "
            "/generate-image :comfyui again.",
            "warning",
        )
    ]
    assert len(store.get_message(message.id).generation_metadata) == 1


def test_regenerate_failure_leaves_message_untouched_and_reports_error(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        image_module, "run_generation_batch", _failing_batch(calls=calls)
    )
    notifications: list = []
    screen.app_instance.notify = lambda text, **kw: notifications.append((text, kw))

    import asyncio

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    assert len(calls) == 1
    untouched = store.get_message(message.id)
    assert len(untouched.generation_metadata) == 1
    assert untouched.attachments[0].data == b"img0"
    assert notifications and notifications[0][1].get("severity") == "error"
    assert message.id not in screen._image._console_imagegen_inflight_message_ids()


def test_regenerate_success_appends_variant_and_browses_to_new_index(monkeypatch):
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        _fake_batch(calls=calls, data=b"appended"),
    )

    import asyncio

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    appended = store.get_message(message.id)
    assert len(appended.generation_metadata) == 2
    assert appended.attachments[1].data == b"appended"
    assert screen._generation_browse[message.id] == 1
    assert message.id not in screen._image._console_imagegen_inflight_message_ids()
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


def test_regenerate_success_inherits_style_from_position_zero_meta(monkeypatch):
    """P2b pin: the appended variant carries the canonical (position-0)
    variant's style, not None.

    Regression coverage for a real bug found while writing this pin:
    ``_regenerate_console_generation_variant`` rebuilds its request from
    position-0's backend/prompt/negative but was dropping ``style``,
    silently downgrading every regenerated card's "Style" field back to
    "Custom" even when the canonical variant carried a named ``@style``.
    """
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    styled_meta = _meta(seed=42, style="Anime Style")
    message = store.append_generation_message(
        session.id,
        content="[image] a red dragon",
        variants=[(b"img0", "image/png", styled_meta)],
        persist=False,
    )
    screen = _bare_generation_screen(store)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    calls: list = []
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        _fake_batch(calls=calls, data=b"appended"),
    )

    import asyncio

    asyncio.run(screen._image._regenerate_console_generation_variant(message.id))

    assert len(calls) == 1
    assert calls[0]["style_name"] == "Anime Style"
    appended = store.get_message(message.id)
    assert len(appended.generation_metadata) == 2
    assert appended.generation_metadata[1].style == "Anime Style"


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
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    monkeypatch.setattr(image_module, "run_generation_batch", _fake_batch(calls=[]))
    button = Button("regen", id=f"console-message-action-regenerate-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    appended = store.get_message(message.id)
    assert len(appended.generation_metadata) == 2


@pytest.mark.asyncio
async def test_handle_console_message_action_blocks_generation_regenerate_when_ephemeral(
    monkeypatch,
):
    """F5 follow-up (task-9 review): the generation-message branch of
    "regenerate" (the circular-arrow button on an image message) calls
    `run_generation_batch` directly -- a FOURTH door onto the same
    disk-writing sink that /generate-image's dispatch gate does not cover,
    since this path never goes through `_dispatch_console_command`. Must
    refuse the same way, with the same registry reason -- and still work
    normally otherwise (the control, which is the pre-existing test just
    above)."""
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    store = ConsoleChatStore()
    temp = store.create_session(title="Temp", ephemeral=True)
    store.switch_session(temp.id)
    message = store.append_generation_message(
        temp.id,
        content="[image] a red dragon",
        variants=[(b"img0", "image/png", _meta(seed=42))],
        persist=False,
    )
    screen = _bare_generation_screen(store)
    notified: list = []
    screen.app_instance.notify = lambda text, **kwargs: notified.append((text, kwargs))
    batch_calls: list = []
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(max_variants_per_message=8),
    )
    monkeypatch.setattr(
        image_module, "run_generation_batch", _fake_batch(calls=batch_calls)
    )
    button = Button("regen", id=f"console-message-action-regenerate-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert batch_calls == [], "must never reach run_generation_batch when ephemeral"
    assert len(store.get_message(message.id).generation_metadata) == 1, (
        "message must be untouched"
    )
    assert notified == [
        (blocked_reason("generate-image", ephemeral=True), {"severity": "warning"})
    ]


# --- TASK-1: speak (TTS) action dispatch --------------------------------------


@pytest.mark.asyncio
async def test_handle_console_message_action_posts_store_issued_speech_snapshot():
    """Speak posts a store-issued snapshot and its bound validator."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="The sky is blue today.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    posted: list = []
    screen.app_instance.post_message = posted.append
    button = Button("speak", id=f"console-message-action-speak-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert len(posted) == 1
    event = posted[0]
    assert isinstance(event, TTSMessageSpeechRequestEvent)
    assert event.snapshot.raw_content == "The sky is blue today."
    assert event.message_id == message.id
    assert not hasattr(event, "text")
    assert event.validator(event.snapshot) == "The sky is blue today."

    store.update_message_content(message.id, "Changed after click.")
    with pytest.raises(ConsoleSpeechSnapshotRejected):
        event.validator(event.snapshot)


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_speak_for_generation_message():
    """Spec §1a: speak also works for a generation-card message, reading
    its ``[image] ...`` marker text."""
    store = ConsoleChatStore()
    _session, message = _seed_generation_message(store, variant_count=1)
    screen = _bare_generation_screen(store)
    posted: list = []
    screen.app_instance.post_message = posted.append
    button = Button("speak", id=f"console-message-action-speak-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert len(posted) == 1
    assert isinstance(posted[0], TTSMessageSpeechRequestEvent)
    assert posted[0].snapshot.raw_content == "[image] a red dragon"
    assert posted[0].message_id == message.id


@pytest.mark.asyncio
async def test_handle_console_message_action_does_not_forge_user_speech_snapshot():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="User-authored text.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    posted: list = []
    notified: list[tuple[tuple, dict]] = []
    screen.app_instance.post_message = posted.append
    screen.app_instance.notify = lambda *args, **kwargs: notified.append((args, kwargs))
    button = Button("speak", id=f"console-message-action-speak-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert posted == []
    assert getattr(screen, "_console_speaking_message_id", None) is None
    screen._sync_native_console_chat_ui.assert_not_awaited()
    assert notified == [
        (
            ("Message changed before speech started; select Speak again.",),
            {"severity": "warning"},
        )
    ]


# --- task-559 unit 2: Console TTS stop toggle dispatch ------------------------


@pytest.mark.asyncio
async def test_handle_console_message_action_speak_marks_message_as_speaking():
    """Dispatching speak records the message id as the screen's ephemeral
    "currently speaking" state and re-syncs the transcript so the action row
    picks up the 🔊 -> ⏹ swap (task-559 unit 2)."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="The sky is blue today.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    screen.app_instance.post_message = lambda *a, **k: None
    button = Button("speak", id=f"console-message-action-speak-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert screen._console_speaking_message_id == message.id
    screen._sync_native_console_chat_ui.assert_awaited()


@pytest.mark.asyncio
async def test_handle_console_message_action_routes_speak_stop_to_tts_playback_event():
    """speak-stop posts the app's existing TTSPlaybackEvent(action="stop")
    -- reuses the legacy stop-button plumbing, no new audio machinery --
    clears the screen's speaking-message tracking so the row swaps back,
    AND (since the screen genuinely believed this message was speaking)
    gives honest "Stopped speaking." feedback."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="The sky is blue today.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    screen._console_speaking_message_id = message.id
    posted: list = []
    screen.app_instance.post_message = posted.append
    notified: list = []
    screen.app_instance.notify = lambda *a, **k: notified.append((a, k))
    button = Button("stop", id=f"console-message-action-speak-stop-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert len(posted) == 1
    event = posted[0]
    assert isinstance(event, TTSPlaybackEvent)
    assert event.action == "stop"
    assert event.message_id == message.id
    event.report_outcome(True)
    assert screen._console_speaking_message_id is None
    await asyncio.sleep(0)
    assert len(notified) == 1
    assert notified[0][0][0] == "Stopped speaking."


@pytest.mark.asyncio
async def test_handle_console_message_action_speak_stop_safe_when_nothing_speaking():
    """speak-stop is a genuinely-idle no-op (fix round 1) when the screen
    never marked any message as speaking -- e.g. a stale/late button press
    after the screen's own state already cleared, or a directly-crafted
    button id bypassing the UI gate that would normally hide this button.
    Still safe to post the stop event (the app-level handler already
    no-ops harmlessly for real) -- but must NOT claim "Stopped speaking."
    for nothing, and must not force an unnecessary transcript re-sync."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="The sky is blue today.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    posted: list = []
    screen.app_instance.post_message = posted.append
    notified: list = []
    screen.app_instance.notify = lambda *a, **k: notified.append((a, k))
    button = Button("stop", id=f"console-message-action-speak-stop-{message.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert len(posted) == 1
    assert posted[0].action == "stop"
    assert getattr(screen, "_console_speaking_message_id", None) is None
    assert notified == []
    screen._sync_native_console_chat_ui.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_console_message_action_speak_stop_does_not_clear_other_message():
    """Stopping message A must not clear message B's tracked speaking
    state -- only an exact id match clears it -- and, since the screen
    never believed message A itself was speaking, no "Stopped speaking."
    feedback is given for A either."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message_a = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="A", persist=False
    )
    message_b = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="B", persist=False
    )
    screen = _bare_generation_screen(store)
    screen._console_speaking_message_id = message_b.id
    screen.app_instance.post_message = lambda *a, **k: None
    notified: list = []
    screen.app_instance.notify = lambda *a, **k: notified.append((a, k))
    button = Button("stop", id=f"console-message-action-speak-stop-{message_a.id}")

    handled = await screen.handle_console_message_action(Button.Pressed(button))

    assert handled is True
    assert screen._console_speaking_message_id == message_b.id
    assert notified == []


@pytest.mark.asyncio
async def test_failed_speech_clears_on_any_next_message_action():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Copy me.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    screen._console_speech_states[message.id] = "failed"
    copied: list[str] = []
    screen.app_instance.copy_to_clipboard = copied.append
    button = Button("copy", id=f"console-message-action-copy-{message.id}")

    assert await screen.handle_console_message_action(Button.Pressed(button)) is True

    assert copied == ["Copy me."]
    assert screen._console_speech_states == {}


@pytest.mark.asyncio
async def test_rejected_stop_post_does_not_claim_stopped():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Speaking.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    screen._console_speaking_message_id = message.id
    screen._console_speech_states[message.id] = "playing"
    screen.app_instance.post_message = lambda *_args, **_kwargs: False
    notified: list[tuple[tuple, dict]] = []
    screen.app_instance.notify = lambda *args, **kwargs: notified.append((args, kwargs))
    button = Button("stop", id=f"console-message-action-speak-stop-{message.id}")

    assert await screen.handle_console_message_action(Button.Pressed(button)) is True

    assert screen._console_speech_states[message.id] == "failed"
    assert screen._console_speaking_message_id is None
    assert not any(args and args[0] == "Stopped speaking." for args, _ in notified)


@pytest.mark.asyncio
async def test_real_handler_stop_order_settles_and_notifies_once(monkeypatch):
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Speaking.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    generation = screen._message._begin_console_speech_presentation(message.id)
    lifecycle = TTSPlaybackLifecycle(
        message_id=message.id,
        request_id=generation,
        validator=lambda: True,
        callback=lambda state: screen._message._settle_console_speech_presentation(
            message.id,
            generation,
            state=state,
        ),
    )
    screen._message._console_speech_owner = lifecycle
    assert lifecycle.report("playing") is True
    handler = TTSEventHandler()
    handler._active_stream_playback_owner = lifecycle
    screen.app_instance.control_tts_playback = handler.handle_tts_playback
    notified: list[tuple[tuple, dict]] = []
    screen.app_instance.notify = lambda *args, **kwargs: notified.append((args, kwargs))
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.stop_live_sink",
        lambda: None,
    )
    button = Button("stop", id=f"console-message-action-speak-stop-{message.id}")

    assert await screen.handle_console_message_action(Button.Pressed(button)) is True

    assert lifecycle.state == "stopped"
    assert screen._console_speech_states[message.id] == "stopped"
    assert screen._console_speaking_message_id is None
    assert [args[0] for args, _kwargs in notified] == ["Stopped speaking."]


@pytest.mark.asyncio
async def test_rejected_owned_stop_retains_lifecycle_for_retry():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Speaking.",
        persist=False,
    )
    screen = _bare_generation_screen(store)
    generation = screen._message._begin_console_speech_presentation(message.id)
    lifecycle = TTSPlaybackLifecycle(
        message_id=message.id,
        request_id=generation,
        validator=lambda: True,
        callback=lambda state: screen._message._settle_console_speech_presentation(
            message.id,
            generation,
            state=state,
        ),
    )
    screen._message._console_speech_owner = lifecycle
    lifecycle.report("playing")

    async def reject_stop(event: TTSPlaybackEvent) -> None:
        event.report_outcome(False)

    screen.app_instance.control_tts_playback = reject_stop
    button = Button("stop", id=f"console-message-action-speak-stop-{message.id}")

    assert await screen.handle_console_message_action(Button.Pressed(button)) is True

    assert lifecycle.state == "playing"
    assert screen._message._console_speech_owner is lifecycle
    assert screen._console_speaking_message_id == message.id
    assert screen._console_speech_states[message.id] == "failed"
    assert screen._message._console_speech_request_generation == generation


# --- F5 (task-9 review): /generate-image dispatch gate in a temporary chat --


@pytest.mark.asyncio
async def test_dispatch_console_command_blocks_generate_image_when_ephemeral():
    """F5: typing /generate-image was not gated -- only the composer-menu
    entry that BUILDS the command was. Gate the command at its actual
    dispatcher (`_dispatch_console_command`) so every path that reaches a
    /generate-image draft (typed by hand, pasted from the Generate Image
    modal, or inserted via the style browser) refuses the same way, with
    the same registry reason -- and still dispatches normally otherwise
    (the control)."""
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    store = ConsoleChatStore()
    temp = store.create_session(title="Temp", ephemeral=True)
    store.switch_session(temp.id)

    screen = _bare_generation_screen(store)
    # `_bare_generation_screen` bypasses `ChatScreen.__init__` (`__new__`, no
    # mounted app), so `screen._workspace` -- the real `ConsoleWorkspace
    # Controller` -- was never constructed. Stub the one method this path
    # reaches, the same narrow-seam discipline the helper's own docstring
    # describes for `_sync_native_console_chat_ui`/`app_instance.notify`.
    screen._workspace = SimpleNamespace(
        _console_initial_session_title_for_workspace=lambda workspace_id: "Console"
    )

    handler_calls: list = []

    async def _spy_handler(parse):
        handler_calls.append(parse)

    screen._console_command_generate_image = _spy_handler

    parse = CommandParse(kind="command", name="generate-image", args="a dragon")
    await screen._dispatch_console_command(parse)

    assert handler_calls == [], "the handler must not run in a temporary chat"
    messages = [m.content for m in store.messages_for_session(temp.id)]
    assert messages == [blocked_reason("generate-image", ephemeral=True)]

    # Control: a normal (non-ephemeral) session still dispatches.
    normal = store.create_session(title="Normal")
    store.switch_session(normal.id)
    await screen._dispatch_console_command(parse)

    assert len(handler_calls) == 1
    assert handler_calls[0] is parse


# --- Task 3: handler-level wiring test for /generate-image style threading ----


@pytest.mark.asyncio
async def test_generate_image_handler_threads_prepared_fields_into_batch(monkeypatch):
    """Handler-level test of _console_command_generate_image: verifies that
    parse -> PreparedGeneration -> run_generation_batch receives the correct
    kwargs with prompt properly composed from style template, and that the
    appended message contains the expected content marker.

    Assembles a bare ChatScreen with minimum stubs following the precedent
    from test_console_generation_actions.py, invokes the handler with a
    /generate-image @anime "a dragon" parse, monkeypatches run_generation_batch
    to capture its kwargs, and asserts the captured call plus the store's
    appended message.
    """
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Media_Creation.generation_templates import get_template

    store = ConsoleChatStore()
    screen = _bare_generation_screen(store)

    # Stub session creation and settings
    def _mock_default_settings():
        return ConsoleSessionSettings(provider="openai")

    screen._session._default_console_session_settings = _mock_default_settings

    # Stub conversation pairs helper to return a simple test pair
    def _mock_conversation_pairs(store, session_id):
        return [("user", "a red dragon by a lake")]

    screen._image._console_generate_image_conversation_pairs = _mock_conversation_pairs

    # Stub status line and composer helpers
    def _mock_composer_or_none():
        return None  # No composer draft to save/restore

    def _mock_clear_draft():
        pass

    screen._console_composer_or_none = _mock_composer_or_none
    screen._clear_console_composer_draft = _mock_clear_draft

    # Stub in-flight tracking
    def _mock_inflight_sessions():
        return set()

    screen._image._console_imagegen_inflight_sessions = _mock_inflight_sessions

    # Capture the batch call and store appends
    captured_kwargs = []
    appended_messages: list = []

    def _mock_batch(**kwargs):
        captured_kwargs.append(kwargs)
        # Return successful batch with one image
        meta = GenerationVariantMeta(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt") or "",
            backend=kwargs["backend"],
            model=None,
            seed=None,
            style=kwargs.get("style_name"),
            params={},
        )
        return BatchResult(successes=[(b"generated_img", "image/png", meta)], errors=[])

    original_append = store.append_generation_message

    def _capture_append(session_id, **kwargs):
        appended_messages.append(kwargs)
        return original_append(session_id, **kwargs)

    store.append_generation_message = _capture_append

    # Monkeypatch run_generation_batch and config
    monkeypatch.setattr(image_module, "run_generation_batch", _mock_batch)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(
            default_backend="swarmui",
            default_batch=1,
            max_variants_per_message=8,
        ),
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [
            {"name": "swarmui", "is_configured": True},
        ],
    )

    # Build the command parse: /generate-image @anime a dragon
    parse = CommandParse(kind="command", name="generate-image", args="@anime a dragon")

    # Invoke the handler
    await screen._console_command_generate_image(parse)

    # Verify run_generation_batch was called once with correct kwargs
    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    template = get_template("style_anime")

    # Prompt should be composed from template + user prompt "a dragon"
    assert "a dragon" in kwargs["prompt"]
    assert "anime style" in kwargs["prompt"]

    # Style name should be "Anime Style"
    assert kwargs["style_name"] == "Anime Style"

    # Width/height/steps/cfg_scale should come from template defaults
    assert kwargs["width"] == template.default_params["width"]  # 768
    assert kwargs["height"] == template.default_params["height"]  # 1024
    assert kwargs["steps"] == template.default_params["steps"]  # 25
    assert kwargs["cfg_scale"] == template.default_params["cfg_scale"]  # 9.0

    # Backend should be the default
    assert kwargs["backend"] == "swarmui"

    # Count should be clamped
    assert kwargs["count"] == 1

    # Verify that append_generation_message was called with correct content marker
    assert len(appended_messages) == 1
    append_kwargs = appended_messages[0]
    assert append_kwargs["content"].startswith("[image] ")
    assert "a dragon" in append_kwargs["content"]


# ---------------------------------------------------------------------------
# Task-559 AC1: LLM-composed conversation-context prompt
# ---------------------------------------------------------------------------


def _imagegen_cfg(**overrides) -> SimpleNamespace:
    """A `get_image_generation_config()` stand-in carrying every field the
    handler + `_console_generate_image_llm_context_options` read."""
    defaults = dict(
        default_backend="swarmui",
        default_batch=1,
        max_variants_per_message=8,
        context_llm_enabled=True,
        context_llm_turns=10,
        context_llm_timeout_seconds=15.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


# --- _console_generate_image_llm_context_options (isolated) ----------------


@pytest.mark.asyncio
async def test_llm_context_options_disabled_by_kill_switch():
    screen = _bare_generation_screen(ConsoleChatStore())
    cfg = _imagegen_cfg(context_llm_enabled=False)
    options = await screen._image._console_generate_image_llm_context_options(cfg)
    assert options.enabled is False
    assert options.provider_ready is False


@pytest.mark.asyncio
async def test_llm_context_options_resolves_ready_provider():
    screen = _bare_generation_screen(ConsoleChatStore())
    screen._build_console_provider_selection = lambda: "selection-sentinel"

    class _FakeGateway:
        async def resolve_for_send(self, selection):
            assert selection == "selection-sentinel"
            return SimpleNamespace(
                ready=True,
                execution_key="openai",
                model="gpt-4o-mini",
                api_key="fake-test-api-key",
            )

    screen._ensure_console_provider_gateway = lambda: _FakeGateway()
    cfg = _imagegen_cfg(context_llm_turns=7, context_llm_timeout_seconds=9.5)
    options = await screen._image._console_generate_image_llm_context_options(cfg)
    assert options.enabled is True
    assert options.provider_ready is True
    assert options.api_endpoint == "openai"
    assert options.model == "gpt-4o-mini"
    assert options.api_key == "fake-test-api-key"
    assert options.turns == 7
    assert options.timeout_seconds == 9.5


@pytest.mark.asyncio
async def test_llm_context_options_provider_not_ready():
    screen = _bare_generation_screen(ConsoleChatStore())
    screen._build_console_provider_selection = lambda: "selection-sentinel"

    class _FakeGateway:
        async def resolve_for_send(self, selection):
            return SimpleNamespace(
                ready=False, execution_key="", model=None, api_key=None
            )

    screen._ensure_console_provider_gateway = lambda: _FakeGateway()
    options = await screen._image._console_generate_image_llm_context_options(
        _imagegen_cfg()
    )
    assert options.enabled is True
    assert options.provider_ready is False


@pytest.mark.asyncio
async def test_llm_context_options_resolution_exception_degrades_gracefully():
    screen = _bare_generation_screen(ConsoleChatStore())

    def _raise():
        raise RuntimeError("selection blew up")

    screen._build_console_provider_selection = _raise
    options = await screen._image._console_generate_image_llm_context_options(
        _imagegen_cfg()
    )
    assert options.enabled is True
    assert options.provider_ready is False


# --- _console_command_generate_image: handler-level wiring ------------------


def _wired_generate_image_screen(store, *, batch_calls, batch_data=b"generated_img"):
    """A `_bare_generation_screen` plus every stub the no-prompt dispatch
    path needs, matching `test_generate_image_handler_threads_prepared_
    fields_into_batch`'s established stubbing style."""
    screen = _bare_generation_screen(store)
    screen._session._default_console_session_settings = lambda: ConsoleSessionSettings(
        provider="openai"
    )
    screen._console_composer_or_none = lambda: None
    screen._clear_console_composer_draft = lambda: None
    screen._image._console_imagegen_inflight_sessions = lambda: set()

    def _mock_batch(**kwargs):
        batch_calls.append(kwargs)
        meta = GenerationVariantMeta(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt") or "",
            backend=kwargs["backend"],
            model=None,
            seed=None,
            style=kwargs.get("style_name"),
            params={},
        )
        return BatchResult(successes=[(batch_data, "image/png", meta)], errors=[])

    return screen, _mock_batch


@pytest.mark.asyncio
async def test_generate_image_handler_no_prompt_uses_llm_composed_context_end_to_end(
    monkeypatch,
):
    """Happy path: a mocked chat_api_call composition reaches BOTH the
    generation request (run_generation_batch's prompt kwarg) and the
    card-visible content marker."""
    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: [
            ("user", "A knight enters a glowing cave."),
            ("assistant", "Crystals shimmer along the walls."),
        ]
    )

    llm_text = "A knight in a glowing crystal cave, dramatic torchlight."

    async def _fake_llm_context_options(cfg):
        assert cfg.context_llm_enabled is True
        return LLMContextOptions(
            enabled=True,
            turns=cfg.context_llm_turns,
            timeout_seconds=cfg.context_llm_timeout_seconds,
            provider_ready=True,
            api_endpoint="openai",
            model="gpt-4o-mini",
            api_key="fake-test-api-key",
            chat_call=lambda **_kwargs: _chat_response(llm_text),
        )

    screen._image._console_generate_image_llm_context_options = (
        _fake_llm_context_options
    )

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module, "get_image_generation_config", lambda: _imagegen_cfg()
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="")
    await screen._console_command_generate_image(parse)

    assert len(batch_calls) == 1
    prompt = batch_calls[0]["prompt"]
    assert llm_text in prompt

    messages = store.messages_for_session(
        store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=screen._session._default_console_session_settings(),
        ).id
    )
    generation_messages = [m for m in messages if m.content.startswith("[image] ")]
    assert len(generation_messages) == 1
    assert llm_text in generation_messages[0].content


@pytest.mark.asyncio
async def test_generate_image_handler_no_prompt_llm_call_raises_falls_back(
    monkeypatch,
):
    """chat_api_call raising -> keyword-extractor result used, generation
    still dispatches, no exception escapes the handler."""
    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    conversation = [("user", "a quiet lakeside cabin at dawn")]
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: conversation
    )

    async def _fake_llm_context_options(cfg):
        return LLMContextOptions(
            enabled=True,
            turns=cfg.context_llm_turns,
            timeout_seconds=cfg.context_llm_timeout_seconds,
            provider_ready=True,
            api_endpoint="openai",
            model="gpt-4o-mini",
            api_key="fake-test-api-key",
            chat_call=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("provider unreachable")
            ),
        )

    screen._image._console_generate_image_llm_context_options = (
        _fake_llm_context_options
    )

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module, "get_image_generation_config", lambda: _imagegen_cfg()
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="")
    await screen._console_command_generate_image(parse)  # must not raise

    assert len(batch_calls) == 1
    assert "a quiet lakeside cabin at dawn" in batch_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_generate_image_handler_no_prompt_llm_timeout_falls_back(monkeypatch):
    """A slow chat_api_call past the configured timeout -> keyword result
    used, generation still dispatches."""
    import time as time_module

    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    conversation = [("user", "a quiet lakeside cabin at dawn")]
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: conversation
    )

    def _slow_call(**_kwargs):
        time_module.sleep(0.3)
        return _chat_response("too late")

    async def _fake_llm_context_options(cfg):
        return LLMContextOptions(
            enabled=True,
            turns=cfg.context_llm_turns,
            timeout_seconds=0.02,
            provider_ready=True,
            api_endpoint="openai",
            model="gpt-4o-mini",
            api_key="fake-test-api-key",
            chat_call=_slow_call,
        )

    screen._image._console_generate_image_llm_context_options = (
        _fake_llm_context_options
    )

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module, "get_image_generation_config", lambda: _imagegen_cfg()
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="")
    await screen._console_command_generate_image(parse)  # must not raise/hang

    assert len(batch_calls) == 1
    assert "a quiet lakeside cabin at dawn" in batch_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_generate_image_handler_no_prompt_llm_empty_response_falls_back(
    monkeypatch,
):
    """An empty/unusable LLM response -> keyword result used, no crash."""
    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    conversation = [("user", "a quiet lakeside cabin at dawn")]
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: conversation
    )

    async def _fake_llm_context_options(cfg):
        return LLMContextOptions(
            enabled=True,
            turns=cfg.context_llm_turns,
            timeout_seconds=cfg.context_llm_timeout_seconds,
            provider_ready=True,
            api_endpoint="openai",
            model="gpt-4o-mini",
            api_key="fake-test-api-key",
            chat_call=lambda **_kwargs: _chat_response("   "),
        )

    screen._image._console_generate_image_llm_context_options = (
        _fake_llm_context_options
    )

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module, "get_image_generation_config", lambda: _imagegen_cfg()
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="")
    await screen._console_command_generate_image(parse)

    assert len(batch_calls) == 1
    assert "a quiet lakeside cabin at dawn" in batch_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_generate_image_handler_no_prompt_kill_switch_off_skips_llm_path(
    monkeypatch,
):
    """context_llm_enabled=False end-to-end through the REAL
    `_console_generate_image_llm_context_options` (not stubbed) -- the
    keyword extractor's result is used and generation still dispatches."""
    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    conversation = [("user", "a quiet lakeside cabin at dawn")]
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: conversation
    )
    # Intentionally NOT stubbing _console_generate_image_llm_context_options --
    # the kill switch must short-circuit before any provider resolution is
    # attempted, so the real method (which would otherwise need
    # _build_console_provider_selection/_ensure_console_provider_gateway) is
    # safe to call as-is here.

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: _imagegen_cfg(context_llm_enabled=False),
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="")
    await screen._console_command_generate_image(parse)

    assert len(batch_calls) == 1
    assert "a quiet lakeside cabin at dawn" in batch_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_generate_image_handler_prompt_present_never_resolves_llm_context(
    monkeypatch,
):
    """When a prompt IS given, the LLM-context resolution seam is never
    consulted at all -- it's a no-prompt-only path."""
    store = ConsoleChatStore()
    batch_calls: list = []
    screen, mock_batch = _wired_generate_image_screen(store, batch_calls=batch_calls)
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: []
    )

    async def _must_not_be_called(cfg):
        raise AssertionError(
            "LLM context resolution must not run when a prompt was given"
        )

    screen._image._console_generate_image_llm_context_options = _must_not_be_called

    monkeypatch.setattr(image_module, "run_generation_batch", mock_batch)
    monkeypatch.setattr(
        image_module, "get_image_generation_config", lambda: _imagegen_cfg()
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    parse = CommandParse(kind="command", name="generate-image", args="a red dragon")
    await screen._console_command_generate_image(parse)

    assert len(batch_calls) == 1
    assert batch_calls[0]["prompt"] == "a red dragon"


class _FakeComposer:
    """Minimal composer double: draft_text()/clear_draft()/insert_text_as_paste()."""

    def __init__(self, text: str = ""):
        self._text = text

    def draft_text(self) -> str:
        return self._text

    def clear_draft(self) -> None:
        self._text = ""

    def insert_text_as_paste(self, text: str) -> None:
        self._text = text


@pytest.mark.asyncio
async def test_generate_image_handler_restores_draft_when_batch_raises(monkeypatch):
    """task-558: today only the zero-success RETURN path restores the
    composer draft; when ``run_generation_batch`` RAISES instead (e.g. an
    unexpected adapter bug, not a caught-and-reported ``ImageGenerationError``
    per-variant failure), the draft stayed cleared and the user's typed
    prompt was lost with no way to recover it. This pins the fix: an
    ``except`` around the batch call restores the draft the same way the
    zero-success path does, reports the error, and the in-flight guard is
    still released via ``finally``."""
    store = ConsoleChatStore()
    screen = _bare_generation_screen(store)

    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    screen._session._default_console_session_settings = lambda: ConsoleSessionSettings(
        provider="openai"
    )
    screen._image._console_generate_image_conversation_pairs = (
        lambda store, session_id: []
    )

    composer = _FakeComposer("a red dragon that got typed before the crash")
    screen._console_composer_or_none = lambda: composer

    system_messages: list[str] = []

    async def _fake_append_system(text: str, *, session_id: str | None = None) -> None:
        # Task 4 (background-write audit): the real handler now threads
        # `session_id=session.id` explicitly on every append in this path
        # (the batch's owning session, captured before the raising call) --
        # this fake must accept the same keyword the real method does.
        system_messages.append(text)

    screen._append_native_console_system_message = _fake_append_system

    monkeypatch.setattr(
        image_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(
            default_backend="swarmui", default_batch=1, max_variants_per_message=8
        ),
    )
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "swarmui", "is_configured": True}],
    )

    def _raising_batch(**kwargs):
        raise RuntimeError("adapter exploded unexpectedly")

    monkeypatch.setattr(image_module, "run_generation_batch", _raising_batch)

    from tldw_chatbook.Chat.console_command_grammar import CommandParse

    parse = CommandParse(kind="command", name="generate-image", args="a red dragon")

    await screen._console_command_generate_image(parse)

    # The draft the user had typed before dispatch must come back.
    assert composer.draft_text() == "a red dragon that got typed before the crash"
    # The failure must be reported, not silently swallowed.
    assert any("adapter exploded unexpectedly" in msg for msg in system_messages)
    # The in-flight guard must still be released -- a crashed batch must
    # never wedge the session against further /generate-image commands.
    assert (
        store.ensure_session().id
        not in screen._image._console_imagegen_inflight_sessions()
    )
    # No orphan generation message from a batch that never produced a result.
    assert store.messages_for_session(store.ensure_session().id) == []
