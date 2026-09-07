"""The Character avatar's PIL work must not run on the Textual event loop.

TASK-22221 (holistic perf review of dev ``a71e62e4b``, finding 22221). Every
distinct Console rail viewport size starts a geometry epoch
(``_ContextOuterBody.on_resize`` -> ``note_character_avatar_viewport_size``),
which is one memo miss in ``_reconcile_character_avatar_geometry``, which was
two synchronous PIL resamples on the event loop:

* ``fit_character_avatar_cell_box`` -> ``scale_image_for_cell_box``: a
  ``copy()`` + LANCZOS ``thumbnail`` whose PIXELS ARE THEN DISCARDED -- only
  ``.width``/``.height`` are read. Measured 0.73 ms median for a 1024px card;
  a 37-step drag paid 27.9 ms of pure waste on the loop.
* the widget build's mosaic render (the pixels the user actually sees):
  6.29 ms median, 187.2 ms across the same drag.

These are permanent gates for both halves, plus the race class the second half
introduces: a resample that finishes AFTER the viewport moved again must never
paint a stale-size avatar.
"""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO

import pytest
from PIL import Image as PILImage

from Tests.UI.test_console_character_avatar import (  # noqa: F401 -- pytest fixtures
    _set_active_console_character,
    avatar_db,
    console_screen_with_db_and_pilot,
)
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.Widgets.Console.console_image_viewer_modal import ClickableAvatarBox

#: A high-resolution character card. The Console render cache caps decoded
#: avatars at IMAGE_DECODE_MAX_DIMENSION (1024), so this is the production
#: worst case, not a synthetic one.
CARD_SIZE = (1024, 1024)


def _card_bytes(size: tuple[int, int] = CARD_SIZE) -> bytes:
    output = BytesIO()
    # Noise, not a flat fill: LANCZOS on a constant image is not representative.
    PILImage.effect_noise(size, 64).convert("RGB").save(output, format="PNG")
    return output.getvalue()


async def _mount_card(screen, db, pilot, size: tuple[int, int] = CARD_SIZE):
    character_id = db.add_character_card(
        {"name": "A roleplay character", "image": _card_bytes(size)}
    )
    _set_active_console_character(screen, character_id, "A roleplay character")
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    await pilot.pause()
    await pilot.pause()
    return screen.query_one("#console-left-rail", ConsoleLeftRail)


# ---------------------------------------------------------------------------
# 1. The fit leg: cell geometry must cost no resample at all
# ---------------------------------------------------------------------------


def test_avatar_cell_fit_never_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole drag's worth of fits must not copy or resample the card.

    Pre-fix this counted 37 LANCZOS thumbnails of a 1024x1024 image, every one
    of them on the event loop and every one of them discarded.
    """

    from tldw_chatbook.UI.Console_Modules.character_avatar_layout import (
        fit_character_avatar_cell_box,
    )

    image = PILImage.effect_noise(CARD_SIZE, 64).convert("RGB")
    resamples: list[str] = []

    for name in ("resize", "thumbnail", "copy", "reduce"):
        original = getattr(PILImage.Image, name)

        def counting(self, *args, _name=name, _original=original, **kwargs):
            resamples.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(PILImage.Image, name, counting)

    boxes = [fit_character_avatar_cell_box(image, cols, 30) for cols in range(24, 61)]

    assert resamples == [], resamples
    # The fit still does its job: monotonic, contained, aspect-true.
    assert boxes[0] != boxes[-1]
    for (cols, lines), box_cols in zip(boxes, range(24, 61)):
        assert 0 < cols <= box_cols
        assert 0 < lines <= 30


def test_prerender_resolves_the_same_box_the_widget_builder_resolves() -> None:
    """Both sides of the thread hop must land on one box, or the token is dead.

    The prerender runs in a worker thread and the widget builder runs on the
    loop; each applies ``fit_character_avatar_cell_box`` exactly once to the
    rail's target box. They must therefore agree for every box -- note that
    the fit is NOT idempotent (a 640x480 card fits (46,30) to (46,17), and
    (46,17) to (45,17)), so "apply it the same number of times" is the
    invariant, not "apply it until stable".
    """

    from tldw_chatbook.UI.Console_Modules.character_avatar_layout import (
        fit_character_avatar_cell_box,
        prerender_character_avatar,
    )

    for size in ((1024, 1024), (300, 1200), (1200, 300), (640, 480)):
        image = PILImage.new("RGB", size)
        for cols in range(4, 61, 7):
            target = fit_character_avatar_cell_box(image, cols, 30)
            if target == (0, 0):
                continue
            token = prerender_character_avatar(image, target, False)
            builder_box = fit_character_avatar_cell_box(image, *target)
            assert token is not None
            assert token.box == builder_box, (size, cols)
            assert token.matches(image, builder_box, False)
            # A different image, box, or colour mode must NOT match.
            assert not token.matches(PILImage.new("RGB", size), builder_box, False)
            assert not token.matches(image, (builder_box[0] + 1, builder_box[1]), False)
            assert not token.matches(image, builder_box, True)


# ---------------------------------------------------------------------------
# 2. The render leg: the visible mosaic must be built off the loop thread
# ---------------------------------------------------------------------------


def _watch_mosaic_thread(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the thread name every mosaic render runs on."""

    from tldw_chatbook.Utils import mosaic_render

    threads: list[str] = []
    original = mosaic_render.mosaic_from_image

    def watched(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(mosaic_render, "mosaic_from_image", watched)
    return threads


@pytest.mark.asyncio
async def test_geometry_replacement_renders_the_avatar_off_the_loop(
    console_screen_with_db_and_pilot,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A viewport-driven avatar replacement must not resample on the loop."""

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)
    holder = screen.query_one("#console-character-avatar", ClickableAvatarBox)
    assert holder.is_mounted

    threads = _watch_mosaic_thread(monkeypatch)
    loop_thread = threading.current_thread().name

    # One genuine viewport epoch, exactly as a resize drag step produces.
    rail.invalidate_character_avatar_geometry()
    rail._character_avatar_box = None
    rail.request_allocation_reconcile()
    for _ in range(10):
        await pilot.pause()

    assert threads, "the replacement must actually render the avatar"
    assert loop_thread not in threads, threads
    assert all(name != "MainThread" for name in threads), threads


@pytest.mark.asyncio
async def test_avatar_still_paints_the_fitted_portrait_after_the_move(
    console_screen_with_db_and_pilot,  # noqa: F811
) -> None:
    """Off-loop rendering must not cost the user the portrait itself."""

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)

    rail.invalidate_character_avatar_geometry()
    rail._character_avatar_box = None
    rail.request_allocation_reconcile()
    for _ in range(10):
        await pilot.pause()

    image = screen.query_one("#console-character-avatar-image")
    box = rail.character_avatar_box
    assert box is not None and box != (0, 0)
    # The mounted widget carries the fitted box, and a real portrait: the
    # coloured mosaic paints the image in BACKGROUND colour, so `str()` of it
    # is all spaces -- the pixels live in the segment styles.
    assert image.styles.width.value == box[0]
    assert image.styles.height.value == box[1]
    rendered = image.renderable
    backgrounds = {
        str(style.bgcolor)
        for _text, style, _control in rendered.render(screen.app.console)
        if style is not None and style.bgcolor is not None
    }
    assert len(backgrounds) > 1, backgrounds


# ---------------------------------------------------------------------------
# 3. The race class off-loop work introduces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resample_finishing_after_the_viewport_moved_paints_nothing(
    console_screen_with_db_and_pilot,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-size render completing late must never reach the DOM.

    Holds one prerender in the worker thread, bumps the geometry generation
    underneath it (what a further drag step does), then releases it. The
    generation fence must drop the finished-but-stale result.
    """

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)
    settled_box = rail.character_avatar_box
    assert settled_box is not None

    from tldw_chatbook.UI.Console_Modules import character_avatar_layout

    entered = threading.Event()
    release = threading.Event()
    original = character_avatar_layout.prerender_character_avatar

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        character_avatar_layout, "prerender_character_avatar", blocking
    )

    stale_box = (max(1, settled_box[0] - 3), max(1, settled_box[1] - 3))
    monkeypatch.setattr(rail, "_character_avatar_fit_box", lambda _c, _l: stale_box)
    rail.invalidate_character_avatar_geometry()
    rail._reconcile_character_avatar_geometry()

    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5.0)
    assert entered.is_set(), "the prerender must run off the loop"

    # The viewport moves again while that render is still in the thread.
    rail._character_avatar_fit_generation += 1
    release.set()
    for _ in range(10):
        await pilot.pause()

    image = screen.query_one("#console-character-avatar-image")
    assert (image.styles.width.value, image.styles.height.value) != stale_box, (
        "a stale-size avatar was painted after the viewport moved"
    )


@pytest.mark.asyncio
async def test_a_superseded_pass_does_not_blank_the_live_avatar(
    console_screen_with_db_and_pilot,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass that loses the race must leave the mounted portrait alone.

    Without the caller-side fence the doomed pass still enters the mount
    lock, unmounts the live avatar, and only THEN declines to mount its
    stale replacement -- so the Character section goes blank until the
    superseding pass lands. The user sees a portrait flicker per drag step.
    """

    _app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)
    holder = screen.query_one("#console-character-avatar", ClickableAvatarBox)
    assert holder.children, "the probe needs a mounted avatar to protect"

    from tldw_chatbook.UI.Console_Modules import character_avatar_layout

    entered = threading.Event()
    release = threading.Event()
    original = character_avatar_layout.prerender_character_avatar

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        character_avatar_layout, "prerender_character_avatar", blocking
    )

    box = rail.character_avatar_box
    assert box is not None
    other_box = (max(1, box[0] - 2), max(1, box[1] - 1))
    monkeypatch.setattr(rail, "_character_avatar_fit_box", lambda _c, _l: other_box)
    rail.invalidate_character_avatar_geometry()
    rail._reconcile_character_avatar_geometry()
    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5.0)
    assert entered.is_set()

    # Supersede it, then let the doomed render finish.
    rail._character_avatar_fit_generation += 1
    release.set()
    for _ in range(10):
        await pilot.pause()

    assert holder.children, "the superseded pass blanked the live avatar"


@pytest.mark.asyncio
async def test_teardown_with_a_resample_in_flight_is_clean(
    console_screen_with_db_and_pilot,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmounting mid-render must not fail a worker or raise into the app."""

    app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)

    from tldw_chatbook.UI.Console_Modules import character_avatar_layout

    entered = threading.Event()
    release = threading.Event()
    original = character_avatar_layout.prerender_character_avatar

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        character_avatar_layout, "prerender_character_avatar", blocking
    )

    failures: list[object] = []
    original_handler = app._handle_exception

    def recording(error):
        failures.append(error)
        return original_handler(error)

    monkeypatch.setattr(app, "_handle_exception", recording)

    rail.invalidate_character_avatar_geometry()
    rail._character_avatar_box = None
    rail._reconcile_character_avatar_geometry()
    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5.0)
    assert entered.is_set()

    # Teardown STARTS while the resample is still running in its thread, and
    # the render only lands afterwards.
    removal = asyncio.ensure_future(rail.remove())
    await pilot.pause()
    release.set()
    await asyncio.wait_for(removal, timeout=5)
    for _ in range(8):
        await pilot.pause()

    assert failures == [], failures
    # `is_mounted` stays True on a removed widget in Textual 8.2.8 -- DOM
    # membership is the honest signal (and is what the mount fence queries).
    assert rail not in screen.query("*").nodes
    # The worker unwound as cancelled/complete, never as a failure.
    assert all(
        worker.state.name != "ERROR"
        for worker in screen.workers
        if worker.group == "console-character-avatar-fit"
    ), [(w.name, w.state) for w in screen.workers]


@pytest.mark.asyncio
async def test_app_exit_with_a_resample_in_flight_raises_nothing(
    console_screen_with_db_and_pilot,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quitting mid-render must not surface a WorkerFailed (the 21122 shape)."""

    app, screen, db, pilot = console_screen_with_db_and_pilot
    rail = await _mount_card(screen, db, pilot)

    from tldw_chatbook.UI.Console_Modules import character_avatar_layout

    entered = threading.Event()
    release = threading.Event()
    original = character_avatar_layout.prerender_character_avatar

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        character_avatar_layout, "prerender_character_avatar", blocking
    )

    failures: list[object] = []
    original_handler = app._handle_exception
    monkeypatch.setattr(
        app, "_handle_exception", lambda error: failures.append(error)
    )

    rail.invalidate_character_avatar_geometry()
    rail._character_avatar_box = None
    rail._reconcile_character_avatar_geometry()
    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5.0)
    assert entered.is_set()

    app.exit()
    await pilot.pause()
    release.set()
    for _ in range(6):
        await pilot.pause()

    assert failures == [], failures
    assert original_handler is not None
