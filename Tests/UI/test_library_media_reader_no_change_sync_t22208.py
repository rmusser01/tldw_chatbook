"""TASK-22208: no-change Reader syncs must not rebuild previews or copy the document.

TASK-22207 made the ``unchanged`` comparison in
``_sync_library_media_viewer_state`` actually reachable for traversal
keystrokes, but the sync still PAID for the compare: it built the full
image-preview projection (``build_media_image_widget``: a PIL LANCZOS
resize plus a per-cell mosaic loop, synchronously on the event loop) BEFORE
the compare and discarded the widget on the no-change path, and it rebuilt
the full viewer display state per sync (``build_library_media_viewer_state``:
``str(content).strip()`` -- a trailing newline forces a full O(document)
copy -- once for the display state and once more inside the
console-representation clause of the compare itself).

These probes drive the real focus-traversal interaction on an image-typed
item and count, inside the pass-through window:

* preview factory invocations (nonzero before this task, 0 after), and
* ``build_library_media_viewer_state`` invocations (each one is at least
  one O(document) copy for trailing-whitespace content; nonzero before,
  0 after -- the display state is memoized per detail arrival).

A change must still rebuild: the settled-row probe pins that a NEW image
reaching the Reader rebuilds the preview from the new source (the memo must
never serve a stale preview), and the memo-key unit probes pin every input
of the mosaic memo (image object identity, box_cols, box_lines) plus the
fact that widget INSTANCES are never reused across builds (a removed
Textual widget cannot be remounted).

Per the 15457 probe rule, the timing test prints per-keystroke wall times
for the task record and asserts only structural counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest
from PIL import Image
from textual.widgets import Button, Static

import tldw_chatbook.UI.Library_Modules.library_media_reader_controller as library_screen_module
from Tests.UI.test_library_media_reader_flow import (
    ControlledDetailMediaService,
    _row_identity,
    _wait_for_detail_call,
)
from Tests.UI.test_library_media_reader_traversal_t22207 import (
    _load_row,
    _release_everything,
)
from Tests.UI.test_library_media_side_by_side import (
    WIDE_SIZE,
    _build_media_test_app,
    _many_media_items,
    _open_media_list,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import library_media_image_preview as preview_module
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)
from tldw_chatbook.Widgets.Library.library_media_image_preview import (
    build_media_image_widget,
)

_PIXELS_CONFIG = {"chat": {"images": {"default_render_mode": "pixels"}}}


def _png_bytes(width: int, height: int, color: str = "navy") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class PreviewControlledDetailMediaService(ControlledDetailMediaService):
    """Gated detail responses plus local-original image seams.

    Only backing ids registered in ``image_bytes`` present an available
    local original -- every other item reads "unavailable", so settling a
    non-image row never schedules a decode.
    """

    def __init__(self, media_items):
        super().__init__(media_items)
        self.image_bytes: dict[int, bytes] = {}

    def check_media_file(self, *, media_id, **kwargs):
        if media_id not in self.image_bytes:
            return {"available": False}
        return {
            "available": True,
            "source": "file_path",
            "content_type": "image/png",
        }

    def download_media_file(self, *, media_id, **kwargs):
        return {
            "content": self.image_bytes[media_id],
            "content_type": "image/png",
            "filename": f"image-{media_id}.png",
        }


def _probe_app(count: int = 12):
    app = _build_media_test_app()
    items = _many_media_items(count)
    _seed_conversations(app, _two_conversations(), media=items)
    service = PreviewControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return app, service


def _counting_factory(calls):
    def factory(image, **kwargs):
        calls.append({"image": image, **kwargs})
        return Static(f"PREVIEW:{image.width}x{image.height}", markup=False)

    return factory


def _make_image_item(service, backing_id: int, *, content: str, png: bytes) -> dict:
    """Reshape one seeded row into an image-typed item with a local original."""
    source = next(
        item
        for item in service.media_items
        if item["id"] == f"media-{backing_id}"
    )
    source["type"] = "image"
    source["content"] = content
    service.image_bytes[backing_id] = png
    return source


@contextmanager
def _count_state_builds():
    """Count ``build_library_media_viewer_state`` calls through the screen module.

    Every call with trailing-whitespace content performs at least one full
    O(document) copy (``str(content).strip()``), so the call count times the
    content length is the copy-bytes metric the task records. The patched
    name is a module-global function, not an ``@on`` handler, so patching is
    dispatch-safe.
    """
    counts = {"calls": 0}
    real = library_screen_module.build_library_media_viewer_state

    def counting(*args, **kwargs):
        counts["calls"] += 1
        return real(*args, **kwargs)

    with patch.object(
        library_screen_module, "build_library_media_viewer_state", counting
    ):
        yield counts


@pytest.mark.asyncio
async def test_no_change_traversal_builds_no_preview_and_copies_no_content():
    """Pass-through keystrokes past a loaded image item: 0 factory calls, 0 state builds."""
    app, service = _probe_app()
    factory_calls: list[dict] = []
    screen = LibraryScreen(app, preview_widget_factory=_counting_factory(factory_calls))
    host = LibraryProductionCSSHarness(app, screen=screen)

    # Trailing newline: forces build_library_media_viewer_state's strip()
    # to copy the whole document on every call.
    content = ("A steady paragraph of stored text for the image item.\n" * 500) + "\n"

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        _, backing_id_0, _ = _row_identity(
            screen.query_one("#library-media-row-0", Button)
        )
        _make_image_item(
            service, backing_id_0, content=content, png=_png_bytes(64, 48)
        )
        await _load_row(screen, pilot, service, 0)
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        await pilot.pause()
        await pilot.pause()

        baseline_factory = len(factory_calls)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()

        final_id, final_backing_id, _ = _row_identity(
            screen.query_one("#library-media-row-5", Button)
        )
        with _count_state_builds() as state_counts:
            for _ in range(5):
                await pilot.press("down")
            pass_through_factory = len(factory_calls) - baseline_factory
            pass_through_state_builds = state_counts["calls"]

        print(
            "TASK-22208 no-change probe: "
            f"factory_calls={pass_through_factory} "
            f"state_builds={pass_through_state_builds} "
            f"copy_bytes~={pass_through_state_builds * len(content)}"
        )

        # Settle the final row so teardown is clean before asserting.
        await _wait_for_detail_call(service, final_backing_id)
        _release_everything(service)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == final_id,
            message="The final traversal row never settled.",
        )
        await pilot.pause()

        # The settled row's content must actually be displayed: the display
        # -state memo is keyed by detail-object identity, and a memo that
        # fails to invalidate on a new arrival would leave the old item's
        # document mounted here.
        final_content = next(
            str(item["content"])
            for item in service.media_items
            if item["id"] == f"media-{final_backing_id}"
        )
        body = screen.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        )
        assert body.content == final_content, (
            "The settled row's content is not displayed -- the display-state "
            "memo served a stale arrival."
        )

        assert pass_through_factory == 0, (
            f"No-change traversal syncs rebuilt the image preview "
            f"{pass_through_factory} time(s); they must build zero widgets."
        )
        assert pass_through_state_builds == 0, (
            f"No-change traversal syncs rebuilt the viewer display state "
            f"{pass_through_state_builds} time(s) "
            f"(~{pass_through_state_builds * len(content)} copied bytes); "
            "the display state must be memoized per detail arrival."
        )


@pytest.mark.asyncio
async def test_new_image_item_still_rebuilds_preview_from_new_source():
    """Settling a DIFFERENT image item rebuilds the preview from the new image."""
    app, service = _probe_app()
    factory_calls: list[dict] = []
    screen = LibraryScreen(app, preview_widget_factory=_counting_factory(factory_calls))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        _, backing_id_0, _ = _row_identity(
            screen.query_one("#library-media-row-0", Button)
        )
        canonical_1, backing_id_1, title_1 = _row_identity(
            screen.query_one("#library-media-row-1", Button)
        )
        _make_image_item(
            service, backing_id_0, content="First image text.\n", png=_png_bytes(64, 48)
        )
        _make_image_item(
            service,
            backing_id_1,
            content="Second image text.\n",
            png=_png_bytes(96, 32, "maroon"),
        )
        await _load_row(screen, pilot, service, 0)
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        await pilot.pause()
        first_widths = {call["image"].width for call in factory_calls}
        assert 64 in first_widths

        await _load_row(screen, pilot, service, 1)
        await _wait_for_condition(
            pilot,
            lambda: any(call["image"].width == 96 for call in factory_calls),
            message="The second image item never rebuilt its preview.",
        )
        assert factory_calls[-1]["image"].width == 96, (
            "The preview served for the newly settled item was not built "
            "from the new item's image (stale-preview hazard)."
        )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_image_item_traversal_wall_time_probe():
    """Per-keystroke wall time on an image item with a large document.

    Printed timings are the task record; the assertions pin the structural
    counts (0 preview builds, 0 state builds in the pass-through window)
    that guarantee the improvement, per the 15457 probe rule. The preview
    factory is the PRODUCTION ``build_media_image_widget`` pinned to the
    mosaic ("pixels") path so the probe measures the real PIL cost.
    """
    app, service = _probe_app()

    def production_pixels_factory(image, *, app_config, box_cols, box_lines):
        return build_media_image_widget(
            image,
            app_config=_PIXELS_CONFIG,
            box_cols=box_cols,
            box_lines=box_lines,
        )

    screen = LibraryScreen(app, preview_widget_factory=production_pixels_factory)
    host = LibraryProductionCSSHarness(app, screen=screen)
    big_content = (
        "A paragraph of steady reading material padding the stored text "
        "toward a large document body.\n" * 10000
    ) + "\n"
    assert len(big_content) > 900_000

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        _, backing_id_0, _ = _row_identity(
            screen.query_one("#library-media-row-0", Button)
        )
        _make_image_item(
            service, backing_id_0, content=big_content, png=_png_bytes(900, 600)
        )
        await _load_row(screen, pilot, service, 0)
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        await pilot.pause()
        await pilot.pause()

        # Direct no-change sync timing: the keystroke wall time below is
        # dominated by the pilot/frame machinery, so also time the sync
        # itself -- the exact code path this task de-costs.
        viewer = screen.query_one("#library-media-viewer")
        direct_sync_ms: list[float] = []
        for _ in range(20):
            started = perf_counter()
            assert screen._sync_library_media_viewer_state(viewer)
            direct_sync_ms.append((perf_counter() - started) * 1000.0)
        print(
            "TASK-22208 direct no-change sync (image item, ~1 MB document): "
            f"median={median(direct_sync_ms):.3f}ms "
            f"samples={','.join(f'{sample:.3f}' for sample in direct_sync_ms[:8])}"
        )
        await pilot.pause()

        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        final_id, final_backing_id, _ = _row_identity(
            screen.query_one("#library-media-row-5", Button)
        )
        per_keystroke_ms: list[float] = []
        with _count_state_builds() as state_counts:
            for _ in range(5):
                started = perf_counter()
                await pilot.press("down")
                per_keystroke_ms.append((perf_counter() - started) * 1000.0)
            pass_through_state_builds = state_counts["calls"]

        # Print BEFORE the settle wait and the asserts so the red run still
        # records the before numbers.
        print(
            "TASK-22208 image-item traversal per-keystroke: "
            f"median={median(per_keystroke_ms):.3f}ms samples="
            + ",".join(f"{sample:.3f}" for sample in per_keystroke_ms)
            + f" state_builds={pass_through_state_builds}"
        )

        await _wait_for_detail_call(service, final_backing_id)
        _release_everything(service)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == final_id,
            timeout=180.0,
            message="The final traversal row never settled past the image item.",
        )
        await pilot.pause()

        assert pass_through_state_builds == 0, (
            "Pass-through keystrokes rebuilt the display state "
            f"{pass_through_state_builds} time(s) over a ~1 MB document."
        )


def _flat_image(width: int, height: int, color: str = "teal"):
    return Image.new("RGB", (width, height), color)


@contextmanager
def _count_mosaic_renders():
    """Count real ``mosaic_from_image`` invocations (patched at its source)."""
    from tldw_chatbook.Utils import mosaic_render

    counts = {"calls": 0}
    real = mosaic_render.mosaic_from_image

    def counting(image, box_cols, box_lines, **kwargs):
        counts["calls"] += 1
        return real(image, box_cols, box_lines, **kwargs)

    with patch.object(mosaic_render, "mosaic_from_image", counting):
        yield counts


def test_mosaic_memo_serves_repeat_builds_and_never_reuses_widget_instances():
    """Same (image, cols, lines): one mosaic render, distinct widget instances."""
    preview_module._MOSAIC_MEMO = None
    image = _flat_image(40, 20)
    with _count_mosaic_renders() as counts:
        first = build_media_image_widget(
            image, app_config=_PIXELS_CONFIG, box_cols=20, box_lines=6
        )
        second = build_media_image_widget(
            image, app_config=_PIXELS_CONFIG, box_cols=20, box_lines=6
        )
    assert counts["calls"] == 1, (
        f"Expected one memoized mosaic render, saw {counts['calls']}."
    )
    # A removed Textual widget cannot be remounted, so the memo must be at
    # the renderable grain: repeat builds share the mosaic but never the
    # widget instance.
    assert first is not second
    assert isinstance(first, Static) and isinstance(second, Static)
    assert str(first.styles.width) == str(second.styles.width)
    assert str(first.styles.height) == str(second.styles.height)


def test_mosaic_memo_invalidates_on_every_key_input():
    """Each memo key input (image identity, cols, lines) forces a rebuild."""
    preview_module._MOSAIC_MEMO = None
    image = _flat_image(40, 20)
    with _count_mosaic_renders() as counts:
        build_media_image_widget(
            image, app_config=_PIXELS_CONFIG, box_cols=20, box_lines=6
        )
        # box_cols changed -> rebuild (a resized Reader must not serve the
        # old width's mosaic).
        build_media_image_widget(
            image, app_config=_PIXELS_CONFIG, box_cols=28, box_lines=6
        )
        assert counts["calls"] == 2
        # box_lines changed -> rebuild.
        build_media_image_widget(
            image, app_config=_PIXELS_CONFIG, box_cols=28, box_lines=9
        )
        assert counts["calls"] == 3
        # A DIFFERENT decoded image object with identical pixels -> rebuild:
        # the key is object identity (a re-decode after eviction or a
        # changed original must never be served the old mosaic).
        build_media_image_widget(
            _flat_image(40, 20), app_config=_PIXELS_CONFIG, box_cols=28, box_lines=9
        )
        assert counts["calls"] == 4
    preview_module._MOSAIC_MEMO = None
