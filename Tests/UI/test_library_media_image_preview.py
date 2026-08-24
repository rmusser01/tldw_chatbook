"""Library Media local-image preview eligibility and rendering contracts."""

from __future__ import annotations

import asyncio
from io import BytesIO
import threading

import pytest
from PIL import Image
from textual.widgets import Button, Static

from Tests.UI.test_library_media_side_by_side import (
    WIDE_SIZE,
    _build_media_test_app,
    _open_media_list,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    StaticLibraryMediaScopeService,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Utils import optional_deps
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)

from tldw_chatbook.Widgets.Library.library_media_image_preview import (
    build_media_image_widget,
    decode_media_image,
    image_preview_eligibility,
)


def _file_check(content_type: str, *, available: bool = True) -> dict[str, object]:
    return {
        "available": available,
        "source": "file_path" if available else None,
        "content_type": content_type,
    }


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/webp"])
def test_eligible_local_original_image_types(mime: str) -> None:
    result = image_preview_eligibility(
        {"type": "image", "url": "file:///tmp/original"},
        _file_check(mime),
        backend="local",
    )

    assert result.eligible is True
    assert result.content_type == mime
    assert result.reason == "eligible"


@pytest.mark.parametrize(
    "mime", ["image/gif", "application/pdf", "audio/mpeg", "video/mp4"]
)
def test_ineligible_original_types_are_rejected_before_download(mime: str) -> None:
    result = image_preview_eligibility(
        {"type": "image", "url": "file:///tmp/original"},
        _file_check(mime),
        backend="local",
    )

    assert result.eligible is False
    assert result.reason == "unsupported"


def test_remote_url_and_server_detail_are_never_eligible() -> None:
    remote = image_preview_eligibility(
        {"type": "image", "url": "https://example.test/image.png"},
        _file_check("image/png"),
        backend="local",
    )
    server = image_preview_eligibility(
        {"type": "image", "url": "file:///tmp/image.png"},
        _file_check("image/png"),
        backend="server",
    )

    assert remote.reason == "remote"
    assert server.reason == "external"
    assert not remote.eligible
    assert not server.eligible


def test_original_must_be_an_available_file_not_stored_text() -> None:
    unavailable = image_preview_eligibility(
        {"type": "png", "url": "file:///tmp/image.png"},
        _file_check("image/png", available=False),
        backend="local",
    )
    stored_text = image_preview_eligibility(
        {"type": "png", "url": "local://media/7"},
        {
            "available": True,
            "source": "stored_content",
            "content_type": "text/plain; charset=utf-8",
        },
        backend="local",
    )

    assert unavailable.reason == "unavailable"
    assert stored_text.reason == "unavailable"


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_decode_accepts_only_supported_raster_formats(format_name: str) -> None:
    buffer = BytesIO()
    Image.new("RGB", (5, 3), "blue").save(buffer, format=format_name)

    decoded = decode_media_image(buffer.getvalue())

    assert decoded.size == (5, 3)
    assert decoded.format == format_name


def test_decode_rejects_an_unsupported_image_even_when_bytes_are_valid() -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="GIF")

    with pytest.raises(ValueError, match="unsupported image format"):
        decode_media_image(buffer.getvalue())


def test_decode_routes_pillow_through_optional_dependency_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def unavailable(module_name: str, feature_name: str | None = None) -> bool:
        calls.append((module_name, feature_name))
        return False

    monkeypatch.setattr(optional_deps, "check_dependency", unavailable)

    with pytest.raises(ImportError):
        decode_media_image(_image_bytes(width=3))

    assert calls == [("PIL", "pillow")]


def test_graphics_widget_routes_textual_image_through_optional_dependency_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def unavailable(module_name: str, feature_name: str | None = None) -> bool:
        calls.append((module_name, feature_name))
        return False

    monkeypatch.setattr(optional_deps, "check_dependency", unavailable)
    image = Image.new("RGB", (3, 2), "blue")

    widget = build_media_image_widget(
        image,
        app_config={"chat": {"images": {"default_render_mode": "regular"}}},
        box_cols=10,
        box_lines=5,
    )

    assert isinstance(widget, Static)
    assert calls == [("textual_image", None)]


def _image_bytes(*, width: int, color: str = "blue") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


class PreviewMediaService(StaticLibraryMediaScopeService):
    """Production-shaped local file seams with optional download gates."""

    def __init__(self, media_items):
        super().__init__(media_items)
        self.check_content_type = "image/png"
        self.check_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.download_outcomes: dict[int, object] = {}
        self.download_entered: dict[int, threading.Event] = {}
        self.download_release: dict[int, threading.Event] = {}
        self.blocked_downloads: set[int] = set()

    def check_media_file(self, *, media_id, **kwargs):
        self.check_calls.append({"media_id": media_id, **kwargs})
        return {
            "available": True,
            "source": "file_path",
            "content_type": self.check_content_type,
        }

    def download_media_file(self, *, media_id, **kwargs):
        self.download_calls.append({"media_id": media_id, **kwargs})
        if media_id in self.blocked_downloads:
            self.download_entered.setdefault(media_id, threading.Event()).set()
            if not self.download_release.setdefault(
                media_id, threading.Event()
            ).wait(5):
                raise RuntimeError("preview download gate timed out")
        outcome = self.download_outcomes.get(media_id, _image_bytes(width=media_id + 3))
        if isinstance(outcome, BaseException):
            raise outcome
        return {
            "content": outcome,
            "content_type": "image/png",
            "filename": f"image-{media_id}.png",
        }


def _preview_items(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "id": f"media-{index}",
            "title": f"Image {index}",
            "type": "image",
            "url": f"file:///tmp/image-{index}.png",
            "last_modified": f"2026-08-{index:02d}T10:00:00Z",
            "content": f"Complete stored text for image {index}.\nSecond line.",
            "version": 1,
        }
        for index in range(1, count + 1)
    ]


def _preview_app():
    app = _build_media_test_app()
    items = _preview_items()
    _seed_conversations(app, _two_conversations(), media=items)
    service = PreviewMediaService(items)
    app.media_reading_scope_service = service
    return app, service


def test_preview_cache_evicts_oldest_image_and_related_session_state() -> None:
    app, _service = _preview_app()
    screen = LibraryScreen(app, preview_widget_factory=_fake_preview_factory([]))
    first_id = "local:media:1"
    screen._library_media_preview_status[first_id] = "cached"
    screen._library_media_preview_hidden.add(first_id)
    screen._library_media_preview_loading[first_id] = 1

    for index in range(1, 22):
        screen._cache_library_media_preview(
            f"local:media:{index}", Image.new("RGB", (1, 1))
        )

    assert len(screen._library_media_preview_images) == 20
    assert first_id not in screen._library_media_preview_images
    assert first_id not in screen._library_media_preview_status
    assert first_id not in screen._library_media_preview_hidden
    assert first_id not in screen._library_media_preview_loading


def _fake_preview_factory(calls):
    def factory(image, **kwargs):
        calls.append((image.width, kwargs))
        return Static(f"PREVIEW:{image.width}", markup=False)

    return factory


@pytest.mark.asyncio
async def test_preview_mounts_above_byte_for_byte_unchanged_complete_text() -> None:
    app, service = _preview_app()
    factory_calls = []
    screen = LibraryScreen(
        app, preview_widget_factory=_fake_preview_factory(factory_calls)
    )
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        backing_id = int(str(row.media_id).rsplit(":", 1)[-1])
        expected_text = next(
            str(item["content"])
            for index, item in enumerate(service.media_items)
            if service._backing_id(item, index) == backing_id
        )
        row.press()
        preview = await _wait_for_selector(
            screen, pilot, "#library-media-image-preview"
        )
        body = screen.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        )

        assert body.content == expected_text
        assert preview.region.y < body.region.y
        assert factory_calls
        assert service.check_calls == [
            {"media_id": backing_id, "mode": "local", "file_type": "original"}
        ]
        assert service.download_calls == [
            {"media_id": backing_id, "mode": "local", "file_type": "original"}
        ]


@pytest.mark.asyncio
async def test_capability_off_keeps_complete_stored_text_without_file_calls() -> None:
    app, service = _preview_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == row.media_id,
            message="Image detail did not load with preview capability off.",
        )

        body = screen.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        )
        assert body.content.startswith("Complete stored text for image")
        assert not screen.query("#library-media-image-preview")
        assert service.check_calls == []
        assert service.download_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_url", "content_type", "expected_check_count"),
    [
        ("https://example.test/image.png", "image/png", 0),
        ("file:///tmp/image.gif", "image/gif", 1),
    ],
)
async def test_remote_and_unsupported_images_never_reach_download(
    source_url: str, content_type: str, expected_check_count: int
) -> None:
    app, service = _preview_app()
    for item in service.media_items:
        item["url"] = source_url
    service.check_content_type = content_type
    screen = LibraryScreen(app, preview_widget_factory=_fake_preview_factory([]))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == row.media_id,
            message="Ineligible image detail did not settle.",
        )
        await pilot.pause()

        assert len(service.check_calls) == expected_check_count
        assert service.download_calls == []
        assert not screen.query("#library-media-image-preview")


@pytest.mark.asyncio
async def test_preview_render_failure_preserves_text_fallback() -> None:
    app, _service = _preview_app()

    def fail_to_render(_image, **_kwargs):
        raise RuntimeError("terminal renderer unavailable")

    screen = LibraryScreen(app, preview_widget_factory=fail_to_render)
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        row.press()
        status = await _wait_for_selector(
            screen, pilot, "#library-media-image-preview-status"
        )

        body = screen.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        )
        assert body.content.startswith("Complete stored text for image")
        assert status.renderable == (
            "Image preview failed — showing complete stored text"
        )
        assert not screen.query("#library-media-image-preview")


@pytest.mark.asyncio
async def test_hide_show_is_per_item_session_state_and_does_not_reload_detail() -> None:
    app, service = _preview_app()
    screen = LibraryScreen(app, preview_widget_factory=_fake_preview_factory([]))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        row.press()
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        detail_calls = len(service.detail_calls)
        file_calls = (len(service.check_calls), len(service.download_calls))

        screen.query_one("#library-media-image-preview-toggle", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-image-preview"),
            message="Hide preview left the image mounted.",
        )
        assert "Show preview" in str(
            screen.query_one("#library-media-image-preview-toggle", Button).label
        )

        screen.query_one("#library-media-image-preview-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        assert len(service.detail_calls) == detail_calls
        assert (len(service.check_calls), len(service.download_calls)) == file_calls


@pytest.mark.asyncio
async def test_preview_failure_keeps_item_loaded_and_retry_is_item_local() -> None:
    app, service = _preview_app()
    screen = LibraryScreen(app, preview_widget_factory=_fake_preview_factory([]))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        backing_id = int(str(row.media_id).rsplit(":", 1)[-1])
        service.download_outcomes[backing_id] = b"not an image"
        row.press()
        status = await _wait_for_selector(
            screen, pilot, "#library-media-image-preview-status"
        )

        assert status.renderable == (
            "Image preview failed — showing complete stored text"
        )
        assert screen._library_media_reader_session.loaded_id == row.media_id
        detail_calls = len(service.detail_calls)
        service.download_outcomes[backing_id] = _image_bytes(width=11)
        screen.query_one("#library-media-image-preview-retry", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")

        assert len(service.detail_calls) == detail_calls
        assert len(service.download_calls) == 2


@pytest.mark.asyncio
async def test_late_preview_for_a_cannot_mount_over_loaded_b() -> None:
    app, service = _preview_app()
    screen = LibraryScreen(app, preview_widget_factory=_fake_preview_factory([]))
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        await _open_media_list(host, pilot)
        row_a = screen.query_one("#library-media-row-0", Button)
        backing_a = int(str(row_a.media_id).rsplit(":", 1)[-1])
        service.blocked_downloads.add(backing_a)
        row_a.press()
        started = await asyncio.to_thread(
            service.download_entered.setdefault(backing_a, threading.Event()).wait,
            2,
        )
        assert started

        row_b = screen.query_one("#library-media-row-1", Button)
        backing_b = int(str(row_b.media_id).rsplit(":", 1)[-1])
        row_b.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == row_b.media_id,
            message="Item B detail did not settle.",
        )
        await _wait_for_selector(screen, pilot, "#library-media-image-preview")
        assert (
            screen.query_one("#library-media-image-preview Static", Static).renderable
            == f"PREVIEW:{backing_b + 3}"
        )

        service.download_release[backing_a].set()
        await _wait_for_condition(
            pilot,
            lambda: row_a.media_id not in screen._library_media_preview_loading,
            message="Stale item A preview worker did not settle.",
        )
        assert screen._library_media_reader_session.loaded_id == row_b.media_id
        assert row_a.media_id not in screen._library_media_preview_images
        assert (
            screen.query_one("#library-media-image-preview Static", Static).renderable
            == f"PREVIEW:{backing_b + 3}"
        )
