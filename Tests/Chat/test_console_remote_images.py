"""Remote inline transcript images (task-1537): setting + URL extraction.

Rendering images referenced by links in assistant replies is OFF by default
and gated behind ``[chat.images] render_remote_images``; extraction accepts
markdown image links and bare image-extension URLs, http(s) only.
"""

import io
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_image_view import (
    extract_image_urls,
    resolve_render_remote_images,
)

PILImage = pytest.importorskip("PIL.Image")


def test_remote_images_default_off():
    """render_remote_images defaults to off for absent/empty config."""
    assert resolve_render_remote_images({}) is False
    assert resolve_render_remote_images({"chat": {"images": {}}}) is False


def test_remote_images_enabled_by_setting():
    """render_remote_images=true in [chat.images] enables the feature."""
    config = {"chat": {"images": {"render_remote_images": True}}}
    assert resolve_render_remote_images(config) is True


def test_extract_markdown_image_links():
    """Markdown ![alt](url) image links extract their http(s) URL."""
    text = "Here you go ![map](https://example.com/city.png) and more"
    assert extract_image_urls(text) == ["https://example.com/city.png"]


def test_extract_bare_image_extension_urls():
    """Bare URLs with image extensions (plus query strings) extract."""
    text = "portrait: https://example.com/a/b/portrait.jpg?size=big done"
    assert extract_image_urls(text) == ["https://example.com/a/b/portrait.jpg?size=big"]


def test_non_image_bare_urls_ignored():
    """Bare URLs without an image extension are not treated as images."""
    assert extract_image_urls("see https://example.com/page.html") == []


def test_non_http_schemes_ignored():
    """data:/file: and other non-http(s) schemes never extract."""
    text = "![x](data:image/png;base64,AAAA) ![y](file:///etc/passwd)"
    assert extract_image_urls(text) == []


def test_dedupe_and_cap():
    """Extraction dedupes repeated URLs and honors the limit cap."""
    url = "https://example.com/i.png"
    text = " ".join(
        [f"![a]({url})", f"![b]({url})"]
        + [f"![n](https://example.com/{i}.png)" for i in range(9)]
    )
    urls = extract_image_urls(text, limit=3)
    assert len(urls) == 3
    assert urls[0] == url
    assert len(set(urls)) == 3


# ---- ChatScreen spec wiring ----


def _bare_screen(*, enabled: bool):
    from Tests.UI.console_controller_stubs import (
        stub_image_controller,
        stub_message_controller,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(
        app_config={"chat": {"images": {"render_remote_images": enabled}}}
    )
    # `_build_console_image_specs` calls `_recent_console_image_messages`,
    # which moved to `ConsoleMessageController` (wave-3 console
    # decomposition, task 1) and is reached through `ChatScreen`'s
    # delegation. `ChatScreen.__new__` skips the construction `__init__`
    # would do. That method touches no injected seam, so nothing is wired.
    stub_message_controller(screen, context="test_console_remote_images._bare_screen")
    stub_image_controller(
        screen,
        context="test_console_remote_images._bare_screen",
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        recent_console_image_messages=(
            lambda messages: screen._message._recent_console_image_messages(messages)
        ),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        console_generation_browse=lambda: screen._console_generation_browse(),
    )
    return screen


def _assistant(content: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content=content, status="complete"
    )


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (16, 16), (0, 119, 226)).save(buf, format="PNG")
    return buf.getvalue()


def test_remote_spec_built_for_cached_link_when_enabled():
    """An enabled setting plus a cached link yields an image-row spec."""
    screen = _bare_screen(enabled=True)
    message = _assistant("map: ![map](https://example.com/pic.png)")
    _state, cache = screen._ensure_console_image_view()
    assert cache.prepare("remote:https://example.com/pic.png", _png_bytes())

    specs = screen._image._build_console_image_specs([message])

    assert message.id in specs


def test_remote_spec_ignored_when_setting_off():
    """With the setting off no spec is built and no fetch dispatches."""
    screen = _bare_screen(enabled=False)
    message = _assistant("map: ![map](https://example.com/pic.png)")
    _state, cache = screen._ensure_console_image_view()
    assert cache.prepare("remote:https://example.com/pic.png", _png_bytes())
    dispatched: list = []
    screen.run_worker = lambda coro, **kw: (dispatched.append(coro), coro.close())

    specs = screen._image._build_console_image_specs([message])

    assert message.id not in specs
    assert dispatched == []


def test_uncached_link_dispatches_fetch_once():
    """An uncached link dispatches exactly one fetch across rebuilds."""
    screen = _bare_screen(enabled=True)
    message = _assistant("see https://example.com/photo.jpg now")
    dispatched: list = []

    def _record(coro, **kwargs):
        dispatched.append(coro)
        coro.close()

    screen.run_worker = _record

    screen._image._build_console_image_specs([message])
    screen._image._build_console_image_specs([message])

    assert len(dispatched) == 1


def test_remote_fetch_attempt_memory_is_bounded():
    """High-cardinality remote links cannot grow controller state forever."""
    attempt_limit = 256
    screen = _bare_screen(enabled=True)
    dispatched: list = []

    def _record(coro, **kwargs):
        dispatched.append(coro)
        coro.close()

    screen.run_worker = _record
    urls = [
        f"https://example.com/photo-{index}.jpg" for index in range(attempt_limit + 1)
    ]
    for url in urls:
        screen._image._build_console_image_specs([_assistant(f"see {url} now")])

    assert len(dispatched) == len(urls)
    assert len(screen._image._remote_image_fetch_attempts) == attempt_limit
    assert urls[0] not in screen._image._remote_image_fetch_attempts
    assert urls[-1] in screen._image._remote_image_fetch_attempts
