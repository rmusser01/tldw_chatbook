"""Remote inline transcript images (task-1537): setting + URL extraction.

Rendering images referenced by links in assistant replies is OFF by default
and gated behind ``[chat.images] render_remote_images``; extraction accepts
markdown image links and bare image-extension URLs, http(s) only.
"""

from tldw_chatbook.Chat.console_image_view import (
    extract_image_urls,
    resolve_render_remote_images,
)


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
    assert extract_image_urls(text) == [
        "https://example.com/a/b/portrait.jpg?size=big"
    ]


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

import io

import pytest
from types import SimpleNamespace

PILImage = pytest.importorskip("PIL.Image")

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)


def _bare_screen(*, enabled: bool):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(
        app_config={"chat": {"images": {"render_remote_images": enabled}}}
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

    specs = screen._build_console_image_specs([message])

    assert message.id in specs


def test_remote_spec_ignored_when_setting_off():
    """With the setting off no spec is built and no fetch dispatches."""
    screen = _bare_screen(enabled=False)
    message = _assistant("map: ![map](https://example.com/pic.png)")
    _state, cache = screen._ensure_console_image_view()
    assert cache.prepare("remote:https://example.com/pic.png", _png_bytes())
    dispatched: list = []
    screen.run_worker = lambda coro, **kw: (dispatched.append(coro), coro.close())

    specs = screen._build_console_image_specs([message])

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

    screen._build_console_image_specs([message])
    screen._build_console_image_specs([message])

    assert len(dispatched) == 1
