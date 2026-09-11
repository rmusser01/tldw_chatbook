"""TASK-588: `recursive_scrape` must close every page it opens.

The link-discovery block inside `recursive_scrape` opened a browser tab,
then did five awaited steps before `page.close()`. Any of them raising --
most realistically `check_url_or_raise_async` rejecting a URL the egress
policy blocks -- skipped the close, and the outer `except Exception` just
logged and moved on, leaking one tab per blocked/failed discovery page for
the lifetime of the crawl's browser context.

Both tests stub the playwright chain at the module seam
(`Article_Extractor_Lib.async_playwright`, the deferred import the module
itself documents as patchable) and count `page.close()` calls, so no real
browser is involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL

BASE_URL = "http://example.test/"


class _FakePage:
    def __init__(self) -> None:
        self.close_calls = 0

    async def goto(self, url: str) -> Any:
        return object()

    async def wait_for_load_state(self, state: str) -> None:
        return None

    async def eval_on_selector_all(self, selector: str, js: str) -> list[str]:
        return []

    async def close(self) -> None:
        self.close_calls += 1


class _FakeContext:
    def __init__(self) -> None:
        self.pages: list[_FakePage] = []

    async def add_cookies(self, cookies: Any) -> None:
        return None

    async def new_page(self) -> _FakePage:
        page = _FakePage()
        self.pages.append(page)
        return page


class _FakeBrowser:
    def __init__(self) -> None:
        self.context = _FakeContext()

    async def new_context(self, **kwargs: Any) -> _FakeContext:
        return self.context

    async def close(self) -> None:
        return None


class _FakeChromium:
    def __init__(self) -> None:
        self.browser = _FakeBrowser()

    async def launch(self, **kwargs: Any) -> _FakeBrowser:
        return self.browser


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()


class _FakeAsyncPlaywright:
    """`async_playwright()` async-context-manager double."""

    def __call__(self) -> _FakeAsyncPlaywright:
        return self

    async def __aenter__(self) -> _FakePlaywright:
        return _FakePlaywright()

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


@pytest.fixture()
def fake_browser_pages(monkeypatch: pytest.MonkeyPatch) -> list[_FakePage]:
    """Patch the playwright seam; return the list of pages the crawl opened."""
    pages_holder: list[_FakePage] = []

    class _RecordingContext(_FakeContext):
        async def new_page(self) -> _FakePage:
            page = _FakePage()
            pages_holder.append(page)
            return page

    class _RecordingBrowser(_FakeBrowser):
        def __init__(self) -> None:
            super().__init__()
            self.context = _RecordingContext()

    class _RecordingChromium(_FakeChromium):
        def __init__(self) -> None:
            super().__init__()
            self.browser = _RecordingBrowser()

    class _RecordingPlaywright(_FakePlaywright):
        def __init__(self) -> None:
            super().__init__()
            self.chromium = _RecordingChromium()

    class _RecordingAsyncPlaywright(_FakeAsyncPlaywright):
        async def __aenter__(self) -> _RecordingPlaywright:
            return _RecordingPlaywright()

    monkeypatch.setattr(AEL, "async_playwright", _RecordingAsyncPlaywright())
    monkeypatch.setattr(AEL, "scrape_article_async", lambda *a, **k: _async_none())
    return pages_holder


async def _async_none() -> None:
    return None


async def test_blocked_link_discovery_page_is_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_browser_pages: list[_FakePage]
) -> None:
    async def _raise_blocked(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("blocked by egress policy")

    monkeypatch.setattr(AEL, "check_url_or_raise_async", _raise_blocked)

    await AEL.recursive_scrape(
        BASE_URL,
        max_pages=1,
        max_depth=1,
        delay=0,
        resume_file=str(tmp_path / "progress.json"),
    )

    assert fake_browser_pages, "the crawl should have opened a discovery page"
    for page in fake_browser_pages:
        assert page.close_calls == 1, (
            "a page whose link discovery failed (here: URL blocked by the "
            "egress guard) must still be closed exactly once"
        )


async def test_successful_link_discovery_page_is_closed_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_browser_pages: list[_FakePage]
) -> None:
    async def _allow(*args: Any, **kwargs: Any) -> None:
        return None

    async def _validate(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(AEL, "check_url_or_raise_async", _allow)
    monkeypatch.setattr(AEL, "collect_navigation_chain", lambda response: [])
    monkeypatch.setattr(AEL, "validate_navigation_chain_async", _validate)

    await AEL.recursive_scrape(
        BASE_URL,
        max_pages=1,
        max_depth=1,
        delay=0,
        resume_file=str(tmp_path / "progress.json"),
    )

    assert fake_browser_pages, "the crawl should have opened a discovery page"
    for page in fake_browser_pages:
        assert page.close_calls == 1
