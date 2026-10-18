"""Terminal evidence must include fresh content and an independently painted layer."""

from __future__ import annotations

import gc
from weakref import ref

import pytest
from playwright.async_api import async_playwright

from Tests.Canvas.browser import canvas_live_harness as harness

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def painted_terminal():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=harness.chromium_executable(playwright.chromium),
        )
        try:
            page = await browser.new_page()
            await page.set_content(
                '<div id="terminal"><canvas class="xterm-text-layer" width="800" '
                'height="120"></canvas><canvas class="xterm-cursor-layer" '
                'width="800" height="120"></canvas><canvas class="xterm-selection-layer" '
                'width="800" height="120"></canvas></div>'
            )
            await page.evaluate("""() => {
                window.paint = (text, layer = 'text') => {
                    const canvas = document.querySelector(`.xterm-${layer}-layer`);
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#121212'; ctx.fillRect(0, 0, 800, 120);
                    ctx.fillStyle = '#e0e0e0'; ctx.font = '16px monospace';
                    ctx.fillText(text, 0, 20);
                };
                paint('previous branch');
            }""")
            yield page
        finally:
            await browser.close()


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "stale",
        "wrong",
        "frozen",
        "blank",
        "background",
        "cursor",
        "selection",
        "hidden",
    ],
)
async def test_terminal_observer_rejects_incomplete_evidence(painted_terminal, failure):
    page = painted_terminal
    assert hasattr(harness, "TerminalPaintObserver"), (
        "missing production-renderer observer"
    )
    observer = harness.TerminalPaintObserver(page)
    observer.receive(b"previous branch")
    if failure == "stale":
        observer.receive(b"\rCANVAS_LIVE_REOPENED")
    await observer.mark()
    if failure not in {"missing", "stale"}:
        observer.receive(
            b"\rwrong result" if failure == "wrong" else b"\rCANVAS_LIVE_REOPENED"
        )
    if failure == "stale":
        observer.receive(b"\x1b[2;1Hunrelated output")
    if failure not in {"frozen", "cursor", "selection", "background"}:
        await page.evaluate(
            "text => paint(text)", "" if failure == "blank" else "new paint"
        )
    if failure == "background":
        await page.evaluate("""() => {
            const ctx = document.querySelector('.xterm-text-layer').getContext('2d');
            ctx.fillStyle = '#121212'; ctx.fillRect(0, 0, 800, 120);
            ctx.fillStyle = '#303030'; ctx.fillRect(0, 0, 400, 120);
        }""")
    if failure in {"cursor", "selection"}:
        await page.evaluate("layer => paint('noise', layer)", failure)
    if failure == "hidden":
        await page.locator("#terminal").evaluate("node => node.style.opacity = '0'")
    with pytest.raises(AssertionError, match="terminal evidence"):
        await observer.wait_for_text("CANVAS_LIVE_REOPENED", timeout_ms=50)


async def test_terminal_observer_accepts_split_fresh_output_and_visible_paint(
    painted_terminal,
):
    page = painted_terminal
    assert hasattr(harness, "TerminalPaintObserver"), (
        "missing production-renderer observer"
    )
    observer = harness.TerminalPaintObserver(page)
    observer.receive(b"previous branch")
    await observer.mark()
    observer.receive(b"\r\x1b[2KCANVAS_LIVE_RE")
    observer.receive(b"OPENED")
    await page.evaluate("() => paint('CANVAS_LIVE_REOPENED')")
    await observer.wait_for_text("CANVAS_LIVE_REOPENED", timeout_ms=500)
    assert "CANVAS_LIVE_REOPENED" in observer.text


@pytest.mark.parametrize(
    ("second_column", "suffix"),
    [(5, ""), (6, ""), (5, " WRONG"), (5, "BAD")],
    ids=["wrapped", "wrong-column", "wrong-fragment", "wrong-boundary"],
)
async def test_terminal_observer_matches_wrapped_words_in_the_same_column(
    painted_terminal, second_column, suffix
):
    page = painted_terminal
    observer = harness.TerminalPaintObserver(page)
    await observer.mark()
    observer.receive(
        f"\x1b[2J\x1b[1;5HRevise the{suffix}\x1b[2;{second_column}Hactive Canvas".encode()
    )
    await page.evaluate("() => paint('Revise the / active Canvas')")
    if second_column == 5 and not suffix:
        await observer.wait_for_text("Revise the active Canvas", timeout_ms=500)
    else:
        with pytest.raises(AssertionError, match="terminal evidence"):
            await observer.wait_for_text("Revise the active Canvas", timeout_ms=50)


async def test_terminal_observer_retires_previous_connection(painted_terminal):
    page = painted_terminal
    assert hasattr(harness, "TerminalPaintObserver"), (
        "missing production-renderer observer"
    )
    observer = harness.TerminalPaintObserver(page)
    observer.receive(b"previous branch")
    await observer.mark()
    observer.receive(b"\rCANVAS_LIVE_REOPENED")
    observer.connect("wss://127.0.0.1/ws?width=80&height=6")
    await page.evaluate("() => paint('new child')")
    with pytest.raises(AssertionError, match="terminal evidence"):
        await observer.wait_for_text("CANVAS_LIVE_REOPENED", timeout_ms=50)


async def test_page_observer_registry_does_not_retain_closed_pages():
    class Page:
        def on(self, _event, callback):
            self.callback = callback

    page = Page()
    page_ref = ref(page)
    observer = harness.terminal_observer(page)
    assert harness.terminal_observer(page) is observer
    del page
    gc.collect()
    assert page_ref() is None


async def test_page_observer_does_not_retain_a_closed_transport_page():
    class Page:
        def on(self, _event, callback):
            self.callback = callback

    class Socket:
        url = "wss://127.0.0.1/ws?width=80&height=24"

        def __init__(self, page):
            self.page = page
            self.callbacks = {}

        def on(self, event, callback):
            self.callbacks[event] = callback

    page = Page()
    page_ref = ref(page)
    _observer = harness.terminal_observer(page)
    page.callback(Socket(page))
    del page
    gc.collect()
    assert page_ref() is None
