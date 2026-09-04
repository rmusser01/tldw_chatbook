"""Real-browser behavior for the trusted preview-first Canvas shell."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    BridgeConfirmationResponse,
    CanvasGateway,
    CanvasGatewayEvent,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from tldw_chatbook.Canvas.limits import sha256_utf8


@dataclass
class _NativeFlowAuthority:
    events: dict[str, CanvasGatewayEvent] = field(default_factory=dict)

    def publish(
        self,
        revision_id: str,
        *,
        kind: str = "updated",
        sequence: int,
        notice: str = "",
    ) -> None:
        self.events[revision_id] = CanvasGatewayEvent(
            event_id=f"event-{revision_id}",
            kind=kind,  # type: ignore[arg-type]
            canvas_id="canvas-a",
            revision_id=revision_id,
            metadata={
                "title": "Release planner",
                "sequence": sequence,
                "source_bytes": 96,
                "content_sha256": "a" * 64,
                "temporary": True,
                "origin_message_id": f"message-{sequence}",
                "origin_turn_id": f"turn-{sequence}",
                "notice": notice,
            },
        )

    async def resolve_render_plan(self, scope: CanvasGatewayScope):
        return compile_canvas_document(
            "<!doctype html><html><body>"
            f"<h1>{scope.revision_id}</h1><p>Isolated preview</p>"
            "</body></html>"
        )

    async def read_source(self, scope: CanvasGatewayScope) -> CanvasSourceResponse:
        source = f"<!doctype html><title>{scope.revision_id}</title>"
        return CanvasSourceResponse(source, sha256_utf8(source))

    async def read_events(
        self, scope: CanvasGatewayScope, *, after_event_id: str | None
    ) -> tuple[CanvasGatewayEvent, ...]:
        event = self.events.get(scope.revision_id)
        if event is None or event.event_id == after_event_id:
            return ()
        return (event,)

    async def confirm_bridge(
        self,
        scope: CanvasGatewayScope,
        request: BridgeConfirmationRequest,
        *,
        settlement: object,
    ) -> BridgeConfirmationResponse:
        del scope, settlement
        return BridgeConfirmationResponse(request.request.request_id, "cancelled")


def _scope(revision_id: str) -> CanvasGatewayScope:
    return CanvasGatewayScope(
        browser_session_id="browser-native-flow",
        conversation_session_id="conversation-native-flow",
        canvas_id="canvas-a",
        revision_id=revision_id,
    )


def _chromium_executable(browser_type: object) -> str:
    configured = os.environ.get("TLDW_CANVAS_CHROMIUM_EXECUTABLE")
    if configured and Path(configured).is_file():
        return configured
    declared = Path(browser_type.executable_path)
    if declared.is_file():
        return str(declared)
    root = Path(__file__).resolve().parents[3]
    caches = [Path.home() / "Library" / "Caches" / "ms-playwright"]
    caches.extend(
        ancestor / "Library" / "Caches" / "ms-playwright"
        for ancestor in root.parents
    )
    candidates = sorted(
        (
            executable
            for cache in caches
            for pattern in (
                "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
                "chromium_headless_shell-*/chrome-mac/headless_shell",
                "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
            )
            for executable in cache.glob(pattern)
        ),
        reverse=True,
    )
    executable = str(candidates[0]) if candidates else shutil.which("chromium")
    if not executable:
        pytest.fail("real Playwright Chromium is required for the Canvas native flow")
    return executable


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_native_canvas_shell_follows_updates_and_keeps_pinned_revision() -> None:
    authority = _NativeFlowAuthority()
    authority.publish("revision-1", kind="selection_changed", sequence=1)
    opened_urls: list[str] = []
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope("revision-1"), opener=opened_urls.append)
    assert opened_urls == [launch.browser_url]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(7_000)
        await page.goto(launch.browser_url)

        await page.get_by_role("heading", name="Chatbook Canvas").wait_for()
        frame = page.frame_locator("#canvas-preview")
        await frame.get_by_role("heading", name="revision-1").wait_for()
        await page.get_by_text("Temporary", exact=True).wait_for()
        await page.get_by_text("Following", exact=True).wait_for()
        await page.locator("#loading-state").wait_for(state="hidden")
        capture_dir = os.environ.get("TLDW_CANVAS_CAPTURE_DIR")
        if capture_dir:
            output = Path(capture_dir)
            output.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=output / "canvas-shell-desktop.png")
            await page.set_viewport_size({"width": 390, "height": 844})
            await page.screenshot(path=output / "canvas-shell-mobile.png")
            await page.locator("#pin-button").focus()
            for _ in range(5):
                await page.keyboard.press("Tab")
            assert await page.evaluate("document.activeElement.id") == "close-button"
            assert await page.locator(".canvas-actions").evaluate(
                "element => element.scrollLeft > 0"
            )
            await page.set_viewport_size({"width": 1440, "height": 900})
        await page.locator("#compatibility").evaluate(
            "element => { element.hidden = false; }"
        )
        await page.get_by_role("button", name="Reopen with scripts disabled").click()
        await page.get_by_text(
            "Opened with generated scripts disabled.", exact=True
        ).wait_for()
        await frame.get_by_role("heading", name="revision-1").wait_for()
        await page.locator("#compatibility").evaluate(
            "element => { element.hidden = true; }"
        )

        await page.get_by_role("button", name="Pin revision").click()
        authority.publish("revision-2", sequence=2)
        gateway.change_selection(
            browser_session_id="browser-native-flow", scope=_scope("revision-2")
        )
        await page.get_by_text("New version available", exact=True).wait_for()
        await frame.get_by_role("heading", name="revision-1").wait_for()

        await page.locator("#follow-button").click()
        await frame.get_by_role("heading", name="revision-2").wait_for()
        await page.get_by_text("Updated · Undo / View previous", exact=True).wait_for()

        authority.publish("revision-3", sequence=3)
        gateway.change_selection(
            browser_session_id="browser-native-flow", scope=_scope("revision-3")
        )
        await frame.get_by_role("heading", name="revision-3").wait_for()

        authority.publish("revision-branch", kind="selection_changed", sequence=4)
        gateway.change_selection(
            browser_session_id="browser-native-flow", scope=_scope("revision-branch")
        )
        await frame.get_by_role("heading", name="revision-branch").wait_for()
        await browser.close()

    await gateway.aclose()
