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
    CanvasGatewayNavigation,
    CanvasGatewayOption,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from tldw_chatbook.Canvas.limits import sha256_utf8
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController


@dataclass
class _NativeFlowAuthority:
    events: dict[str, CanvasGatewayEvent] = field(default_factory=dict)
    following: bool = True
    latest_revision: str = "revision-1"

    def publish(
        self,
        revision_id: str,
        *,
        kind: str = "updated",
        sequence: int,
        notice: str = "",
    ) -> None:
        self.latest_revision = revision_id
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
        if scope.revision_id == "runtime-failure":
            return compile_canvas_document(
                "<!doctype html><h1>Failure fixture</h1><script>throw new Error('fixture')</script>"
            )
        return compile_canvas_document(
            "<!doctype html><html><body>"
            f"<h1>{scope.revision_id}</h1><p>Isolated preview</p>"
            "</body></html>"
        )

    async def read_source(self, scope: CanvasGatewayScope) -> CanvasSourceResponse:
        source = f"<!doctype html><title>{scope.revision_id}</title>"
        return CanvasSourceResponse(source, sha256_utf8(source))

    async def describe_selection(self, scope: CanvasGatewayScope):
        event = self.events.get(scope.revision_id)
        metadata = event.metadata if event is not None else {}
        return CanvasGatewayProjection(
            scope=scope,
            options=(CanvasGatewayOption("canvas-a", scope.revision_id, "Release planner"),),
            title="Release planner",
            sequence=int(metadata.get("sequence", 1)),
            parent_revision_id={
                "revision-2": "revision-1",
                "revision-3": "revision-2",
                "revision-branch": "revision-3",
            }.get(scope.revision_id),
            source_bytes=96,
            content_sha256="a" * 64,
            origin_message_id=str(metadata.get("origin_message_id", "message-1")),
            origin_turn_id=str(metadata.get("origin_turn_id", "turn-1")),
            temporary=True,
            following=self.following,
        )

    async def navigate(self, scope: CanvasGatewayScope, *, action: str, canvas_id=None, title=None):
        del canvas_id, title
        next_scope = scope
        if action == "pin":
            self.following = False
        elif action in {"follow", "select"}:
            self.following = True
            next_scope = _scope(self.latest_revision)
        if action == "previous":
            parent = (await self.describe_selection(scope)).parent_revision_id
            if parent is None:
                raise ValueError("no previous")
            next_scope = _scope(parent)
            self.following = False
        return CanvasGatewayNavigation(next_scope, await self.describe_selection(next_scope))

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
            primary = await page.locator(".canvas-primary-row").bounding_box()
            secondary = await page.locator(".canvas-secondary-row").bounding_box()
            assert primary is not None and secondary is not None
            assert secondary["y"] >= primary["y"] + primary["height"] - 1
            assert await page.get_by_text("More actions →", exact=True).is_visible()
            assert await page.locator(".canvas-controls-scroll").evaluate(
                "element => element.scrollWidth > element.clientWidth"
            )
            await page.locator("#pin-button").focus()
            for _ in range(5):
                await page.keyboard.press("Tab")
            assert await page.evaluate("document.activeElement.id") == "close-button"
            assert await page.locator(".canvas-controls-scroll").evaluate(
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
        await page.get_by_text("Pinned", exact=True).wait_for()
        authority.publish("revision-2", sequence=2)
        gateway.change_selection(
            browser_session_id="browser-native-flow", scope=_scope("revision-2")
        )
        await page.get_by_text("New version available", exact=True).wait_for()
        await frame.get_by_role("heading", name="revision-1").wait_for()

        await page.locator("#follow-button").click()
        await frame.get_by_role("heading", name="revision-2").wait_for()
        await page.get_by_text("Updated · View previous", exact=True).wait_for()
        await page.get_by_role("button", name="View previous").click()
        await frame.get_by_role("heading", name="revision-1").wait_for()
        await page.get_by_text("Pinned", exact=True).wait_for()
        await page.get_by_text("Revision 1", exact=True).wait_for()
        await page.get_by_role("button", name="Follow latest").click()
        await frame.get_by_role("heading", name="revision-2").wait_for()

        await page.get_by_role("button", name="Inspect source").click()
        await page.locator("#source-view").wait_for()
        for _ in range(30):
            if "revision-2" in await page.locator("#source-view").input_value():
                break
            await page.wait_for_timeout(50)
        assert await page.locator("#source-panel").get_attribute("role") == "dialog"
        assert await page.locator("#source-view").input_value() == (
            "<!doctype html><title>revision-2</title>"
        )
        assert await page.locator(".canvas-toolbar").get_attribute("inert") is not None
        await page.keyboard.press("Escape")
        assert await page.evaluate("document.activeElement.id") == "source-button"

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


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_production_authority_import_selector_rename_and_hot_reload() -> None:
    session_id = "temporary-browser-session"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    counter = 0

    def scope_resolver(requested: str) -> CanvasScope:
        nonlocal counter
        assert requested == session_id
        counter += 1
        return CanvasScope(
            session_id=session_id,
            conversation_id=session_id,
            active_message_ids=("user-1", "assistant-1"),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id=f"interaction-{counter}",
        )

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=scope_resolver,
        canvas_controller=controller,
    )
    first = authority.import_html(
        session_id=session_id,
        source="<!doctype html><title>First board</title><h1>First version</h1>",
        create_new=True,
    )
    second = authority.import_html(
        session_id=session_id,
        source="<!doctype html><title>Second board</title><h1>Second version</h1>",
        create_new=True,
    )
    scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-production-authority",
        canvas_id=second.canvas_id,
        revision_id=second.revision_id,
    )
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(scope)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1100, "height": 760})
        page.set_default_timeout(7_000)
        await page.goto(launch.browser_url)
        frame = page.frame_locator("#canvas-preview")
        await frame.get_by_role("heading", name="Second version").wait_for()
        assert await page.locator("#canvas-selector option").count() == 2

        await page.locator("#canvas-selector").select_option(first.canvas_id)
        await frame.get_by_role("heading", name="First version").wait_for()
        await page.locator("#canvas-title").fill("Renamed board")
        await page.locator("#canvas-title").press("Enter")
        await page.get_by_text("Title saved as a new revision.", exact=True).wait_for()
        await page.get_by_text("Revision 2", exact=True).wait_for()

        updated = authority.import_html(
            session_id=session_id,
            source="<!doctype html><title>Renamed board</title><h1>Hot reloaded</h1>",
            create_new=False,
        )
        await frame.get_by_role("heading", name="Hot reloaded").wait_for()
        await page.get_by_text("Revision 3", exact=True).wait_for()
        assert updated.parent_revision_id is not None
        await browser.close()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_runtime_failure_and_javascript_disabled_have_visible_recovery() -> None:
    authority = _NativeFlowAuthority()
    authority.publish("runtime-failure", sequence=1)
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope("runtime-failure"))
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 900, "height": 600})
        await page.goto(launch.browser_url)
        await page.get_by_role("button", name="Reopen with scripts disabled").wait_for()
        await page.get_by_role("button", name="Reopen with scripts disabled").click()
        await page.get_by_text("Opened with generated scripts disabled.", exact=True).wait_for()
        await page.frame_locator("#canvas-preview").get_by_role(
            "heading", name="Failure fixture"
        ).wait_for()

        no_js = await browser.new_context(java_script_enabled=False)
        no_js_page = await no_js.new_page()
        await no_js_page.goto(launch.clean_url)
        recovery = no_js_page.get_by_role(
            "heading", name="Chatbook Canvas needs trusted shell scripts"
        )
        await recovery.wait_for()
        box = await recovery.bounding_box()
        assert box is not None and box["y"] < 600
        await no_js.close()
        await browser.close()
    await gateway.aclose()
