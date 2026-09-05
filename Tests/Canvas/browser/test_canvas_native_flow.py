"""Real-browser behavior for the trusted preview-first Canvas shell."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from Tests.Chatbooks.test_chatbook_canvas_round_trip import _seed_canvas_graph
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
from tldw_chatbook.Canvas.service import CanvasService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleMessageRole,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@dataclass
class _NativeFlowAuthority:
    events: dict[str, CanvasGatewayEvent] = field(default_factory=dict)
    following: bool = True
    latest_revision: str = "revision-1"
    temporary: bool = True

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
                "temporary": self.temporary,
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
        if scope.revision_id == "active-script":
            return compile_canvas_document(
                "<!doctype html><output id='ticks'>0</output>"
                "<script>setInterval(() => { const node = document.querySelector('#ticks'); "
                "node.textContent = String(Number(node.textContent) + 1); }, 10);</script>"
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
            temporary=self.temporary,
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
async def test_native_gateway_shutdown_destroys_an_actively_running_preview() -> None:
    authority = _NativeFlowAuthority()
    authority.publish("active-script", kind="selection_changed", sequence=1)
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope("active-script"))

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 760})
        page.set_default_timeout(7_000)
        await page.goto(launch.browser_url)
        ticks = page.frame_locator("#canvas-preview").locator("#ticks")
        await expect(ticks).not_to_have_text("0")

        await gateway.aclose()

        await expect(page.locator("#canvas-preview")).to_have_attribute(
            "src", "about:blank"
        )
        await expect(page.get_by_text("Disconnected", exact=True)).to_be_visible()
        assert await page.locator("#canvas-preview").evaluate(
            "frame => frame.contentDocument === null || frame.contentDocument.body.textContent === ''"
        )
        await page.evaluate(
            """() => {
                const probe = document.createElement('button');
                probe.id = 'disconnect-focus-probe';
                document.body.append(probe);
                probe.focus();
            }"""
        )
        await page.wait_for_timeout(800)
        assert (
            await page.evaluate("document.activeElement.id") == "disconnect-focus-probe"
        )
        await browser.close()


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

        authority.temporary = False
        await page.get_by_role("button", name="Pin revision").click()
        await page.get_by_text("Pinned", exact=True).wait_for()
        await expect(page.locator("#temporary-badge")).to_be_hidden()
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
async def test_native_temporary_promotion_reopens_and_unsaved_close_destroys(
    tmp_path: Path,
) -> None:
    """Show both atomic promotion and unsaved destruction in the real shell."""

    database = CharactersRAGDB(tmp_path / "native-lifecycle.sqlite", "native-browser")
    controller = ConsoleCanvasController(durable_service=CanvasService(database))
    holder: dict[str, NativeConsoleCanvasAuthority] = {}
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(database),
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
        on_canvas_context_changed=lambda session_id: (
            holder.get("authority")
            and holder["authority"].sync_live_context(session_id)
        ),
    )
    scopes: dict[str, CanvasScope] = {}

    def resolve(session_id: str) -> CanvasScope:
        return scopes[session_id]

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=resolve,
        canvas_controller=controller,
    )
    holder["authority"] = authority
    gateway = CanvasGateway(authority=authority)
    authority.bind_gateway_invalidator(gateway.mark_browser_session_unavailable)
    try:
        saved = store.create_session(ephemeral=True)
        saved_message = store.append_message(
            saved.id, role=ConsoleMessageRole.ASSISTANT, content="temporary Canvas"
        )
        scopes[saved.id] = CanvasScope(
            session_id=saved.id,
            conversation_id=saved.id,
            active_message_ids=(saved_message.id,),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="temporary-import",
        )
        root = authority.import_html(
            session_id=saved.id,
            source="<!doctype html><title>Temporary board</title><main>root</main>",
            create_new=True,
        )
        scopes[saved.id] = replace(scopes[saved.id], run_id="temporary-rename")
        selected = authority.gateway_scope(
            session_id=saved.id,
            browser_session_id="temporary-before-save",
            canvas_id=root.canvas_id,
            revision_id=root.revision_id,
        )
        renamed = authority.navigate(selected, action="rename", title="Saved board")
        launch = await gateway.open_shell(renamed.scope)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=_chromium_executable(playwright.chromium),
            )
            page = await browser.new_page()
            await page.goto(launch.browser_url)
            await expect(page.get_by_text("Temporary", exact=True)).to_be_visible()
            await expect(page.get_by_text("Revision 2", exact=True)).to_be_visible()

            conversation_id = store.promote_ephemeral_session(saved.id)
            persisted_message_id = store.get_message(
                saved_message.id
            ).persisted_message_id
            assert conversation_id is not None and persisted_message_id is not None
            scopes[saved.id] = replace(
                scopes[saved.id],
                conversation_id=conversation_id,
                active_message_ids=(persisted_message_id,),
                selected_canvas_id=root.canvas_id,
                selected_revision_id=renamed.scope.revision_id,
                run_id="durable-reopen",
            )
            durable_scope = authority.gateway_scope(
                session_id=saved.id,
                browser_session_id="durable-after-save",
                canvas_id=root.canvas_id,
                revision_id=renamed.scope.revision_id,
                follow_latest=False,
            )
            durable_launch = await gateway.open_shell(durable_scope)
            await page.goto(durable_launch.browser_url)
            await expect(page.locator("#temporary-badge")).to_be_hidden()
            await expect(page.get_by_text("Revision 2", exact=True)).to_be_visible()
            await expect(page.get_by_text("Pinned", exact=True)).to_be_visible()
            await expect(
                page.frame_locator("#canvas-preview").locator("main")
            ).to_have_text("root")

            discarded = store.create_session(ephemeral=True)
            discarded_message = store.append_message(
                discarded.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="discard this Canvas",
            )
            scopes[discarded.id] = CanvasScope(
                session_id=discarded.id,
                conversation_id=discarded.id,
                active_message_ids=(discarded_message.id,),
                selected_canvas_id=None,
                selected_revision_id=None,
                run_id="discard-import",
            )
            doomed = authority.import_html(
                session_id=discarded.id,
                source="<!doctype html><title>Unsaved</title><main>doomed</main>",
                create_new=True,
            )
            doomed_scope = authority.gateway_scope(
                session_id=discarded.id,
                browser_session_id="temporary-discard",
                canvas_id=doomed.canvas_id,
                revision_id=doomed.revision_id,
            )
            doomed_launch = await gateway.open_shell(doomed_scope)
            doomed_page = await browser.new_page()
            await doomed_page.goto(doomed_launch.browser_url)
            await expect(
                doomed_page.frame_locator("#canvas-preview").locator("main")
            ).to_have_text("doomed")
            store.close_session(discarded.id)
            await expect(
                doomed_page.get_by_text("Disconnected", exact=True)
            ).to_be_visible()
            await expect(doomed_page.locator("#canvas-preview")).to_have_attribute(
                "src", "about:blank"
            )
            assert controller.promotion_contribution(discarded.id) is None
            await browser.close()
    finally:
        await gateway.aclose()
        controller.close_runtime()
        database.close_connection()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_imported_archive_reopens_exact_branch_after_source_purge(
    tmp_path: Path,
) -> None:
    """Prove an imported branch remains runnable after the source DB is gone."""

    source_path = tmp_path / "archive-source.sqlite"
    expected = _seed_canvas_graph(source_path)
    archive_path = tmp_path / "archive.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path / "archive-create"
    creator.temp_dir.mkdir()
    assert creator.create_chatbook(
        name="Canvas browser archive",
        description="browser reopen",
        content_selections={
            ContentType.CONVERSATION: [str(expected["conversation_id"])]
        },
        output_path=archive_path,
    )[0]
    source_path.unlink()
    assert not source_path.exists()

    target_path = tmp_path / "archive-target.sqlite"
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "archive-import"
    importer.temp_dir.mkdir()
    assert importer.import_chatbook(
        archive_path,
        conflict_resolution=ConflictResolution.RENAME,
    )[0]
    database = CharactersRAGDB(target_path, "archive-browser")
    gateway = None
    try:
        conversation = database.get_conversation_by_name("Canvas archive graph")[0]
        conversation_id = str(conversation["id"])
        repository = CanvasService(database)
        active_message_ids: list[str] = []
        message_id = str(conversation["active_leaf_message_id"])
        while message_id:
            active_message_ids.append(message_id)
            row = (
                database.get_connection()
                .execute(
                    "SELECT parent_message_id FROM messages WHERE id = ?",
                    (message_id,),
                )
                .fetchone()
            )
            assert row is not None
            message_id = str(row[0]) if row[0] is not None else ""
        active_message_ids.reverse()
        canvases = repository.list_canvases(
            CanvasScope(
                session_id="archive-browser",
                conversation_id=conversation_id,
                active_message_ids=tuple(active_message_ids),
                selected_canvas_id=None,
                selected_revision_id=None,
                run_id="archive-reopen",
            )
        )
        canvas = next(item for item in canvases if item.title == "Planner alternate")
        scope = CanvasScope(
            session_id="archive-browser",
            conversation_id=conversation_id,
            active_message_ids=tuple(active_message_ids),
            selected_canvas_id=canvas.canvas_id,
            selected_revision_id=canvas.revision_id,
            run_id="archive-reopen",
        )
        controller = ConsoleCanvasController(durable_service=repository)
        controller.activate_session(scope.session_id)
        authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _requested: scope,
            canvas_controller=controller,
        )
        browser_scope = authority.gateway_scope(
            session_id=scope.session_id,
            browser_session_id="archive-browser-route",
            canvas_id=canvas.canvas_id,
            revision_id=canvas.revision_id,
            follow_latest=False,
        )
        gateway = CanvasGateway(authority=authority)
        launch = await gateway.open_shell(browser_scope)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=_chromium_executable(playwright.chromium),
            )
            page = await browser.new_page()
            await page.goto(launch.browser_url)
            await expect(
                page.frame_locator("#canvas-preview").locator("main")
            ).to_have_text("right 🌿")
            await expect(page.get_by_text("Revision 3", exact=True)).to_be_visible()
            await expect(page.get_by_text("Pinned", exact=True)).to_be_visible()
            await browser.close()
    finally:
        if gateway is not None:
            await gateway.aclose()
        database.close_connection()


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
async def test_real_branch_transition_clears_unreachable_preview_until_reopened() -> (
    None
):
    controller = ConsoleCanvasController()
    holder = {}
    store = ConsoleChatStore(
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
        on_canvas_context_changed=lambda session_id: (
            holder.get("authority")
            and holder["authority"].sync_live_context(session_id)
        ),
    )
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="root")
    left = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="left"
    )

    def resolve(requested: str) -> CanvasScope:
        if store.active_session_id != requested:
            raise RuntimeError("Canvas session is no longer active")
        return CanvasScope(
            session_id=requested,
            conversation_id=requested,
            active_message_ids=tuple(store.active_path_message_ids(requested)),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="browser-branch-transition",
        )

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=resolve,
        canvas_controller=controller,
    )
    holder["authority"] = authority
    created = authority.import_html(
        session_id=session.id,
        source="<!doctype html><h1>Left branch preview</h1>",
        source_message_id=left.id,
        origin_message_id=left.id,
        source_turn_id="left-browser-import",
        block_index=0,
        block_identity=f"{left.id}:canvas-html:0",
    )
    scope = authority.gateway_scope(
        session_id=session.id,
        browser_session_id="browser-unreachable-production",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    gateway = CanvasGateway(authority=authority)
    authority.bind_gateway_invalidator(gateway.mark_browser_session_unavailable)
    launch = await gateway.open_shell(scope)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1000, "height": 700})
        page.set_default_timeout(7_000)
        await page.goto(launch.browser_url)
        await (
            page.frame_locator("#canvas-preview")
            .get_by_role("heading", name="Left branch preview")
            .wait_for()
        )
        await page.get_by_role("button", name="Inspect source").click()
        await page.locator("#source-panel").wait_for(state="visible")
        assert await page.evaluate("document.activeElement.id") == (
            "source-close-button"
        )

        store.create_sibling(
            left.id, role=ConsoleMessageRole.ASSISTANT, content="right"
        )
        recovery = page.locator("#loading-state")
        await recovery.get_by_text(
            "Unavailable on this branch. Return to Chatbook and reopen this "
            "Canvas from a reachable transcript card.",
            exact=True,
        ).wait_for()
        assert await recovery.get_attribute("role") == "status"
        assert await recovery.get_attribute("aria-live") == "polite"
        assert await recovery.get_attribute("aria-atomic") == "true"
        assert await page.locator("#source-panel").is_hidden()
        assert await page.evaluate("document.activeElement.id") == "close-button"
        assert await page.get_by_role("button", name="Close").is_enabled()
        assert await page.locator("#canvas-preview").get_attribute("src") in {
            None,
            "about:blank",
        }
        assert await page.get_by_text("Disconnected", exact=True).is_visible()

        store.set_active_leaf(session.id, left.id)
        await page.wait_for_timeout(500)
        assert await recovery.is_visible()

        reopened_scope = authority.gateway_scope(
            session_id=session.id,
            browser_session_id="browser-reopened-production",
            canvas_id=created.canvas_id,
            revision_id=created.revision_id,
        )
        reopened = await gateway.open_shell(reopened_scope)
        await page.goto(reopened.browser_url)
        await (
            page.frame_locator("#canvas-preview")
            .get_by_role("heading", name="Left branch preview")
            .wait_for()
        )
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
        await page.get_by_text(
            "Opened with generated scripts disabled.", exact=True
        ).wait_for()
        await (
            page.frame_locator("#canvas-preview")
            .get_by_role("heading", name="Failure fixture")
            .wait_for()
        )

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


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_bridge_confirmation_submits_exact_draft_and_downloads_passive_blob() -> (
    None
):
    session_id = "temporary-confirmation-session"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    composer = {"generation": 1}
    drafts: list[str] = []

    def capture(_target):
        generation = composer["generation"]

        def apply(text: str) -> None:
            if composer["generation"] != generation:
                raise RuntimeError("composer changed")
            drafts.append(text)

        return apply

    def scope_resolver(requested: str) -> CanvasScope:
        assert requested == session_id
        return CanvasScope(
            session_id=session_id,
            conversation_id=session_id,
            active_message_ids=("user-1", "assistant-1"),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="confirmation-run",
        )

    source = """<!doctype html><html><head><title>Bridge tools</title></head><body>
    <button id="submit-text">Submit text</button>
    <button id="submit-json">Submit JSON</button>
    <button id="submit-max">Submit maximum text</button>
    <button id="submit-exponent">Submit exponent</button>
    <button id="submit-key-order">Submit key order</button>
    <button id="submit-negative-zero">Submit negative zero</button>
    <button id="submit-recovery-threshold">Submit recovery threshold</button>
    <button id="submit-final-expiry">Submit final expiry</button>
    <button id="submit-mismatch">Submit mismatched presentation</button>
    <button id="download-text">Download text</button>
    <script>
      document.getElementById("submit-text").addEventListener("click", () => canvas.submit("  exact\\ntext  "));
      document.getElementById("submit-json").addEventListener("click", () => canvas.submit({z: 1, a: [true, null]}));
      document.getElementById("submit-max").addEventListener("click", () => canvas.submit("x".repeat(16 * 1024)));
      document.getElementById("submit-exponent").addEventListener("click", () => canvas.submit({small: 1e-7, negative_zero: -0}));
      document.getElementById("submit-key-order").addEventListener("click", () => {
        const value = {};
        value["\\u{10000}"] = "supplementary";
        value["\\uE000"] = "bmp";
        canvas.submit(value);
      });
      document.getElementById("submit-negative-zero").addEventListener("click", () => canvas.submit(-0));
      document.getElementById("submit-recovery-threshold").addEventListener("click", () => canvas.submit("recovery threshold"));
      document.getElementById("submit-final-expiry").addEventListener("click", () => canvas.submit("final expiry"));
      document.getElementById("submit-mismatch").addEventListener("click", () => canvas.submit({mismatch: 1}));
      document.getElementById("download-text").addEventListener("click", () => canvas.download({filename: " result.txt ", mime_type: "text/plain", data: "full result\\n"}));
    </script></body></html>"""
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=scope_resolver,
        canvas_controller=controller,
        bridge_prepare=capture,
    )
    created = authority.import_html(
        session_id=session_id,
        source=source,
        create_new=True,
    )
    scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-confirmation",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(scope)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 780})
        page.set_default_timeout(7_000)

        async def shape_preparation(route) -> None:
            response = await route.fetch()
            presentation = await response.json()
            if presentation.get("complete_text") == "recovery threshold":
                presentation["expires_in_seconds"] = 11
            elif presentation.get("complete_text") == "final expiry":
                presentation["expires_in_seconds"] = 0.25
            elif presentation.get("complete_text") == '{"mismatch":1}':
                presentation["complete_text"] = '{"mismatch":2}'
                presentation["byte_size"] = 14
            await route.fulfill(response=response, json=presentation)

        await page.route("**/api/bridge/prepare", shape_preparation)
        prepare_request_ids: list[str] = []
        page.on(
            "request",
            lambda request: (
                prepare_request_ids.append(
                    request.post_data_json["request"]["request_id"]
                )
                if request.url.endswith("/api/bridge/prepare")
                else None
            ),
        )
        await page.add_init_script(
            """(() => {
              const revoke = URL.revokeObjectURL.bind(URL);
              URL.revokeObjectURL = (value) => { window.__canvasLastObjectUrlRevoked = true; revoke(value); };
              Document.prototype.execCommand = function(command) {
                if (command !== "copy") return false;
                const text = this.getElementById("bridge-complete-text");
                window.__canvasCopiedText = text.value.slice(text.selectionStart, text.selectionEnd);
                return true;
              };
              window.close = () => { window.__canvasCloseAttempted = true; };
            })();"""
        )
        await page.goto(launch.browser_url)
        frame = page.frame_locator("#canvas-preview")
        await frame.get_by_role("button", name="Submit text").click()
        dialog = page.get_by_role("dialog", name="Send result to chat")
        await dialog.wait_for()
        assert (
            await page.evaluate("document.activeElement.id") == "bridge-cancel-button"
        )
        assert (
            await page.locator("#bridge-complete-text").input_value()
            == "  exact\ntext  "
        )
        assert await page.locator("#bridge-target").text_content() == (
            f"Conversation {session_id} · Canvas “Bridge tools” · Revision 1 · "
            f"Canvas ID {created.canvas_id} · Revision ID {created.revision_id}"
        )
        assert await page.locator("#bridge-expiry").text_content() in {
            "Review expires in 5:00",
            "Review expires in 4:59",
        }
        assert await page.locator(".canvas-toolbar").get_attribute("inert") is not None
        await page.get_by_role("button", name="Copy result").click()
        assert await page.evaluate("window.__canvasCopiedText") == "  exact\ntext  "
        await page.keyboard.press("Escape")
        await dialog.wait_for(state="hidden")
        assert drafts == []

        await frame.get_by_role("button", name="Submit text").click()
        await page.get_by_role("button", name="Send to composer").click()
        await dialog.wait_for(state="hidden")
        await page.get_by_text("Draft inserted · Review it in Chatbook before sending.", exact=True).wait_for()
        assert drafts == ["  exact\ntext  "]

        await frame.get_by_role("button", name="Submit JSON").click()
        await expect(page.locator("#bridge-complete-text")).to_have_value('{"a":[true,null],"z":1}')
        composer["generation"] += 1
        await page.get_by_role("button", name="Send to composer").click()
        await page.get_by_text("The Chatbook draft changed. Nothing was inserted.", exact=True).wait_for()
        assert await page.get_by_role("button", name="Copy result").is_visible()
        assert drafts == ["  exact\ntext  "]
        refused_request_id = prepare_request_ids[-1]
        composer["generation"] = 1
        await page.get_by_role("button", name="Retry confirmation").click()
        await page.get_by_role("dialog", name="Send result to chat").wait_for()
        assert prepare_request_ids[-1] != refused_request_id
        assert (
            await page.locator("#bridge-complete-text").input_value()
            == '{"a":[true,null],"z":1}'
        )
        await page.get_by_role("button", name="Send to composer").click()
        await dialog.wait_for(state="hidden")
        await page.get_by_text(
            "Draft inserted · Review it in Chatbook before sending.", exact=True
        ).wait_for()
        assert drafts == ["  exact\ntext  ", '{"a":[true,null],"z":1}']

        for button_name, expected_text in (
            ("Submit key order", '{"\ue000":"bmp","\U00010000":"supplementary"}'),
            ("Submit exponent", '{"negative_zero":0,"small":1e-07}'),
            ("Submit negative zero", "0"),
        ):
            await frame.get_by_role("button", name=button_name).click()
            await page.get_by_role("dialog", name="Send result to chat").wait_for()
            assert (
                await page.locator("#bridge-complete-text").input_value()
                == expected_text
            )
            async with page.expect_response(
                lambda response: response.url.endswith("/api/bridge")
            ):
                await page.keyboard.press("Escape")

        await frame.get_by_role("button", name="Submit maximum text").click()
        await page.get_by_role("dialog", name="Send result to chat").wait_for()
        assert (
            len(await page.locator("#bridge-complete-text").input_value()) == 16 * 1024
        )
        assert await page.locator("#bridge-expiry").text_content() in {
            "Review expires in 5:00",
            "Review expires in 4:59",
        }
        assert await page.locator("#bridge-expiry").get_attribute("aria-live") is None
        assert await page.locator("#bridge-expiry").get_attribute("role") is None
        described_by = (await dialog.get_attribute("aria-describedby") or "").split()
        assert {"bridge-summary", "bridge-expiry"}.issubset(described_by)
        await page.wait_for_timeout(1_100)
        assert await page.locator("#bridge-recovery").is_hidden()
        async with page.expect_response(
            lambda response: response.url.endswith("/api/bridge")
        ):
            await page.keyboard.press("Escape")

        await frame.get_by_role("button", name="Submit JSON").click()
        await page.get_by_role("dialog", name="Send result to chat").wait_for()
        composer["generation"] += 1
        await page.get_by_role("button", name="Send to composer").click()
        await page.get_by_text("The Chatbook draft changed. Nothing was inserted.", exact=True).wait_for()
        await page.get_by_role("button", name="Close Canvas and return").click()
        await page.get_by_text(
            "This browser could not return to Chatbook automatically. Return to the matching Chatbook conversation; the result was not inserted.",
            exact=True,
        ).wait_for()
        assert await page.evaluate("window.__canvasCloseAttempted === true")
        assert await page.locator(".canvas-toolbar").get_attribute("inert") is None

        await frame.get_by_role("button", name="Download text").click()
        await page.get_by_role("dialog", name="Download generated file").wait_for()
        assert await page.get_by_text("result.txt", exact=True).is_visible()
        assert await page.get_by_text("text/plain", exact=True).is_visible()
        assert await page.get_by_text("12 bytes", exact=True).is_visible()
        async with page.expect_download() as download_info:
            await page.get_by_role("button", name="Download file").click()
        download = await download_info.value
        assert download.suggested_filename == "result.txt"
        assert await page.evaluate("window.__canvasLastObjectUrlRevoked === true")

        await frame.get_by_role("button", name="Submit recovery threshold").click()
        await page.get_by_role("dialog", name="Send result to chat").wait_for()
        composer["generation"] += 1
        await page.get_by_role("button", name="Send to composer").click()
        await page.get_by_text(
            "Canvas confirmation expires in 10 seconds.", exact=True
        ).wait_for()
        assert await page.locator("#bridge-recovery").text_content() == (
            "The Chatbook draft changed. Nothing was inserted."
        )
        assert await page.locator("#bridge-recovery").is_visible()
        assert await page.get_by_role("button", name="Copy result").is_visible()
        assert await page.get_by_role("button", name="Retry confirmation").is_visible()
        assert await page.get_by_role(
            "button", name="Close Canvas and return"
        ).is_visible()
        assert (
            await page.locator("#bridge-expiry-status").get_attribute("role")
            == "status"
        )
        assert (
            await page.locator("#bridge-expiry-status").get_attribute("aria-live")
            == "polite"
        )
        composer["generation"] -= 1
        await page.keyboard.press("Escape")

        capture_dir = os.environ.get("TLDW_CANVAS_CONFIRMATION_CAPTURE_DIR")
        if capture_dir:
            output = Path(capture_dir)
            output.mkdir(parents=True, exist_ok=True)
            await frame.get_by_role("button", name="Submit JSON").click()
            await page.get_by_role("dialog", name="Send result to chat").wait_for()
            await page.screenshot(path=output / "canvas-confirmation-desktop.png")
            await page.set_viewport_size({"width": 390, "height": 844})
            await page.screenshot(path=output / "canvas-confirmation-narrow.png")
            async with page.expect_response(
                lambda response: response.url.endswith("/api/bridge")
            ):
                await page.keyboard.press("Escape")

        await frame.get_by_role("button", name="Submit mismatched presentation").click()
        await page.get_by_text(
            "Canvas request could not be confirmed. Reload the preview and try again.",
            exact=True,
        ).wait_for()
        assert await page.get_by_role("dialog").count() == 0
        await page.get_by_role("button", name="Reload").click()
        await frame.get_by_role("button", name="Submit final expiry").click()
        await page.get_by_text(
            "Canvas confirmation expired. Request it again from the preview.", exact=True
        ).wait_for()
        assert await page.locator("#notice").get_attribute("role") == "status"
        assert await page.locator("#notice").get_attribute("aria-live") == "polite"
        assert await page.get_by_role("dialog").count() == 0
        await browser.close()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "javascript_value",
        "process_text",
        "tampered_text",
        "expected_parser_entries",
    ),
    (
        pytest.param(
            "{large: 9007199254740992}",
            '{"large":9007199254740992}',
            '{"large":9007199254740993}',
            None,
            id="large-integer-collision",
        ),
        pytest.param(
            "{decimal: 0.1}",
            '{"decimal":0.1}',
            '{"decimal":0.10000000000000001}',
            None,
            id="rounded-decimal-collision",
        ),
        pytest.param(
            "{safe: true}",
            '{"safe":true}',
            '{"oversized":"' + ("x" * (16 * 1024)) + '"}',
            0,
            id="oversized-presentation-skips-parser",
        ),
    ),
)
async def test_trusted_shell_bounds_and_compares_structured_presentation_losslessly(
    javascript_value: str,
    process_text: str,
    tampered_text: str,
    expected_parser_entries: int | None,
) -> None:
    session_id = "temporary-numeric-collision-session"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)

    def scope_resolver(requested: str) -> CanvasScope:
        assert requested == session_id
        return CanvasScope(
            session_id=session_id,
            conversation_id=session_id,
            active_message_ids=("user-1", "assistant-1"),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="numeric-collision-run",
        )

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=scope_resolver,
        canvas_controller=controller,
        bridge_prepare=lambda _target: lambda _text: None,
    )
    created = authority.import_html(
        session_id=session_id,
        source=(
            "<!doctype html><button id='submit'>Submit</button><script>"
            "document.getElementById('submit').addEventListener('click', () => "
            f"canvas.submit({javascript_value}));"
            "</script>"
        ),
        create_new=True,
    )
    scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-numeric-collision",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(scope)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 900, "height": 700})
        page.set_default_timeout(7_000)

        async def instrument_lossless_parser(route) -> None:
            response = await route.fetch()
            source = await response.text()
            entry = "  function parseLosslessBridgeJson(source) {\n"
            assert source.count(entry) == 1
            instrumented = source.replace(
                entry,
                entry
                + "    window.__canvasLosslessParseEntries = "
                + "(window.__canvasLosslessParseEntries || 0) + 1;\n",
            )
            await route.fulfill(response=response, body=instrumented)

        async def tamper_preparation(route) -> None:
            request = route.request.post_data_json["request"]
            assert request["value"] == json.loads(process_text)
            await route.fulfill(
                json={
                    "request_id": request["request_id"],
                    "kind": "submit",
                    "conversation_id": session_id,
                    "canvas_id": created.canvas_id,
                    "revision_id": created.revision_id,
                    "canvas_title": "Canvas",
                    "revision_number": 1,
                    "complete_text": tampered_text,
                    "filename": None,
                    "mime_type": None,
                    "byte_size": len(tampered_text.encode("utf-8")),
                    "expires_in_seconds": 300,
                }
            )

        if expected_parser_entries is not None:
            await page.route("**/static/canvas_shell.js", instrument_lossless_parser)
        await page.route("**/api/bridge/prepare", tamper_preparation)
        await page.goto(launch.browser_url)
        await page.get_by_text("Connected", exact=True).wait_for()
        await (
            page.frame_locator("#canvas-preview")
            .get_by_role("button", name="Submit")
            .click()
        )
        await page.get_by_text(
            "Canvas request could not be confirmed. Reload the preview and try again.",
            exact=True,
        ).wait_for()
        assert await page.get_by_role("dialog").count() == 0
        if expected_parser_entries is not None:
            assert (
                await page.evaluate("window.__canvasLosslessParseEntries || 0")
                == expected_parser_entries
            )
        await browser.close()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_trusted_shell_rejects_forged_bridge_values_before_server_preparation() -> (
    None
):
    authority = _NativeFlowAuthority()
    authority.publish("revision-1", sequence=1)
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope("revision-1"))

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1000, "height": 700})
        prepare_requests: list[str] = []
        page.on(
            "request",
            lambda request: (
                prepare_requests.append(request.url)
                if request.url.endswith("/api/bridge/prepare")
                else None
            ),
        )
        await page.add_init_script(
            """(() => {
              const NativeMessageChannel = MessageChannel;
              window.MessageChannel = class extends NativeMessageChannel {
                constructor() {
                  super();
                  window.__canvasShellPort = this.port1;
                  const nativePost = this.port1.postMessage.bind(this.port1);
                  this.port1.postMessage = (message, transfer) => {
                    if (message && message.nonce) window.__canvasLoadNonce = message.nonce;
                    return nativePost(message, transfer);
                  };
                }
              };
            })();"""
        )
        await page.goto(launch.browser_url)
        await page.wait_for_function("() => Boolean(window.__canvasLoadNonce)")

        for forged in (
            "outer-extra",
            "nonfinite",
            "cycle",
            "deep",
            "oversize",
            "control-name",
            "fake-raster",
            "invalid-json-download",
        ):
            before = len(prepare_requests)
            await page.evaluate(
                """(kind) => {
                  let value = "safe";
                  if (kind === "nonfinite") value = {answer: Number.POSITIVE_INFINITY};
                  if (kind === "cycle") { value = {}; value.self = value; }
                  if (kind === "deep") {
                    value = "leaf";
                    for (let index = 0; index < 17; index += 1) value = [value];
                  }
                  if (kind === "oversize") value = "x".repeat(16 * 1024 + 1);
                  let message = {
                    type: "canvas:bridge-request",
                    nonce: window.__canvasLoadNonce,
                    request_id: `forged-${kind}`,
                    kind: "submit",
                    value,
                  };
                  if (kind === "outer-extra") message.extra = true;
                  if (kind === "control-name") {
                    message.kind = "download";
                    message.value = {filename: "\\nreport.txt", mime_type: "text/plain", data: "safe"};
                  }
                  if (kind === "fake-raster") {
                    message.kind = "download";
                    message.value = {filename: "pixel.png", mime_type: "image/png", data: "data:image/png;base64,PGh0bWw+"};
                  }
                  if (kind === "invalid-json-download") {
                    message.kind = "download";
                    message.value = {filename: "report.json", mime_type: "application/json", data: "not json"};
                  }
                  window.__canvasShellPort.onmessage({data: message});
                }""",
                forged,
            )
            await page.wait_for_timeout(75)
            assert len(prepare_requests) == before, forged
            assert await page.get_by_role("dialog").count() == 0, forged
        await browser.close()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_source_download_defaults_inert_and_runnable_html_requires_warning() -> (
    None
):
    authority = _NativeFlowAuthority()
    authority.publish("revision-1", sequence=1)
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope("revision-1"))
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1000, "height": 700})
        await page.goto(launch.browser_url)
        async with page.expect_download() as inert_info:
            await page.get_by_role("button", name="Download").click()
        assert (await inert_info.value).suggested_filename.endswith(".canvas.html.txt")

        await page.get_by_role("button", name="Inspect source").click()
        await page.get_by_role("button", name="Download as runnable HTML").click()
        warning = page.get_by_role("dialog", name="Run outside the Canvas sandbox?")
        await warning.wait_for()
        await page.get_by_text(
            "This HTML runs outside Chatbook and bypasses Canvas zero-egress and sandbox protections.",
            exact=True,
        ).wait_for()
        assert (
            await page.evaluate("document.activeElement.id") == "bridge-cancel-button"
        )
        await page.get_by_role("button", name="Cancel").click()
        await warning.wait_for(state="hidden")
        await browser.close()
    await gateway.aclose()
