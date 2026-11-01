"""The five external actions bind arguments to their real Textual workers."""

from __future__ import annotations

import asyncio
import json
import threading
import webbrowser

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Markdown
from textual.worker import WorkerState, get_current_worker

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_conversation_inspector import (
    InspectorHarness,
    _default_kwargs,
)
from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
    LibraryCollectionsController,
)
from tldw_chatbook.UI.Library_Modules.library_media_controller import (
    LibraryMediaController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    TAB_NEXT_SEND,
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleMarkdownMessage

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.asyncio,
    pytest.mark.usefixtures("retire_test_app_owners"),
]


class _TranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleMarkdownMessage(
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="A [source](https://example.test/transcript)",
                id="worker-binding",
            )
        )


@pytest.mark.parametrize(
    "kind", ["transcript", "settings", "media", "collections", "inspector"]
)
async def test_external_action_uses_its_owned_thread_worker(
    kind, monkeypatch, wait_for_condition
) -> None:
    url = f"https://example.test/{kind}?v=original"
    ordinary = {"role": "user", "content": "ordinary user content"}
    automatic_body = "AUTOMATIC-PROJECT-INSTRUCTION-MUST-NOT-BE-COPIED"

    async def snapshot() -> ConsoleContextSnapshot:
        return ConsoleContextSnapshot(
            current_messages=[],
            next_send_payload={
                "messages": [
                    ordinary,
                    {
                        "role": "user",
                        "content": automatic_body,
                        EPHEMERAL_ORIGIN_KEY: "project_instructions",
                    },
                ]
            },
        )

    if kind == "transcript":
        app = _TranscriptHarness()
    elif kind == "inspector":
        app = InspectorHarness(
            **_default_kwargs(initial_tab=TAB_NEXT_SEND, snapshot_factory=snapshot)
        )
    else:
        app_instance = _build_test_app()
        if kind == "collections":
            scope = app_instance.collections_capture_scope_service
            authority = scope.active_authority
            assert authority is not None
            saved = await scope.save_capture(
                CaptureSaveRequest(
                    authority.key,
                    url,
                    title="Worker binding",
                    text_content="Local body",
                )
            )
            assert saved.capture is not None
        app = DestinationHarness(
            app_instance, "settings" if kind == "settings" else "library"
        )

    calls = []
    entered = threading.Event()
    release = threading.Event()
    issuer_thread = threading.get_ident()

    def record(value):
        calls.append((value, threading.get_ident(), get_current_worker()))
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("external-action recorder was not released")

    if kind == "inspector":
        import pyperclip

        monkeypatch.setattr(pyperclip, "copy", record)
    else:
        monkeypatch.setattr(webbrowser, "open", record)

    async with app.run_test(size=(120, 40)):
        owner = app.screen
        if kind == "transcript":
            owner = app.query_one(ConsoleMarkdownMessage)
            handler = owner._open_link
            event = Markdown.LinkClicked(owner.query_one(Markdown), url)
        elif kind == "settings":
            assert type(owner) is SettingsScreen
            owner.active_category = SettingsCategoryId.ABOUT.value
            await wait_for_condition(
                lambda: bool(owner.query("#settings-about-markdown"))
            )
            handler = owner._handle_about_link
            event = Markdown.LinkClicked(
                owner.query_one("#settings-about-markdown", Markdown), f"  {url}  "
            )
        elif kind == "inspector":
            assert type(owner) is ConsoleConversationInspector
            await wait_for_condition(lambda: owner._snapshot_ready)
            handler = owner._copy_json
            event = Button.Pressed(
                owner.query_one("#console-inspector-next-send-copy", Button)
            )
        else:
            assert type(owner) is LibraryScreen
            if kind == "media":
                controller = owner._media_controller
                assert type(controller) is LibraryMediaController
                controller._library_media_detail = {"url": url}
                handler = controller.handle_library_media_open_original
                event = Button.Pressed(Button(id="library-media-open-original"))
            else:
                await wait_for_condition(
                    lambda: bool(owner.query("#library-row-browse-collections"))
                )
                owner.query_one("#library-row-browse-collections", Button).press()
                capture_controller = owner._library_collections_capture_controller
                assert capture_controller is not None
                await wait_for_condition(
                    lambda: (
                        capture_controller.state.loaded_detail is not None
                        and capture_controller.state.loaded_detail.capture.identity
                        == saved.capture.identity
                        and bool(owner.query("#library-collections-open-original"))
                    )
                )
                controller = owner._collections_controller
                assert type(controller) is LibraryCollectionsController
                assert capture_controller.state.identity_actions_enabled
                handler = controller.open_library_collection_capture_original
                event = Button.Pressed(
                    owner.query_one("#library-collections-open-original", Button)
                )
            assert controller._screen is owner

        assert owner.is_mounted
        name = {"collections": "open-capture-url", "inspector": "copy-json"}.get(
            kind, "open-link"
        )
        before = set(app.workers)
        issued = []
        try:
            try:
                handler(event)
            finally:
                # No await between invocation and capture: unrelated jobs cannot run.
                issued = [worker for worker in app.workers if worker not in before]
            assert len(issued) == 1
            worker = issued[0]
            assert worker.node is owner
            assert worker.name == name
            assert worker.exit_on_error is False
            await wait_for_condition(entered.is_set)
            assert len(calls) == 1
            value, thread_id, active_worker = calls[0]
            assert thread_id != issuer_thread
            assert active_worker is worker
            if kind == "inspector":
                assert isinstance(value, str)
                payload = json.loads(value)
                assert payload["messages"] == [ordinary]
                assert (
                    payload["project_instructions_export"] == "automatic body omitted"
                )
                assert automatic_body not in value
            else:
                assert value == url
        finally:
            release.set()
            for worker in issued:
                await wait_for_condition(
                    lambda worker=worker: worker.state is not WorkerState.PENDING
                )
                await asyncio.wait_for(asyncio.shield(worker.wait()), timeout=5.0)
        assert worker.state is WorkerState.SUCCESS
        assert len(calls) == 1
