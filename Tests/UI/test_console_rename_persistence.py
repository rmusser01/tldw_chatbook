"""TASK-341: the tab rename modal must persist a saved conversation's title.

Renaming looked successful (tab + transcript header updated) but only
mutated the in-memory session; on restart the rail showed the old title
(UX review finding j2-rename-tab-only-silently-lost).
"""

import pytest

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleRenameSessionModal


class _TitleRecorder:
    def __init__(self):
        self.updates = []

    def update_conversation_title(self, *, conversation_id, title):
        self.updates.append({"conversation_id": conversation_id, "title": title})
        return True

    def create_conversation(self, **kwargs):
        return "conv-render"

    def create_message(self, **kwargs):
        return "msg-render"


@pytest.mark.asyncio
async def test_rename_modal_persists_saved_conversation_title():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        store = console._ensure_console_chat_store()
        recorder = _TitleRecorder()
        store.persistence = recorder
        session = store.restore_persisted_session(
            title="Websocket reconnect strategy",
            workspace_id=None,
            persisted_conversation_id="conv-341",
            all_nodes=[],
            active_leaf_persisted_id=None,
        )

        console._open_console_session_rename_modal(session.id)
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRenameSessionModal)
        modal.dismiss("Reconnect deep dive")
        await pilot.pause()
        await pilot.pause()

        assert store.sessions()[-1].title == "Reconnect deep dive"
        assert recorder.updates == [
            {"conversation_id": "conv-341", "title": "Reconnect deep dive"}
        ]
