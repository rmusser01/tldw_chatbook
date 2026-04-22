from unittest.mock import Mock

import pytest

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel


def _text(widget: Static) -> str:
    return str(widget.render())


class _ChatWindowHarnessApp(App):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self._app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield ChatWindowEnhanced(self._app_instance)


@pytest.fixture
def mock_chat_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7,
        }
    }
    host.chat_sidebar_collapsed = False
    host.chat_right_sidebar_collapsed = False
    host.notify = Mock()
    host.run_worker = Mock()
    host.bell = Mock()
    return host


@pytest.fixture
def chat_window_settings(monkeypatch):
    def get_setting(section, key, default=None):
        overrides = {
            ("chat_defaults", "enable_tabs"): False,
            ("chat.voice", "show_mic_button"): True,
            ("chat.images", "show_attach_button"): True,
        }
        return overrides.get((section, key), default)

    providers = {"openai": ["gpt-4.1"]}

    monkeypatch.setattr("tldw_chatbook.UI.Chat_Window_Enhanced.get_cli_setting", get_setting)
    monkeypatch.setattr("tldw_chatbook.Widgets.compact_model_bar.get_cli_setting", get_setting)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.compact_model_bar.get_cli_providers_and_models",
        lambda: providers,
    )
    monkeypatch.setattr("tldw_chatbook.Widgets.enhanced_settings_sidebar.get_cli_setting", get_setting)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.enhanced_settings_sidebar.get_cli_providers_and_models",
        lambda: providers,
    )


def test_chat_screen_state_round_trip_preserves_task_resume_and_pending_approval():
    state = ChatScreenState(
        task_resume_state=TaskResumeState(
            summary="Continue the UX rescue work",
            last_step="Landed the study dashboard shell",
            next_action="Review the privileged file-write diff",
            diff_summary="3 files changed in chat UI",
            pending_approval={
                "summary": "Allow workspace write for chat task cards",
                "details": "Adds inline approval and resume widgets to chat",
            },
        )
    )

    restored = ChatScreenState.from_dict(state.to_dict())

    assert restored.task_resume_state.summary == "Continue the UX rescue work"
    assert restored.task_resume_state.last_step == "Landed the study dashboard shell"
    assert restored.task_resume_state.next_action == "Review the privileged file-write diff"
    assert restored.task_resume_state.diff_summary == "3 files changed in chat UI"
    assert restored.task_resume_state.pending_approval == {
        "summary": "Allow workspace write for chat task cards",
        "details": "Adds inline approval and resume widgets to chat",
    }


def test_chat_screen_updates_and_syncs_task_resume_state(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen.chat_window = Mock()

    task_state = TaskResumeState(
        summary="Resume the chat-first migration",
        last_step="Prepared the inline task surface",
        next_action="Confirm the approval copy and continue implementation",
    )

    screen.set_task_resume_state(task_state)

    assert screen.chat_state.task_resume_state == task_state
    screen.chat_window.sync_task_resume_state.assert_called_once_with(task_state)


@pytest.mark.asyncio
async def test_chat_window_mounts_inline_task_surface_above_chat_log(
    chat_window_settings,
    mock_chat_host,
):
    app = _ChatWindowHarnessApp(mock_chat_host)

    async with app.run_test() as pilot:
        main_content = pilot.app.query_one("#chat-main-content", Container)
        task_surface = pilot.app.query_one("#chat-task-surface", Container)
        chat_log = pilot.app.query_one("#chat-log", VerticalScroll)

        children = list(main_content.children)
        assert children.index(task_surface) < children.index(chat_log)


@pytest.mark.asyncio
async def test_chat_window_sync_task_state_shows_inline_approval_and_resume_content(
    chat_window_settings,
    mock_chat_host,
):
    app = _ChatWindowHarnessApp(mock_chat_host)
    task_state = TaskResumeState(
        summary="Resume the chat safety layer",
        last_step="Drafted approval and continuity widgets",
        next_action="Review details and continue the implementation",
        diff_summary="2 files changed in the chat shell",
        pending_approval={
            "summary": "Allow workspace write for chat task cards",
            "details": "Touches the chat window and chat screen state",
        },
    )

    async with app.run_test() as pilot:
        chat_window = pilot.app.query_one(ChatWindowEnhanced)
        chat_window.sync_task_resume_state(task_state)
        await pilot.pause(0.05)

        approval_card = pilot.app.query_one(ChatApprovalCard)
        resume_panel = pilot.app.query_one(ChatResumePanel)

        assert "Allow workspace write for chat task cards" in _text(
            approval_card.query_one("#approval-summary", Static)
        )
        assert "Drafted approval and continuity widgets" in _text(
            resume_panel.query_one("#resume-last-step", Static)
        )
        assert "Review details and continue the implementation" in _text(
            resume_panel.query_one("#resume-next-action", Static)
        )
