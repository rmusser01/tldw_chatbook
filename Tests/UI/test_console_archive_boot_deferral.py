"""Blank Console refresh and empty resume queues do not load archive actions."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_archive_actions_load_only_for_a_saved_conversation():
    script = textwrap.dedent(
        """
        import asyncio
        import sys
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, Mock

        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
        from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
        from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore

        module = "tldw_chatbook.Chat.conversation_archive_actions"
        assert module not in sys.modules, "controller imports eagerly load archive actions"

        async def exercise():
            app = SimpleNamespace(pending_handoffs=PendingHandoffStore())
            screen = SimpleNamespace(app_instance=app)
            screen.app = SimpleNamespace(screen=screen)
            archive_module = "tldw_chatbook.UI.Console_Modules.archive"
            assert archive_module not in sys.modules
            await ChatScreen._consume_pending_conversation_resume(screen)
            assert archive_module not in sys.modules, "empty visible resume poll loads archive controller"
            from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
            await consume_conversation_resume(screen)
            assert module not in sys.modules, "empty resume queue loads archive actions"

            sessions = [SimpleNamespace(persisted_conversation_id=None)]
            service = SimpleNamespace(get_conversation_archive_states=Mock(return_value={"saved": False}))
            app.local_chat_conversation_service = service
            result = ([], 0, "")
            owner = SimpleNamespace(
                app_instance=app,
                _compute_persisted_console_browser_rows=lambda *_: None,
                _canonical_membership_revision=0,
                _console_persisted_rows_cache_token=0,
                _persisted_console_browser_rows=AsyncMock(return_value=result),
                _record_canonical_owner_rows=Mock(),
                _console_chat_store=SimpleNamespace(sessions=lambda: sessions),
            )
            refresh = ConsoleWorkspaceController._refresh_console_persisted_rows_cache
            assert await refresh(owner) == result
            assert module not in sys.modules, "unsaved session refresh loads archive actions"
            service.get_conversation_archive_states.assert_not_called()

            sessions[0].persisted_conversation_id = "saved"
            assert await refresh(owner) == result
            assert module in sys.modules, "saved-conversation refresh must load its archive authority"
            service.get_conversation_archive_states.assert_called_once_with(["saved"])
            assert app._conversation_archive_states == {"saved": False}

        asyncio.run(exercise())
        """
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
