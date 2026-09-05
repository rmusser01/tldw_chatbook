"""Console /research flag parsing (task-16793)."""

import pytest

from tldw_chatbook.UI.Console_Modules.research_command import (
    ResearchCommandIntent,
    parse_research_command,
)


def test_plain_question():
    intent = parse_research_command("what is RAG?")
    assert intent.question == "what is RAG?"
    assert intent.source_policy == "balanced"
    assert intent.providers is None


def test_policy_flag():
    intent = parse_research_command("--policy academic_only why do neurons die?")
    assert intent.source_policy == "academic_only"
    assert intent.question == "why do neurons die?"


def test_providers_flag_with_categories():
    intent = parse_research_command(
        "--policy web_first --providers biomedical,zenodo tau aggregation"
    )
    assert intent.source_policy == "web_first"
    assert intent.providers == ["biomedical", "zenodo"]
    assert intent.question == "tau aggregation"


def test_flags_after_question():
    intent = parse_research_command("how do gnn work --policy web_only")
    assert intent.source_policy == "web_only"
    assert intent.question == "how do gnn work"


def test_invalid_policy_rejected():
    with pytest.raises(ValueError, match="policy"):
        parse_research_command("--policy sideways a question")


def test_empty_question_rejected():
    with pytest.raises(ValueError, match="question"):
        parse_research_command("--policy web_only")


def test_intent_shape():
    intent = ResearchCommandIntent(question="q", source_policy="balanced")
    assert intent.provider_overrides() is None
    intent = ResearchCommandIntent(
        question="q", source_policy="web_first", providers=["biomedical"]
    )
    assert intent.provider_overrides() == {"academic_providers": ["biomedical"]}


def test_dangling_flag_token_is_a_usage_error():
    with pytest.raises(ValueError, match="--policy needs a value"):
        parse_research_command("--policy")  # flag with no value, no question
    with pytest.raises(ValueError, match="--providers needs a value"):
        parse_research_command("a question --providers")  # dangling at end


def test_oversized_args_rejected_by_shared_validator():
    with pytest.raises(ValueError, match="too long"):
        parse_research_command("x" * 5000 + " --policy web_only")


# --- TASK-21105 review fix: corrupt store must not kill the app -------------
#
# The research database opens on FIRST USE now, so a corrupt/unreadable
# store no longer fails at boot (where construction failure used to null
# the service and trip the handler's unavailable-guard) -- it raises from
# ``launch_run`` inside the /research worker. That worker runs with
# Textual's default ``exit_on_error=True``: an unhandled raise there tears
# down the whole app. This drives the real handler with a REAL
# LocalResearchService over a genuinely corrupt file and asserts the
# worker coroutine degrades to the logged warning instead of raising.
# Reverting the guard (moving ``launch_run`` back outside the worker's
# try) makes this test fail on the un-swallowed DatabaseError.


class _RecordingScreen:
    """Duck-typed stand-in for the ChatScreen surface the handler touches."""

    def __init__(self, app) -> None:
        self.app = app
        self.system_messages: list[str] = []
        self.worker_coroutines: list[object] = []

    async def _append_native_console_system_message(self, text: str) -> None:
        self.system_messages.append(text)

    def _current_console_conversation_id(self) -> str:
        return "conv-console-1"

    def run_worker(self, coroutine, **kwargs):
        self.worker_coroutines.append(coroutine)
        return None


class _StubApp:
    research_window_academic_enabled = False
    chachanotes_db = None

    def __init__(self, local_research_service) -> None:
        self.local_research_service = local_research_service


async def test_research_worker_survives_a_corrupt_research_store(tmp_path):
    from types import SimpleNamespace

    from loguru import logger as loguru_logger

    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )
    from tldw_chatbook.UI.Console_Modules.wiring import build_console_commands_controller

    db_path = tmp_path / "research.db"
    db_path.write_bytes(b"this is not a sqlite database at all\x00\x01")
    db_path.chmod(0o600)

    # Lazy open means construction succeeds -- exactly the shape that used
    # to be impossible (eager construction failed in app.py's except and
    # the handler's None-guard fired instead).
    service = LocalResearchService(db_path)
    screen = _RecordingScreen(_StubApp(service))
    build_console_commands_controller(screen)

    warnings: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: warnings.append(str(message)), level="WARNING"
    )
    try:
        await screen._commands._console_command_research(
            SimpleNamespace(args="why is the sky blue")
        )

        assert len(screen.worker_coroutines) == 1
        # The launch guard is the subject: awaiting the worker body must
        # NOT raise even though launch_run's first-use open fails on the
        # corrupt file.
        await screen.worker_coroutines[0]
    finally:
        loguru_logger.remove(sink_id)

    assert any("Console research run failed" in entry for entry in warnings), (
        f"expected the degraded-run warning, got: {warnings}"
    )
    # The user-facing start message was already posted; the run simply
    # never delivers -- same degrade as any failed research run.
    assert any("Deep research started" in text for text in screen.system_messages)


@pytest.mark.asyncio
async def test_invalid_saved_backend_is_shown_before_console_research_launch(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    app = _build_test_app()
    host = ConsoleHarness(app)
    launches = Mock(return_value={"id": "unexpected-run"})
    # Keep an accidental baseline launch offline; dispatch itself must be absent.
    monkeypatch.setattr(LocalResearchEngine, "execute_run", AsyncMock())
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            host,
            "local_research_service",
            SimpleNamespace(launch_run=launches),
            raising=False,
        )
        monkeypatch.setattr(
            console, "_current_console_conversation_id", lambda: "conversation-1"
        )
        assert SettingsConfigAdapter().save_values(
            "SearchSettings", {"search_provider_default": "brvae"}
        )

        await console._console_command_research(
            CommandParse(kind="command", name="research", args="routing")
        )
        await pilot.pause()
        store = console._ensure_console_chat_store()
        messages = [
            m.content
            for m in store.messages_for_session(store.active_session_id)
            if m.role is ConsoleMessageRole.SYSTEM
        ]
        assert any(
            "search_provider_default" in message and "brvae" in message
            for message in messages
        ), (messages, launches.call_count)
        assert not any("Deep research started" in message for message in messages)
        launches.assert_not_called()
