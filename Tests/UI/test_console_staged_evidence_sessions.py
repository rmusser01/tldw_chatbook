"""TASK-33620.7: staged evidence belongs to one Console chat and one turn.

Live on dev 1eaa6a9429 (2026-10-30, Anthropic claude-haiku-4-5): a Library
search staged two sources, the next send consumed them and the reply cited
[S1]. The chip then read "Sources: 0" and Inspect "Sources: None staged",
while the composer strip kept "Staged for next send · 2 sources" through
later sourceless sends, a new Ctrl+T tab and a new workspace chat. Un-stage
on that strip turned it into a permanent "Evidence sent with this message".
Evidence staged in one chat was also staged in every other chat.

Two causes: ``ConsoleRuntime`` held ONE app-wide staged value and receipt,
and the runtime's release (every production send goes through it) never
told the view, so nothing redrew the strip or the Inspect cards.

These tests send through the production path -- the composer's Send, the
runtime's turn custody and its frozen capture -- with only the Library
lookup at the capture seam faked, and read what the strip, the Sources chip,
Inspect and the transcript show at each step.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _select_llamacpp_console,
)
from Tests.UI.test_console_staged_evidence_strip import (
    STRIP_ID,
    UNSTAGE_ID,
    _launch,
    _strip_text,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_display_state import (
    build_console_staged_evidence_strip_state,
)
from tldw_chatbook.Chat.console_live_work import (
    PENDING_LAUNCH_CARD_ID,
    SOURCE_READINESS_CARD_ID,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
)
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)
from tldw_chatbook.Widgets.Console.console_staged_evidence_strip import (
    ConsoleStagedEvidenceStrip,
)

pytestmark = pytest.mark.bootstrap_profile

EVIDENCE = "[S1] MEDIA — Source 1\nBody 1\n\n[S2] MEDIA — Source 2\nBody 2"
SENT_LINE = "sent with the last message"


def _plain(widget: Static) -> str:
    renderable = widget.renderable
    return str(getattr(renderable, "plain", renderable))


def _rail_text(screen) -> str:
    """Every Inspect fact row and card line, shown or not."""
    rail = screen.query_one("#console-inspector-rail-body")
    return "\n".join(_plain(widget) for widget in rail.query(Static))


def _summary_text(screen) -> str:
    """The "What happens if I send now?" card."""
    summary = screen.query_one(ConsoleSendAuthoritySummary)
    return "\n".join(_plain(widget) for widget in summary.query(Static))


def _last_request_text(gateway) -> str:
    return "\n".join(
        str(message.get("content", "")) for message in gateway.sent_messages[-1]
    )


def _chip_text(screen) -> str:
    return _plain(screen.query_one("#console-sources-label", Static))


def _reply_footers(screen) -> list[str]:
    """The dim footer under each assistant reply, in transcript order."""
    transcript = screen.query_one("#console-native-transcript")
    return [_plain(footer) for footer in transcript.query(".console-markdown-footer")]


async def _wait_for_replies(screen, pilot, count: int) -> None:
    store = screen._ensure_console_chat_store()
    for _ in range(120):
        replies = [
            message
            for message in store.messages_for_session(store.active_session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "complete"
            and message.content == "accepted"
        ]
        if (
            len(replies) >= count
            and len(_reply_footers(screen)) >= count
            # The turn leaves runtime custody after its reply lands; a send
            # admitted before that is refused as a second turn.
            and not screen._console_runtime().has_custodied_turns(
                store.active_session_id
            )
        ):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.05)
    rows = [
        (message.role.value, message.status, message.content[:60])
        for message in store.messages_for_session(store.active_session_id)
    ]
    raise AssertionError(f"{count} replies did not arrive: {rows}")


async def _send(screen, pilot, draft: str, replies: int) -> None:
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(draft)
    send = screen.query_one("#console-send-message", Button)
    for _ in range(80):  # Send stays "Sending..." until the last turn settles
        await pilot.pause(0.05)
        if not send.disabled:
            break
    send.press()
    await _wait_for_replies(screen, pilot, replies)


async def _settle(pilot) -> None:
    for _ in range(4):
        await pilot.pause()


async def _wait_for_new_chat(store, pilot, previous: str | None) -> None:
    for _ in range(40):
        if store.active_session_id not in (None, previous):
            return
        await pilot.pause(0.05)
    raise AssertionError("no new chat became active")


async def _open_inspect(screen, pilot) -> None:
    """Inspect keeps its cards current only while the rail is open."""
    if not screen._is_console_widget_displayed("console-right-rail"):
        await screen.action_toggle_console_inspector_rail()
    await _settle(pilot)


def _assert_staged(screen, count: int) -> None:
    strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
    assert strip.display is True
    assert f"Staged for next send · {count} sources" in _strip_text(screen)
    assert f"Sources: {count}" in _chip_text(screen)
    assert f"Sources: {count} staged" in _summary_text(screen)
    assert screen.query(f"#{PENDING_LAUNCH_CARD_ID}")


def _assert_nothing_staged(screen, *, sent: int | None) -> None:
    """Strip, chip and every Inspect source card agree nothing is staged.

    ``sent`` is the source count the Run card may report for this chat's
    LAST message, or None when that message carried none.
    """
    strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
    assert strip.display is False, _strip_text(screen)
    assert "Staged for next send" not in _strip_text(screen)
    assert "Evidence sent with this message" not in _strip_text(screen)
    assert "Sources: 0" in _chip_text(screen)
    assert "Sources: None staged" in _summary_text(screen)
    tray = screen.query_one("#console-staged-context-tray", ConsoleStagedContextTray)
    assert tray.state.is_empty is True
    # The live-work card shows source readiness, not the consumed launch.
    assert not screen.query(f"#{PENDING_LAUNCH_CARD_ID}")
    assert screen.query(f"#{SOURCE_READINESS_CARD_ID}")
    rail = _rail_text(screen)
    if sent is None:
        assert SENT_LINE not in rail, rail
    else:
        assert f"{SENT_LINE} · {sent} sources" in rail, rail


def test_runtime_keeps_staged_evidence_and_its_receipt_per_chat() -> None:
    """One chat's staging is not another's, and no fence crosses chats."""
    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    early, later = _launch(2), _launch(3)

    # A cold Console's handoff lands before any chat is active ...
    runtime.stage_console_staged_evidence(early)
    first = store.create_session(title="Chat 1").id
    # ... and belongs to the first chat that becomes active.
    assert runtime.snapshot_console_staged_evidence()[0] is early

    second = store.create_session(title="Chat 2").id
    assert store.active_session_id == second
    assert runtime.snapshot_console_staged_evidence() == (None, 0, None)
    assert runtime._has_staged_evidence(second) is False
    assert runtime._has_staged_evidence(first) is True
    later_revision = runtime.stage_console_staged_evidence(later)

    _launch_a, first_revision, _ = runtime.snapshot_console_staged_evidence(first)
    sent = LocalRagContextResult(context=EVIDENCE, citation_builder=None)
    # The first chat's revision never releases the second chat's launch.
    assert not runtime.release_console_staged_evidence(
        later, sent, revision=first_revision, session_id=second
    )
    assert runtime.release_console_staged_evidence(
        early, sent, revision=first_revision, session_id=first
    )
    assert runtime.snapshot_console_staged_evidence(first)[::2] == (None, 2)
    assert runtime.snapshot_console_staged_evidence(second) == (
        later,
        later_revision,
        None,
    )


def test_runtime_redraws_on_a_chat_switch_only_when_evidence_is_shown() -> None:
    """Boot creates and activates chats while the Console is still mounting.

    A redraw posted then reached the Inspect card swap before its viewport
    was mounted (MountError: the UI-ready census boot crashed). With nothing
    staged or received in any chat a switch has nothing to redraw.
    """
    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    posted: list[object] = []
    runtime.view = SimpleNamespace(
        console_view_hooks=lambda: {"staged_evidence": lambda: True},
        call_later=posted.append,
    )

    store.create_session(title="Chat 1")
    store.create_session(title="Chat 2")
    assert posted == []

    runtime.stage_console_staged_evidence(_launch(1))
    store.create_session(title="Chat 3")
    assert len(posted) == 1


@pytest.fixture
def capture(monkeypatch) -> AsyncMock:
    """Fake only the Library lookup behind the runtime's frozen capture."""
    fake = AsyncMock(
        return_value=LocalRagContextResult(context=EVIDENCE, citation_builder=None)
    )
    monkeypatch.setattr(
        chat_rag_events, "capture_console_staged_evidence_for_chat", fake
    )
    return fake


def _console_app(gateway):
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    # A second turn in a saved chat reads its archive state, and the factory
    # app has no service for it (`_allow_second_turns` in
    # test_console_send_admission_off_pump.py; lessons-testing-evidence.md).
    app.local_chat_conversation_service = SimpleNamespace(
        db=SimpleNamespace(is_memory_db=True),
        get_conversation_archive_states=lambda _ids: {},
    )
    return app


@pytest.mark.asyncio
async def test_a_send_in_one_chat_leaves_another_chats_staging_alone(
    capture,
) -> None:
    """The other chat's staged launch is neither sent nor consumed."""
    gateway = CapturingGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)
        store = console._ensure_console_chat_store()
        first_chat = store.active_session_id
        console._retrieval._stage_console_library_rag_launch(_launch(2))
        await _settle(pilot)

        await pilot.press("ctrl+t")
        await _wait_for_new_chat(store, pilot, first_chat)
        # A new tab starts on the app default provider; make it sendable.
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)
        await _settle(pilot)
        await _send(console, pilot, "unrelated", replies=1)

        assert capture.await_count == 0
        assert EVIDENCE not in _last_request_text(gateway)
        assert "Sources (" not in _reply_footers(console)[0]
        await console._session._activate_native_console_session(first_chat)
        await _settle(pilot)
        _assert_staged(console, 2)


@pytest.mark.asyncio
async def test_a_temporary_chats_send_shows_its_staged_sources_on_its_reply(
    capture,
) -> None:
    """A Temporary chat accepts its turn on the non-durable path; same claim."""
    gateway = CapturingGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        saved_chat = store.active_session_id
        console.action_new_temporary_console_tab()
        await _wait_for_new_chat(store, pilot, saved_chat)
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)

        launch = _launch(2)
        console._retrieval._stage_console_library_rag_launch(launch)
        await _settle(pilot)
        _assert_staged(console, 2)

        # The send consumes the evidence: the claim moves onto its reply.
        await _send(console, pilot, "question", replies=1)
        assert capture.await_count == 1
        assert capture.await_args_list[0].args[1] is launch
        assert EVIDENCE in _last_request_text(gateway)
        _assert_nothing_staged(console, sent=2)
        assert _reply_footers(console) == ["Sources (2) · sent with this turn"]


@pytest.mark.asyncio
async def test_staged_evidence_follows_one_turn_and_one_chat(capture) -> None:
    """Stage -> send -> sourceless send -> new tab -> workspace chat."""
    gateway = CapturingGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)
        store = console._ensure_console_chat_store()
        first_chat = store.active_session_id

        launch = _launch(2)
        console._retrieval._stage_console_library_rag_launch(launch)
        await _settle(pilot)
        _assert_staged(console, 2)

        # The send consumes the evidence: the claim moves onto its reply.
        await _send(console, pilot, "question", replies=1)
        assert capture.await_count == 1
        assert capture.await_args_list[0].args[1] is launch
        assert EVIDENCE in _last_request_text(gateway)
        _assert_nothing_staged(console, sent=2)
        assert _reply_footers(console) == ["Sources (2) · sent with this turn"]

        # A later send with nothing staged claims no evidence anywhere.
        await _send(console, pilot, "follow up", replies=2)
        assert capture.await_count == 1
        assert EVIDENCE not in _last_request_text(gateway)
        _assert_nothing_staged(console, sent=None)
        assert _reply_footers(console)[0] == "Sources (2) · sent with this turn"
        assert "Sources (" not in _reply_footers(console)[1]

        # Evidence staged here stays in this chat ...
        console._retrieval._stage_console_library_rag_launch(_launch(3))
        await _settle(pilot)
        _assert_staged(console, 3)

        # ... a new Ctrl+T tab starts clean ...
        await pilot.press("ctrl+t")
        await _wait_for_new_chat(store, pilot, first_chat)
        new_tab = store.active_session_id
        await _settle(pilot)
        _assert_nothing_staged(console, sent=None)

        # ... so does a newly entered workspace chat ...
        console._workspace._activate_console_session_for_workspace("workspace-u15")
        await _settle(pilot)
        assert store.active_session_id not in (first_chat, new_tab)
        _assert_nothing_staged(console, sent=None)

        # ... and the first chat still has its own staging when it returns.
        await console._session._activate_native_console_session(first_chat)
        await _settle(pilot)
        assert store.active_session_id == first_chat
        _assert_staged(console, 3)


@pytest.mark.asyncio
async def test_unstage_on_a_strip_with_nothing_staged_hides_it(capture) -> None:
    """AC#4: a stale strip's Un-stage never becomes a permanent notice."""
    gateway = CapturingGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)
        console._retrieval._stage_console_library_rag_launch(_launch(2))
        await _settle(pilot)
        await _send(console, pilot, "question", replies=1)

        # Whatever left the strip showing a consumed launch, its Un-stage
        # press is the user's way out of it.
        strip = console.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        strip.sync_state(build_console_staged_evidence_strip_state(_launch(2)))
        await _settle(pilot)
        console.query_one(UNSTAGE_ID, Button).press()
        await _settle(pilot)

        assert strip.display is False, _strip_text(console)
        assert "Evidence sent with this message" not in _strip_text(console)
        _assert_nothing_staged(console, sent=2)


@pytest.mark.asyncio
async def test_unstage_on_a_stale_strip_redraws_every_staged_surface(
    capture, monkeypatch
) -> None:
    """A release whose redraw never reached the view is healed by Un-stage.

    Before this fix Un-stage with nothing staged resynced only the strip,
    leaving Inspect's "Sources — next send" tray and "Live work" card on
    the consumed launch.
    """
    gateway = CapturingGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        await _open_inspect(console, pilot)
        console._retrieval._stage_console_library_rag_launch(_launch(2))
        await _settle(pilot)
        hooks = console.console_view_hooks
        # Lose the release's redraw, as a view away or covered at release.
        monkeypatch.setattr(
            console,
            "console_view_hooks",
            lambda: {k: v for k, v in hooks().items() if k != "staged_evidence"},
        )
        await _send(console, pilot, "question", replies=1)
        assert console.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display

        console.query_one(UNSTAGE_ID, Button).press()
        await _settle(pilot)

        _assert_nothing_staged(console, sent=2)
