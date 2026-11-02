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
    _configure_native_ready_console,
    _sync_console_control_bar_now,
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
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
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
SENT_LINE = "attached to the last message"


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

    from Tests.UI.test_console_hook_review_send_freeze import _until

    assert await _until(
        lambda: screen._console_attach_reconciled
        and not screen._console_attach_reconcile_running
        and screen._hooks._permissions().snapshot().ready,
        15,
    )
    composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(draft)
    await _sync_console_control_bar_now(screen)
    send = screen.query_one("#console-send-message", Button)
    assert await _until(lambda: not send.disabled, 15)
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

    ``sent`` is the accepted captured source count the Run card may report for this chat's
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
    _configure_native_ready_console(app, model="test-model")
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
        assert _reply_footers(console) == ["Sources (2) · attached to this turn"]


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
        assert _reply_footers(console) == ["Sources (2) · attached to this turn"]

        # A later send with nothing staged claims no evidence anywhere.
        await _send(console, pilot, "follow up", replies=2)
        assert capture.await_count == 1
        assert EVIDENCE not in _last_request_text(gateway)
        _assert_nothing_staged(console, sent=None)
        assert _reply_footers(console)[0] == "Sources (2) · attached to this turn"
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


def test_a_retried_reply_drops_the_sources_line_its_retry_did_not_send() -> None:
    """The count belongs to the accepted failed attempt only.

    Retry rebuilds the request from history without the released evidence,
    so "Sources (N) · attached to this turn" would describe an answer written
    without them.
    """
    store = ConsoleChatStore()
    session = store.ensure_session()
    reply = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.set_evidence_source_count(reply.id, 2)
    store.mark_message_failed(reply.id)
    assert store.get_message(reply.id).evidence_source_count == 2

    token = store.begin_generation_attempt(reply.id)
    retried = store.prepare_message_retry(reply.id, generation_token=token)

    assert retried.evidence_source_count is None
    assert store.get_message(reply.id).evidence_source_count is None


def test_a_restored_reply_carries_no_session_only_source_count() -> None:
    """Session-only like ``citation_presentation``: never restored."""
    store = ConsoleChatStore()
    session = ConsoleChatSession(id="session-a", title="Chat 1")
    reply = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer [S1]",
        evidence_source_count=2,
    )

    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: [reply]},
        active_session_id=session.id,
    )

    assert store.get_message(reply.id).evidence_source_count is None


def test_a_failing_reply_source_count_never_costs_the_send(monkeypatch) -> None:
    """The count is display only; its docstring promises the send survives."""
    from tldw_chatbook.Chat import console_display_state
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    def broken(*_args, **_kwargs):
        raise RuntimeError("count unavailable")

    monkeypatch.setattr(console_display_state, "console_sent_source_count", broken)
    store = ConsoleChatStore()
    session = store.ensure_session()
    reply = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    sent = LocalRagContextResult(context=EVIDENCE, citation_builder=None)
    lease = SimpleNamespace(released=False, capture_result=sent, launch=_launch(2))
    controller = SimpleNamespace(store=store)

    ConsoleChatController._note_reply_evidence_sources(
        controller, reply.id, SimpleNamespace(staged_evidence=lease), None
    )

    assert store.get_message(reply.id).evidence_source_count is None


def test_a_failing_staged_evidence_redraw_never_reaches_the_view_pump() -> None:
    """The redraw is posted to the view's pump, where an exception ends the app.

    The screen's own release used to swallow a surface-refresh failure
    ("rail body vanished mid-send"); the runtime-routed redraw must too.
    """
    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    posted: list = []

    def explode() -> None:
        raise RuntimeError("rail body vanished mid-send")

    runtime.view = SimpleNamespace(
        console_view_hooks=lambda: {"staged_evidence": explode},
        call_later=posted.append,
    )
    store.create_session(title="Chat 1")
    runtime.stage_console_staged_evidence(_launch(1))
    store.create_session(title="Chat 2")
    assert len(posted) == 1

    posted[0]()  # must not raise


def test_cold_staging_binds_first_activation_without_read_or_view() -> None:
    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    launch = _launch(2)
    revision = runtime.stage_console_staged_evidence(launch)
    first = store.create_session(title="First").id
    second = store.create_session(title="Second").id
    # No snapshot/helper/view runs between those synchronous activations.
    assert runtime.snapshot_console_staged_evidence(second) == (None, 0, None)
    assert runtime.snapshot_console_staged_evidence(first) == (launch, revision, None)
    assert None not in runtime._staged_evidence


def test_cold_staging_ignores_none_activation_and_preserves_keyed_state() -> None:
    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    cold, explicit = _launch(1), _launch(3)
    cold_revision = runtime.stage_console_staged_evidence(cold)
    runtime._on_active_session_changed()
    runtime._on_active_session_changed()
    assert runtime._staged_evidence[None].launch is cold
    first = store.create_session(title="First", activate=False).id
    keyed_revision = runtime.stage_console_staged_evidence(explicit, first)
    store.switch_session(first)
    assert runtime.snapshot_console_staged_evidence(first) == (
        explicit,
        keyed_revision,
        None,
    )
    assert keyed_revision > cold_revision
    assert None not in runtime._staged_evidence
    second = store.create_session(title="Second").id
    assert runtime.snapshot_console_staged_evidence(second) == (None, 0, None)


def _staged_close_runtime():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store = ConsoleChatStore()
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime.set_chat_controller(controller)
    return runtime, store, controller


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["background", "active", "last", "cold"])
async def test_successful_close_retires_exact_staging_immediately(position) -> None:
    runtime, store, controller = _staged_close_runtime()
    launch, other_launch = _launch(2), _launch(3)
    if position == "cold":
        runtime.stage_console_staged_evidence(launch)
    first = store.create_session(title="First", ephemeral=True).id
    if position != "cold":
        runtime.stage_console_staged_evidence(launch, first)
    other = None
    if position in {"background", "active"}:
        other = store.create_session(title="Other", ephemeral=True).id
        runtime.stage_console_staged_evidence(other_launch, other)
        runtime.set_console_staged_evidence_notice(3, other)
        other_state = runtime.snapshot_console_staged_evidence(other)
        if position == "active":
            store.switch_session(first)
    revision = controller.lifecycle_impact(session_id=first).revision
    result = await runtime.close_session(first, expected_revision=revision)
    # Inspect map before any read/activation could defensively prune it.
    assert first not in {session.id for session in store.sessions()}
    assert first not in runtime._staged_evidence
    assert None not in runtime._staged_evidence
    if other is not None:
        assert result.id == other
        assert runtime.snapshot_console_staged_evidence(other) == other_state
    else:
        assert result is None  # normal finalization of the last chat succeeds


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["revision", "voice", "native", "preparation"])
async def test_unfinished_close_keeps_exact_staging(refusal, monkeypatch) -> None:
    runtime, store, controller = _staged_close_runtime()
    first = store.create_session(title="First", ephemeral=True).id
    launch = _launch(2)
    revision = runtime.stage_console_staged_evidence(launch)
    lifecycle = controller.lifecycle_impact(session_id=first).revision
    if refusal == "revision":
        controller._advance_lifecycle_revision(first)
    elif refusal == "voice":
        from Tests.UI.test_console_session_tab_close import _UndrainedVoiceOwner

        runtime._voice_promotion_owner = _UndrainedVoiceOwner()
    else:

        async def refuse(*_args, **_kwargs):
            assert runtime._staged_evidence[first].launch is launch
            raise RuntimeError("retained owner refuses finalization")

        if refusal == "native":
            monkeypatch.setattr(runtime, "_drain_ordinary_native_commits", refuse)
        else:
            from tldw_chatbook.Chat import console_preparation_reads

            monkeypatch.setattr(
                console_preparation_reads, "drain_preparation_reads", refuse
            )
    if refusal == "voice":
        assert await runtime.close_session(first, expected_revision=lifecycle) is None
    else:
        with pytest.raises(RuntimeError):
            await runtime.close_session(first, expected_revision=lifecycle)
    assert first in {session.id for session in store.sessions()}
    assert runtime.snapshot_console_staged_evidence(first) == (launch, revision, None)


@pytest.mark.asyncio
async def test_close_retires_staging_before_cancellation_redelivery(
    monkeypatch,
) -> None:
    import asyncio

    runtime, store, controller = _staged_close_runtime()
    first = store.create_session(title="First", ephemeral=True).id
    runtime.stage_console_staged_evidence(_launch(2))
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime._bounded_wait

    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_bounded_wait", held)
    closing = asyncio.create_task(
        runtime.close_session(
            first,
            expected_revision=controller.lifecycle_impact(session_id=first).revision,
        )
    )
    await entered.wait()
    closing.cancel()
    assert first in runtime._staged_evidence
    assert first in {session.id for session in store.sessions()}
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert first not in runtime._staged_evidence
    assert first not in {session.id for session in store.sessions()}


@pytest.mark.asyncio
@pytest.mark.parametrize("ephemeral", [False, True], ids=["saved", "temporary"])
async def test_accepted_sources_are_attached_when_prerequisite_refuses_before_contact(
    tmp_path, monkeypatch, ephemeral
) -> None:
    from Tests.Chat.test_console_staged_evidence_dispatch_pin import (
        _runtime,
        _admit,
        _library_search,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import _citation_notice
    from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController

    runtime, store, controller, first = _runtime(tmp_path, ephemeral=ephemeral)
    _library_search(monkeypatch)
    runtime.stage_console_staged_evidence(SimpleNamespace(evidence=EVIDENCE))
    releases = []
    original = runtime.release_console_staged_evidence

    def release(*args, **kwargs):
        result = original(*args, **kwargs)
        releases.append(result)
        return result

    monkeypatch.setattr(runtime, "release_console_staged_evidence", release)
    refused = AsyncMock(side_effect=RuntimeError("trace maintenance unavailable"))
    monkeypatch.setattr(controller, "_wait_for_trace_maintenance_dispatch", refused)
    await runtime.wait_for_turn(_admit(runtime, controller, first, "question"))
    replies = [
        m
        for m in store.messages_for_session(first)
        if m.role is ConsoleMessageRole.ASSISTANT
    ]
    assert len(replies) == 1
    reply = replies[0]
    assert reply.evidence_source_count == 1
    assert refused.await_count == 1
    assert controller.provider_gateway.calls == 0
    assert releases == [True]
    assert runtime.snapshot_console_staged_evidence(first)[::2] == (None, 1)
    if not ephemeral:
        row = (
            store.persistence.db.get_connection()
            .execute("SELECT state FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert row is not None and row["state"] == "accepted"
    assert _citation_notice(reply) == "Sources (1) · attached to this turn"
    receipt = ConsoleRetrievalController._console_rag_source_status(None, None, 1)
    assert receipt == "attached to the last message · 1 source"
    assert "sent" not in _citation_notice(reply) + receipt


def test_accepted_source_run_receipt_uses_attached_vocabulary() -> None:
    from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController

    assert ConsoleRetrievalController._console_rag_source_status(None, None, 2) == (
        "attached to the last message · 2 sources"
    )


@pytest.mark.asyncio
async def test_original_stock_handler_accepts_b_without_consuming_a_staging(
    capture, record_property
) -> None:
    """Real mounted source checks and acceptance; natural button is separate."""
    import asyncio
    import hashlib

    from Tests.UI.test_console_hook_review_send_freeze import _until
    from tldw_chatbook.Chat.console_received_dispatch import stock_native_methods
    from tldw_chatbook.UI.Screens.chat_screen import _CONSOLE_RECEIVED_SCREEN_SELECTORS
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
    from tldw_chatbook.Widgets.Console.console_transcript import _citation_notice

    class HeldGateway(CapturingGateway):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            self.sent_messages.append(list(messages))
            self.entered.set()
            await self.release.wait()
            yield "accepted"

    gateway = HeldGateway()
    host = ConsoleHarness(_console_app(gateway))
    async with host.run_test(size=(200, 56)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        assert stock_native_methods(console, _CONSOLE_RECEIVED_SCREEN_SELECTORS)
        store = console._ensure_console_chat_store()
        runtime = console._console_runtime()
        first = store.active_session_id
        launch = _launch(2)
        console._retrieval._stage_console_library_rag_launch(launch)
        first_state = runtime.snapshot_console_staged_evidence(first)
        assert first_state[0] is launch and first_state[1] > 0

        await pilot.press("ctrl+t")
        await _wait_for_new_chat(store, pilot, first)
        second = store.active_session_id
        await _select_llamacpp_console(console)
        assert await _until(
            lambda: console._console_visible_draft_session_id == second
            and console._console_attach_reconciled
            and not console._console_attach_reconcile_running
            and console._hooks._permissions().snapshot().ready,
            15,
        )
        assert runtime.snapshot_console_staged_evidence(second) == (None, 0, None)
        assert runtime.snapshot_console_staged_evidence(first) == first_state
        strip = console.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert await _until(lambda: not strip.display, 15)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("unrelated")
        await _sync_console_control_bar_now(console)
        send = console.query_one("#console-send-message", Button)
        assert not send.disabled
        worker = console.run_worker(
            console.handle_console_send_message(Button.Pressed(send)),
            group="pr3331-original-handler",
            exit_on_error=False,
        )
        turn_task = None
        try:
            reached = await _until(gateway.entered.is_set, 15)
            record_property("pr3331_route", "original-stock-handler-worker")
            record_property("pr3331_gateway_entered", reached)
            if not reached:
                recoveries = runtime.recoveries_for_session(second)
                record_property("pr3331_recovery_count", len(recoveries))
                record_property(
                    "pr3331_recovery_reason_sha256",
                    ";".join(
                        hashlib.sha256(row.reason.encode()).hexdigest()
                        for row in recoveries
                    ),
                )
                preparation = store.preparation_for_session(second)
                record_property(
                    "pr3331_preparation_state", str(getattr(preparation, "state", None))
                )
                record_property(
                    "pr3331_preparation_pause_kind",
                    str(getattr(preparation, "pause_kind", None)),
                )
                record_property("pr3331_handler_state", worker.state.name)
            assert reached, "Original stock handler did not reach the held gateway"
            records = [
                row
                for row in runtime._turn_custody.values()
                if row.session_id == second
            ]
            assert len(records) == 1
            record = records[0]
            turn_task = record.task
            assert turn_task is not None and not turn_task.done()
            assert record.store is store
            assert record.request is not None and record.request.session_id == second
            assert record.request.staged_evidence_launch is None
            if record.received_intent is not None:
                assert record.received_intent.staged_evidence_launch is None
            assert record.inputs.staged_evidence_revision is None
            # Issued only by the real acceptance owner, never set by this test.
            assert record.inputs.durable_accepted is True
            record_property("pr3331_durable_accepted", record.inputs.durable_accepted)
            record_property(
                "pr3331_received_intent", record.received_intent is not None
            )
            assert len(gateway.sent_messages) == 1
            assert EVIDENCE not in _last_request_text(gateway)
            assert capture.await_count == 0
            replies = [
                m
                for m in store.messages_for_session(second)
                if m.role is ConsoleMessageRole.ASSISTANT
            ]
            assert len(replies) == 1
            assert replies[0].evidence_source_count is None
            assert "Sources (" not in _citation_notice(replies[0])
            assert runtime.snapshot_console_staged_evidence(first) == first_state
        finally:
            gateway.release.set()
            await worker.wait()
            if turn_task is not None:
                await asyncio.shield(turn_task)
        assert await _until(lambda: not runtime.has_custodied_turns(second), 15)
        assert runtime.snapshot_console_staged_evidence(first) == first_state
        assert (
            console._ensure_console_chat_controller()._ordinary_native_commit_retirements(
                second
            )
            == ()
        )
        assert len(gateway.sent_messages) == 1
        record_property("pr3331_joined_cleanup", True)
