"""TASK-33661: Resend re-runs a broken last Console turn in place.

Every broken shape runs through a real ``ConsoleChatController`` and a
``ConsoleChatStore`` persisted to a real SQLite ChaChaNotes database, and the
restart shapes rebuild the store from that database through the production
hydration path. "In place" is proven three ways: the user message keeps its
id, the new reply parents directly under it with no sibling, and the
database holds exactly one live user row and one live assistant row.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
)
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_turn_resend import (
    RESEND_COMPOSER_BUSY_COPY,
    RESEND_NOT_BROKEN_COPY,
    RESEND_STAGED_ATTACHMENTS_COPY,
    is_refused_echo,
    resend_refused_echo,
    resend_target_id,
    resend_turn,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# Drives the real controller and store (lessons-testing-evidence.md).
pytestmark = pytest.mark.bootstrap_profile

USER = ConsoleMessageRole.USER
ASSISTANT = ConsoleMessageRole.ASSISTANT
SYSTEM = ConsoleMessageRole.SYSTEM
TOOL = ConsoleMessageRole.TOOL
REPLY = "fresh reply"
BLOCKED_COPY = "Provider blocked: select a model"


class _Gateway:
    """Provider double whose behavior the test flips between attempts."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.emits_thinking = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.seen: list[list[dict]] = []
        self.on_stream = None

    async def resolve_for_send(self, _selection):
        if self.mode == "blocked":
            return SimpleNamespace(ready=False, visible_copy=BLOCKED_COPY)
        return SimpleNamespace(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
            may_emit_thinking=self.emits_thinking,
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.seen.append(list(messages))
        if self.on_stream is not None:
            self.on_stream()
        if self.mode == "error":
            raise RuntimeError("provider exploded")
        if self.mode == "partial-error":
            yield "partial text"
            raise RuntimeError("provider exploded mid-stream")
        if self.mode == "hang":
            self.started.set()
            await self.release.wait()
        if self.mode == "ok":
            yield REPLY


class _NoThinkingPersistence:
    """Real persistence that cannot round-trip thinking (preflight blocks)."""

    def __init__(self, delegate: ChatPersistenceService) -> None:
        self._delegate = delegate
        self.db = delegate.db

    @staticmethod
    def thinking_round_trip_version() -> int:
        return 0

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


@pytest.fixture
def databases():
    opened: list[CharactersRAGDB] = []
    yield opened
    for db in reversed(opened):
        db.close()


def _controller(store, gateway) -> ConsoleChatController:
    return ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )


def _console(tmp_path, databases, mode="ok", *, persistence_wrapper=None):
    db = CharactersRAGDB(tmp_path / "resend.sqlite", client_id="resend-test")
    databases.append(db)
    persistence = ChatPersistenceService(db)
    if persistence_wrapper is not None:
        persistence = persistence_wrapper(persistence)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Resend")
    gateway = _Gateway(mode)
    return SimpleNamespace(
        db=db,
        store=store,
        session_id=session.id,
        gateway=gateway,
        controller=_controller(store, gateway),
    )


def _restart(console) -> SimpleNamespace:
    """Rebuild the session from the database the way a relaunch does."""
    conversation_id = console.store._sessions[
        console.session_id
    ].persisted_conversation_id
    return _restore(console.db, conversation_id)


def _restore(db, conversation_id) -> SimpleNamespace:
    tree = ChatConversationService(db).get_conversation_tree(
        conversation_id, root_limit=100, depth_cap=100
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Resend",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=console_messages_from_conversation_tree(tree, db=db),
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    gateway = _Gateway("ok")
    return SimpleNamespace(
        db=db,
        store=store,
        session_id=session.id,
        gateway=gateway,
        controller=_controller(store, gateway),
    )


def _path(console) -> list[ConsoleChatMessage]:
    return console.store.messages_for_session(console.session_id)


def _live_db_rows(console) -> list[tuple[str, str | None]]:
    conversation_id = console.store._sessions[
        console.session_id
    ].persisted_conversation_id
    rows = (
        console.db.get_connection()
        .execute(
            "SELECT id, parent_message_id FROM messages "
            "WHERE conversation_id = ? AND deleted = 0",
            (conversation_id,),
        )
        .fetchall()
    )
    return [(row[0], row[1]) for row in rows]


def _assert_resent_in_place(console, user_id: str) -> ConsoleChatMessage:
    """One user row, one fresh reply directly under it, nothing else."""
    path = _path(console)
    assert [(row.role, row.status) for row in path] == [
        (USER, "complete"),
        (ASSISTANT, "complete"),
    ]
    user, reply = path
    assert user.id == user_id
    assert reply.content == REPLY
    assert reply.parent_message_id == user.persisted_message_id
    assert reply.sibling_count == 1
    assert console.store.active_path_message_ids(console.session_id) == [
        user.id,
        reply.id,
    ]
    assert sorted(_live_db_rows(console), key=lambda row: row[1] or "") == [
        (user.persisted_message_id, None),
        (reply.persisted_message_id, user.persisted_message_id),
    ]
    return reply


def _user(console) -> ConsoleChatMessage:
    return next(row for row in _path(console) if row.role is USER)


async def _stopped_empty_turn(console) -> None:
    console.gateway.mode = "hang"
    task = asyncio.create_task(console.controller.submit_draft("hello there"))
    await asyncio.wait_for(console.gateway.started.wait(), 2)
    assert console.controller.stop_active_run() is True
    await asyncio.wait_for(task, 2)
    console.gateway.mode = "ok"


# --- broken-turn detection -------------------------------------------------


def _m(role, *, status="complete", content="x", persisted=True, state=None):
    return ConsoleChatMessage(
        role=role,
        content=content,
        status=status,
        persisted_message_id="p" if persisted else None,
        assistant_generation_state=state,
    )


@pytest.mark.parametrize(
    ("label", "rows", "broken"),
    [
        ("refused echo", [_m(USER, status="failed", persisted=False)], True),
        ("refused echo + block row", [
            _m(USER, status="failed", persisted=False), _m(SYSTEM),
        ], True),
        ("persisted, no reply", [_m(USER)], True),
        ("failed reply", [_m(USER), _m(ASSISTANT, status="failed", content="")], True),
        ("failed partial reply", [_m(USER), _m(ASSISTANT, status="failed")], True),
        ("empty stopped reply", [
            _m(USER), _m(ASSISTANT, status="stopped", content=""), _m(SYSTEM),
        ], True),
        ("restored failed", [_m(USER), _m(ASSISTANT, content="", state="failed")], True),
        ("restored empty stopped", [
            _m(USER), _m(ASSISTANT, content="", state="stopped"),
        ], True),
        ("discarded", [
            _m(USER), _m(ASSISTANT, content="Response discarded.", state="discarded"),
        ], True),
        ("healthy", [_m(USER), _m(ASSISTANT, state="complete")], False),
        ("partial stopped", [_m(USER), _m(ASSISTANT, status="stopped")], False),
        ("streaming", [_m(USER), _m(ASSISTANT, status="streaming", content="")], False),
        ("in-flight echo", [_m(USER, persisted=False)], False),
        ("dispatch recovery owner", [
            _m(USER), _m(ASSISTANT, content="", state="accepted"),
        ], False),
        ("mid-path broken turn", [
            _m(USER), _m(ASSISTANT, status="failed", content=""), _m(SYSTEM),
            _m(USER), _m(ASSISTANT),
        ], False),
        ("no user row", [_m(ASSISTANT, status="failed")], False),
        # Review C1: a Continue chain whose earlier reply has text is healthy
        # history, never "no reply", even when its last reply failed.
        ("continue chain, last failed", [
            _m(USER), _m(ASSISTANT, state="complete"),
            _m(ASSISTANT, status="failed", content="partial text"),
        ], False),
        ("continue chain, restored failed", [
            _m(USER), _m(ASSISTANT, state="complete"),
            _m(ASSISTANT, content="partial text", state="failed"),
        ], False),
        # Review I1: tool output is a partial reply.
        ("stopped agent turn with tool output", [
            _m(USER), _m(ASSISTANT, status="stopped", content=""),
            _m(TOOL, content="read_file -> 3 lines"), _m(SYSTEM),
        ], False),
        ("tool output, no reply", [
            _m(USER), _m(TOOL, content="read_file -> 3 lines"),
        ], False),
        # Qodo #2: any tool output keeps the turn partial, whatever the reply.
        ("failed reply with tool output", [
            _m(USER), _m(ASSISTANT, status="failed", content=""),
            _m(TOOL, content="read_file -> 3 lines"), _m(SYSTEM),
        ], False),
        ("restored failed reply with tool output", [
            _m(USER), _m(ASSISTANT, content="", state="failed"),
            _m(TOOL, content="read_file -> 3 lines"),
        ], False),
        # Qodo #4: a restored failed reply with text is partial (Continue).
        ("restored failed reply with text", [
            _m(USER), _m(ASSISTANT, content="partial text", state="failed"),
        ], False),
    ],
)
def test_resend_target_is_the_last_user_row_of_a_broken_turn_only(label, rows, broken):
    expected = (
        next(row.id for row in reversed(rows) if row.role is USER) if broken else None
    )
    assert resend_target_id(rows) == expected, label


# --- each broken shape, real controller + SQLite ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "empty", "partial-error"])
async def test_resend_retries_a_failed_reply_in_place(tmp_path, databases, mode):
    console = _console(tmp_path, databases, mode)
    await console.controller.submit_draft("hello there")
    user = _user(console)
    failed = _path(console)[1]
    assert failed.status == "failed"
    assert resend_target_id(_path(console)) == user.id
    console.gateway.mode = "ok"

    result = await resend_turn(console.controller, user.id)

    assert result.accepted
    reply = _assert_resent_in_place(console, user.id)
    # Retried in place: the same assistant row, not a new one.
    assert reply.id == failed.id
    assert resend_target_id(_path(console)) is None


@pytest.mark.asyncio
async def test_resend_replaces_an_empty_stopped_reply(tmp_path, databases):
    console = _console(tmp_path, databases)
    await _stopped_empty_turn(console)
    user = _user(console)
    assert [row.role for row in _path(console)] == [USER, ASSISTANT, SYSTEM]
    assert resend_target_id(_path(console)) == user.id

    result = await resend_turn(console.controller, user.id)

    assert result.accepted
    _assert_resent_in_place(console, user.id)


@pytest.mark.asyncio
async def test_resend_after_restart_replaces_a_restored_failed_reply(
    tmp_path, databases
):
    console = _console(tmp_path, databases, "error")
    await console.controller.submit_draft("hello there")
    restarted = _restart(console)
    user = _user(restarted)
    restored = _path(restarted)[1]
    assert (restored.status, restored.assistant_generation_state) == (
        "complete",
        "failed",
    )

    result = await resend_turn(restarted.controller, user.id)

    assert result.accepted
    _assert_resent_in_place(restarted, user.id)


@pytest.mark.asyncio
async def test_resend_after_restart_answers_a_user_message_with_no_reply(
    tmp_path, databases
):
    console = _console(tmp_path, databases)
    await console.controller.submit_draft("hello there")
    console.store.delete_message(_path(console)[1].id)
    restarted = _restart(console)
    user = _user(restarted)
    assert [row.role for row in _path(restarted)] == [USER]
    assert resend_target_id(_path(restarted)) == user.id

    result = await resend_turn(restarted.controller, user.id)

    assert result.accepted
    _assert_resent_in_place(restarted, user.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("relaunch", [False, True])
async def test_resend_after_a_discarded_dispatch_recovery(
    tmp_path, databases, relaunch
):
    db, conversation_id, repository = _database(tmp_path / "discard.sqlite")
    databases.append(db)
    _insert(db, repository, _acceptance(conversation_id))
    # TASK-33662: restored through the production hydration (fresh native
    # ids), not ``_restored_store``, whose native ids equal the persisted ids.
    console = _restore(db, conversation_id)
    assert resend_target_id(_path(console)) is None
    discarded = await console.controller.discard_dispatch_recovery(console.session_id)
    assert discarded.accepted, discarded.visible_copy
    if relaunch:
        console = _restore(db, conversation_id)
    user = _user(console)
    assert _path(console)[1].assistant_generation_state == "discarded"
    assert resend_target_id(_path(console)) == user.id

    result = await resend_turn(console.controller, user.id)

    assert result.accepted
    _assert_resent_in_place(console, user.id)


class _Recoveries:
    """The runtime's unsent-turn recovery surface (``ConsoleRuntime``)."""

    def __init__(self, *entries) -> None:
        self.entries = list(entries)
        self.discarded: list[str] = []

    def recoveries_for_session(self, _session_id):
        return tuple(self.entries)

    def discard_turn_recovery(self, turn_id):
        self.discarded.append(turn_id)
        self.entries = [entry for entry in self.entries if entry.turn_id != turn_id]
        return True


class _Composer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.captures: list[object] = []

    def draft_text(self) -> str:
        return self.text

    def capture_draft_for_send(self):
        stash = SimpleNamespace(text=self.text)
        self.captures.append(stash)
        return stash

    def load_draft(self, text: str) -> None:
        self.text = text


def _echo_resender(console, *, runtime=None, composer=None, dispatched=None):
    async def dispatch(draft, stash, session_id):
        if dispatched is not None:
            dispatched.append((draft, stash))
        result = await console.controller.submit_draft(draft, session_id=session_id)
        return result.accepted

    return lambda echo: resend_refused_echo(
        echo,
        store=console.store,
        runtime=runtime,
        composer=composer,
        dispatch=dispatch,
    )


@pytest.mark.asyncio
async def test_resend_re_sends_a_refused_echo_as_one_user_message(
    tmp_path, databases
):
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    assert (echo.status, echo.persisted_message_id) == ("failed", None)
    assert [row.role for row in _path(console)] == [USER, SYSTEM]
    console.gateway.mode = "ok"

    result = await resend_turn(
        console.controller, echo.id, resend_echo=_echo_resender(console)
    )

    assert result.accepted
    path = _path(console)
    assert [(row.role, row.status, row.content) for row in path] == [
        (USER, "complete", "hello there"),
        (ASSISTANT, "complete", REPLY),
    ]
    assert path[0].persisted_message_id is not None
    assert path[1].parent_message_id == path[0].persisted_message_id
    assert len(_live_db_rows(console)) == 2


@pytest.mark.asyncio
async def test_refused_echo_resend_consumes_its_recovery_and_the_composer_copy(
    tmp_path, databases, monkeypatch
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda _p, _m: True)
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("look at this")
    echo = _user(console)
    image = PendingAttachment(
        file_path="/tmp/photo.png",
        display_name="photo.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
    )
    runtime = _Recoveries(
        SimpleNamespace(turn_id="turn-1", draft="look at this", attachments=(image,))
    )
    composer = _Composer("look at this")
    dispatched: list = []
    console.gateway.mode = "ok"

    result = await resend_turn(
        console.controller,
        echo.id,
        resend_echo=_echo_resender(
            console, runtime=runtime, composer=composer, dispatched=dispatched
        ),
    )

    assert result.accepted
    assert runtime.discarded == ["turn-1"]
    # The composer's own copy is handed to the send path, which commits it.
    assert dispatched == [("look at this", composer.captures[0])]
    user, reply = _path(console)
    assert (user.content, reply.content) == ("look at this", REPLY)
    assert [attachment.display_name for attachment in user.attachments] == [
        "photo.png"
    ]
    assert console.store.pending_attachments(console.session_id) == []


@pytest.mark.asyncio
async def test_refused_echo_resend_never_overwrites_a_different_draft(
    tmp_path, databases
):
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    runtime = _Recoveries(
        SimpleNamespace(turn_id="turn-1", draft="hello there", attachments=())
    )
    composer = _Composer("a newer thought")
    console.gateway.mode = "ok"

    result = await resend_turn(
        console.controller,
        echo.id,
        resend_echo=_echo_resender(console, runtime=runtime, composer=composer),
    )

    assert (result.accepted, result.visible_copy) == (False, RESEND_COMPOSER_BUSY_COPY)
    assert composer.text == "a newer thought"
    assert runtime.discarded == []
    assert _user(console).id == echo.id


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError, RuntimeError])
async def test_refused_echo_text_survives_a_dispatch_that_raises(
    tmp_path, databases, error
):
    """Review I2: the echo is deleted and its recovery consumed before the
    send-path await. A cancelled or failing dispatch must leave the text
    recoverable, never gone."""
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    runtime = _Recoveries(
        SimpleNamespace(turn_id="turn-1", draft="hello there", attachments=())
    )
    composer = _Composer()

    async def explode(_draft, _stash, _session_id):
        raise error("dispatch interrupted")

    with pytest.raises(error):
        await resend_refused_echo(
            echo,
            store=console.store,
            runtime=runtime,
            composer=composer,
            dispatch=explode,
        )

    assert composer.text == "hello there"
    assert console.store.session_draft(console.session_id) == "hello there"


@pytest.mark.asyncio
async def test_a_refused_echo_resend_never_overwrites_text_typed_during_the_send(
    tmp_path, databases
):
    """Review M2: the send path can refuse after the user typed more."""
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    composer = _Composer()

    async def refuse_after_typing(_draft, _stash, _session_id):
        composer.text = "newer typing"
        return False

    copy = await resend_refused_echo(
        echo,
        store=console.store,
        runtime=None,
        composer=composer,
        dispatch=refuse_after_typing,
    )

    assert copy is None
    assert composer.text == "newer typing"


@pytest.mark.asyncio
async def test_refused_echo_resend_keeps_the_draft_when_the_send_path_refuses(
    tmp_path, databases
):
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    composer = _Composer()

    async def refuse(_draft, _stash, _session_id):
        return False

    copy = await resend_refused_echo(
        echo, store=console.store, runtime=None, composer=composer, dispatch=refuse
    )

    assert copy is None
    assert composer.text == "hello there"
    assert console.store.session_draft(console.session_id) == "hello there"


# --- every normal send gate applies ----------------------------------------


@pytest.mark.asyncio
async def test_refused_resend_shows_the_normal_send_readiness_copy(
    tmp_path, databases
):
    console = _console(tmp_path, databases, "error")
    await console.controller.submit_draft("hello there")
    user = _user(console)
    console.gateway.mode = "blocked"

    result = await resend_turn(console.controller, user.id)

    assert (result.accepted, result.visible_copy) == (False, BLOCKED_COPY)
    assert _path(console)[-1].role is SYSTEM
    assert _path(console)[-1].content == BLOCKED_COPY
    # Still broken, so Resend stays on offer.
    assert resend_target_id(_path(console)) == user.id


@pytest.mark.asyncio
async def test_resend_applies_the_vision_gate_to_the_turns_attachments(
    tmp_path, databases, monkeypatch
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda _p, _m: True)
    console = _console(tmp_path, databases, "error")
    console.store.add_pending_attachment(
        console.session_id,
        PendingAttachment(
            file_path="/tmp/photo.png",
            display_name="photo.png",
            file_type="image",
            insert_mode="attachment",
            data=b"\x89PNG-bytes",
            mime_type="image/png",
        ),
    )
    await console.controller.submit_draft("look at this")
    user = _user(console)
    assert user.attachments
    failed = _path(console)[1]
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda _p, _m: False)
    console.gateway.mode = "ok"

    result = await resend_turn(console.controller, user.id)

    assert not result.accepted
    assert "can't accept images" in result.visible_copy
    assert console.store.get_message(failed.id).status == "failed"
    assert console.gateway.seen == [console.gateway.seen[0]]


@pytest.mark.asyncio
async def test_resend_applies_skill_refusal_like_a_normal_send(
    tmp_path, databases, monkeypatch
):
    console = _console(tmp_path, databases, "error")
    await console.controller.submit_draft("hello there")
    restarted = _restart(console)
    user = _user(restarted)

    async def refuse(messages, _context):
        return messages, "Skill blocked: review it first.", (), (), ""

    monkeypatch.setattr(restarted.controller, "_apply_skill_substitution", refuse)

    result = await resend_turn(restarted.controller, user.id)

    assert (result.accepted, result.visible_copy) == (
        False,
        "Skill blocked: review it first.",
    )
    assert restarted.gateway.seen == []
    assert resend_target_id(_path(restarted)) == user.id


@pytest.mark.asyncio
async def test_resend_applies_the_thinking_persistence_preflight(
    tmp_path, databases
):
    console = _console(
        tmp_path, databases, persistence_wrapper=_NoThinkingPersistence
    )
    await _stopped_empty_turn(console)
    user = _user(console)
    console.gateway.emits_thinking = True
    contacts = len(console.gateway.seen)

    result = await resend_turn(console.controller, user.id)

    assert not result.accepted
    assert "thinking" in result.visible_copy.lower()
    assert len(console.gateway.seen) == contacts
    assert resend_target_id(_path(console)) == user.id


@pytest.mark.asyncio
async def test_resend_applies_the_pinned_prefill(tmp_path, databases):
    console = _console(tmp_path, databases, "error")
    await console.controller.submit_draft("hello there")
    restarted = _restart(console)
    restarted.store.set_session_pinned_prefill(restarted.session_id, "PINNED")

    result = await resend_turn(restarted.controller, _user(restarted).id)

    assert result.accepted
    assert restarted.gateway.seen[-1][-1] == {"role": "assistant", "content": "PINNED"}
    assert _path(restarted)[-1].content == "PINNED" + REPLY


@pytest.mark.asyncio
async def test_a_refused_clear_is_reported_without_re_running(
    tmp_path, databases, monkeypatch
):
    console = _console(tmp_path, databases)
    await _stopped_empty_turn(console)
    user = _user(console)
    contacts = len(console.gateway.seen)

    def refuse(_message_id):
        raise ValueError("Resolve pending dispatch before deleting this message.")

    monkeypatch.setattr(console.store, "delete_message", refuse)

    result = await resend_turn(console.controller, user.id)

    assert (result.accepted, result.visible_copy) == (
        False,
        "Resolve pending dispatch before deleting this message.",
    )
    assert len(console.gateway.seen) == contacts


@pytest.mark.parametrize(
    "function",
    [
        is_refused_echo,
        resend_target_id,
        resend_turn,
        resend_refused_echo,
        ConsoleChatController.continue_from_message,
    ],
)
def test_resend_entry_points_document_args_and_returns(function):
    """Qodo #1: Google-style Args/Returns name every parameter."""
    doc = inspect.getdoc(function) or ""
    assert "Args:" in doc and "Returns:" in doc, function.__name__
    for name in inspect.signature(function).parameters:
        if name != "self":
            assert f"{name}:" in doc, (function.__name__, name)


def _all_db_rows(console) -> list[tuple]:
    conversation_id = console.store._sessions[
        console.session_id
    ].persisted_conversation_id
    return sorted(
        tuple(row)
        for row in console.db.get_connection().execute(
            "SELECT id, content, deleted FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
    )


def _path_snapshot(console) -> list[tuple]:
    return [(row.id, row.role, row.status, row.content) for row in _path(console)]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["failed", "stopped"])
async def test_a_backup_pause_refuses_resend_before_anything_is_cleared(
    tmp_path, databases, shape
):
    """Qodo #3: the maintenance boundary used to refuse inside retry/continue,
    after the reply and its failure rows were already tombstoned."""
    console = _console(tmp_path, databases, "error")
    if shape == "failed":
        await console.controller.submit_draft("hello there")
    else:
        await _stopped_empty_turn(console)
    user = _user(console)
    assert resend_target_id(_path(console)) == user.id
    before_path, before_db = _path_snapshot(console), _all_db_rows(console)
    console.controller.maintenance_close_admission()

    result = await resend_turn(console.controller, user.id)

    assert _path_snapshot(console) == before_path
    assert _all_db_rows(console) == before_db
    assert (result.accepted, result.visible_copy) == (
        False,
        "Console generation is paused for backup maintenance.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("relaunch", [False, True])
async def test_resend_never_offers_a_failed_turn_with_tool_output(
    tmp_path, databases, relaunch
):
    """Qodo #2: a failed reply with tool output kept stale tool markers on the
    in-place retry. Any tool output now makes the turn partial; the live
    failed reply keeps its own Retry."""
    console = _console(tmp_path, databases, "error")

    def tool_marker():
        console.store.append_message(
            console.session_id, role=TOOL, content="read_file -> 3 lines"
        )

    console.gateway.on_stream = tool_marker
    await console.controller.submit_draft("hello there")
    if relaunch:
        console = _restart(console)
        # The relaunch overlays the run's tool markers on the restored reply.
        tool_marker()
    user = _user(console)
    console.gateway.mode = "ok"
    before_path, before_db = _path_snapshot(console), _all_db_rows(console)

    result = await resend_turn(console.controller, user.id)

    assert _path_snapshot(console) == before_path
    assert _all_db_rows(console) == before_db
    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert resend_target_id(_path(console)) is None


@pytest.mark.asyncio
async def test_resend_never_offers_a_restored_failed_reply_that_has_text(
    tmp_path, databases
):
    """Qodo #4 (data loss): after a relaunch a failed reply with partial text
    reads complete+failed; Resend used to tombstone it. It is partial."""
    console = _console(tmp_path, databases, "partial-error")
    await console.controller.submit_draft("hello there")
    restarted = _restart(console)
    reply = _path(restarted)[1]
    assert (reply.status, reply.assistant_generation_state, reply.content) == (
        "complete",
        "failed",
        "partial text",
    )
    before_path, before_db = _path_snapshot(restarted), _all_db_rows(restarted)

    result = await resend_turn(restarted.controller, _user(restarted).id)

    assert _path_snapshot(restarted) == before_path
    assert _all_db_rows(restarted) == before_db
    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert resend_target_id(_path(restarted)) is None


def _image(name: str, data: bytes) -> PendingAttachment:
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_recovery", [True, False])
async def test_refused_echo_resend_refuses_newer_staged_attachments(
    tmp_path, databases, with_recovery
):
    """Qodo #5: an attachment staged after the refusal rode along with (or
    replaced) the echo's own on Resend."""
    console = _console(tmp_path, databases, "blocked")
    await console.controller.submit_draft("hello there")
    echo = _user(console)
    console.store.add_pending_attachment(
        console.session_id, _image("newer.png", b"newer-bytes")
    )
    runtime = (
        _Recoveries(
            SimpleNamespace(turn_id="turn-1", draft="hello there", attachments=())
        )
        if with_recovery
        else None
    )
    dispatched: list = []

    async def dispatch(draft, stash, _session_id):
        dispatched.append(draft)
        return True

    copy = await resend_refused_echo(
        echo,
        store=console.store,
        runtime=runtime,
        composer=_Composer(),
        dispatch=dispatch,
    )

    assert copy == RESEND_STAGED_ATTACHMENTS_COPY
    assert dispatched == []
    assert [item.display_name for item in console.store.pending_attachments(
        console.session_id
    )] == ["newer.png"]
    assert _user(console).id == echo.id
    if runtime is not None:
        assert runtime.discarded == []


@pytest.mark.asyncio
async def test_refused_echo_resend_sends_its_own_restaged_attachments_once(
    tmp_path, databases, monkeypatch
):
    """The staged-attachment refusal must not refuse the echo's own files when
    they are what is staged (a Restore from the unsent-turn shelf)."""
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda _p, _m: True)
    console = _console(tmp_path, databases, "blocked")
    console.store.add_pending_attachment(
        console.session_id, _image("photo.png", b"own-bytes")
    )
    await console.controller.submit_draft("look at this")
    echo = _user(console)
    assert [item.display_name for item in console.store.pending_attachments(
        console.session_id
    )] == ["photo.png"]
    console.gateway.mode = "ok"

    result = await resend_turn(
        console.controller,
        echo.id,
        resend_echo=_echo_resender(console, composer=_Composer("look at this")),
    )

    assert result.accepted
    user, _reply = _path(console)
    assert [item.display_name for item in user.attachments] == ["photo.png"]


# --- negatives -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("relaunch", [False, True])
async def test_resend_never_touches_a_continue_chain_with_a_healthy_reply(
    tmp_path, databases, relaunch
):
    """Review C1 (data loss): U -> A "fresh reply" -> Continue -> B fails with
    partial text. After a relaunch, B reads complete+failed, and Resend used to
    anchor its clear at U, tombstoning A's whole subtree."""
    console = _console(tmp_path, databases)
    await console.controller.submit_draft("hello there")
    healthy = _path(console)[1]
    console.gateway.mode = "partial-error"
    await console.controller.continue_from_message(healthy.id)
    if relaunch:
        console = _restart(console)
    user = _user(console)
    before = sorted(_live_db_rows(console))

    result = await resend_turn(console.controller, user.id)

    assert sorted(_live_db_rows(console)) == before
    contents = [row.content for row in _path(console) if row.role is ASSISTANT]
    assert contents[0] == REPLY
    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert resend_target_id(_path(console)) is None


@pytest.mark.asyncio
async def test_resend_never_offers_a_stopped_turn_with_tool_output(
    tmp_path, databases
):
    """Review I1: tool output is a partial reply; Resend would wipe the tool
    trace and re-run the tools."""
    console = _console(tmp_path, databases, "hang")
    task = asyncio.create_task(console.controller.submit_draft("hello there"))
    await asyncio.wait_for(console.gateway.started.wait(), 2)
    console.store.append_message(
        console.session_id, role=TOOL, content="read_file -> 3 lines"
    )
    assert console.controller.stop_active_run() is True
    await asyncio.wait_for(task, 2)
    user = _user(console)
    assert [(row.role, row.content) for row in _path(console)][1:] == [
        (ASSISTANT, ""),
        (TOOL, "read_file -> 3 lines"),
        (SYSTEM, "Response stopped by user."),
    ]
    before_rows = [row.id for row in _path(console)]
    before_db = sorted(_live_db_rows(console))
    console.gateway.mode = "ok"
    contacts = len(console.gateway.seen)

    result = await resend_turn(console.controller, user.id)

    assert [row.id for row in _path(console)] == before_rows
    assert sorted(_live_db_rows(console)) == before_db
    assert len(console.gateway.seen) == contacts
    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert resend_target_id(_path(console)) is None


@pytest.mark.asyncio
async def test_resend_refuses_a_healthy_turn(tmp_path, databases):
    console = _console(tmp_path, databases)
    await console.controller.submit_draft("hello there")
    before = [(row.id, row.content) for row in _path(console)]

    result = await resend_turn(console.controller, _user(console).id)

    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert [(row.id, row.content) for row in _path(console)] == before
    assert len(console.gateway.seen) == 1


@pytest.mark.asyncio
async def test_resend_refuses_a_broken_turn_above_the_last_one(tmp_path, databases):
    console = _console(tmp_path, databases, "error")
    await console.controller.submit_draft("first")
    first_user = _user(console)
    console.gateway.mode = "ok"
    await console.controller.submit_draft("second")
    before = [row.id for row in _path(console)]

    result = await resend_turn(console.controller, first_user.id)

    assert (result.accepted, result.visible_copy) == (False, RESEND_NOT_BROKEN_COPY)
    assert [row.id for row in _path(console)] == before


@pytest.mark.asyncio
async def test_resend_refuses_while_a_run_is_live(tmp_path, databases):
    console = _console(tmp_path, databases, "hang")
    task = asyncio.create_task(console.controller.submit_draft("hello there"))
    await asyncio.wait_for(console.gateway.started.wait(), 2)
    user = _user(console)
    assert resend_target_id(_path(console)) is None

    result = await resend_turn(console.controller, user.id)

    assert (result.accepted, result.visible_copy) == (
        False,
        "A run is already running in this tab.",
    )
    console.controller.stop_active_run()
    await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_resend_refuses_while_a_dispatch_recovery_is_unresolved(
    tmp_path, databases
):
    db, conversation_id, repository = _database(tmp_path / "recovery.sqlite")
    databases.append(db)
    _insert(db, repository, _acceptance(conversation_id))
    console = _restore(db, conversation_id)
    assert console.store.dispatch_recovery_for_session(console.session_id)
    user = _user(console)
    assert resend_target_id(_path(console)) is None

    result = await resend_turn(console.controller, user.id)

    assert not result.accepted
    assert result.visible_copy.startswith("Finish or discard the pending response")
    assert console.gateway.seen == []
