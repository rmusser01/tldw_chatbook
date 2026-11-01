"""Real edit/fork persistence across the live controller's key-readiness await."""

import asyncio
import base64
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_citation_key_first_use import _Keyring, _compose
from Tests.Chat.test_citation_trace_repository import _repository, _sealed_write
from Tests.Chat.test_console_chat_fork_persistence import _commit
from Tests.UI.console_controller_stubs import stub_message_controller
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_trace_identity import (
    CITATION_FINGERPRINT_KEYRING_SERVICE,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.session import (
    ConsoleSessionController,
    _ConsoleForkRequest,
)
from tldw_chatbook.Widgets.Console.console_edit_message_modal import ConsoleEditResult

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def cited_store(tmp_path):
    db = CharactersRAGDB(tmp_path / "readiness.sqlite", client_id="readiness")
    ready = _repository(db)
    service = ChatPersistenceService(db, citation_repository=ready)
    store = ConsoleChatStore(persistence=service)
    session = store.create_session(
        title="Original",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
    )
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Question", persist=True
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Answer [S1].",
        persist=True,
    )
    prepared = ready.prepare_write(_sealed_write())
    row = db.get_message_by_id(message.persisted_message_id)
    with db.transaction() as cursor:
        ready.write_prepared(
            cursor,
            prepared,
            message_id=row["id"],
            message_revision=row["version"],
            message_body=row["content"],
        )
    entered, release = threading.Event(), threading.Event()

    class HeldKeyring(_Keyring):
        __module__ = "keyring.backends.macOS"

        def get_password(self, service, account):
            entered.set()
            assert release.wait(15)
            return super().get_password(service, account)

    backend = HeldKeyring()
    backend.secrets[
        (
            CITATION_FINGERPRINT_KEYRING_SERVICE,
            ready.identity_context.fingerprint_key_id,
        )
    ] = base64.b64encode(b"k" * 32).decode()
    repository = _compose(db, tmp_path, backend)
    service.citation_repository = repository
    yield db, store, user, message, repository, entered, release
    release.set()
    if repository._key_acquisition is not None:
        repository._key_acquisition.thread.join(5)
        assert not repository._key_acquisition.thread.is_alive()
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["none", "session", "store", "controller", "body", "cancel"]
)
async def test_edit_waits_for_original_key_and_rechecks_original_target(
    cited_store, change
):
    db, store, _user, message, repository, entered, release = cited_store
    opened = []
    current = {"store": store, "controller": None}

    async def push_screen(modal, callback):
        opened.append((modal, callback))

    async def sync():
        pass

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(notify=lambda *args, **kwargs: None),
        app=SimpleNamespace(push_screen=push_screen),
        run_worker=lambda coroutine, **kwargs: coroutine.close(),
    )
    controller = stub_message_controller(
        screen,
        chat_store_accessor=lambda: current["store"],
        current_chat_store_accessor=lambda: current["store"],
        current_chat_controller_accessor=lambda: current["controller"],
        sync_native_console_chat_ui=sync,
    )
    before = dict(db.get_message_by_id(message.persisted_message_id))
    task = asyncio.create_task(
        controller._open_console_message_edit_modal(
            message_id=message.id, content=message.content
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert opened == []
        assert dict(db.get_message_by_id(message.persisted_message_id)) == before
        # The held key owns no write lock.
        db.add_note(title="Unrelated", content="Writer remains available")
        if change == "session":
            store.create_session(title="Another")
        elif change == "store":
            current["store"] = ConsoleChatStore()
        elif change == "controller":
            current["controller"] = object()
        elif change == "body":
            # Simulate a refreshed native projection without changing the saved owner.
            store._message_or_raise(message.id).content = "Changed"
        elif change == "cancel":
            task.cancel()
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.1)
            assert task in done
        release.set()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        if change == "none":
            assert outcomes == [None]
            assert len(opened) == 1
            opened[0][1](ConsoleEditResult(text=message.content, resend=False))
            after = db.get_message_by_id(message.persisted_message_id)
            assert after["version"] > before["version"]
            reopened = _repository(db)
            active = reopened.get_active_trace_for_current_message(
                after["id"], after["content"]
            )
            assert reopened.verify_active_trace_result(active)
            assert active.summary.namespace.trace_id == "trace-1"
        else:
            assert opened == []
            assert dict(db.get_message_by_id(message.persisted_message_id)) == before
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "source", "store", "request", "cancel"])
async def test_fork_classifies_only_after_key_readiness_and_revalidates_fence(
    cited_store, change
):
    db, store, user, message, repository, entered, release = cited_store
    states = []
    modal = SimpleNamespace(
        state="validating",
        show_stale_source=lambda: states.append("stale"),
        show_committing=lambda: states.append("committing"),
        show_precommit_error=lambda error: states.append(error),
    )
    request = _ConsoleForkRequest(
        store.issue_fork_fence(message.id),
        "fork-session",
        "fork-conversation",
        modal,
        "Fork",
    )

    async def commit(store, request):
        _commit(store.persistence, request.snapshot)
        request.committed = True
        return True

    async def register(*args):
        registrations.append(True)
        return False

    registrations = []
    controller = SimpleNamespace(
        _active_fork_request=request,
        _fork_validation_generation=1,
        _ensure_console_chat_store=lambda: store,
        _fork_prefix_messages=lambda s,
        f: ConsoleSessionController._fork_prefix_messages(None, s, f),
        _validate_fork_image_selections_fn=lambda *_: True,
        _commit_durable_console_chat_fork=commit,
        _register_console_chat_fork=register,
        _show_created_not_opened=lambda *_: None,
    )
    controller._fork_request_is_current = (
        lambda r, g: ConsoleSessionController._fork_request_is_current(controller, r, g)
    )
    before = [
        tuple(row)
        for row in db.get_connection().execute("SELECT * FROM rag_message_trace_owners")
    ]
    task = asyncio.create_task(
        ConsoleSessionController._run_console_chat_fork(controller, request, 1)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert request.snapshot is None and not request.committed
        assert [
            tuple(row)
            for row in db.get_connection().execute(
                "SELECT * FROM rag_message_trace_owners"
            )
        ] == before
        if change == "source":
            store.update_message_content(user.id, "Changed question")
        elif change == "store":
            controller._ensure_console_chat_store = lambda: object()
        elif change == "request":
            controller._fork_validation_generation += 1
        elif change == "cancel":
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.1)
            assert task in done
        release.set()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        if change == "none":
            assert outcomes == [None]
            assert request.committed
            fork_reply = request.snapshot.messages[-1]
            reopened = _repository(db)
            active = reopened.get_active_trace_for_current_message(
                fork_reply.persisted_message_id, "Answer [S1]."
            )
            assert reopened.verify_active_trace_result(active)
            assert active.summary.namespace.trace_id == "trace-1"
        else:
            assert request.snapshot is None and not request.committed
            assert registrations == []
            assert db.get_conversation_by_id("fork-conversation") is None
            assert [
                tuple(row)
                for row in db.get_connection().execute(
                    "SELECT * FROM rag_message_trace_owners"
                )
            ] == before
            if change in {"source", "store"}:
                assert states == ["stale"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_direct_edit_and_resend_refuses_cited_assistant_without_touching_owner(
    cited_store,
):
    from Tests.Chat.test_console_edit_resend import StreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    db, store, _user, message, repository, entered, _release = cited_store
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    before = dict(db.get_message_by_id(message.persisted_message_id))
    owners = [
        tuple(row)
        for row in db.get_connection().execute("SELECT * FROM rag_message_trace_owners")
    ]
    result = await controller.edit_and_resend_message(message.id, "New answer")
    assert not result.accepted
    assert "Only your messages" in result.visible_copy
    assert dict(db.get_message_by_id(message.persisted_message_id)) == before
    assert [
        tuple(row)
        for row in db.get_connection().execute("SELECT * FROM rag_message_trace_owners")
    ] == owners
    assert repository.fingerprint_key_pending and not entered.is_set()
