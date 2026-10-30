"""Persisted 'Cited sources' on the real Console send paths (TASK-33620.8).

Owner decision 2026-10-27: citations persist by default. Canonical citation
records ship on, the fingerprint key is created in the secure system keyring
on first use, and an Automatic (prepared-bundle) send gets the same canonical
capture as a manually staged one. Before this task a default install sent
[Sn]-labelled evidence on both paths and no 'Cited sources' row ever
appeared: the staged path had records off, and the Automatic path never built
a trace at all.

These tests drive the real ChatScreen, controller, store, runtime and
citation repository against a file-backed ChaChaNotes database (a
``:memory:`` one is per-thread, and durable commits run off the loop), with
the shipped default config. Only the provider gateway, the Library search
service and the keyring backend are doubles. The keyring double reports a
secure backend, as the macOS Keychain would; the null keyring the test suite
installs is exactly the fail-closed case and is covered separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _select_llamacpp_console,
    _wait_for_selector,
)
from Tests.UI.test_console_send_acknowledgement import eager_tasks, press, until
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_citation_sources_modal import (
    CITATION_SOURCES_UNAVAILABLE_COPY,
    ConsoleCitationSourcesModal,
)

pytestmark = pytest.mark.bootstrap_profile

NOTE_TITLE = "Zephyr Orchard launch checkpoint"
NOTE_BODY = "Verification code marigold-73. Launch review Thursday at 14:30 UTC."
REPLY = "The verification code is marigold-73 [S1]."
QUESTION = "What is the verification code?"


class _SecureKeyringDouble:
    """A secure-classified keyring that keeps secrets in memory.

    ``is_secure_keyring_backend`` classifies a backend by its module path, so
    this reports the macOS backend's module. Secrets outlive an app instance
    (one double serves the app before and after a restart), as a real
    Keychain's do.
    """

    __module__ = "keyring.backends.macOS"
    priority = 5

    def __init__(self) -> None:
        self.secrets: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.secrets.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.secrets[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.secrets.pop((service, account), None)


class _CitingGateway(_ReadyResolutionGateway):
    """Ready provider that answers with an [S1] citation and records requests."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def resolve_context_window(self, settings):
        return self.cached_context_window(settings)

    async def stream_chat(self, resolution, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        yield REPLY


class _NoteLibrary:
    """Library search double that finds the one seeded note."""

    def __init__(self, note_id: str) -> None:
        self.note_id = note_id
        self.calls = 0

    async def search(self, query, source_types, mode, **kwargs):
        self.calls += 1
        return {
            "runtime_backend": "local",
            "results": [
                {
                    "source_id": self.note_id,
                    "source_type": "notes",
                    "title": NOTE_TITLE,
                    "snippet": NOTE_BODY,
                    "score": 0.91,
                    "runtime_backend": "local",
                }
            ],
        }


@pytest.fixture
def secure_keyring(monkeypatch) -> _SecureKeyringDouble:
    import keyring

    backend = _SecureKeyringDouble()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    return backend


@pytest.fixture
def direct_sends():
    """Route sends straight to the provider double (no agent loop)."""

    # A file-backed DB makes the runtime build an agent bridge, whose loop
    # calls the real provider; this keeps sends on the gateway double. The
    # bootstrap profile restores the config file after the test.
    assert config_module.save_setting_to_cli_config("console", "agent_runtime", False)
    yield


def _build_app(db_path: Path):
    """Build a factory app over a file-backed ChaChaNotes DB.

    The citation services are composed by the app's own wiring, so the
    repository, its policy (the shipped default config) and its key provider
    are the production ones.
    """

    app = _build_test_app()
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    db = CharactersRAGDB(db_path, client_id="cited-sources-ui")
    app.chachanotes_db = db
    app._wire_chat_conversation_services()
    gateway = _CitingGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    _allow_second_turns(host)
    return host, app, db, gateway


def _note_payload(note_id: str) -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="notes",
        item_type="note",
        title=NOTE_TITLE,
        body=NOTE_BODY,
        source_id=note_id,
        suggested_prompt="Use this note as context.",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
    )


async def _ready_console(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    await _select_llamacpp_console(console)
    await pilot.pause(0.3)
    return console


def _completed_assistant(console):
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    for message in store.messages_for_session(session_id):
        if (
            message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "complete"
            and message.persisted_message_id
        ):
            return message
    return None


def _active_conversation_id(console) -> str:
    store = console._ensure_console_chat_store()
    session = next(
        item for item in store.sessions() if item.id == store.active_session_id
    )
    assert session.persisted_conversation_id
    return session.persisted_conversation_id


def _completed_assistants(console) -> int:
    store = console._ensure_console_chat_store()
    return sum(
        1
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
        and message.status == "complete"
        and message.persisted_message_id
    )


def _cited_sources_buttons(console):
    return list(console.query(".console-transcript-citation-sources"))


async def _send(host, pilot, console, text: str) -> None:
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    composer.load_draft(text)
    await pilot.pause()
    press(host, "enter", "\r")
    await until(lambda: _completed_assistant(console) is not None, timeout=20)


async def _open_sources(host, console):
    await until(lambda: bool(_cited_sources_buttons(console)), timeout=20)
    button = _cited_sources_buttons(console)[0]
    label = str(button.label)
    button.press()
    await until(
        lambda: isinstance(host.screen, ConsoleCitationSourcesModal)
        and bool(host.screen.display_rows),
        timeout=20,
    )
    return label, host.screen.display_rows


def _assert_note_row(rows, note_id: str) -> None:
    assert len(rows) == 1
    row = rows[0]
    assert row.display_marker == "[S1]"
    assert row.title.endswith(NOTE_TITLE)
    assert "marigold-73" in row.snapshot_text
    assert row.open_source_type == "notes"
    assert row.source_id == note_id


def _trace_count(db: CharactersRAGDB) -> int:
    return db.get_connection().execute(
        "SELECT COUNT(*) FROM rag_citation_traces"
    ).fetchone()[0]


def _assert_export_carries_the_record(db_path: Path, conversation_id: str) -> None:
    """What a Chatbook export writes for the cited reply (its citation payload)."""
    from tldw_chatbook.Chat.citation_service_factory import (
        build_local_citation_conversation_service,
    )
    from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator

    db = CharactersRAGDB(db_path, client_id="cited-sources-export")
    try:
        service, _repository, _migration = build_local_citation_conversation_service(
            db, sidecar_path=db_path.with_name("rag_context.json")
        )
        messages = service.get_messages_with_context(conversation_id)
        cited = [
            ChatbookCreator._message_citation_export_payload(message)
            for message in messages
            if "[S1]" in str(message.get("content") or "")
        ]
        assert cited and cited[0]["citations"]
        references = cited[0]["evidence_bundle"]["references"]
        assert references[0]["title"].endswith(NOTE_TITLE)
        assert "marigold-73" in references[0]["snippet"]
    finally:
        db.close_connection()


async def _reopen_after_restart(db_path: Path, conversation_id: str, note_id: str):
    host, app, db, _gateway = _build_app(db_path)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                assert await console._workspace._resume_console_workspace_conversation(
                    conversation_id
                )
                label, rows = await _open_sources(host, console)
                assert label == "Cited sources (1)"
                _assert_note_row(rows, note_id)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_staged_note_reply_shows_cited_sources_and_survives_restart(
    tmp_path, secure_keyring, direct_sends
):
    """AC#1/#3/#6: staged Library evidence -> persisted Cited sources row."""
    assert config_module.get_rag_citation_canonical_writes_enabled() is True
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION)
                label, rows = await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
                await host.screen.request_safe_cancel(source="visible")
                await until(
                    lambda: not isinstance(host.screen, ConsoleCitationSourcesModal)
                )
                # AC#7: sealing the record must also settle the turn, or the
                # next send is refused ("active dispatch checkpoint").
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                composer.focus()
                composer.load_draft("And when is the review?")
                await pilot.pause()
                press(host, "enter", "\r")
                await until(lambda: _completed_assistants(console) == 2, timeout=20)
        assert label == "Cited sources (1)"
        _assert_note_row(rows, note_id)
        assert "[S1]" in gateway.requests[0][-1]["content"]
        assert len(gateway.requests) == 2
        assert _trace_count(db) == 1
    finally:
        db.close_connection()
    _assert_export_carries_the_record(db_path, conversation_id)
    await _reopen_after_restart(db_path, conversation_id, note_id)


@pytest.mark.asyncio
async def test_automatic_note_reply_shows_cited_sources_and_survives_restart(
    tmp_path, secure_keyring, direct_sends
):
    """AC#2/#3/#6: Automatic retrieval gets the staged path's canonical capture."""
    assert config_module.get_rag_citation_canonical_writes_enabled() is True
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    library = _NoteLibrary(note_id)
    app.library_rag_search_service = library
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                store = console._ensure_console_chat_controller().store
                store.stage_session_library_policy(
                    store.active_session_id,
                    ConsoleLibraryPolicyCandidate(
                        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
                        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                    ),
                )
                await _send(host, pilot, console, QUESTION)
                label, rows = await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
        assert library.calls == 1
        assert label == "Cited sources (1)"
        _assert_note_row(rows, note_id)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()
    await _reopen_after_restart(db_path, conversation_id, note_id)


def _assistant_counted(console) -> bool:
    """Count discovery has resolved the completed reply (to any value)."""
    message = _completed_assistant(console)
    return message is not None and message.id in console._console_citation_counts


@pytest.mark.asyncio
async def test_without_a_secure_keyring_evidence_goes_out_unlabelled_and_unrecorded(
    tmp_path, direct_sends
):
    """AC#4: the suite's null keyring is the fail-closed case.

    No key can be created, so no citation record will exist: the model gets
    the evidence by title with no [S#] labels, nothing is written, and no
    Sources row claims otherwise.
    """
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION)
                await until(lambda: _assistant_counted(console), timeout=20)
                await pilot.pause(0.3)
                buttons = _cited_sources_buttons(console)
                repository = app.citation_trace_repository
        prompt = gateway.requests[0][-1]["content"]
        assert "marigold-73" in prompt
        assert "not saved as citations for this reply" in prompt
        assert f"[S1] NOTE — {NOTE_TITLE}" not in prompt
        assert "[S#] labels only" not in prompt
        assert buttons == []
        assert _trace_count(db) == 0
        assert getattr(repository, "fingerprint_key_state", None) == (
            "keyring_unavailable"
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_a_record_whose_key_is_gone_says_so_and_names_the_next_step(
    tmp_path, secure_keyring, direct_sends
):
    """AC#4/#5: a saved record the key no longer verifies is not hidden.

    The reply keeps its [S1] marker, so instead of an unexplained marker it
    shows 'Cited sources unavailable', and the window names the cause (the
    key can't be read) and the next step (unlock or restore the keyring).
    """
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, _gateway = _build_app(db_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION)
                await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()

    secure_keyring.secrets.clear()  # the keychain was reset between runs
    host, app, db, _gateway = _build_app(db_path)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                assert await console._workspace._resume_console_workspace_conversation(
                    conversation_id
                )
                await until(lambda: bool(_cited_sources_buttons(console)), timeout=20)
                button = _cited_sources_buttons(console)[0]
                label = str(button.label)
                button.press()
                await until(
                    lambda: isinstance(host.screen, ConsoleCitationSourcesModal)
                    and host.screen.unavailable_cause != "load_failed",
                    timeout=20,
                )
                modal = host.screen
                await pilot.pause(0.2)
                state = str(
                    modal.query_one("#console-citation-sources-state").renderable
                )
        assert label == "Cited sources unavailable"
        assert modal.unavailable_cause == "key_unreadable"
        assert state == CITATION_SOURCES_UNAVAILABLE_COPY["key_unreadable"]
        assert "Unlock or restore your system keyring" in state
        assert secure_keyring.secrets == {}  # never silently replaced
    finally:
        db.close_connection()
