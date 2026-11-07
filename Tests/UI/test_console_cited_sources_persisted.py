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
the shipped default config. That default includes ``[console] agent_runtime``
(absent, so on): with a file-backed DB every send runs through the real
``ConsoleAgentBridge``, whose model adapter streams from the provider
gateway. The two send-path tests also run once with ``agent_runtime = false``
(the provider-direct Console stream), the one non-default dispatcher a user
can choose. Only the provider gateway, the Library search service (Automatic
test only) and the keyring backend are doubles. The keyring double reports a
secure backend, as the macOS Keychain would; the null keyring the test suite
installs is exactly the fail-closed case and is covered separately.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
import json
from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _select_llamacpp_console,
    _wait_for_selector,
)
from Tests.UI.test_console_send_acknowledgement import eager_tasks, press, until
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
        self.calls: list[str] = []

    def get_password(self, service: str, account: str) -> str | None:
        self.calls.append("get_password")
        return self.secrets.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.calls.append("set_password")
        self.secrets[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.calls.append("delete_password")
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


@dataclass
class _SendPath:
    """Which Console dispatcher a test's sends take, and what it observed."""

    name: str
    agent_turns: list[str] = field(default_factory=list)

    @property
    def direct(self) -> bool:
        return self.name == DIRECT

    def assert_taken(self, sends: int) -> None:
        """Every send went through the agent loop, or none of them did."""
        expected = 0 if self.direct else sends
        assert len(self.agent_turns) == expected, (self.name, self.agent_turns)


AGENT_RUNTIME = "agent_runtime"
DIRECT = "direct"


def _spy_agent_turns(monkeypatch) -> list[str]:
    """Record each turn the real agent bridge runs (it still runs it)."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    turns: list[str] = []
    run_reply = ConsoleAgentBridge.run_reply

    def recording_run_reply(self, *args, **kwargs):
        turns.append(kwargs.get("assistant_message_id", ""))
        return run_reply(self, *args, **kwargs)

    monkeypatch.setattr(ConsoleAgentBridge, "run_reply", recording_run_reply)
    return turns


@pytest.fixture
def shipped_send_path(monkeypatch) -> _SendPath:
    """The shipped default dispatcher: ``[console] agent_runtime`` on."""
    return _SendPath(AGENT_RUNTIME, _spy_agent_turns(monkeypatch))


@pytest.fixture(params=[AGENT_RUNTIME, DIRECT])
def send_path(request, monkeypatch) -> _SendPath:
    """The shipped agent-runtime dispatcher, then the provider-direct one.

    ``direct`` is ``[console] agent_runtime = false``; the bootstrap profile
    restores the config file after the test.
    """
    path = _SendPath(request.param, _spy_agent_turns(monkeypatch))
    if path.direct:
        assert config_module.save_setting_to_cli_config(
            "console", "agent_runtime", False
        )
    return path


def _build_app(db_path: Path, send_path: _SendPath):
    """Build a factory app over a file-backed ChaChaNotes DB.

    The citation services are composed by the app's own wiring, so the
    repository, its policy (the shipped default config) and its key provider
    are the production ones.
    """

    app = _build_test_app()
    if send_path.direct:
        app.app_config.setdefault("console", {})["agent_runtime"] = False
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    db = CharactersRAGDB(db_path, client_id="cited-sources-ui")
    app.chachanotes_db = db
    app._wire_chat_conversation_services()
    # The factory initially composes without a chat DB. Rebind its real
    # dictionary participant alongside the real conversation service.
    from tldw_chatbook.Backup_Recovery.chat_source_participants import (
        build_dictionary_service,
    )

    app.local_chat_dictionary_service = build_dictionary_service(db)
    app.chat_dictionary_scope_service.local_service = app.local_chat_dictionary_service
    gateway = _CitingGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
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


async def _send(host, pilot, console, text: str, record_property) -> None:
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    composer.load_draft(text)
    await pilot.pause()
    press(host, "enter", "\r")
    started = asyncio.get_running_loop().time()
    # Real Windows admission exceeded the former 20s first-send bound while
    # progressing through guarded native work (PR3323 r-ceca463c). Keep the
    # bound finite and retain per-send timing in the receipt, not only stdout.
    observations = []
    for interval in range(3 if os.name == "nt" else 1):
        try:
            await until(lambda: _completed_assistant(console) is not None, timeout=20)
            break
        except AssertionError:
            observations.append(_send_phase(console, started))
    else:
        record_property("first_send_phases", json.dumps(observations))
        raise AssertionError(
            "Citation send did not complete within its native readiness bound"
        )
    record_property("first_send_phases", json.dumps(observations))
    record_property(
        "first_send_seconds", round(asyncio.get_running_loop().time() - started, 3)
    )


def _send_phase(console, started):
    """Content-free phase evidence; no locals, arguments, source IDs or body."""
    chains = []
    for frame in sys._current_frames().values():
        chain = []
        while frame is not None:
            chain.append(
                f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{frame.f_lineno}"
            )
            frame = frame.f_back
        if any(
            "storage_admission.py" in item or "console_agent_bridge.py" in item
            for item in chain
        ):
            chains.append(chain[:8])
    return {
        "elapsed": round(asyncio.get_running_loop().time() - started, 3),
        "state": console._ensure_console_chat_controller().run_state.status.value,
        "native_frames": chains,
    }


async def _open_sources(host, console):
    await until(lambda: bool(_cited_sources_buttons(console)), timeout=20)
    button = _cited_sources_buttons(console)[0]
    label = str(button.label)
    button.press()
    try:
        await until(
            lambda: isinstance(host.screen, ConsoleCitationSourcesModal)
            and bool(host.screen.display_rows),
            timeout=20,
        )
    except AssertionError:
        modal = host.screen
        raise AssertionError(
            f"Sources did not open: screen={type(modal).__name__}; "
            f"cause={getattr(modal, 'unavailable_cause', None)}; "
            f"mounted={modal.is_mounted}; "
            f"current={modal._request_is_current(modal._request_generation) if isinstance(modal, ConsoleCitationSourcesModal) else None}"
        ) from None
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
    return (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM rag_citation_traces")
        .fetchone()[0]
    )


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


async def _reopen_after_restart(
    db_path: Path, conversation_id: str, note_id: str, send_path: _SendPath
):
    host, app, db, _gateway = _build_app(db_path, send_path)
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
    tmp_path, secure_keyring, send_path, record_property
):
    """AC#1/#3/#6/#7: staged Library evidence -> persisted Cited sources row."""
    assert config_module.get_rag_citation_canonical_writes_enabled() is True
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                # Boot composed the citation services without the keyring:
                # the key is created on first use, not at startup.
                assert secure_keyring.calls == []
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION, record_property)
                assert "set_password" in secure_keyring.calls
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
        send_path.assert_taken(sends=2)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()
    _assert_export_carries_the_record(db_path, conversation_id)
    await _reopen_after_restart(db_path, conversation_id, note_id, send_path)


@pytest.mark.asyncio
async def test_automatic_note_reply_shows_cited_sources_and_survives_restart(
    tmp_path, secure_keyring, send_path, record_property
):
    """AC#2/#3/#6: Automatic retrieval gets the staged path's canonical capture."""
    assert config_module.get_rag_citation_canonical_writes_enabled() is True
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, send_path)
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
                await _send(host, pilot, console, QUESTION, record_property)
                label, rows = await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
        assert library.calls == 1
        assert label == "Cited sources (1)"
        _assert_note_row(rows, note_id)
        assert "[S1]" in gateway.requests[0][-1]["content"]
        send_path.assert_taken(sends=1)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()
    await _reopen_after_restart(db_path, conversation_id, note_id, send_path)


def _assistant_counted(console) -> bool:
    """Count discovery has resolved the completed reply (to any value)."""
    message = _completed_assistant(console)
    return message is not None and message.id in console._console_citation_counts


@pytest.mark.asyncio
async def test_without_a_secure_keyring_evidence_goes_out_unlabelled_and_unrecorded(
    tmp_path, shipped_send_path, record_property
):
    """AC#4: the suite's null keyring is the fail-closed case.

    No key can be created, so no citation record will exist: the model gets
    the evidence by title with no [S#] labels, nothing is written, and no
    Sources row claims otherwise.
    """
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION, record_property)
                await until(lambda: _assistant_counted(console), timeout=20)
                await pilot.pause(0.3)
                buttons = _cited_sources_buttons(console)
                repository = app.citation_trace_repository
        prompt = gateway.requests[0][-1]["content"]
        assert "marigold-73" in prompt
        assert "not saved as citations for this reply" in prompt
        assert f"[S1] NOTES — {NOTE_TITLE}" not in prompt
        assert "[S#] labels only" not in prompt
        assert buttons == []
        shipped_send_path.assert_taken(sends=1)
        assert _trace_count(db) == 0
        assert getattr(repository, "fingerprint_key_state", None) == (
            "keyring_unavailable"
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_a_record_whose_key_is_gone_says_so_and_names_the_next_step(
    tmp_path, secure_keyring, shipped_send_path, record_property
):
    """AC#4/#5: a saved record the key no longer verifies is not hidden.

    The reply keeps its [S1] marker, so instead of an unexplained marker it
    shows 'Cited sources unavailable', and the window names the cause (the
    key can't be read) and the next step (unlock or restore the keyring).
    """
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, _gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send(host, pilot, console, QUESTION, record_property)
                await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
        shipped_send_path.assert_taken(sends=1)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()

    secure_keyring.secrets.clear()  # the keychain was reset between runs
    host, app, db, _gateway = _build_app(db_path, shipped_send_path)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["staged", "automatic"])
@pytest.mark.parametrize(
    "failure", [None, "record_retrieval_run", "record_prompt_evidence_set"]
)
async def test_authority_shrink_preserves_rank_or_dispatches_only_unlabelled_survivor(
    tmp_path,
    secure_keyring,
    shipped_send_path,
    monkeypatch,
    adapter,
    failure,
    record_property,
):
    from tldw_chatbook.Chat.citation_evidence_models import (
        EvidenceBundle,
        EvidenceReference,
    )
    from tldw_chatbook.Chat.citation_trace_builder import CitationTraceBuilder
    from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch

    db_path = tmp_path / "shrunk.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    missing_title = "Removed source must never reach dispatch"
    references = (
        EvidenceReference(
            evidence_id="S1",
            source_id="missing-note",
            source_type="note",
            title=missing_title,
            snippet="REMOVED_PAYLOAD",
            authority_label="local",
        ),
        EvidenceReference(
            evidence_id="S2",
            source_id=note_id,
            source_type="note",
            title=NOTE_TITLE,
            snippet=NOTE_BODY,
            authority_label="local",
        ),
    )

    class TwoNoteLibrary:
        async def search(self, *args, **kwargs):
            return {
                "runtime_backend": "local",
                "results": [
                    {
                        "source_id": ref.source_id,
                        "source_type": "notes",
                        "title": ref.title,
                        "snippet": ref.snippet,
                        "score": 0.9,
                        "runtime_backend": "local",
                    }
                    for ref in references
                ],
            }

    if failure:

        def reject(*args, **kwargs):
            raise RuntimeError("controlled recording failure")

        monkeypatch.setattr(CitationTraceBuilder, failure, reject)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                if adapter == "staged":
                    console._retrieval._stage_console_library_rag_launch(
                        ConsoleLiveWorkLaunch.from_values(
                            source="Library Search/RAG",
                            title="Two sources",
                            payload={
                                "evidence_bundle": EvidenceBundle(
                                    bundle_id="shrinking",
                                    query=QUESTION,
                                    references=references,
                                ).to_payload()
                            },
                        )
                    )
                else:
                    app.library_rag_search_service = TwoNoteLibrary()
                    store = console._ensure_console_chat_store()
                    store.stage_session_library_policy(
                        store.active_session_id,
                        ConsoleLibraryPolicyCandidate(
                            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
                            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                        ),
                    )
                await _send(host, pilot, console, QUESTION, record_property)
                if failure is None:
                    label, rows = await _open_sources(host, console)
                    assert label == "Cited sources (1)"
                    _assert_note_row(rows, note_id)
                    conversation_id = _active_conversation_id(console)
                else:
                    await until(lambda: _assistant_counted(console), timeout=20)
                    assert _cited_sources_buttons(console) == []
        prompt = gateway.requests[0][-1]["content"]
        assert NOTE_BODY in prompt
        assert missing_title not in prompt and "REMOVED_PAYLOAD" not in prompt
        shipped_send_path.assert_taken(sends=1)
        if failure is None:
            assert f"[S1] NOTES — {NOTE_TITLE}" in prompt
            payloads = [
                json.loads(row[0])
                for row in db.get_connection().execute(
                    "SELECT run_payload_json FROM rag_evidence_runs"
                )
            ]
            candidates = [
                item for payload in payloads for item in payload["candidates"]
            ]
            assert [
                (item["rank"], item["source_identity"]["source_id"])
                for item in candidates
            ] == [(2, note_id)]
        else:
            assert "not saved as citations for this reply" in prompt
            assert f"[S1] NOTES — {NOTE_TITLE}" not in prompt
            assert "[S#] labels only" not in prompt
            assert _trace_count(db) == 0
    finally:
        db.close_connection()
    if failure is None:
        await _reopen_after_restart(
            db_path, conversation_id, note_id, shipped_send_path
        )
