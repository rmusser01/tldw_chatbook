"""The one-time citation-records question for upgraded installs (TASK-33620.8.3).

Owner decision 2026-10-30. Installs created before citation records shipped on
carry the old ``[rag_citations] canonical_writes_enabled = false`` that first
run wrote into their config.toml, so they kept records off after upgrading.
On the first reply sent with Library evidence they are asked once, "Keep a
record of which sources replies cite?"; the answer is saved beside
config.toml, which is never rewritten for it.

These tests drive the real ChatScreen, controller, store, runtime and citation
repository over a file-backed ChaChaNotes database, with the shared bootstrap
profile's config.toml carrying the old explicit ``false`` (restored after each
test by conftest). Only the provider gateway, the Library search service
(Automatic test) and the keyring backend are doubles, as in
``test_console_cited_sources_persisted.py``, whose harness they reuse.
"""

# ruff: noqa: F811 - the persisted-sources harness's fixtures are imported.
from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.test_console_cited_sources_persisted import (  # noqa: F401
    NOTE_BODY,
    NOTE_TITLE,
    QUESTION,
    _active_conversation_id,
    _assert_note_row,
    _assistant_counted,
    _build_app,
    _cited_sources_buttons,
    _completed_assistants,
    _note_payload,
    _open_sources,
    _ready_console,
    _send,
    _trace_count,
    secure_keyring,
    shipped_send_path,
)
from Tests.UI.test_console_send_acknowledgement import eager_tasks, press, until
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.bootstrap_profile

QUESTION_TITLE = "Keep a record of which sources replies cite?"
ANSWER_FILENAME = "citation_records_choice.json"
FOLLOW_UP = "And when is the review?"
LABELLED_EVIDENCE = f"[S1] NOTES — {NOTE_TITLE}"


def _answer_path() -> Path:
    return Path(config_module._get_effective_config_path()).parent / ANSWER_FILENAME


def _saved_answer() -> str | None:
    import json

    path = _answer_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("answer")


@pytest.fixture
def old_shipped_false():
    """config.toml as first run wrote it before records shipped on.

    The bootstrap profile is shared by every ``bootstrap_profile`` test in the
    process; conftest restores its config.toml afterwards, and this fixture
    removes the answer file this feature keeps beside it.
    """
    _answer_path().unlink(missing_ok=True)
    assert config_module.save_setting_to_cli_config(
        "rag_citations", "canonical_writes_enabled", False
    )
    config_path = Path(config_module._get_effective_config_path())
    assert _citation_section(config_path) == {"canonical_writes_enabled": False}
    yield config_path
    _answer_path().unlink(missing_ok=True)


def _citation_section(config_path: Path) -> dict:
    import tomllib

    return tomllib.loads(config_path.read_text(encoding="utf-8"))["rag_citations"]


def _question_open(host) -> bool:
    screen = host.screen
    return isinstance(screen, ConfirmationDialog) and screen.title == QUESTION_TITLE


def _question_ready(host) -> bool:
    """Open and mounted: a modal answers only once its mount has run."""
    return _question_open(host) and host.screen.is_mounted


async def _answer(host, button_id: str, expected: str, config_path: Path) -> None:
    """Answer, then wait until it is saved and applied: config.toml is untouched."""
    before = config_path.read_bytes()
    host.screen.query_one(button_id).press()
    await until(lambda: not _question_open(host), timeout=10)
    await until(lambda: _saved_answer() == expected, timeout=10)
    assert config_path.read_bytes() == before


async def _staged_send(host, pilot, console, note_id, text, record_property):
    console._stage_handoff_as_console_live_work(_note_payload(note_id))
    await pilot.pause()
    await _send(host, pilot, console, text, record_property)


async def _send_more(host, pilot, console, text: str, replies: int) -> None:
    """A later send in the same chat (``_send`` waits only for the first reply)."""
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    composer.load_draft(text)
    await pilot.pause()
    press(host, "enter", "\r")
    await until(lambda: _completed_assistants(console) == replies, timeout=20)


async def _settled_without_question(host, pilot, console) -> None:
    """Count discovery has run for the newest reply, and nothing asked."""
    await until(lambda: _assistant_counted(console), timeout=20)
    await pilot.pause(0.5)
    assert not _question_open(host)


@pytest.mark.asyncio
async def test_old_shipped_false_asks_once_and_keep_turns_on_cited_sources(
    tmp_path, secure_keyring, shipped_send_path, old_shipped_false, record_property
):
    """AC#3/#4/#2: asked once after the first Library reply; Keep sticks."""
    config_path = old_shipped_false
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                assert app.citation_trace_repository.canonical_writes_enabled is False
                await _staged_send(
                    host, pilot, console, note_id, QUESTION, record_property
                )
                # The reply went out with Library evidence and no record.
                await until(lambda: _question_ready(host), timeout=20)
                await pilot.pause()
                # No button is focused, so typing meant for the composer
                # cannot answer it.
                assert host.screen.focused is not None
                assert host.screen.focused.id == "confirmation-dialog"
                await _answer(host, "#confirm-button", "keep", config_path)
                assert app.citation_trace_repository.canonical_writes_enabled is True
                # Keep applies at once: the next Library reply is recorded.
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send_more(host, pilot, console, FOLLOW_UP, replies=2)
                label, rows = await _open_sources(host, console)
                conversation_id = _active_conversation_id(console)
                await host.screen.request_safe_cancel(source="visible")
                await pilot.pause(0.3)
                assert not _question_open(host)
        first, second = (
            gateway.requests[0][-1]["content"],
            gateway.requests[1][-1]["content"],
        )
        assert "not saved as citations for this reply" in first
        assert LABELLED_EVIDENCE not in first
        assert LABELLED_EVIDENCE in second
        assert label == "Cited sources (1)"
        _assert_note_row(rows, note_id)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()
    # The answer lives beside config.toml, whose old false stays as it was.
    assert _citation_section(config_path) == {"canonical_writes_enabled": False}
    host, app, db, _gateway = _build_app(db_path, shipped_send_path)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                assert app.citation_trace_repository.canonical_writes_enabled is True
                assert await console._workspace._resume_console_workspace_conversation(
                    conversation_id
                )
                label, rows = await _open_sources(host, console)
                assert not _question_open(host)
        assert label == "Cited sources (1)"
        _assert_note_row(rows, note_id)
    finally:
        db.close_connection()
    assert _citation_section(config_path) == {"canonical_writes_enabled": False}


@pytest.mark.asyncio
async def test_dont_keep_is_never_asked_again_and_the_palette_changes_it(
    tmp_path, secure_keyring, shipped_send_path, old_shipped_false, record_property
):
    """AC#4/#6: Don't keep is saved, never asked again; the palette reopens it."""
    from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider

    config_path = old_shipped_false
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                await _staged_send(
                    host, pilot, console, note_id, QUESTION, record_property
                )
                await until(lambda: _question_ready(host), timeout=20)
                await _answer(host, "#cancel-button", "off", config_path)
                console._stage_handoff_as_console_live_work(_note_payload(note_id))
                await pilot.pause()
                await _send_more(host, pilot, console, FOLLOW_UP, replies=2)
                await _settled_without_question(host, pilot, console)
        assert app.citation_trace_repository.canonical_writes_enabled is False
        assert len(gateway.requests) == 2
        assert all(
            LABELLED_EVIDENCE not in request[-1]["content"]
            for request in gateway.requests
        )
        assert _trace_count(db) == 0
    finally:
        db.close_connection()

    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                await _staged_send(
                    host, pilot, console, note_id, QUESTION, record_property
                )
                await _settled_without_question(host, pilot, console)
                assert app.citation_trace_repository.canonical_writes_enabled is False
                provider = ConsoleCommandProvider(console)
                command = next(
                    callback
                    for label, callback, _help in provider._commands(console)
                    if label == "Console: Citation records…"
                )
                command()
                await until(lambda: _question_ready(host), timeout=10)
                await _answer(host, "#confirm-button", "keep", config_path)
                await until(
                    lambda: app.citation_trace_repository.canonical_writes_enabled,
                    timeout=10,
                )
    finally:
        db.close_connection()
    assert _citation_section(config_path) == {"canonical_writes_enabled": False}


@pytest.mark.asyncio
async def test_automatic_library_reply_asks_and_escape_leaves_it_unanswered(
    tmp_path, secure_keyring, shipped_send_path, old_shipped_false, record_property
):
    """AC#3: the Automatic path asks too; Escape closes the question unanswered."""
    from Tests.UI.test_console_cited_sources_persisted import _NoteLibrary

    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
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
                await until(lambda: _question_ready(host), timeout=20)
                await host.screen.request_safe_cancel(source="escape")
                await until(lambda: not _question_open(host), timeout=10)
                await pilot.pause(0.3)
        assert library.calls == 1
        assert (
            "not saved as citations for this reply"
            in gateway.requests[0][-1]["content"]
        )
        assert _saved_answer() is None
        assert app.citation_trace_repository.canonical_writes_enabled is False
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_records_of_its_own_then_turned_off_is_never_asked(
    tmp_path, secure_keyring, shipped_send_path, record_property
):
    """AC#5: on by default (never asked), then turned off in config.toml."""
    _answer_path().unlink(missing_ok=True)
    db_path = tmp_path / "chachanotes.sqlite"
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    note_id = db.add_note(title=NOTE_TITLE, content=NOTE_BODY)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                await _staged_send(
                    host, pilot, console, note_id, QUESTION, record_property
                )
                await until(lambda: bool(_cited_sources_buttons(console)), timeout=20)
                await pilot.pause(0.3)
                assert not _question_open(host)
        assert _trace_count(db) == 1
    finally:
        db.close_connection()

    assert config_module.save_setting_to_cli_config(
        "rag_citations", "canonical_writes_enabled", False
    )
    host, app, db, gateway = _build_app(db_path, shipped_send_path)
    try:
        async with host.run_test(size=(140, 44)) as pilot:
            with eager_tasks():
                console = await _ready_console(host, pilot)
                await _staged_send(
                    host, pilot, console, note_id, QUESTION, record_property
                )
                await _settled_without_question(host, pilot, console)
        assert app.citation_trace_repository.canonical_writes_enabled is False
        assert (
            "not saved as citations for this reply"
            in gateway.requests[0][-1]["content"]
        )
        assert _saved_answer() is None
    finally:
        db.close_connection()
        _answer_path().unlink(missing_ok=True)
