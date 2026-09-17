"""App-owned authoring drains, local exchange and real navigation."""

import asyncio
import json
import os
import sqlite3
import stat
from pathlib import Path
from threading import Event, get_ident

import pytest

from Tests.Workflows.helpers import prompt_definition
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows.authoring import WorkflowAuthoring
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.models import DraftWriteFailed


@pytest.mark.parametrize(
    "name",
    [
        "x" * 257,
        " " * 257,
        "",
        " \t\n",
        None,
        42,
        True,
        b"name",
        [],
        pytest.param(chr(0xD800), id="high-surrogate-start"),
        pytest.param(chr(0xDBFF), id="high-surrogate-end"),
        pytest.param(chr(0xDC00), id="low-surrogate-start"),
        pytest.param(chr(0xDFFF), id="low-surrogate-end"),
        pytest.param(" name" + chr(0xD800) + " ", id="embedded-surrogate"),
        pytest.param(chr(0xD83D) + chr(0xDE00), id="uncombined-surrogate-pair"),
    ],
)
async def test_creation_rejects_invalid_name_before_opening_storage(tmp_path, name):
    path = tmp_path / "unopened.sqlite3"
    opened = []

    def path_factory():
        opened.append(True)
        return path

    owner = WorkflowAuthoring(path_factory)
    try:
        with pytest.raises(ValueError) as failure:
            await owner.create(name)
        assert len(str(failure.value)) < 512
        assert "x" * 257 not in str(failure.value)
        assert opened == []
        assert not path.exists()
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("  A name\t", "A name"),
        ("é" * 256, "é" * 256),
        ("e\u0301", "e\u0301"),
        ("😀" * 256, "😀" * 256),
    ],
)
async def test_creation_persists_trimmed_bounded_name_without_normalization(
    tmp_path, name, expected
):
    path = tmp_path / "named.sqlite3"
    owner = WorkflowAuthoring(lambda: path)
    try:
        revision = await owner.create(name)
        assert json.loads(revision.raw_json)["name"] == expected
        assert owner.documents.get_head(revision.workflow_id) == revision
    finally:
        await owner.close()


async def test_invalid_creation_retains_current_draft_and_saved_heads(tmp_path):
    owner = WorkflowAuthoring(lambda: tmp_path / "retained.sqlite3")
    try:
        revision = await owner.create("Original")
        pending = owner.drafts.update('{"unfinished":')
        with pytest.raises(ValueError):
            await owner.create("x" * 257)
        assert owner.drafts.current == pending
        assert owner.documents.list_workflows() == (revision,)
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "name", ["é" * 1024, pytest.param(chr(0xD800), id="legacy-surrogate")]
)
async def test_creation_name_limit_does_not_rewrite_imported_names(tmp_path, name):
    source = tmp_path / "portable.json"
    content = prompt_definition()
    content["name"] = name
    source.write_text(json.dumps(content), encoding="utf-8")
    owner = WorkflowAuthoring(lambda: tmp_path / "imported.sqlite3")
    try:
        revision = await owner.import_file(source)
        assert json.loads(revision.raw_json)["name"] == content["name"]
    finally:
        await owner.close()


@pytest.mark.parametrize("name", ["x" * 257, pytest.param(chr(0xDFFF), id="surrogate")])
async def test_standalone_controller_rejects_invalid_creation_name(tmp_path, name):
    from tldw_chatbook.UI.Workflows_Modules.controller import WorkflowsController

    owner = WorkflowAuthoring(lambda: tmp_path / "controller.sqlite3")
    try:
        revision = await owner.create("Retained")
        pending = owner.drafts.update('{"unfinished":')
        controller = WorkflowsController(owner.documents, owner.drafts)
        with pytest.raises(ValueError):
            await controller.create(name)
        assert owner.drafts.current == pending
        assert owner.documents.list_workflows() == (revision,)
    finally:
        await owner.close()


async def test_unused_authoring_close_does_not_create_a_database(tmp_path):
    path = tmp_path / "unused.sqlite3"
    owner = WorkflowAuthoring(lambda: path)
    await owner.flush()
    await owner.close()
    assert not path.exists()


async def test_failed_open_without_a_draft_does_not_block_quit_flush(tmp_path):
    path = tmp_path / "not-a-database.sqlite3"
    path.write_text("This is not SQLite")
    owner = WorkflowAuthoring(lambda: path)
    with pytest.raises(sqlite3.DatabaseError):
        await owner.open()
    assert owner.drafts is None
    # The app calls flush before its close guard, even after setup refused.
    await owner.flush()
    await owner.close()
    assert path.read_text() == "This is not SQLite"


async def test_cancelled_open_is_retained_and_close_waits_for_constructor(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Workflows.authoring as module

    path = tmp_path / "opening.sqlite3"
    started, release = Event(), Event()
    constructor = module.WorkflowsDB
    connections = []

    def blocked(selected):
        started.set()
        assert release.wait(5)
        db = constructor(selected)
        connections.append(db)
        return db

    monkeypatch.setattr(module, "WorkflowsDB", blocked)
    owner = WorkflowAuthoring(lambda: path)
    opening = asyncio.create_task(owner.open())
    closing = None
    try:
        assert await asyncio.to_thread(started.wait, 3)
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        if closing:
            await closing
    assert len(connections) == 1
    with pytest.raises(RuntimeError, match="closed"), connections[0].transaction():
        pass


async def test_failed_close_retains_exact_buffer_and_retry_persists(
    tmp_path, monkeypatch
):
    path = tmp_path / "retry.sqlite3"
    owner = WorkflowAuthoring(lambda: path)
    await owner.open()
    revision = owner.documents.create(json.dumps(prompt_definition()))
    await owner.drafts.select(revision.workflow_id, revision.revision_id)
    pending = owner.drafts.update('{"unfinished":')
    write = owner.documents.put_draft

    def refuse(*args, **kwargs):
        raise OSError("private storage detail")

    monkeypatch.setattr(owner.documents, "put_draft", refuse)
    with pytest.raises(DraftWriteFailed):
        await owner.close()
    assert owner.drafts.current == pending
    assert "Not saved" in owner.drafts.status
    # Still interactive after refused persistence.
    owner.drafts.update('{"unfinished": ')
    monkeypatch.setattr(owner.documents, "put_draft", write)
    await owner.close()
    db = WorkflowsDB(path)
    try:
        assert (
            DocumentService(db)
            .get_draft(revision.workflow_id, revision.revision_id)
            .raw_text
            == '{"unfinished": '
        )
    finally:
        db.close()


async def test_exchange_keeps_opaque_metadata_step_id_and_saved_revision(
    tmp_path, monkeypatch
):
    source = tmp_path / "incoming.json"
    definition = prompt_definition()
    definition["metadata"]["opaque"] = {"vendor": ["unchanged", 9007199254740993]}
    source.write_text(json.dumps(definition), encoding="utf-8")
    owner = WorkflowAuthoring(lambda: tmp_path / "authoring.sqlite3")
    loop_thread = get_ident()
    validate = owner._exchange_path

    def validate_off_loop(path):
        assert get_ident() != loop_thread, (
            "Exchange metadata must not block the app loop"
        )
        return validate(path)

    monkeypatch.setattr(owner, "_exchange_path", validate_off_loop)
    try:
        revision = await owner.import_file(source)
        owner.drafts.update(
            owner.documents.edit_field(revision.raw_json, "/name", "Edited name")
        )
        saved = await owner.drafts.save_revision()
        owner.drafts.update('{"unfinished":')
        target = tmp_path / "export.json"
        await owner.export_file(target, saved)
        exported = json.loads(target.read_text())
        assert exported["name"] == "Edited name"
        assert exported["steps"][0]["id"] == "prepare"
        assert exported["metadata"]["opaque"] == {
            "vendor": ["unchanged", 9007199254740993]
        }
        assert exported["metadata"]["tldw_workflow"]["revision_id"] == saved.revision_id
        assert "raw_text" not in exported
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        await owner.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file identity and aliases")
@pytest.mark.parametrize("operation", ["import", "export"])
@pytest.mark.parametrize(
    "suffix,alias_kind",
    [
        ("", "hardlink"),
        ("-wal", "hardlink"),
        ("-shm", "hardlink"),
        ("-journal", "hardlink"),
        ("", "directory"),
    ],
)
async def test_exchange_refuses_database_alias_before_generic_file_io(
    tmp_path, monkeypatch, operation, suffix, alias_kind
):
    import tldw_chatbook.Workflows.authoring as module

    directory = tmp_path / "store"
    directory.mkdir()
    path = directory / "authoring.json"
    owner = WorkflowAuthoring(lambda: path)
    alias = tmp_path / "selected.json"
    try:
        await owner.open()
        # PERSIST leaves a real rollback journal available after the write.
        mode = "PERSIST" if suffix == "-journal" else "WAL"
        assert (
            owner._db._connection.execute(f"PRAGMA journal_mode={mode}").fetchone()[0]
            == mode.lower()
        )
        revision = await owner.create("Protected authoring store")
        await owner.flush()
        before = owner.drafts.current
        protected = Path(str(path) + suffix)
        assert protected.is_file()
        if alias_kind == "hardlink":
            alias.hardlink_to(protected)
            selected = alias
        else:
            alias.symlink_to(directory, target_is_directory=True)
            selected = alias / path.name
            assert selected.stat().st_nlink == 1

        # Audit the actual unsafe boundary, forwarding unchanged if reached.
        # Rejection by the generic helper itself is already too late for import.
        boundary = (
            "open_private_binary"
            if operation == "import"
            else "atomic_private_write_text"
        )
        original = getattr(module, boundary)
        calls = []

        def observe(*args, **kwargs):
            calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(module, boundary, observe)
        with pytest.raises((ValueError, OSError)):
            if operation == "import":
                await owner.import_file(selected)
            else:
                await owner.export_file(selected, revision)
        assert calls == [], "Database aliases must be refused before generic file I/O"
        assert owner.drafts.current == before
        assert (
            owner.documents.get_revision(revision.workflow_id, revision.revision_id)
            == revision
        )
    finally:
        alias.unlink(missing_ok=True)
        await owner.close()


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_close_waiters_share_failure_and_next_close_can_retry(
    tmp_path, monkeypatch, cancel_first
):
    owner = WorkflowAuthoring(lambda: tmp_path / "waiters.sqlite3")
    revision = await owner.create("Retained")
    pending = owner.drafts.update('{"unfinished":')
    write = owner.documents.put_draft
    started, release = Event(), Event()

    def refuse(*args, **kwargs):
        started.set()
        assert release.wait(5)
        raise OSError("private storage detail")

    monkeypatch.setattr(owner.documents, "put_draft", refuse)
    first = asyncio.create_task(owner.close())
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 3)
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            second = asyncio.create_task(owner.close())
            await asyncio.sleep(0)
        drain = owner._closing
    finally:
        release.set()
    results = await asyncio.gather(
        drain, *([first, second] if second else []), return_exceptions=True
    )
    assert all(isinstance(result, DraftWriteFailed) for result in results)
    assert owner.drafts.current == pending
    monkeypatch.setattr(owner.documents, "put_draft", write)
    await owner.close()
    db = WorkflowsDB(tmp_path / "waiters.sqlite3")
    try:
        assert (
            DocumentService(db).get_draft(revision.workflow_id, revision.revision_id)
            == pending
        )
    finally:
        db.close()


async def test_oversized_import_cannot_change_selected_draft(tmp_path):
    owner = WorkflowAuthoring(lambda: tmp_path / "size.sqlite3")
    source = tmp_path / "too-large.json"
    source.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    try:
        await owner.open()
        with pytest.raises(ValueError, match="large|limit|size"):
            await owner.import_file(source)
        assert owner.drafts.current is None
        assert owner.documents.list_workflows() == ()
    finally:
        await owner.close()


async def test_cancelled_export_finishes_before_store_closes(tmp_path, monkeypatch):
    import tldw_chatbook.Workflows.authoring as module

    owner = WorkflowAuthoring(lambda: tmp_path / "drain.sqlite3")
    await owner.open()
    revision = owner.documents.create(json.dumps(prompt_definition()))
    started, release = Event(), Event()
    write = module.atomic_private_write_text
    target = tmp_path / "export.json"

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return write(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_private_write_text", blocked)
    exporting = asyncio.create_task(owner.export_file(target, revision))
    closing = None
    try:
        assert await asyncio.to_thread(started.wait, 3)
        exporting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await exporting
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        if closing:
            await closing
    assert json.loads(target.read_text())["steps"][0]["id"] == "prepare"


@pytest.fixture
def real_authoring_app(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.config import save_setting_to_cli_config

    path = tmp_path / "real-app.sqlite3"
    save_setting_to_cli_config("splash_screen", "enabled", False)
    monkeypatch.setattr("tldw_chatbook.config.get_workflows_db_path", lambda: path)
    app = _build_test_app("home")
    monkeypatch.setattr(app, "_refresh_model_catalogs", AsyncMock())
    return app, path


async def settle_authoring(app, pilot):
    """The full app also owns long-lived, unrelated workers."""
    await pilot.pause()
    await asyncio.gather(
        *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
    )
    await pilot.pause()


async def test_real_app_creation_recovers_after_oversized_name(real_authoring_app):
    from textual.widgets import Input

    from Tests.UI.test_screen_navigation import _wait_for_initial_screen
    from Tests.UI.test_workflows_editor import painted_text
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Workflows_Modules.library import ChoiceModal

    app, _path = real_authoring_app
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_initial_screen(pilot)
        app.post_message(NavigateToScreen("workflows"))
        await settle_authoring(app, pilot)
        assert await pilot.click("#workflow-new")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceModal)
        app.screen.query_one(Input).value = "x" * 257
        await pilot.press("enter")
        await settle_authoring(app, pilot)
        assert "256" in painted_text(app.screen)
        assert app.workflow_documents.list_workflows() == ()
        assert await pilot.click("#workflow-new")
        await pilot.pause()
        app.screen.query_one(Input).value = "  Short enough  "
        await pilot.press("enter")
        await settle_authoring(app, pilot)
        assert json.loads(app.workflow_drafts.base.raw_json)["name"] == "Short enough"
        assert len(app.workflow_documents.list_workflows()) == 1


async def test_real_app_create_edit_navigate_quit_and_restart(real_authoring_app):
    from textual.widgets import Input

    from Tests.UI.test_screen_navigation import _wait_for_initial_screen
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
    from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor

    app, path = real_authoring_app
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_initial_screen(pilot)
        assert not path.exists()
        app.post_message(NavigateToScreen("workflows"))
        await settle_authoring(app, pilot)
        assert isinstance(app.screen, WorkflowsScreen)
        assert app.workflow_documents is not None
        assert await pilot.click("#workflow-new")
        await pilot.pause()
        await pilot.press(*"Local workflow", "enter")
        await settle_authoring(app, pilot)
        editor = app.screen.query_one(WorkflowEditor)
        identifier = next(
            key
            for key, (pointer, _) in editor.field_bindings.items()
            if pointer == "/name"
        )
        editor.query_one("#" + identifier, Input).value = "Changed in real app"
        await pilot.pause()
        revision = app.workflow_drafts.base
        pending = app.workflow_drafts.current
        app.post_message(NavigateToScreen("home"))
        await settle_authoring(app, pilot)
        assert not isinstance(app.screen, WorkflowsScreen)
        app.action_quit()
        await pilot.pause()
        assert app._shutting_down
        assert app._workflow_authoring._closed
    reopened = WorkflowsDB(path)
    try:
        documents = DocumentService(reopened)
        assert (
            documents.get_draft(revision.workflow_id, revision.revision_id) == pending
        )
        assert documents.field_text(pending.raw_text, "/name", as_json=False) == (
            "Changed in real app"
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("action", ["navigate", "quit"])
async def test_real_app_write_refusal_keeps_screen_buffer_and_allows_retry(
    real_authoring_app, monkeypatch, action
):
    from unittest.mock import Mock

    from Tests.UI.test_screen_navigation import _wait_for_initial_screen
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app, path = real_authoring_app
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_initial_screen(pilot)
        await app.handle_screen_navigation(NavigateToScreen("workflows"))
        await settle_authoring(app, pilot)
        await app._workflow_authoring.create("Retained draft")
        screen = app.screen
        pending = app.workflow_drafts.update('{"private unfinished":')
        write = app.workflow_documents.put_draft
        notify = Mock(wraps=app.notify)
        monkeypatch.setattr(app, "notify", notify)

        def refuse(*args, **kwargs):
            raise OSError("private storage detail")

        monkeypatch.setattr(app.workflow_documents, "put_draft", refuse)
        if action == "navigate":
            await app.handle_screen_navigation(NavigateToScreen("home"))
        else:
            app.action_quit()
            for _ in range(100):
                await pilot.pause(0.02)
                if not app._quit_in_progress:
                    break
            assert not app._quit_in_progress
        assert app.screen is screen
        assert not app._shutting_down
        assert app.workflow_drafts.current == pending
        assert "Not saved" in app.workflow_drafts.status
        assert "private storage detail" not in str(notify.call_args_list)
        assert all(
            call.kwargs.get("severity") != "information"
            for call in notify.call_args_list
        )
        monkeypatch.setattr(app.workflow_documents, "put_draft", write)
        await app.handle_screen_navigation(NavigateToScreen("home"))
        assert app.screen is not screen
        app.action_quit()
        await pilot.pause()
        assert app._shutting_down
    db = WorkflowsDB(path)
    try:
        assert (
            DocumentService(db).get_draft(pending.workflow_id, pending.base_revision_id)
            == pending
        )
    finally:
        db.close()


@pytest.mark.parametrize("approve_replace", [False, True])
async def test_real_app_import_edit_save_export_with_actual_file_pickers(
    real_authoring_app, tmp_path, approve_replace
):
    from textual.widgets import Input, OptionList

    from Tests.UI.test_screen_navigation import _wait_for_initial_screen
    from Tests.UI.test_workflows_editor import field_for
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor
    from tldw_chatbook.UI.Workflows_Modules.library import ChoiceModal
    from tldw_chatbook.Widgets.enhanced_file_picker import (
        EnhancedFileOpen,
        EnhancedFileSave,
    )

    definition = prompt_definition()
    definition["opaque"] = {"preserve": ["verbatim", 9007199254740993]}
    source = tmp_path / "incoming.json"
    source.write_text(json.dumps(definition))
    target = tmp_path / "outgoing.json"
    target.write_text("existing content")
    app, _ = real_authoring_app
    async with app.run_test(size=(160, 48)) as pilot:
        await _wait_for_initial_screen(pilot)
        await app.handle_screen_navigation(NavigateToScreen("workflows"))
        await settle_authoring(app, pilot)
        assert await pilot.click("#workflow-import")
        await pilot.pause()
        assert isinstance(app.screen, EnhancedFileOpen)
        app.screen.query_one("#filename-input", Input).value = str(source)
        assert await pilot.click("#select")
        await settle_authoring(app, pilot)
        field_for(
            app.screen.query_one(WorkflowEditor), "/name"
        ).value = "Edited locally"
        await pilot.pause()
        assert await pilot.click("#workflow-save-revision")
        await settle_authoring(app, pilot)
        saved = app.workflow_drafts.base
        app.workflow_drafts.update('{"not exported":')
        assert await pilot.click("#workflow-export")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceModal)
        assert "secrets" in app.screen.detail
        app.screen.query_one(OptionList).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, EnhancedFileSave)
        app.screen.query_one("#filename-input", Input).value = str(target)
        assert await pilot.click("#select")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceModal)
        assert "Replace" in app.screen.title_text
        assert target.read_text() == "existing content"
        if not approve_replace:
            await pilot.press("escape")
            await settle_authoring(app, pilot)
            assert target.read_text() == "existing content"
            return
        app.screen.query_one(OptionList).focus()
        await pilot.press("enter")
        await settle_authoring(app, pilot)
        exported = json.loads(target.read_text())
        assert exported["name"] == "Edited locally"
        assert exported["opaque"] == definition["opaque"]
        assert exported["steps"][0]["id"] == "prepare"
        assert exported["metadata"]["tldw_workflow"]["revision_id"] == saved.revision_id
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
