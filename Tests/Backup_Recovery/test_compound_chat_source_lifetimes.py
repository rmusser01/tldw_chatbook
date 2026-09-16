"""Actual core/sidecar boundaries for the bounded chat source cohort."""

import json
import select

import pytest

from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_participant_lifetimes import local_root
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_dictionary_admitted_create_finishes_history_after_core_commit_pause(
    tmp_path, local_root, monkeypatch, launch
):
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.Backup_Recovery.chat_source_participants import (
        build_dictionary_service,
    )

    (tmp_path / "data").mkdir(mode=0o700)
    target = tmp_path / "config.toml"
    target.write_text(
        f'[paths]\ndata_dir = "{tmp_path / "data"}"\n[database]\nchachanotes_db_path = "{tmp_path / "chat.sqlite"}"\n'
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    install_config_source(monkeypatch)
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "phase10-test")
    service = build_dictionary_service(db)
    path = service.history_store_path
    native = db.get_connection()
    original = module.cdl.save_chat_dictionary
    pauses = []
    observed_ids = []

    def committed_then_pause(*args, **kwargs):
        result = original(*args, **kwargs)
        assert (
            native.execute(
                "SELECT name FROM chat_dictionaries WHERE id = ?", (result,)
            ).fetchone()[0]
            == "coherent"
        )
        observed_ids.append(result)
        pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", committed_then_pause)
    error = None
    result = None
    child = None
    try:
        try:
            result = service.create_dictionary({"name": "coherent"})
        except Exception as caught:
            error = caught
        child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
        assert observed_ids
        assert not select.select([child.stdout], [], [], 0.05)[0]
        # Existing escaped borrower remains live and reflects the durable commit.
        assert (
            native.execute(
                "SELECT name FROM chat_dictionaries WHERE id = ?", (observed_ids[0],)
            ).fetchone()[0]
            == "coherent"
        )
        db.close_connection()
        assert not select.select([child.stdout], [], [], 0.02)[
            0
        ]  # Actual startup remains.
        assert error is None, (
            f"durable dictionary committed but history publication failed: {error!r}; "
            f"sidecar_exists={path.exists()} cached_history={service._history!r}"
        )
        assert result["id"] == observed_ids[0]
        assert (
            json.loads(path.read_text())["dictionaries"][str(result["id"])]["versions"][
                0
            ]["snapshot"]["name"]
            == "coherent"
        )
    finally:
        db.close_connection()
        for pause in reversed(pauses):
            pause.resume()
        if child is not None:
            child.kill()
            child.wait(timeout=5)


def test_persona_pause_refuses_before_cache_and_sidecar_mutation(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    path = tmp_path / "personas.json"
    service = LocalCharacterPersonaService(None, persona_store_path=path)
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            service.create_persona_profile({"name": "must remain absent"})
        assert service._persona_profiles == []
        assert not path.exists()
    finally:
        pause.resume()


@pytest.fixture
def configured_db(tmp_path, monkeypatch, local_root):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    (tmp_path / "data").mkdir(mode=0o700)
    target = tmp_path / "config.toml"
    target.write_text(
        f'[paths]\ndata_dir = "{tmp_path / "data"}"\n'
        f'[database]\nchachanotes_db_path = "{tmp_path / "chat.sqlite"}"\n'
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config = install_config_source(monkeypatch)
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "phase10-test")
    try:
        yield db, config
    finally:
        db.close_connection()


def test_actual_citation_reader_refuses_during_pause(configured_db):
    from Tests.Chat.test_citation_legacy_migration import _repository
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, config = configured_db
    path = config.get_user_data_dir() / "tldw_chatbook_chat_rag_context.json"
    path.write_text('{"version":1,"conversations":{}}')
    migration = CitationLegacyMigrationService(
        db=db, repository=_repository(db), sidecar_path=path
    )
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            migration._raw_sidecar()
    finally:
        pause.resume()


def test_ordinary_optimistic_conflict_does_not_poison_source(configured_db):
    import time
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

    db, _ = configured_db
    service = chat.build_dictionary_service(db)
    created = service.create_dictionary({"name": "original"})
    before = service.history_store_path.read_bytes()
    with pytest.raises(ConflictError):
        service.update_dictionary(
            created["id"], {"name": "wrong"}, expected_version=9999
        )
    assert service.history_store_path.read_bytes() == before
    participant = raw._raw_participant(service)
    participant.close_admission()
    try:
        assert participant.drain(time.monotonic() + 0.1)
    finally:
        participant.resume()


@pytest.mark.asyncio
async def test_running_dictionary_scope_cancellation_waits_for_source_close(
    configured_db, monkeypatch
):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
        ChatDictionaryScopeService,
    )
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    db, _ = configured_db
    service = chat.build_dictionary_service(db)
    scope = ChatDictionaryScopeService(local_service=service, server_service=None)
    entered, finish = threading.Event(), threading.Event()
    original = module.cdl.save_chat_dictionary
    worker_connections = []

    def hold_after_commit(*args, **kwargs):
        result = original(*args, **kwargs)
        worker_connections.append(db.get_connection())
        entered.set()
        assert finish.wait(5)
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", hold_after_commit)
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    task = asyncio.create_task(scope.create_dictionary({"name": "cancelled awaiter"}))
    try:
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), (
            "cancelled awaiter retired while actual core/file work is still running"
        )
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.history_store_path.exists()

        def verify_closed():
            import sqlite3

            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                worker_connections[0].execute("SELECT 1")

        await loop.run_in_executor(executor, verify_closed)
    finally:
        finish.set()
        try:
            await task
        except BaseException:
            pass
        await loop.run_in_executor(executor, db.close_connection)
        loop._default_executor = previous_executor
        executor.shutdown(wait=True)


def test_dictionary_import_finishes_core_after_parser_pause(configured_db, monkeypatch):
    cdl = fresh_dictionary_library(monkeypatch)

    db, config = configured_db
    folder = config.get_user_data_dir() / "chat_dicts"
    folder.mkdir(exist_ok=True)
    path = folder / "coherent.md"
    path.write_text("alpha: beta\n")
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    original = raw._file
    pauses = []

    @contextmanager
    def pause_after_parser(token, selected, mode):
        with original(token, selected, mode) as stream:
            yield stream
            if raw._states[token].source is cdl and mode == "r" and not pauses:
                pauses.append(storage._begin_local_pause())

    monkeypatch.setattr(raw, "_file", pause_after_parser)
    try:
        result = cdl.import_dictionary_from_file(db, str(path))
        assert result is not None, (
            "parser read completed but the corresponding core import was refused"
        )
        native = getattr(db._local, "conn")
        assert (
            native.execute(
                "SELECT content FROM chat_dictionaries WHERE id = ?", (result,)
            ).fetchone()[0]
            == path.read_text()
        )
    finally:
        for pause in reversed(pauses):
            pause.resume()


def test_dictionary_parser_external_input_refuses_before_read(
    tmp_path, local_root, monkeypatch
):
    cdl = fresh_dictionary_library(monkeypatch)
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    path = tmp_path / "external.md"
    path.write_text("alpha: beta\n")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            cdl.parse_user_dict_markdown_file(str(path), str(tmp_path))
    finally:
        pause.resume()


def fresh_dictionary_library(monkeypatch):
    import importlib.util
    import sys
    import tldw_chatbook.Character_Chat as package
    from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as original

    spec = importlib.util.spec_from_file_location(original.__name__, original.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(package, "Chat_Dictionary_Lib", module)
    spec.loader.exec_module(module)
    return module


def test_failed_dictionary_publication_keeps_cache_and_sticky_core_evidence(
    configured_db,
):
    import copy
    import time
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    original = source.create_dictionary({"name": "before"})
    before = copy.deepcopy(source._history)
    before_bytes = source.history_store_path.read_bytes()
    foreign = source.history_store_path.with_suffix(".json.tmp")
    foreign.write_bytes(b"another actor")
    with pytest.raises(FileExistsError):
        source.create_dictionary({"name": "committed without history"})
    assert source._history == before
    assert source.history_store_path.read_bytes() == before_bytes
    assert foreign.read_bytes() == b"another actor"
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM chat_dictionaries")
        .fetchone()[0]
        == 2
    )
    foreign.unlink()  # Test actor removes only its own sentinel.
    source.update_dictionary(
        original["id"], {"name": "later success"}, expected_version=original["version"]
    )
    participant = raw._raw_participant(source)
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.02)
        assert source._chat_persistence_error == "chat_publication_incomplete"
    finally:
        participant.resume()


@pytest.mark.parametrize("selector", ["path", "db", "profile", "no_file"])
def test_installed_source_cannot_retarget_or_demote(
    configured_db, monkeypatch, selector
):
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, config = configured_db
    source = chat.build_dictionary_service(db)
    original = source.history_store_path
    if selector == "path":
        source.history_store_path = original.with_name("different.json")
    elif selector == "no_file":
        source.history_store_path = None
    elif selector == "db":
        source.db = None
    else:
        monkeypatch.setenv(
            "TLDW_CONFIG_PATH", str(original.with_name("different.toml"))
        )
    with pytest.raises(RecoveryRequired, match="chat_source_selection_changed"):
        source.create_dictionary({"name": "denied"})
    assert source._history == {}
    assert not original.exists()
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM chat_dictionaries")
        .fetchone()[0]
        == 0
    )


def test_same_path_second_source_cannot_inherit_paused_pair(configured_db, monkeypatch):
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    db, _ = configured_db
    first, second = chat.build_dictionary_service(db), chat.build_dictionary_service(db)
    original = module.cdl.save_chat_dictionary
    pauses = []

    def committed(*args, **kwargs):
        result = original(*args, **kwargs)
        pauses.append(storage._begin_local_pause())
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            second.create_dictionary({"name": "must not run"})
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", committed)
    try:
        result = first.create_dictionary({"name": "first only"})
        assert (
            json.loads(first.history_store_path.read_text())["dictionaries"][
                str(result["id"])
            ]["versions"][0]["snapshot"]["name"]
            == "first only"
        )
        assert second._history == {}
    finally:
        for pause in pauses:
            pause.resume()


def test_pause_between_pair_acquisitions_has_no_core_or_cache_effect(
    configured_db, monkeypatch
):
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import (
        raw_participants as raw,
        chat_source_participants as chat,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    original = raw._scope
    pauses = []

    @contextmanager
    def interrupted(actual_source, route, **kwargs):
        if actual_source is source:
            pauses.append(storage._begin_local_pause())
        with original(actual_source, route, **kwargs) as token:
            yield token

    monkeypatch.setattr(raw, "_scope", interrupted)
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            source.create_dictionary({"name": "not committed"})
        assert (
            db._local.conn.execute("SELECT COUNT(*) FROM chat_dictionaries").fetchone()[
                0
            ]
            == 0
        )
        assert source._history == {}
        assert not source.history_store_path.exists()
    finally:
        for pause in pauses:
            pause.resume()


def test_citation_pair_migration_and_compatibility_writer_keep_exact_relationship(
    configured_db, monkeypatch
):
    from Tests.Chat.test_citation_legacy_migration import (
        _repository,
        _conversation_with_messages,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
    )
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, config = configured_db
    conversation, messages = _conversation_with_messages(db, 1)
    path = config.get_user_data_dir() / "tldw_chatbook_chat_rag_context.json"
    migration = CitationLegacyMigrationService(
        db=db, repository=_repository(db, enabled=False), sidecar_path=path
    )
    source = ChatConversationService(
        db, rag_context_store_path=path, citation_legacy_migration=migration
    )
    chat.bind_citation_services(source, migration)
    original = db.get_message_by_id
    pauses = []

    def message_then_pause(*args, **kwargs):
        result = original(*args, **kwargs)
        if not pauses:
            pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(db, "get_message_by_id", message_then_pause)
    try:
        result = source.record_message_rag_context(
            conversation, messages[0], rag_context={"query": "paired"}
        )
        assert (
            json.loads(path.read_text())["conversations"][conversation][messages[0]]
            == result
        )
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            migration._raw_sidecar()  # A new root job cannot reuse completed pair.
    finally:
        for pause in pauses:
            pause.resume()
    source.citation_legacy_migration = None
    with pytest.raises(RecoveryRequired, match="chat_source_relationship_changed"):
        source._load_rag_context_store()


@pytest.mark.parametrize("mode", ["queued", "preexisting", "executor_cancel"])
@pytest.mark.asyncio
async def test_actual_dictionary_scope_job_boundaries(configured_db, monkeypatch, mode):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
        ChatDictionaryScopeService,
    )
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    scope = ChatDictionaryScopeService(local_service=source, server_service=None)
    loop = asyncio.get_running_loop()
    old_executor = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered, finish = threading.Event(), threading.Event()
    original = module.cdl.save_chat_dictionary
    actual_calls = []

    def actual_save(*args, **kwargs):
        actual_calls.append(True)
        result = original(*args, **kwargs)
        if mode == "executor_cancel":
            entered.set()
            assert finish.wait(5)
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", actual_save)
    occupied = None
    task = None
    try:
        if mode == "queued":

            def occupy():
                entered.set()
                assert finish.wait(5)

            occupied = pool.submit(occupy)
            assert entered.wait(1)
        elif mode == "preexisting":
            prior = await loop.run_in_executor(pool, db.get_connection)
        futures = []
        run = loop.run_in_executor

        def observed_run(*args, **kwargs):
            result = run(*args, **kwargs)
            futures.append(result)
            return result

        monkeypatch.setattr(loop, "run_in_executor", observed_run)
        task = asyncio.create_task(scope.create_dictionary({"name": mode}))
        await asyncio.sleep(0.02)
        if mode == "queued":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            finish.set()
            await run(pool, lambda: None)
            assert not actual_calls
            assert not source.history_store_path.exists()
        elif mode == "preexisting":
            result = await task

            def still_usable():
                assert db.get_connection() is prior
                assert (
                    prior.execute(
                        "SELECT name FROM chat_dictionaries WHERE id = ?",
                        (result["id"],),
                    ).fetchone()[0]
                    == mode
                )

            await run(pool, still_usable)
        else:
            for _ in range(500):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            futures[0].cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert source.history_store_path.exists()
    finally:
        finish.set()
        if task is not None:
            try:
                await task
            except BaseException:
                pass
        await loop.run_in_executor(pool, db.close_connection)
        loop._default_executor = old_executor
        pool.shutdown(wait=True)
        if occupied is not None:
            occupied.result()


@pytest.mark.parametrize("timing", ["before", "after"])
def test_native_pair_close_failure_retains_exclusion(
    configured_db, local_root, monkeypatch, launch, request, timing
):
    import os
    import subprocess
    import sys
    import time
    import copy
    from pathlib import Path
    from tldw_chatbook.Backup_Recovery import (
        raw_participants as raw,
        chat_source_participants as chat,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    if os.environ.get("TASK10_NATIVE_PAIR_CHILD") != timing:
        env = dict(
            os.environ, TASK10_NATIVE_PAIR_CHILD=timing, PYTHONDONTWRITEBYTECODE="1"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                request.node.nodeid,
                "-q",
                "-o",
                "cache_dir=/private/tmp/task10-phase10-native-child-cache",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        Path(f"/private/tmp/task10-phase10-native-{timing}.log").write_text(
            result.stdout + result.stderr
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    source.create_dictionary({"name": "before"})
    before = copy.deepcopy(source._history)
    # Diagnostic-only retirement of this child's known successful startup token.
    # Actual source/core holds now provide the independent native exclusion proof.
    startup = storage._startups[(os.getpid(), str(local_root))]
    startup.close()
    real_replace, real_close = raw._replace, os.close
    target = []

    def arm_after_publication(token, temporary, destination):
        result = real_replace(token, temporary, destination)
        if raw._states[token].source is source:
            target.append(raw._states[token].pins[source.history_store_path.parent])
        return result

    def ambiguous_close(fd):
        if target and fd == target[0]:
            if timing == "after":
                real_close(fd)
            raise OSError("injected native close ambiguity")
        return real_close(fd)

    monkeypatch.setattr(raw, "_replace", arm_after_publication)
    monkeypatch.setattr(os, "close", ambiguous_close)
    with pytest.raises(RecoveryRequired, match="raw_resources_not_retired"):
        source.create_dictionary({"name": "durably published"})
    monkeypatch.setattr(os, "close", real_close)
    assert source._history == before
    assert source._chat_persistence_error == "chat_publication_incomplete"
    assert (
        db._local.conn.execute("SELECT COUNT(*) FROM chat_dictionaries").fetchone()[0]
        == 2
    )
    assert len(json.loads(source.history_store_path.read_text())["dictionaries"]) == 2
    if timing == "before":
        assert os.fstat(target[0])
    db.close_connection()
    pause = storage._begin_local_pause()
    participant = raw._raw_participant(source)
    participant.close_admission()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        assert not participant.drain(time.monotonic() + 0.03)
        assert not pause.drain(time.monotonic() + 0.03)
        assert not select.select([child.stdout], [], [], 0.08)[0]
    finally:
        child.kill()
        child.wait(timeout=5)
        pause.resume()
        # Retained uncertain descriptors are intentionally reaped with this child.


def test_library_custom_export_and_default_export_match_real_record(
    configured_db, monkeypatch, tmp_path
):
    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    ident = cdl.save_chat_dictionary(db, name="Export Name", content="alpha: beta\n")
    output = tmp_path / "outside-profile.md"
    assert cdl.export_dictionary_to_file(db, ident, str(output)) == str(output)
    assert output.read_text() == "alpha: beta\n"
    default = config.get_user_data_dir() / "chat_dicts" / "Export-Name.md"
    assert cdl.export_dictionary_to_file(db, ident) == str(default)
    assert default.read_bytes() == output.read_bytes()


def test_library_copy_preserves_actual_external_source_bytes_and_metadata(
    configured_db, monkeypatch, tmp_path
):
    import os
    import stat

    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    source = tmp_path / "relative.md"
    source.write_bytes(b"alpha: beta\r\n")
    source.chmod(0o640)
    os.utime(source, ns=(1234567800000000000, 1234567900000000000))
    # The existing relative import route parses under its configured base and
    # copies its caller-relative input; preflight must account for both files.
    destination = root / source.name
    destination.write_bytes(source.read_bytes())
    monkeypatch.chdir(tmp_path)
    result = cdl.import_dictionary_from_file(db, source.name)
    assert result is not None
    assert destination.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns
    assert db.get_connection().execute(
        "SELECT file_path FROM chat_dictionaries WHERE id = ?", (result,)
    ).fetchone()[0] == str(destination)


def test_custom_export_missing_parent_keeps_original_failure_semantics(
    configured_db, monkeypatch, tmp_path
):
    cdl = fresh_dictionary_library(monkeypatch)
    db, _ = configured_db
    ident = cdl.save_chat_dictionary(db, name="custom", content="a: b\n")
    output = tmp_path / "uncreated" / "custom.md"
    assert cdl.export_dictionary_to_file(db, ident, str(output)) is None
    assert not output.parent.exists()


def test_independent_maintainer_observes_coherent_dictionary_pair_after_native_close(
    configured_db, local_root, monkeypatch, request
):
    if _diagnostic_child(request, "dictionary"):
        return
    import os
    import subprocess
    import sys
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    source.create_dictionary({"name": "first generation"})
    # Diagnostic-only known startup retirement; core and raw source leases remain.
    storage._startups[(os.getpid(), str(local_root))].close()
    script = """
import json, sqlite3, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print("attempting", flush=True)
with Admission(Path(sys.argv[1])).maintenance(("bootstrap.unbound",), 10):
    with sqlite3.connect(sys.argv[2]) as db:
        rows = db.execute("SELECT id, name FROM chat_dictionaries ORDER BY id").fetchall()
    payload = json.loads(Path(sys.argv[3]).read_text())
    print(json.dumps({"rows":rows,"history":payload}), flush=True)
"""
    children = []
    pauses = []
    original = module.cdl.save_chat_dictionary

    def after_commit(*args, **kwargs):
        result = original(*args, **kwargs)
        pauses.append(storage._begin_local_pause())
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                script,
                str(local_root / "admission"),
                str(db.db_path),
                str(source.history_store_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        assert line(child) == "attempting"
        assert not select.select([child.stdout], [], [], 0.04)[0]
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", after_commit)
    try:
        created = source.create_dictionary({"name": "second generation"})
        child = children[0]
        assert not select.select([child.stdout], [], [], 0.04)[
            0
        ]  # Native DB borrower is still live.
        db.close_connection()
        observed = json.loads(line(child))
        assert child.wait(timeout=5) == 0
        assert observed["rows"][-1] == [created["id"], "second generation"]
        for ident, name in observed["rows"]:
            assert (
                observed["history"]["dictionaries"][str(ident)]["versions"][-1][
                    "snapshot"
                ]["name"]
                == name
            )
    finally:
        db.close_connection()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("failure", ["paused", "stale_temp", "serialization"])
def test_persona_all_seven_cached_categories_survive_refused_publication(
    configured_db, monkeypatch, failure
):
    import copy
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, config = configured_db
    path = config.get_user_data_dir() / "tldw_chatbook_personas.json"
    payload = {
        "profiles": [{"id": "p", "name": "prior", "version": 3}],
        "exemplars": [{"id": "e", "persona_id": "p", "content": "prior"}],
        "character_exemplars": [{"id": "c", "character_id": 1, "content": "prior"}],
        "chat_settings": {"chat": {"temperature": 0.2}},
        "chat_greeting_selections": {"chat": 1},
        "chat_presets": [{"id": "preset", "name": "prior"}],
        "character_memories": [{"id": "m", "character_id": 1, "content": "prior"}],
    }
    path.write_text(json.dumps(payload))
    source = chat.build_persona_service(db)
    names = (
        "_persona_profiles",
        "_persona_exemplars",
        "_character_exemplars",
        "_chat_settings",
        "_chat_greeting_selections",
        "_chat_presets",
        "_character_memories",
    )
    before = {name: copy.deepcopy(getattr(source, name)) for name in names}
    before_bytes = path.read_bytes()
    pause = None
    error = FileExistsError
    if failure == "paused":
        pause = storage._begin_local_pause()
        error = RecoveryRequired
    elif failure == "stale_temp":
        path.with_suffix(".json.tmp").write_bytes(b"not our temporary")
    else:
        original = json.dumps

        def failed_serialization(value, *args, **kwargs):
            if isinstance(value, dict) and set(value) == set(payload):
                raise ValueError("injected serialization failure")
            return original(value, *args, **kwargs)

        monkeypatch.setattr(json, "dumps", failed_serialization)
        error = ValueError
    try:
        with pytest.raises(error):
            source.create_persona_profile({"name": "must not publish"})
        assert {name: getattr(source, name) for name in names} == before
        assert path.read_bytes() == before_bytes
    finally:
        if pause is not None:
            pause.resume()


def test_pause_inside_real_dictionary_insert_still_finishes_pair(configured_db):
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    connection = db.get_connection()
    pauses = []

    def observe(sql):
        if "INSERT INTO chat_dictionaries" in sql and not pauses:
            pauses.append(storage._begin_local_pause())

    connection.set_trace_callback(observe)
    try:
        result = source.create_dictionary({"name": "inside native statement"})
        assert pauses
        assert (
            json.loads(source.history_store_path.read_text())["dictionaries"][
                str(result["id"])
            ]["versions"][0]["snapshot"]["name"]
            == result["name"]
        )
        assert (
            connection.execute(
                "SELECT name FROM chat_dictionaries WHERE id = ?", (result["id"],)
            ).fetchone()[0]
            == result["name"]
        )
    finally:
        connection.set_trace_callback(None)
        for pause in pauses:
            pause.resume()


@pytest.mark.asyncio
async def test_foreign_task_cannot_reuse_chat_pair(configured_db):
    import asyncio
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    with chat.operation(source):

        async def foreign():
            with pytest.raises(
                RecoveryRequired, match="raw_operation_provenance_invalid"
            ):
                source.create_dictionary({"name": "foreign task"})

        await asyncio.create_task(foreign())
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM chat_dictionaries")
        .fetchone()[0]
        == 0
    )
    assert not source.history_store_path.exists()


def test_subclass_and_explicit_memory_sources_keep_ordinary_semantics(
    tmp_path, local_root, monkeypatch
):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    install_config_source(monkeypatch)
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    class CustomPersona(LocalCharacterPersonaService):
        pass

    source = CustomPersona(None, persona_store_path=tmp_path / "custom.json")
    created = source.create_persona_profile({"name": "ordinary"})
    assert source.get_persona_profile(created["id"])["name"] == "ordinary"
    with pytest.raises(RecoveryRequired, match="raw_participant_not_installed"):
        raw._raw_participant(source)
    memory = CharactersRAGDB(":memory:", "explicit-memory")
    try:
        ordinary = LocalChatDictionaryService(memory)
        created = ordinary.create_dictionary({"name": "same-thread memory"})
        assert ordinary.get_dictionary(created["id"])["name"] == "same-thread memory"
        assert ordinary.history_store_path is None
    finally:
        memory.close_connection()


def test_library_missing_input_preserves_parser_and_import_results(
    configured_db, monkeypatch
):
    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    missing = config.get_user_data_dir() / "chat_dicts" / "missing.md"
    missing.parent.mkdir(exist_ok=True)
    assert cdl.parse_user_dict_markdown_file(str(missing)) == {}
    assert cdl.import_dictionary_from_file(db, str(missing)) is None


def test_library_listing_pause_refuses_before_glob(configured_db, monkeypatch):
    from pathlib import Path

    cdl = fresh_dictionary_library(monkeypatch)
    _, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    (root / "present.md").write_text("a: b")
    pauses = []
    real_folder = cdl.get_chat_dicts_folder
    real_glob = Path.glob
    observed = []

    def pause_after_folder():
        selected = real_folder()
        pauses.append(storage._begin_local_pause())
        return selected

    def observe_glob(path, pattern):
        observed.append(bool(storage._raw_operations))
        return real_glob(path, pattern)

    monkeypatch.setattr(cdl, "get_chat_dicts_folder", pause_after_folder)
    monkeypatch.setattr(Path, "glob", observe_glob)
    try:
        # A preadmitted directory listing may finish, or refusal must precede glob.
        result = cdl.list_available_dictionary_files()
        assert not observed or all(observed)
        assert result == [] or result[0]["name"] == "present"
    finally:
        for pause in pauses:
            pause.resume()


def test_library_copy_real_flags_and_checked_errno(
    configured_db, monkeypatch, tmp_path
):
    import errno
    import os
    import stat
    from tldw_chatbook.Backup_Recovery import dictionary_file_participants as files

    if not hasattr(os, "chflags"):
        pytest.skip("BSD file flags unavailable")
    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    source = tmp_path / "flags.md"
    source.write_text("a: b\n")
    os.chflags(source, stat.UF_NODUMP)
    destination = root / source.name
    destination.write_text("a: b\n")
    monkeypatch.chdir(tmp_path)
    try:
        assert cdl.import_dictionary_from_file(db, source.name) is not None
        assert destination.stat().st_flags & stat.UF_NODUMP
        with pytest.raises(OSError) as invalid:
            files._copy_flags(-1, stat.UF_NODUMP)
        assert invalid.value.errno == errno.EBADF
    finally:
        os.chflags(source, 0)
        os.chflags(destination, 0)


@pytest.mark.parametrize("point", ["core_selection", "raw_admission"])
def test_library_export_preflight_pause_has_no_output(
    configured_db, monkeypatch, tmp_path, point
):
    from tldw_chatbook.Backup_Recovery import dictionary_file_participants as files
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    cdl = fresh_dictionary_library(monkeypatch)
    db, _ = configured_db
    ident = cdl.save_chat_dictionary(db, name="preflight", content="a: b")
    pauses = []
    if point == "core_selection":
        original = cdl.load_chat_dictionary

        def paused(*args, **kwargs):
            result = original(*args, **kwargs)
            pauses.append(storage._begin_local_pause())
            return result

        monkeypatch.setattr(cdl, "load_chat_dictionary", paused)
    else:
        original = files.pin_inputs

        def paused(state):
            original(state)
            pauses.append(storage._begin_local_pause())

        monkeypatch.setattr(files, "pin_inputs", paused)
    output = tmp_path / "never-created.md"
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            cdl.export_dictionary_to_file(db, ident, str(output))
        assert not output.exists()
        assert not output.with_suffix(".md.tmp").exists()
    finally:
        for pause in pauses:
            pause.resume()


def test_library_export_identity_swap_preserves_foreign_destination(
    configured_db, monkeypatch, tmp_path
):
    from tldw_chatbook.Backup_Recovery import dictionary_file_participants as files

    cdl = fresh_dictionary_library(monkeypatch)
    db, _ = configured_db
    ident = cdl.save_chat_dictionary(db, name="identity", content="a: b")
    output = tmp_path / "identity.md"
    output.write_text("previous")
    foreign = tmp_path / "foreign.md"
    foreign.write_text("foreign")
    original = files.check_destination

    def replace_target(plan):
        foreign.replace(output)
        return original(plan)

    monkeypatch.setattr(files, "check_destination", replace_target)
    assert cdl.export_dictionary_to_file(db, ident, str(output)) is None
    assert output.read_text() == "foreign"
    assert not output.with_suffix(".md.tmp").exists()


def test_library_copy_then_core_failure_stays_unqualified(
    configured_db, monkeypatch, tmp_path
):
    import time
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    source = tmp_path / "mixed.md"
    source.write_text("a: b\n")
    target = root / source.name
    target.write_text("a: old\n")
    monkeypatch.chdir(tmp_path)
    original = cdl.save_chat_dictionary

    def fail(*args, **kwargs):
        raise OSError("core publication failed")

    monkeypatch.setattr(cdl, "save_chat_dictionary", fail)
    assert cdl.import_dictionary_from_file(db, source.name) is None
    assert target.read_bytes() == source.read_bytes()
    monkeypatch.setattr(cdl, "save_chat_dictionary", original)
    assert cdl.import_dictionary_from_file(db, source.name) is not None
    participant = raw._raw_participant(cdl)
    pause = storage._begin_local_pause()
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.05)
    finally:
        pause.resume()
        participant.resume()


def test_library_missing_flags_primitive_keeps_ordinary_import(
    configured_db, monkeypatch, tmp_path
):
    from tldw_chatbook.Backup_Recovery import dictionary_file_participants as files

    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    source = tmp_path / "portable.md"
    source.write_text("a: b\n")
    (root / source.name).write_text("a: b\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(files, "_flags_function", lambda: None)
    assert cdl.import_dictionary_from_file(db, source.name) is not None
    assert files.binding(cdl) is None
    assert cdl in files._errors


@pytest.mark.asyncio
async def test_dictionary_job_close_outcome_failure_restores_prior_cache(
    configured_db, monkeypatch
):
    import asyncio
    import copy
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
        ChatDictionaryScopeService,
    )

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    source.create_dictionary({"name": "before"})
    before = copy.deepcopy(source._history)
    scope = ChatDictionaryScopeService(local_service=source, server_service=None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(executor)
    real_close = db.close_connection
    thread = threading.current_thread()

    def close_then_fail():
        real_close()
        if threading.current_thread() is not thread:
            raise OSError("positive close followed by outcome failure")

    monkeypatch.setattr(db, "close_connection", close_then_fail)
    try:
        with pytest.raises(OSError, match="outcome failure"):
            await scope.create_dictionary({"name": "committed"})
        assert source._history == before
        assert source._chat_persistence_error == "chat_publication_incomplete"
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM chat_dictionaries")
            .fetchone()[0]
            == 2
        )
    finally:
        monkeypatch.setattr(db, "close_connection", real_close)
        await loop.run_in_executor(executor, real_close)
        loop._default_executor = previous
        executor.shutdown(wait=True)


def test_dictionary_sidecar_identity_swap_preserves_foreign_bytes(
    configured_db, monkeypatch, tmp_path
):
    import copy
    from tldw_chatbook.Backup_Recovery import chat_source_participants as chat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Character_Chat import local_chat_dictionary_service as module

    db, _ = configured_db
    source = chat.build_dictionary_service(db)
    source.create_dictionary({"name": "before"})
    before = copy.deepcopy(source._history)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("foreign sidecar")
    real_save = module.cdl.save_chat_dictionary

    def replace_after_commit(*args, **kwargs):
        result = real_save(*args, **kwargs)
        foreign.replace(source.history_store_path)
        return result

    monkeypatch.setattr(module.cdl, "save_chat_dictionary", replace_after_commit)
    with pytest.raises(RecoveryRequired, match="chat_sidecar_identity_changed"):
        source.create_dictionary({"name": "committed"})
    assert source.history_store_path.read_text() == "foreign sidecar"
    assert source._history == before
    assert source._chat_persistence_error == "chat_publication_incomplete"
    assert not source.history_store_path.with_suffix(".json.tmp").exists()


def _diagnostic_child(request, kind):
    import os
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("TASK10_SOURCE_ONLY_CHILD") == kind:
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            request.node.nodeid,
            "-q",
            "-o",
            "cache_dir=/private/tmp/task10-phase10-source-child-cache",
        ],
        env={
            **os.environ,
            "TASK10_SOURCE_ONLY_CHILD": kind,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    Path(f"/private/tmp/task10-phase10-source-only-{kind}.log").write_text(
        result.stdout + result.stderr
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def test_library_external_copy_native_maintenance_lifetime(
    configured_db, monkeypatch, tmp_path, local_root, launch, request
):
    if _diagnostic_child(request, "library"):
        return
    import os
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import (
        raw_participants as raw,
        dictionary_file_participants as files,
    )

    cdl = fresh_dictionary_library(monkeypatch)
    db, config = configured_db
    root = config.get_user_data_dir() / "chat_dicts"
    root.mkdir(exist_ok=True)
    external = tmp_path / "external.md"
    external.write_bytes(b"a: coherent\n")
    target = root / external.name
    target.write_bytes(external.read_bytes())
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    storage._startups[(os.getpid(), str(local_root))].close()
    real_file = raw._file
    pauses, children = [], []

    @contextmanager
    def parse_pause(token, path, mode):
        with real_file(token, path, mode) as stream:
            yield stream
            if raw._states[token].source is cdl and mode == "r" and not pauses:
                pauses.append(storage._begin_local_pause())
                child = launch(
                    local_root / "admission", "maintenance", ("bootstrap.unbound",)
                )
                children.append(child)
                assert not select.select([child.stdout], [], [], 0.05)[0]

    monkeypatch.setattr(raw, "_file", parse_pause)
    try:
        ident = cdl.import_dictionary_from_file(db, external.name)
        assert ident is not None
        assert target.read_bytes() == external.read_bytes()
        assert (
            db._local.conn.execute(
                "SELECT content FROM chat_dictionaries WHERE id = ?", (ident,)
            ).fetchone()[0]
            == external.read_text()
        )
        assert files.binding(cdl) == ("chat.dictionaries", root, True)
        assert external.parent != root
        child = children[0]
        assert not select.select([child.stdout], [], [], 0.05)[0]
        db.close_connection()
        assert line(child) == "entered"
        release(child)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        for pause in pauses:
            pause.resume()
