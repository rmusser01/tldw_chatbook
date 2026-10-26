"""Native in-place consolidation owns exact stores and durable interruption fences."""

from contextlib import contextmanager

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation import preview_selected_mobile
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections
from tldw_chatbook.Backup_Recovery.profile_paths import (
    canonical_console_paths,
    legacy_mobile_dir,
)


@contextmanager
def prepared(
    tmp_path,
    monkeypatch,
    *,
    fresh=False,
    enrolled=True,
    ordinary_execution=False,
    run_mode=None,
    destination_mode=None,
    wal=False,
):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import (
        seed_canonical_console,
        seed_legacy_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
    )

    profile = tmp_path / "profile"
    profile.mkdir(mode=448)
    config = profile / "config.toml"
    config.write_text(
        f'''[general]\nusers_name="shared"\n[paths]\ndata_dir="{profile / "data"}"\n[database]\nchachanotes_db_path="{profile / "new" / "nested" / "chat.db"}"\nworkspaces_db_path="{profile / "new" / "nested" / "workspaces.db"}"\n'''
    )
    config.chmod(384)
    values = read_profile_config(config)
    source = legacy_mobile_dir(values)
    seed_legacy_mobile(source)
    stores = CanonicalConsoleStores(*canonical_console_paths(values))
    if not fresh:
        seed_canonical_console(stores)
    if wal:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0'); c.execute(\"INSERT INTO mobile_consolidation_batches VALUES ('test-owned-wal','test','{}','test','test')\"); c.commit(); os._exit(0)",
                str(stores.workspace_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for suffix in ("-wal", "-shm"):
            from pathlib import Path

            sidecar = Path(f"{stores.workspace_path}{suffix}")
            assert sidecar.exists()
            sidecar.chmod(0o644)
    if destination_mode is not None:
        for path in stores.paths:
            path.chmod(destination_mode)
    if run_mode is not None:
        stores.run_path.chmod(run_mode)
    root = tmp_path / "bootstrap"
    control = tmp_path / "control"
    control.mkdir(mode=448)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("mobile.profile", (profile,))
    if enrolled:
        bind_profile(root, config, ("mobile.profile",), authority.control_root)
    preview = preview_selected_mobile(config, config)
    inventory = discover((config,))
    journal = Journal(control, "mobile-test")
    from contextlib import ExitStack

    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner

    with ExitStack() as resources:
        if ordinary_execution:
            resources.enter_context(ConsoleExecutionOwner.acquire(stores.run_path))
        with authority.maintenance(
            ("bootstrap.unbound", "mobile.profile"), 2
        ) as session:
            yield (session, journal, config, stores, preview, inventory, authority)


def prepare(session, journal, config, preview, inventory):
    session.prepare_mobile_consolidation(
        journal,
        source_config=config,
        target_config=config,
        selections=DiscoverySelections(),
        approved_scope=inventory.scope_digest,
        source_fingerprint=preview.source_fingerprint,
        destination_witnesses=preview.destination_witnesses,
        destination_sidecars=preview.destination_sidecars,
    )


@pytest.mark.parametrize(
    "owner_id", ["db.chachanotes.primary", "db.workspaces", "db.agent_runs"]
)
def test_exact_installed_owner_transaction_and_durable_fence(
    tmp_path, monkeypatch, owner_id
):
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        prepare(session, journal, config, preview, inventory)
        with pytest.raises(bootstrap.RecoveryRequired):
            acquire_storage(stores.chat_path)
        with session.mobile_consolidation_owner(journal, owner_id) as database:
            with database.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO mobile_consolidation_batches VALUES (?, ?, ?, ?, ?)",
                    ("test-owned-native-row", "test", "{}", "test", "test"),
                )
            connection = (
                database._get_thread_connection()
                if owner_id == "db.chachanotes.primary"
                else database._held_connection()
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches"
                ).fetchone()[0]
                == 1
            )
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        assert bootstrap.startup_permission(config, authority.control_root.parent) == (
            False,
            "recovery_pending",
        )
    service = RecoveryService(journal.root.parent)
    try:
        state = service.status(journal.operation_id)
        assert state["mode"] == "mobile_consolidation"
        assert state["state"] == "recovery_required"
        assert state["actions"] == ()
        for action in ("finish", "rollback", "abort"):
            with pytest.raises(ValueError, match="recovery_action_unavailable"):
                service.start_recovery(journal.operation_id, action=action)
    finally:
        service.close()


def test_unenrolled_source_cannot_mint_in_place_authority(tmp_path, monkeypatch):
    with (
        prepared(tmp_path, monkeypatch, enrolled=False) as (
            session,
            journal,
            config,
            _,
            preview,
            inventory,
            _,
        ),
        pytest.raises(bootstrap.RecoveryRequired, match="close_old_instance"),
    ):
        prepare(session, journal, config, preview, inventory)


def test_fresh_parents_and_databases_have_ordered_creation_evidence(
    tmp_path, monkeypatch
):
    with prepared(tmp_path, monkeypatch, fresh=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        assert not stores.chat_path.parent.exists()
        prepare(session, journal, config, preview, inventory)
        with session.mobile_consolidation_owner(
            journal, "db.chachanotes.primary"
        ) as database:
            assert database.get_all_conversation_ids() == []
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        creations = [row for row in records if row.event.startswith("mobile_path_")]
        assert len(creations) == 6
        for intended, observed in zip(creations[::2], creations[1::2], strict=True):
            assert intended.event == "mobile_path_intended"
            assert observed.event == "mobile_path_observed"
            assert intended.evidence["before"]["inode"] is None
            assert (
                intended.evidence["before"]["path"]
                == observed.evidence["after"]["path"]
            )
            assert observed.evidence["after"]["inode"] is not None


def test_grant_refuses_foreign_repository_path_factory_thread_and_stale_config(
    tmp_path, monkeypatch
):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        with session.mobile_consolidation_owner(journal, "db.workspaces") as database:
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mobile_repository_scope_invalid"
            ):
                WorkspaceDB(stores.workspace_path)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mobile_connection_scope_invalid"
            ):
                connect_private_sqlite("db.base", stores.chat_path)

            class ForeignFactory(sqlite3.Connection):
                pass

            with pytest.raises(
                bootstrap.RecoveryRequired, match="mobile_connection_scope_invalid"
            ):
                connect_private_sqlite(
                    "db.base", stores.workspace_path, factory=ForeignFactory
                )
            with (
                ThreadPoolExecutor(max_workers=1) as pool,
                pytest.raises(bootstrap.RecoveryRequired),
            ):
                pool.submit(database._held_connection).result()
            original = config.read_text()
            config.write_text(original + "\n# changed\n")
            with pytest.raises(bootstrap.RecoveryRequired, match="close_old_instance"):
                database._held_connection()
            config.write_text(original)
        with pytest.raises(bootstrap.RecoveryRequired):
            database._held_connection()


def test_reopen_same_operation_keeps_pending_and_exact_witnesses(tmp_path, monkeypatch):
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        prepare(session, journal, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(journal, "db.workspaces") as database,
            database.transaction() as cursor,
        ):
            cursor.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?, ?, ?, ?, ?)",
                ("test-owned", "fingerprint", "batch", "{}", "now"),
            )
    reopened = Journal(journal.root.parent, journal.operation_id)
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        prepare(session, reopened, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(reopened, "db.workspaces") as database,
            database.connection() as connection,
        ):
            assert (
                connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches"
                ).fetchone()[0]
                == 1
            )
        replacement = stores.chat_path.with_name("replaced.db")
        replacement.write_bytes(stores.chat_path.read_bytes())
        replacement.chmod(384)
        replacement.replace(stores.chat_path)
        with (
            pytest.raises(bootstrap.RecoveryRequired, match="mobile_object_changed"),
            session.mobile_consolidation_owner(reopened, "db.workspaces"),
        ):
            pass


def test_journal_cannot_claim_owner_retirement_or_completion_out_of_order(
    tmp_path, monkeypatch
):
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        owner = {"owner_id": "db.workspaces", "path": str(stores.workspace_path)}
        with pytest.raises(ValueError, match="mobile_native_session_required"):
            journal.record("mobile_owner_opened", owner)
        with journal._locked(exclusive=True) as parent:
            for event, evidence in (
                ("mobile_owner_retired", owner),
                ("committed", {}),
                ("mobile_batch_committed", {}),
            ):
                with pytest.raises(ValueError):
                    journal._append(parent, event, evidence)
        with (
            session.mobile_consolidation_owner(journal, "db.workspaces"),
            journal._locked(exclusive=True) as parent,
            pytest.raises(ValueError, match="mobile_owner_not_retired"),
        ):
            journal._append(parent, "mobile_owner_opened", owner)


def test_live_capability_rejects_copied_scope_pid_foreign_operation_and_source_write(
    tmp_path, monkeypatch
):
    import copy
    from concurrent.futures import ThreadPoolExecutor

    from Tests.Backup_Recovery.mobile_consolidation_fixture import fingerprints
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_execution_owner import (
        ConsoleExecutionOwner,
        ConsoleExecutionUnavailable,
    )
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _stores,
        preview,
        inventory,
        _,
    ):
        before = fingerprints(preview.source_dir)
        prepare(session, journal, config, preview, inventory)
        with pytest.raises(ConsoleExecutionUnavailable, match="scope_required"):
            ConsoleExecutionOwner._acquire_consolidation(object())
        with pytest.raises(bootstrap.RecoveryRequired):
            ConsoleExecutionOwner._acquire_consolidation(
                object.__new__(storage._MobileScope)
            )
        foreign = Journal(journal.root.parent, "foreign-operation")
        with (
            pytest.raises(bootstrap.RecoveryRequired, match="native_preparation"),
            session.mobile_consolidation_owner(foreign, "db.workspaces"),
        ):
            pass
        with session.mobile_consolidation_owner(
            journal, "db.chachanotes.primary"
        ) as database:
            scope = database._mobile_consolidation_scope
            connection = database._get_thread_connection()
            cursor = connection.cursor()
            with (
                ThreadPoolExecutor(max_workers=1) as pool,
                pytest.raises(bootstrap.RecoveryRequired),
            ):
                pool.submit(cursor.execute, "SELECT 1").result()
            with monkeypatch.context() as changed:
                changed.setattr(storage._local, "mobile_scope", copy.copy(scope))
                with pytest.raises(bootstrap.RecoveryRequired):
                    cursor.execute("SELECT 1")
            with monkeypatch.context() as changed:
                changed.setattr(session, "_pid", -1)
                with pytest.raises(bootstrap.RecoveryRequired):
                    cursor.execute("SELECT 1")
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mobile_connection_scope_invalid"
            ):
                connect_private_sqlite(
                    "db.chachanotes.primary", preview.source_dir / "chat.db"
                )
            cursor.close()
        assert fingerprints(preview.source_dir) == before
        with pytest.raises(bootstrap.RecoveryRequired):
            ConsoleExecutionOwner._acquire_consolidation(scope)


@pytest.mark.parametrize("failure", ["connection", "execution_user"])
def test_unretired_physical_resources_keep_native_fences(
    tmp_path, monkeypatch, failure
):
    import os
    import sqlite3
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery.admission import AdmissionError, AdmissionTimeout
    from tldw_chatbook.Utils.platform_files import fcntl

    if os.environ.get("TLDW_TASK8B_RETIREMENT_CHILD") != failure:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_unretired_physical_resources_keep_native_fences[{failure}]",
                "-q",
                "--basetemp",
                str(tmp_path / "child"),
            ],
            env={**os.environ, "TLDW_TASK8B_RETIREMENT_CHILD": failure},
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    with (
        pytest.raises(AdmissionError, match="resources_not_retired"),
        prepared(tmp_path, monkeypatch) as (
            session,
            journal,
            config,
            stores,
            preview,
            inventory,
            authority,
        ),
    ):
        prepare(session, journal, config, preview, inventory)
        with session.mobile_consolidation_owner(journal, "db.agent_runs") as database:
            scope = database._mobile_consolidation_scope
            connection = database._held_connection()
            if failure == "connection":

                def fail_close(self):
                    raise OSError("injected actual close refusal")

                monkeypatch.setattr(type(connection), "close", fail_close)
            else:
                user = scope.execution_owner._use(stores.run_path)
                user.__enter__()
    if failure == "connection":
        assert sqlite3.Connection.in_transaction.__get__(connection) is False
    assert not scope.execution_owner._stream.closed
    lock = stores.run_path.with_name(f".{stores.run_path.name}.execution.lock")
    with lock.open("r+") as contender, pytest.raises(BlockingIOError):
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (
        pytest.raises(AdmissionTimeout),
        authority.maintenance(("bootstrap.unbound", "mobile.profile"), 0.05),
    ):
        raise AssertionError("retained maintenance authority was released")
    if failure == "execution_user":
        user.__exit__(None, None, None)
        assert scope.execution_owner._stream.closed
    assert (
        bootstrap.startup_permission(config, authority.control_root.parent)[0] is False
    )


def test_cooperative_native_lease_retires_before_maintenance_and_incompatible_refuses(
    tmp_path,
):
    import threading

    from tldw_chatbook.Backup_Recovery.admission import AdmissionError, AdmissionTimeout

    root = tmp_path / "bootstrap"
    source = tmp_path / "source"
    source.mkdir(mode=448)
    authority = admission_authority(root)
    authority.register("mobile.source", (source,))
    entered, retire = (threading.Event(), threading.Event())

    def producer():
        with authority.normal(("mobile.source",)):
            entered.set()
            assert retire.wait(5)

    worker = threading.Thread(target=producer)
    worker.start()
    assert entered.wait(5)
    try:
        with (
            pytest.raises(AdmissionTimeout),
            authority.maintenance(("mobile.source",), 0.05),
        ):
            raise AssertionError("live lease admitted maintenance")
    finally:
        retire.set()
        worker.join(5)
    assert not worker.is_alive()
    with authority.maintenance(("mobile.source",), 1) as session:
        session._check()
    entered.clear()
    retire.clear()

    def incompatible():
        with authority.incompatible(("mobile.source",)):
            entered.set()
            assert retire.wait(5)

    worker = threading.Thread(target=incompatible)
    worker.start()
    assert entered.wait(5)
    try:
        with (
            pytest.raises(AdmissionError, match="known_incompatible_client"),
            authority.maintenance(("mobile.source",), 1),
        ):
            pass
    finally:
        retire.set()
        worker.join(5)


@pytest.mark.asyncio
async def test_cancelled_observer_does_not_retire_worker_connections_or_execution_lock(
    tmp_path, monkeypatch
):
    import asyncio
    import sqlite3
    import threading

    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

    entered, release = (threading.Event(), threading.Event())
    actual = {}

    def worker():
        with prepared(tmp_path, monkeypatch) as (
            session,
            journal,
            config,
            _,
            preview,
            inventory,
            authority,
        ):
            prepare(session, journal, config, preview, inventory)
            with session.mobile_consolidation_owner(
                journal, "db.agent_runs"
            ) as database:
                actual.update(
                    authority=authority,
                    connection=database._held_connection(),
                    execution=database._execution_owner,
                )
                entered.set()
                assert release.wait(15)

    worker_task = asyncio.create_task(asyncio.to_thread(worker))

    async def observe():
        return await asyncio.shield(worker_task)

    observer = asyncio.create_task(observe())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not worker_task.done()
        assert not actual["execution"]._stream.closed
        with (
            pytest.raises(AdmissionTimeout),
            actual["authority"].maintenance(
                ("bootstrap.unbound", "mobile.profile"), 0.05
            ),
        ):
            raise AssertionError("cancelled observer released native ownership")
    finally:
        release.set()
        await worker_task
    assert actual["execution"]._stream.closed
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.in_transaction.__get__(actual["connection"])
    with actual["authority"].maintenance(("bootstrap.unbound", "mobile.profile"), 1):
        pass


@pytest.mark.parametrize("damage", ["journal", "selector", "control"])
def test_corrupt_durable_evidence_cannot_reopen_or_clear_pending(
    tmp_path, monkeypatch, damage
):
    import json

    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        prepare(session, journal, config, preview, inventory)
        with session.mobile_consolidation_owner(journal, "db.workspaces"):
            pass
        pointer = authority.control_root.parent / (
            "pending-" + bootstrap._key(journal.operation_id) + ".json"
        )
        target = journal.root / "000000.json" if damage == "journal" else pointer
        record = json.loads(target.read_text())
        if damage == "journal":
            record["evidence"]["unknown"] = "not accepted"
        elif damage == "selector":
            record["selectors"] = [str(config.with_name("foreign.toml"))]
        else:
            record["control_root"] = str(journal.root.parent / "foreign")
        target.write_text(json.dumps(record))
        with pytest.raises((ValueError, bootstrap.RecoveryRequired)):
            prepare(session, journal, config, preview, inventory)
        assert pointer.exists()
        assert (
            bootstrap.startup_permission(config, authority.control_root.parent)[0]
            is False
        )
    if damage == "journal":
        service = RecoveryService(journal.root.parent)
        try:
            service._states[journal.operation_id] = {"state": "succeeded"}
            with pytest.raises(ValueError):
                service.status(journal.operation_id)
        finally:
            service.close()


def test_ordinary_execution_owner_prevents_native_maintenance(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

    with (
        pytest.raises(AdmissionTimeout),
        prepared(tmp_path, monkeypatch, ordinary_execution=True),
    ):
        pytest.fail("ordinary execution ownership must prevent maintenance admission")


@pytest.mark.parametrize(
    "owner_id", ["db.chachanotes.primary", "db.workspaces", "db.agent_runs"]
)
def test_reviewed_mode_hardening_is_journaled_before_actual_owner_construction(
    tmp_path, monkeypatch, owner_id
):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import fingerprints

    with prepared(tmp_path, monkeypatch, destination_mode=0o644) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        source_bytes = fingerprints(preview.source_dir)
        source_modes = {
            path: path.stat().st_mode for path in preview.source_dir.rglob("*")
        }
        target = dict(
            zip(
                ("db.chachanotes.primary", "db.workspaces", "db.agent_runs"),
                stores.paths,
                strict=True,
            )
        )[owner_id]
        append = journal._append
        transitions = []

        def observed_append(parent, event, evidence):
            if event.startswith("mobile_mode_"):
                transitions.append((event, target.stat().st_mode & 0o777))
            return append(parent, event, evidence)

        monkeypatch.setattr(journal, "_append", observed_append)
        prepare(session, journal, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(journal, owner_id) as database,
            database.transaction() as cursor,
        ):
            cursor.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?, ?, ?, ?, ?)",
                ("test-mode-change", "test", "{}", "test", "test"),
            )
        assert transitions == [
            ("mobile_mode_intended", 0o644),
            ("mobile_mode_observed", 0o600),
        ]
        assert target.stat().st_mode & 0o777 == 0o600
        assert fingerprints(preview.source_dir) == source_bytes
        assert {
            path: path.stat().st_mode for path in preview.source_dir.rglob("*")
        } == source_modes
    reopened = Journal(journal.root.parent, journal.operation_id)
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        prepare(session, reopened, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(reopened, owner_id) as database,
            database.transaction() as cursor,
        ):
            assert (
                cursor.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id='test-mode-change'"
                ).fetchone()[0]
                == 1
            )
    assert bootstrap.startup_permission(config, authority.control_root.parent) == (
        False,
        "recovery_pending",
    )


def test_reopen_rejects_changed_review_and_creation_rejects_swapped_parent(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import native_files

    with prepared(tmp_path, monkeypatch, fresh=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        with pytest.raises(ValueError, match="mobile_review_changed"):
            session.prepare_mobile_consolidation(
                journal,
                source_config=config,
                target_config=config,
                selections=DiscoverySelections(),
                approved_scope="0" * 64,
                source_fingerprint=preview.source_fingerprint,
                destination_witnesses=preview.destination_witnesses,
                destination_sidecars=preview.destination_sidecars,
            )
        native_pinned = native_files.pinned_directory

        @contextmanager
        def changed_parent(path):
            if path == stores.chat_path.parent.parent.parent:
                path.rename(path.with_name("retired-profile"))
                path.mkdir(mode=448)
            with native_pinned(path) as descriptor:
                yield descriptor

        monkeypatch.setattr(native_files, "pinned_directory", changed_parent)
        with (
            pytest.raises(bootstrap.RecoveryRequired, match="mobile_object_changed"),
            session.mobile_consolidation_owner(journal, "db.chachanotes.primary"),
        ):
            pass
        assert not stores.chat_path.exists()
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "mobile_path_intended"


def test_killed_native_owner_reopens_same_journal_under_fresh_exclusion(
    tmp_path, monkeypatch
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    marker = os.environ.get("TLDW_TASK8B_CRASH_MARKER")
    if marker:
        with prepared(tmp_path, monkeypatch) as (
            session,
            journal,
            config,
            _stores,
            preview,
            inventory,
            authority,
        ):
            prepare(session, journal, config, preview, inventory)
            with session.mobile_consolidation_owner(
                journal, "db.agent_runs"
            ) as database:
                with database.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO mobile_consolidation_batches VALUES (?, ?, ?, ?, ?)",
                        ("test-owned-crash-row", "test", "{}", "test", "test"),
                    )
                Path(marker).write_text(
                    json.dumps(
                        {
                            "bootstrap": str(authority.control_root.parent),
                            "control": str(journal.root.parent),
                            "config": str(config),
                            "fingerprint": preview.source_fingerprint,
                            "witnesses": preview.destination_witnesses,
                            "sidecars": preview.destination_sidecars,
                            "inventory": inventory.scope_digest,
                        }
                    )
                )
                os._exit(23)
    marker = tmp_path / "crash-evidence.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_killed_native_owner_reopens_same_journal_under_fresh_exclusion",
            "-q",
            "--basetemp",
            str(tmp_path / "child"),
        ],
        env={**os.environ, "TLDW_TASK8B_CRASH_MARKER": str(marker)},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    evidence = json.loads(marker.read_text())
    root = Path(evidence["bootstrap"])
    config = Path(evidence["config"])
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    assert bootstrap.startup_permission(config, root) == (False, "recovery_pending")
    journal = Journal(Path(evidence["control"]), "mobile-test")
    with journal._locked(exclusive=False) as parent:
        assert journal._records(parent)[-1].event == "mobile_owner_opened"
    authority = admission_authority(root)
    preview = SimpleNamespace(
        source_fingerprint=evidence["fingerprint"],
        destination_sidecars=tuple(
            (path, tuple(tuple(item) for item in rows))
            for path, rows in evidence["sidecars"]
        ),
        destination_witnesses=tuple(
            (
                (path, tuple((tuple(item) for item in chain)))
                for path, chain in evidence["witnesses"]
            )
        ),
    )
    inventory = SimpleNamespace(scope_digest=evidence["inventory"])
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        prepare(session, journal, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(journal, "db.agent_runs") as database,
            database.connection() as connection,
        ):
            assert (
                connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id='test-owned-crash-row'"
                ).fetchone()[0]
                == 1
            )
    assert bootstrap.startup_permission(config, root) == (False, "recovery_pending")


def test_real_wal_hardening_lifecycle_and_original_review_reopen(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import fingerprints

    with prepared(tmp_path, monkeypatch, wal=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        original_main_witnesses = preview.destination_witnesses
        original_sidecars = dict(preview.destination_sidecars)[
            str(stores.workspace_path)
        ]
        assert all(row[3] & 0o777 == 0o644 for row in original_sidecars[:2])
        source_bytes = fingerprints(preview.source_dir)
        source_modes = {
            path: path.stat().st_mode for path in preview.source_dir.rglob("*")
        }
        prepare(session, journal, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(journal, "db.workspaces") as database,
            database.transaction() as cursor,
        ):
            assert (
                cursor.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id='test-owned-wal'"
                ).fetchone()[0]
                == 1
            )
            cursor.execute(
                "INSERT INTO mobile_consolidation_batches VALUES ('native-wal-write','test','{}','test','test')"
            )
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        modes = [record for record in records if record.event == "mobile_mode_observed"]
        assert [row.evidence["after"]["path"] for row in modes] == [
            str(stores.workspace_path) + suffix for suffix in ("-wal", "-shm")
        ]
        assert all(row.evidence["after"]["mode"] & 0o777 == 0o600 for row in modes)
        assert any(
            row.event == "mobile_sidecars_observed"
            and row.evidence["phase"] == "retired"
            for row in records
        )
        assert preview.destination_witnesses == original_main_witnesses
        assert fingerprints(preview.source_dir) == source_bytes
        assert {
            path: path.stat().st_mode for path in preview.source_dir.rglob("*")
        } == source_modes
    reopened = Journal(journal.root.parent, journal.operation_id)
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        prepare(session, reopened, config, preview, inventory)
        with (
            session.mobile_consolidation_owner(reopened, "db.workspaces") as database,
            database.transaction() as cursor,
        ):
            assert (
                cursor.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id='native-wal-write'"
                ).fetchone()[0]
                == 1
            )


@pytest.mark.parametrize("change", ["mode", "generation"])
def test_helper_handoff_refuses_unjournaled_sidecar_changes(
    tmp_path, monkeypatch, change
):
    from pathlib import Path

    from tldw_chatbook.DB import private_sqlite

    with prepared(tmp_path, monkeypatch, wal=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        native_prepare = private_sqlite.prepare_in_helper
        target = Path(f"{stores.workspace_path}-wal")
        changed = []

        def change_before_helper(request, **kwargs):
            if request.require_private_mode:
                assert target.stat().st_mode & 0o777 == 0o600
                if change == "generation":
                    replacement = target.with_name("replacement-wal")
                    replacement.write_bytes(target.read_bytes())
                    replacement.chmod(0o644)
                    replacement.replace(target)
                else:
                    target.chmod(0o644)
                changed.append((target.stat().st_ino, target.read_bytes()))
            return native_prepare(request, **kwargs)

        monkeypatch.setattr(private_sqlite, "prepare_in_helper", change_before_helper)
        with (
            pytest.raises(Exception, match="private_sqlite_mode_change_refused"),
            session.mobile_consolidation_owner(journal, "db.workspaces"),
        ):
            pytest.fail("unreviewed sidecar reached SQLite owner")
        assert changed and (target.stat().st_ino, target.read_bytes()) == changed[0]
        assert target.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("change", ["before", "after"])
def test_interrupted_native_mode_transition_remains_fenced(
    tmp_path, monkeypatch, change
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    with prepared(tmp_path, monkeypatch, destination_mode=0o644) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        prepare(session, journal, config, preview, inventory)
        native_chmod = storage.os.fchmod
        inode = stores.workspace_path.stat().st_ino

        def interrupt_chmod(fd, mode):
            if storage.os.fstat(fd).st_ino == inode:
                if change == "after":
                    native_chmod(fd, mode)
                raise OSError("interrupted-mode-transition")
            return native_chmod(fd, mode)

        with monkeypatch.context() as changed:
            changed.setattr(storage.os, "fchmod", interrupt_chmod)
            with (
                pytest.raises(OSError, match="interrupted-mode-transition"),
                session.mobile_consolidation_owner(journal, "db.workspaces"),
            ):
                pass
        assert stores.workspace_path.stat().st_mode & 0o777 == (
            0o644 if change == "before" else 0o600
        )
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "mobile_mode_intended"
    assert bootstrap.startup_permission(config, authority.control_root.parent) == (
        False,
        "recovery_pending",
    )
    with (
        authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session,
        pytest.raises(bootstrap.RecoveryRequired),
    ):
        prepare(
            session,
            Journal(journal.root.parent, journal.operation_id),
            config,
            preview,
            inventory,
        )


def test_preview_rejects_absent_main_sidecar_name_overlap(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    target = CanonicalConsoleStores(
        tmp_path / "chat.db", tmp_path / "chat.db-wal", tmp_path / "run.db"
    )
    preview = preview_mobile_consolidation(tmp_path / "source", target)
    assert "mobile_unsafe_alias" in preview.issues
    assert not any(path.exists() for path in target.paths)


@pytest.mark.parametrize("kind", ["symlink", "main_link", "sidecar_link", "directory"])
def test_sidecar_preview_refuses_unsafe_objects_and_aliases(tmp_path, kind):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    stores = CanonicalConsoleStores(
        tmp_path / "chat.db", tmp_path / "workspace.db", tmp_path / "run.db"
    )
    stores.chat_path.touch(mode=0o600)
    sidecar = tmp_path / "chat.db-wal"
    if kind == "symlink":
        sidecar.symlink_to(stores.chat_path)
    elif kind == "main_link":
        sidecar.hardlink_to(stores.chat_path)
    elif kind == "sidecar_link":
        sidecar.touch(mode=0o600)
        (tmp_path / "workspace.db-shm").hardlink_to(sidecar)
    else:
        sidecar.mkdir(mode=0o700)
    preview = preview_mobile_consolidation(tmp_path / "absent-source", stores)
    assert "mobile_unsafe_alias" in preview.issues


def test_changed_sidecar_generation_refuses_before_any_native_hardening(
    tmp_path, monkeypatch
):
    from pathlib import Path

    with prepared(tmp_path, monkeypatch, wal=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        sidecar = Path(f"{stores.workspace_path}-wal")
        replacement = sidecar.with_name("changed-wal")
        replacement.write_bytes(sidecar.read_bytes())
        replacement.chmod(0o644)
        replacement.replace(sidecar)
        with (
            pytest.raises(bootstrap.RecoveryRequired, match="mobile_sidecar_changed"),
            session.mobile_consolidation_owner(journal, "db.workspaces"),
        ):
            pass
        assert sidecar.stat().st_mode & 0o777 == 0o644
        with journal._locked(exclusive=False) as parent:
            assert not any(
                record.event.startswith("mobile_mode_")
                for record in journal._records(parent)
            )
