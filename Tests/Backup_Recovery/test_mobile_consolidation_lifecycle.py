"""Private native lifecycle: reviewed selectors remain locators, never authority."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared
from tldw_chatbook.Backup_Recovery.mobile_consolidation import preview_selected_mobile


def test_selected_preview_preserves_complete_immutable_selector_provenance(
    tmp_path, monkeypatch
):
    with prepared(tmp_path, monkeypatch) as (_, _, config, _, preview, _, _):
        provenance = getattr(preview, "selector_provenance", None)
        assert provenance, "selected preview must retain checked selector provenance"
        assert preview.source_selector == config
        assert preview.target_selector == config
        selected = next(row for row in provenance if row.selector == config)
        assert len(selected.fingerprint) == 64
        assert selected.destination == preview.destination
        assert selected.destination_witnesses == preview.destination_witnesses
        with pytest.raises(FrozenInstanceError):
            selected.fingerprint = "changed"
        with config.open("a") as stream:
            stream.write("\n# Changed review evidence\n")
    current = preview_selected_mobile(config, config)
    assert current.selector_provenance != provenance


def test_physical_preview_has_no_selector_provenance(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    preview = preview_mobile_consolidation(
        tmp_path / "absent-source",
        CanonicalConsoleStores(*(tmp_path / leaf for leaf in ("c", "w", "r"))),
    )
    assert getattr(preview, "selector_provenance", None) == ()
    assert preview.source_selector is None
    assert preview.target_selector is None


def _prepare_review(session, journal, config, preview, inventory):
    from tldw_chatbook.Backup_Recovery.models import DiscoverySelections

    session.prepare_mobile_consolidation(
        journal,
        source_config=config,
        target_config=config,
        selections=DiscoverySelections(),
        approved_scope=inventory.scope_digest,
        source_fingerprint=preview.source_fingerprint,
        destination_witnesses=preview.destination_witnesses,
        destination_sidecars=preview.destination_sidecars,
        reviewed_preview=preview,
    )


def test_native_preparation_refuses_missing_selector_provenance(tmp_path, monkeypatch):
    from dataclasses import replace

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        with pytest.raises(ValueError, match="mobile_selector_review_required"):
            _prepare_review(
                session,
                journal,
                config,
                replace(preview, selector_provenance=()),
                inventory,
            )
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent) == []


def test_native_preparation_refuses_new_candidate_selector(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        candidate = config.with_name("candidate.toml")
        candidate.write_bytes(config.read_bytes())
        candidate.chmod(0o600)
        bind_profile(
            authority.control_root.parent,
            candidate,
            ("mobile.profile",),
            authority.control_root,
            session=session,
        )
        with pytest.raises(ValueError, match="mobile_selector_review_changed"):
            _prepare_review(session, journal, config, preview, inventory)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent) == []


def test_prepared_before_pending_recovers_under_fresh_native_exclusion(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import bootstrap, control_records

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        with monkeypatch.context() as fault:

            def interrupted(*args, **kwargs):
                raise OSError("injected before pending publication")

            fault.setattr(control_records, "register_pending", interrupted)
            with pytest.raises(OSError, match="injected"):
                _prepare_review(session, journal, config, preview, inventory)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "mobile_prepared"
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        _prepare_review(fresh, journal, config, preview, inventory)
        assert bootstrap.startup_permission(config, authority.control_root.parent) == (
            False,
            "recovery_pending",
        )
    with journal._locked(exclusive=False) as parent:
        assert len(journal._records(parent)) == 1


@pytest.mark.parametrize("phase", ["before_publication", "after_publication"])
def test_native_created_inode_recovers_from_durable_staging_witness(
    tmp_path, monkeypatch, phase
):
    from tldw_chatbook.Backup_Recovery import native_files

    with prepared(tmp_path, monkeypatch, fresh=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        publish = native_files.publish_new

        def interrupted(source, target, **kwargs):
            if target == stores.chat_path.parent.parent:
                if phase == "after_publication":
                    publish(source, target, **kwargs)
                raise OSError("injected creation publication interruption")
            return publish(source, target, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(native_files, "publish_new", interrupted)
            with (
                pytest.raises(OSError, match="injected creation"),
                session.mobile_consolidation_owner(journal, "db.chachanotes.primary"),
            ):
                pytest.fail("interrupted creation exposed an owner")
        with journal._locked(exclusive=False) as parent:
            staged = journal._records(parent)[-1]
            assert staged.event == "mobile_path_staged"
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        _prepare_review(fresh, journal, config, preview, inventory)
        with fresh.mobile_consolidation_owner(journal, "db.chachanotes.primary") as db:
            assert db.get_all_conversation_ids() == []
        assert (
            stores.chat_path.parent.parent.stat().st_ino
            == staged.evidence["temporary"]["inode"]
        )


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("owner_id", ["db.workspaces", "db.agent_runs"])
def test_native_mode_intent_recovers_same_inode_under_actual_owner_lease(
    tmp_path, monkeypatch, phase, owner_id
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
        _prepare_review(session, journal, config, preview, inventory)
        target = (
            stores.run_path if owner_id == "db.agent_runs" else stores.workspace_path
        )
        inode = target.stat().st_ino
        chmod = storage.os.fchmod

        def interrupted(fd, mode):
            if storage.os.fstat(fd).st_ino == inode:
                if phase == "after":
                    chmod(fd, mode)
                raise OSError("injected native mode interruption")
            return chmod(fd, mode)

        with monkeypatch.context() as fault:
            fault.setattr(storage.os, "fchmod", interrupted)
            with (
                pytest.raises(OSError, match="injected native mode"),
                session.mobile_consolidation_owner(journal, owner_id),
            ):
                pytest.fail("interrupted hardening exposed an owner")
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        _prepare_review(fresh, journal, config, preview, inventory)
        with fresh.mobile_consolidation_owner(journal, owner_id) as db:
            with db.transaction() as cursor:
                assert cursor.execute("PRAGMA user_version").fetchone()[0] >= 0
            connection = db._held_connection()
        assert target.stat().st_ino == inode
        assert target.stat().st_mode & 0o777 == 0o600
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_source_claim_is_physical_and_cannot_redirect_to_another_control(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_consolidation_apply as lifecycle
    from tldw_chatbook.Backup_Recovery.journal import Journal

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        with lifecycle._claim_source(session, journal, preview) as claim:
            original = claim.path.read_bytes()
            lock_inode = claim.lock_path.stat().st_ino
            assert claim.operation_id == journal.operation_id
        other_control = tmp_path / "other-control"
        other_control.mkdir(mode=0o700)
        other = Journal(other_control, "another-operation")
        with (
            pytest.raises(ValueError, match="mobile_source_already_claimed"),
            lifecycle._claim_source(session, other, preview),
        ):
            pytest.fail("same physical source acquired a new operation")
        with lifecycle._claim_source(session, journal, preview) as repeated:
            assert repeated.path.read_bytes() == original
            assert repeated.lock_path.stat().st_ino == lock_inode
        with journal._locked(exclusive=False) as parent:
            assert [row.event for row in journal._records(parent)].count(
                "mobile_claim_observed"
            ) == 1


@contextmanager
def _capture_fixture(tmp_path, monkeypatch, **options):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials, profile_paths
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    credential_store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Evals_DB import EvalsDB
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB

    source_state = options.pop("source_state", None)
    excluded_citation = options.pop("excluded_citation", False)
    with prepared(tmp_path, monkeypatch, **options) as values:
        _, journal, config, stores, _, _, authority = values
    if source_state is not None:
        import shutil

        from tldw_chatbook.Backup_Recovery.profile_paths import legacy_mobile_dir

        source = legacy_mobile_dir(read_profile_config(config))
        shutil.rmtree(
            source
        )  # Test fixture construction, before any reviewed operation.
        if source_state == "empty":
            source.mkdir(mode=0o700)
    if excluded_citation:
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        owner = CharactersRAGDB(stores.chat_path, "excluded-citation-fixture")
        with owner.transaction() as cursor:
            cursor.execute(
                "UPDATE rag_identity_context SET fingerprint_key_id=? WHERE context_name='default'",
                (credentials.RECOVERY_SETUP_REQUIRED,),
            )
        owner.close_connection()
    config_values = read_profile_config(config)
    for key, owner_type in (
        ("media_db_path", MediaDatabase),
        ("prompts_db_path", PromptsDatabase),
        ("evals_db_path", EvalsDB),
        ("library_collections_db_path", LibraryCollectionsDB),
        ("subscriptions_db_path", SubscriptionsDB),
        ("scheduled_tasks_db_path", ScheduledTasksDB),
    ):
        owner = owner_type(
            profile_paths.database_path(config_values, key), "mobile-capture-fixture"
        )
        owner.close()
    unused_config = tmp_path / "unused-default-config"
    unused_config.mkdir(mode=0o700)
    monkeypatch.setattr(
        profile_paths, "default_config_path", lambda: unused_config / "config.toml"
    )
    from tldw_chatbook.Backup_Recovery import config_adapter

    monkeypatch.setattr(
        config_adapter, "default_config_path", profile_paths.default_config_path
    )
    install_adapters()
    with _preview_reads():
        inventory = discover((config,))
    preview = preview_selected_mobile(config, config)
    assert inventory.complete, [
        (item.owner, str(item.path), item.status)
        for item in inventory.items
        if item.status in {"unsupported", "missing_required", "unavailable"}
    ]
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        yield session, journal, config, stores, preview, inventory, authority


def test_private_lifecycle_verifies_fixed_native_partitions_before_owners(
    tmp_path, monkeypatch
):
    import json
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _config,
        _stores,
        preview,
        inventory,
        _,
    ):
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        before = {
            path: (path.read_bytes(), path.stat().st_mode)
            for path in preview.capture_scope
            if path.is_file()
        }
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle:
            legacy = json.loads(lifecycle.source.manifest_bytes)
            rollback = json.loads(lifecycle.canonical.manifest_bytes)
            versions = lambda doc: {
                owner["owner_id"]: owner["schema_version"] for owner in doc["owners"]
            }
            assert {
                key: versions(legacy)[key]
                for key in ("db.chachanotes.primary", "db.workspaces", "db.agent_runs")
            } == {
                "db.chachanotes.primary": 81,
                "db.workspaces": 8,
                "db.agent_runs": 25,
            }
            assert {
                key: versions(rollback)[key]
                for key in ("db.chachanotes.primary", "db.workspaces", "db.agent_runs")
            } == {
                "db.chachanotes.primary": 82,
                "db.workspaces": 9,
                "db.agent_runs": 26,
            }
            assert lifecycle.source.path.is_file()
            assert lifecycle.canonical.path.is_file()
            with lifecycle.owner("db.workspaces") as db:
                assert db is not None
        assert before == {
            path: (path.read_bytes(), path.stat().st_mode) for path in before
        }
        with journal._locked(exclusive=False) as fd:
            events = [row.event for row in journal._records(fd)]
        assert events.count("mobile_capture_verified") == 2
        assert events.index("mobile_owner_opened") > max(
            index
            for index, event in enumerate(events)
            if event == "mobile_capture_verified"
        )


def test_native_source_claim_does_not_expand_reviewed_inventory(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _claim_source
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        with _claim_source(session, journal, preview):
            with session._discovery_reads():
                current = discover((config,))
            old = {item.logical_id: item for item in inventory.items}
            new = {item.logical_id: item for item in current.items}
            differences = [
                (key, old.get(key), new.get(key))
                for key in old.keys() | new.keys()
                if old.get(key) != new.get(key)
            ]
            assert current.scope_digest == inventory.scope_digest, (
                current.issues,
                differences,
            )


@pytest.mark.parametrize("phase", ["before", "after"])
def test_native_archive_publication_recovers_exact_verified_inode(
    tmp_path, monkeypatch, phase
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery import archive_writer
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        authority,
    ):
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        publish = archive_writer.publish_new

        def interrupted(source, target, **kwargs):
            if phase == "after":
                publish(source, target, **kwargs)
            raise OSError("injected archive publication interruption")

        with monkeypatch.context() as fault:
            fault.setattr(archive_writer, "publish_new", interrupted)
            with (
                pytest.raises(OSError, match="injected archive"),
                _captured_mobile(
                    session,
                    journal,
                    preview,
                    inventory.scope_digest,
                    destination=destination,
                    password=b"test-private-mobile-rollback",
                    cancel=Event(),
                ),
            ):
                pytest.fail("unverified captures exposed lifecycle")
        with journal._locked(exclusive=False) as fd:
            staged = journal._records(fd)[-1]
        assert staged.event == "mobile_capture_staged"
    with (
        authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh,
        _captured_mobile(
            fresh,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle,
    ):
        assert (
            lifecycle.source.path.stat().st_ino == staged.evidence["temporary"]["inode"]
        )
        assert lifecycle.canonical.path.is_file()
    with journal._locked(exclusive=False) as fd:
        assert (
            sum(row.event == "mobile_capture_verified" for row in journal._records(fd))
            == 2
        )

    assert not Path(staged.evidence["temporary_parent"]["path"]).exists()


@pytest.mark.parametrize(
    "phase", ["intent", "unwitnessed", "staged", "published", "observed"]
)
def test_actual_process_death_preserves_exact_creation_evidence(
    tmp_path, monkeypatch, phase
):
    import subprocess
    import sys
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import bootstrap

    with prepared(tmp_path, monkeypatch, fresh=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        _prepare_review(session, journal, config, preview, inventory)
    script = r"""
import os, sys
from pathlib import Path
assert os.environ.get("TLDW_TEST_CONFIG_ROOT")
assert "test" in os.environ["TLDW_CONFIG_PATH"]
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap, native_files
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _review_mobile_operation
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
authority = admission_authority(Path(sys.argv[1]))
journal = _journal(Path(sys.argv[2]), "mobile-test")
phase = sys.argv[3]
with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
    review = _review_mobile_operation(session, journal)
    with journal._locked(exclusive=False) as fd:
        scope = journal._records(fd)[0].evidence["inventory_digest"]
    session.prepare_mobile_consolidation(journal, source_config=review.source_selector,
        target_config=review.target_selector, selections=DiscoverySelections(), approved_scope=scope,
        source_fingerprint=review.source_fingerprint, destination_witnesses=review.destination_witnesses,
        destination_sidecars=review.destination_sidecars, reviewed_preview=review)
    append = journal._append
    def fault(fd, event, evidence):
        if phase == "unwitnessed" and event == "mobile_path_staged":
            os._exit(73)
        result = append(fd, event, evidence)
        if (phase, event) in {("intent", "mobile_path_intended"), ("staged", "mobile_path_staged"), ("observed", "mobile_path_observed")}:
            os._exit(73)
        return result
    journal._append = fault
    publish = native_files.publish_new
    def published(*args, **kwargs):
        result = publish(*args, **kwargs)
        if phase == "published":
            os._exit(73)
        return result
    native_files.publish_new = published
    with session.mobile_consolidation_owner(journal, "db.chachanotes.primary"):
        raise AssertionError("death seam was not reached")
"""
    import textwrap

    script = (
        "try:\n"
        + textwrap.indent(script, "    ")
        + "\nexcept BaseException:\n    import traceback; traceback.print_exc(); raise\n"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(authority.control_root.parent),
            str(journal.root.parent),
            phase,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 73, child.stdout + child.stderr
    assert bootstrap.startup_permission(config, authority.control_root.parent) == (
        False,
        "recovery_pending",
    )
    with journal._locked(exclusive=False) as fd:
        records = journal._records(fd)
    intent = next(row for row in records if row.event == "mobile_path_intended")
    temporary = Path(intent.evidence["temporary"])
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        if phase in {"intent", "unwitnessed"}:
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mobile_transition_unsettled"
            ):
                _prepare_review(fresh, journal, config, preview, inventory)
            assert temporary.exists() == (phase == "unwitnessed")
        else:
            _prepare_review(fresh, journal, config, preview, inventory)
            with fresh.mobile_consolidation_owner(
                journal, "db.chachanotes.primary"
            ) as db:
                assert db.get_all_conversation_ids() == []
            witnessed = next(
                row.evidence["temporary"]
                for row in records
                if row.event == "mobile_path_staged"
            )
            assert stores.chat_path.parent.parent.stat().st_ino == witnessed["inode"]


def test_generic_generation_reads_still_refuse_mobile_pending(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import generation_witnesses, storage_admission

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        arguments = (
            preview.destination.chat_path,
            authority.control_root.parent,
            ("mobile.profile",),
            config,
        )
        with pytest.raises(ValueError, match="projection_generation_unavailable"):
            generation_witnesses._paired_witnesses(*arguments)
        with session._discovery_reads():
            assert generation_witnesses._paired_witnesses(*arguments) == []
            with pytest.raises(ValueError, match="projection_generation_unavailable"):
                generation_witnesses._paired_witnesses(
                    *arguments[:-1], config.with_name("foreign.toml")
                )
        assert not storage_admission._mobile_discovery_pending(
            arguments[1], arguments[2], config
        )


def test_authentic_plain_legacy_profile_has_unused_default_key_and_capture_is_complete(
    tmp_path, monkeypatch
):
    from contextlib import closing
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        source = preview.source_dir / "chat.db"
        assert source.is_file(), preview.source_dir
        with (
            session._discovery_reads(private_sqlite=True),
            closing(
                connect_private_sqlite(
                    "recovery.mobile.preview", source, read_only=True
                )
            ) as connection,
        ):
            context = connection.execute(
                "SELECT fingerprint_key_id FROM rag_identity_context WHERE context_name='default'"
            ).fetchone()
            assert context and context[0].startswith("fpkey_")

            class ReadOnlyDatabase:
                def get_connection(self):
                    return connection

            repository = object.__new__(CitationTraceRepository)
            repository.db = ReadOnlyDatabase()
            assert not repository.fingerprint_bearing_rows_exist()
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        from tldw_chatbook.Backup_Recovery import credentials

        before = {
            path: (path.read_bytes(), path.stat().st_mode)
            for path in preview.capture_scope
            if path.is_file()
        }
        backend = credentials._credential_store()._keyring
        assert backend.values == {}
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle:
            assert (
                lifecycle.source.path.is_file() and lifecycle.canonical.path.is_file()
            )
            assert not lifecycle.manual_recovery_required
        assert backend.values == {} and backend.deleted == []
        assert before == {
            path: (path.read_bytes(), path.stat().st_mode) for path in before
        }
        assert len(tuple(destination.iterdir())) == 2


def test_credential_acknowledgement_is_immutable_partitioned_preparation(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        issues = (("legacy", ("credential_manual_recovery_required:" + "a" * 64,)),)
        reviewed = replace(preview, credential_acknowledgements=issues)
        _prepare_review(session, journal, config, reviewed, inventory)
        with journal._locked(exclusive=False) as fd:
            record = journal._records(fd)[0]
        assert record.evidence["credential_review"] == [
            {"purpose": "legacy", "issues": list(issues[0][1])}
        ]
        with pytest.raises(ValueError, match="mobile_credential_review_changed"):
            _prepare_review(session, journal, config, preview, inventory)


@pytest.mark.parametrize(
    "kind", ["missing", "unreadable", "unlock_required", "excluded"]
)
def test_nonmanual_credential_acknowledgements_are_never_native_mobile_consent(
    tmp_path, monkeypatch, kind
):
    from dataclasses import replace

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        reviewed = replace(
            preview,
            credential_acknowledgements=(
                ("legacy", ("credential_" + kind + ":" + "a" * 64,)),
            ),
        )
        with pytest.raises(ValueError, match="mobile_credential_review_invalid"):
            _prepare_review(session, journal, config, reviewed, inventory)
        with journal._locked(exclusive=False) as fd:
            assert journal._records(fd) == []


def test_chat_exports_are_inert_until_explicit_runtime_service_access():
    import subprocess
    import sys

    script = r"""
import os, sys
assert os.environ.get("TLDW_TEST_CONFIG_ROOT")
from Tests.real_profile_guard import install
install()
import tldw_chatbook.Chat as chat
assert "tldw_chatbook.config" not in sys.modules
assert "tldw_chatbook.runtime_policy.bootstrap" not in sys.modules
from tldw_chatbook.Chat import ServerChatConversationService, ServerChatLoopService
from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService as Conversation
from tldw_chatbook.Chat.server_chat_loop_service import ServerChatLoopService as Loop
assert ServerChatConversationService is Conversation is chat.ServerChatConversationService
assert ServerChatLoopService is Loop is chat.ServerChatLoopService
assert {"ServerChatConversationService", "ServerChatLoopService"} <= set(chat.__all__)
try:
    chat.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown export was accepted")
assert ServerChatLoopService(client=object()).client is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_credential_preflight_has_no_claim_pending_or_publication(
    tmp_path, monkeypatch
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _preview_mobile_credential_issues,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        issues = _preview_mobile_credential_issues(
            session, preview, inventory.scope_digest, cancel=Event()
        )
        assert tuple(purpose for purpose, _ in issues) == ("legacy", "canonical")
        assert all(
            all(issue.startswith("credential_") for issue in rows) for _, rows in issues
        )
        with journal._locked(exclusive=False) as fd:
            assert journal._records(fd) == []
        assert bootstrap.startup_permission(config, authority.control_root.parent)[0]
        assert preview.credential_acknowledgements == ()


def _seed_existing_citation_credentials(session, preview):
    """Dedicated credential fixtures; plain mobile capture never calls this helper."""
    import base64
    import secrets
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.credential_policies import CITATION_SERVICE
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    backend = credentials._credential_store()._keyring
    with session._discovery_reads(private_sqlite=True):
        for path in (preview.source_dir / "chat.db", preview.destination.chat_path):
            with closing(
                connect_private_sqlite("recovery.mobile.preview", path, read_only=True)
            ) as connection:
                key_id = connection.execute(
                    "SELECT fingerprint_key_id FROM rag_identity_context WHERE context_name='default'"
                ).fetchone()[0]
            backend.set_password(
                CITATION_SERVICE,
                key_id,
                base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
            )
    return backend


def test_exact_manual_credential_acknowledgements_capture_actual_encrypted_material(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _preview_mobile_credential_issues,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        backend = _seed_existing_citation_credentials(session, preview)
        original_keys = dict(backend.values)
        issues = _preview_mobile_credential_issues(
            session, preview, inventory.scope_digest, cancel=Event()
        )
        assert all(
            rows
            and all(
                issue.startswith("credential_manual_recovery_required:")
                for issue in rows
            )
            for _, rows in issues
        )
        reviewed = replace(preview, credential_acknowledgements=issues)
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            reviewed,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle:
            assert (
                lifecycle.source.path.is_file() and lifecycle.canonical.path.is_file()
            )
            assert lifecycle.credential_issues == issues
            assert lifecycle.manual_recovery_required
        assert backend.values == original_keys


@pytest.mark.parametrize(
    "consent", ["implicit", "wrong_partition", "changed_set", "unreadable"]
)
def test_native_capture_rechecks_exact_manual_credential_consent(
    tmp_path, monkeypatch, consent
):
    from dataclasses import replace
    from threading import Event

    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _preview_mobile_credential_issues,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        backend = _seed_existing_citation_credentials(session, preview)
        issues = _preview_mobile_credential_issues(
            session, preview, inventory.scope_digest, cancel=Event()
        )
        if consent == "wrong_partition":
            issues = (("legacy", issues[1][1]), ("canonical", issues[0][1]))
        elif consent == "changed_set":
            issues = (
                ("legacy", ("credential_manual_recovery_required:" + "a" * 64,)),
                ("canonical", issues[1][1]),
            )
        elif consent == "unreadable":
            backend.values.update({key: "not-a-key" for key in backend.values})
        reviewed = (
            preview
            if consent == "implicit"
            else replace(preview, credential_acknowledgements=issues)
        )
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with (
            pytest.raises(CaptureReviewRequired) as error,
            _captured_mobile(
                session,
                journal,
                reviewed,
                inventory.scope_digest,
                destination=destination,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ),
        ):
            pytest.fail("credential acknowledgement changed native coverage")
        if consent == "unreadable":
            assert any(
                issue.startswith("credential_unreadable:")
                for issue in error.value.issues
            )
        assert not tuple(destination.iterdir())


@pytest.mark.parametrize("source_state", ["absent", "empty"])
def test_empty_source_has_native_no_data_proof_without_legacy_creation(
    tmp_path, monkeypatch, source_state
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch, source_state=source_state) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        assert not preview.has_data
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle:
            assert lifecycle.source is None and lifecycle.canonical.path.is_file()
        assert preview.source_dir.exists() is (source_state == "empty")
        assert (
            not preview.source_dir.exists() or list(preview.source_dir.iterdir()) == []
        )
        with journal._locked(exclusive=False) as fd:
            events = journal._records(fd)
        assert sum(row.event == "mobile_source_empty" for row in events) == 1
        assert [
            row.evidence["purpose"]
            for row in events
            if row.event == "mobile_capture_verified"
        ] == ["canonical"]


def test_source_claim_retains_actual_process_lock_and_refuses_replaced_receipt(
    tmp_path, monkeypatch
):
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _claim_source

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        with _claim_source(session, journal, preview) as claim:
            script = "import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            child = subprocess.run(
                [sys.executable, "-c", script, str(claim.lock_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            assert child.returncode != 0 and "BlockingIOError" in child.stderr
            original = claim.path.read_bytes()
            claim.path.rename(claim.path.with_name("retained-original-claim"))
            claim.path.write_bytes(original)
            claim.path.chmod(0o600)
            with pytest.raises(ValueError, match="mobile_source_claim_changed"):
                claim.check()
        child = subprocess.run(
            [sys.executable, "-c", script, str(claim.lock_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert child.returncode == 0, child.stderr


def test_wal_rollback_captures_committed_rows_without_source_or_sidecar_mutation(
    tmp_path, monkeypatch
):
    import json
    import sqlite3
    import zipfile
    from contextlib import closing
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _read_mobile_archive,
    )

    with _capture_fixture(tmp_path, monkeypatch, wal=True) as (
        session,
        journal,
        _,
        stores,
        preview,
        inventory,
        _,
    ):
        paths = (
            *preview.capture_scope,
            stores.workspace_path,
            stores.workspace_path.with_name(stores.workspace_path.name + "-wal"),
            stores.workspace_path.with_name(stores.workspace_path.name + "-shm"),
        )
        before = {
            path: (path.read_bytes(), path.stat().st_mode)
            for path in paths
            if path.is_file()
        }
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as lifecycle,
            _read_mobile_archive(
                lifecycle.canonical.path, b"test-private-mobile-rollback", Event()
            ) as sealed,
        ):
            doc = json.loads(sealed.manifest_bytes)
            payload = next(
                row for row in doc["files"] if row["owner_id"] == "db.workspaces"
            )
            candidate = tmp_path / "inspected-workspace.sqlite"
            with zipfile.ZipFile(sealed.path) as archive:
                candidate.write_bytes(archive.read(payload["payload"]))
            with closing(
                sqlite3.connect(candidate.as_uri() + "?mode=ro&immutable=1", uri=True)
            ) as connection:
                assert connection.execute(
                    "SELECT operation_id FROM mobile_consolidation_batches WHERE operation_id='test-owned-wal'"
                ).fetchone() == ("test-owned-wal",)
        assert before == {
            path: (path.read_bytes(), path.stat().st_mode) for path in before
        }


def test_live_partition_refuses_copied_session_scope_journal_and_retirement(
    tmp_path, monkeypatch
):
    import copy
    from threading import Event

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _MobileCapturePartition,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as lifecycle:
            partition = next(
                scope
                for scope in session._scopes
                if type(scope) is _MobileCapturePartition
            )
            with pytest.raises(ValueError, match="mobile_native_partition_required"):
                copy.copy(partition).check()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="maintenance_session_inactive"
            ):
                copy.copy(session)._check()
            partition.journal = Journal(journal.root.parent, journal.operation_id)
            with pytest.raises(ValueError, match="mobile_native_partition_required"):
                partition.check()
            partition.journal = journal
        with pytest.raises(ValueError, match="mobile_source_claim_inactive"):
            lifecycle.check()


def test_source_close_failure_keeps_real_lock_and_native_quarantine_until_process_exit(
    tmp_path, monkeypatch
):
    import subprocess
    import sys

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        _prepare_review(session, journal, config, preview, inventory)
    script = r"""
import os,sys,subprocess
from pathlib import Path
assert os.environ.get("TLDW_TEST_CONFIG_ROOT")
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _review_mobile_operation,_claim_source
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
authority=admission_authority(Path(sys.argv[1]))
journal=_journal(Path(sys.argv[2]),"mobile-test")
try:
    with authority.maintenance(("bootstrap.unbound","mobile.profile"),2) as session:
        review=_review_mobile_operation(session,journal)
        with journal._locked(exclusive=False) as fd:
            scope=journal._records(fd)[0].evidence["inventory_digest"]
        session.prepare_mobile_consolidation(journal,source_config=review.source_selector,target_config=review.target_selector,selections=DiscoverySelections(),approved_scope=scope,source_fingerprint=review.source_fingerprint,destination_witnesses=review.destination_witnesses,destination_sidecars=review.destination_sidecars,reviewed_preview=review)
        try:
            with _claim_source(session,journal,review) as claim:
                actual_close=storage_admission.os.close
                def failed_close(fd):
                    if fd==claim.descriptor:
                        raise OSError("injected close before actual retirement")
                    return actual_close(fd)
                storage_admission.os.close=failed_close
        except (bootstrap.RecoveryRequired, AdmissionError) as error:
            assert str(error)=="capture_resources_not_retired"
        else:
            raise AssertionError("close failure released native claim")
        child=subprocess.run([sys.executable,"-c","import os,fcntl,sys; fd=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(claim.lock_path)],capture_output=True,text=True,check=False)
        assert child.returncode!=0 and "BlockingIOError" in child.stderr
except (bootstrap.RecoveryRequired, AdmissionError) as error:
    assert str(error)=="capture_resources_not_retired"
else:
    raise AssertionError("failed retirement released maintenance")
assert storage_admission._failed_capture_holds
os._exit(73)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(authority.control_root.parent),
            str(journal.root.parent),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert child.returncode == 73, child.stdout + child.stderr
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        _prepare_review(fresh, journal, config, preview, inventory)
        from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
            _claim_source,
        )

        with _claim_source(fresh, journal, preview) as claim:
            claim.check()


def test_cold_native_capture_resume_reuses_witnessed_ciphertext(tmp_path, monkeypatch):
    import subprocess
    import sys
    from threading import Event

    from tldw_chatbook.Backup_Recovery import archive_writer
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        authority,
    ):
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)

        def interrupted(*args, **kwargs):
            raise OSError("injected prepublication interruption")

        with monkeypatch.context() as fault:
            fault.setattr(archive_writer, "publish_new", interrupted)
            with (
                pytest.raises(OSError, match="prepublication"),
                _captured_mobile(
                    session,
                    journal,
                    preview,
                    inventory.scope_digest,
                    destination=destination,
                    password=b"test-private-mobile-rollback",
                    cancel=Event(),
                ),
            ):
                pytest.fail("unverified archive yielded lifecycle")
        with journal._locked(exclusive=False) as fd:
            staged = journal._records(fd)[-1]
        assert staged.event == "mobile_capture_staged"
    script = r"""
import os,sys
from pathlib import Path
from threading import Event
assert os.environ.get("TLDW_TEST_CONFIG_ROOT")
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap,profile_paths,credentials
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
profile_paths.default_config_path=lambda:Path(sys.argv[3])
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
class EmptyIsolatedKeyring:
    def get_password(self,*args): return None
store=KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
credentials._credential_store=lambda:store
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
install_adapters()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _resume_captured_mobile
journal=_journal(Path(sys.argv[2]),"mobile-test")
with admission_authority(Path(sys.argv[1])).maintenance(("bootstrap.unbound","mobile.profile"),2) as session:
    with _resume_captured_mobile(session,journal,destination=Path(sys.argv[4]),password=b'test-private-mobile-rollback',cancel=Event()) as lifecycle:
        assert lifecycle.source.path.stat().st_ino==int(sys.argv[5])
        assert lifecycle.canonical.path.is_file()
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(authority.control_root.parent),
            str(journal.root.parent),
            str(tmp_path / "unused-default-config" / "config.toml"),
            str(destination),
            str(staged.evidence["temporary"]["inode"]),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    with journal._locked(exclusive=False) as fd:
        assert (
            sum(row.event == "mobile_capture_verified" for row in journal._records(fd))
            == 2
        )


@pytest.mark.parametrize("published", [False, True])
def test_native_creation_recovery_refuses_foreign_replacement(
    tmp_path, monkeypatch, published
):
    from tldw_chatbook.Backup_Recovery import bootstrap, native_files

    with prepared(tmp_path, monkeypatch, fresh=True) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        _prepare_review(session, journal, config, preview, inventory)
        publish = native_files.publish_new

        def interrupted(source, target, **kwargs):
            if target == stores.chat_path.parent.parent:
                if published:
                    publish(source, target, **kwargs)
                raise OSError("injected native publication interruption")
            return publish(source, target, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(native_files, "publish_new", interrupted)
            with (
                pytest.raises(OSError, match="native publication"),
                session.mobile_consolidation_owner(journal, "db.chachanotes.primary"),
            ):
                pytest.fail("interrupted creation exposed owner")
        with journal._locked(exclusive=False) as fd:
            evidence = journal._records(fd)[-1].evidence
        path = (
            stores.chat_path.parent.parent
            if published
            else Path(evidence["temporary"]["path"])
        )
        path.rename(path.with_name(path.name + "-retained-original"))
        path.mkdir(mode=0o700)
        foreign = path.stat().st_ino
    with (
        authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh,
        pytest.raises((bootstrap.RecoveryRequired, ValueError)),
    ):
        _prepare_review(fresh, journal, config, preview, inventory)
    assert path.stat().st_ino == foreign


def test_mobile_capture_refuses_explicitly_excluded_citation_material(
    tmp_path, monkeypatch
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch, excluded_citation=True) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with (
            pytest.raises(ValueError, match="credential_material_excluded"),
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ),
        ):
            pytest.fail("excluded material exposed a complete lifecycle")
        with journal._locked(exclusive=False) as fd:
            records = journal._records(fd)
        assert not any(
            row.event == "mobile_capture_verified"
            and row.evidence["purpose"] == "canonical"
            for row in records
        )


@pytest.mark.parametrize(
    "field",
    [
        "has_data",
        "source_dir",
        "destination",
        "source_witnesses",
        "capture_scope",
        "rollback_scope",
        "candidate_destinations",
        "counts",
        "mappings",
    ],
)
def test_native_preparation_rejects_changed_preview_authority_before_effects(
    tmp_path, monkeypatch, field
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
    )

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        authority,
    ):
        assert preview.has_data
        changes = {
            "has_data": False,
            "source_dir": tmp_path / "foreign-source",
            "destination": CanonicalConsoleStores(
                *(tmp_path / leaf for leaf in ("c", "w", "r"))
            ),
            "source_witnesses": (),
            "capture_scope": (),
            "rollback_scope": (),
            "candidate_destinations": (),
            "counts": (),
            "mappings": (),
        }
        forged = replace(preview, **{field: changes[field]})
        assert getattr(forged, field) != getattr(preview, field)
        with pytest.raises(ValueError):
            _prepare_review(session, journal, config, forged, inventory)
        with journal._locked(exclusive=False) as fd:
            assert journal._records(fd) == []
        assert bootstrap.startup_permission(config, authority.control_root.parent)[0]


@pytest.mark.parametrize("entry", ["preflight", "acknowledged_capture"])
def test_excluded_citation_with_actual_manual_issues_refuses(
    tmp_path, monkeypatch, entry
):
    from dataclasses import replace
    from threading import Event

    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _preview_mobile_credential_issues,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        stores,
        preview,
        inventory,
        authority,
    ):
        _, service, names = GENERATION_KEYRINGS[0]
        backend = credentials._credential_store()._keyring
        backend.set_password(service, names[0], "test-owned-generation-key")
        original_keys = dict(backend.values)
        issues = _preview_mobile_credential_issues(
            session, preview, inventory.scope_digest, cancel=Event()
        )
        assert all(
            rows
            and all(
                value.startswith("credential_manual_recovery_required:")
                for value in rows
            )
            for _, rows in issues
        )
    # Change only test-owned setup through the real owner outside maintenance.
    # The fresh review sees this state; existing exact native manual issues remain
    # insufficient consent for excluded material and cannot suppress its validation.
    owner = CharactersRAGDB(stores.chat_path, "excluded-with-manual-fixture")
    with owner.transaction() as cursor:
        cursor.execute(
            "UPDATE rag_identity_context SET fingerprint_key_id=? WHERE context_name='default'",
            (credentials.RECOVERY_SETUP_REQUIRED,),
        )
    owner.close_connection()
    with _preview_reads():
        inventory = discover((config,))
    preview = preview_selected_mobile(config, config)
    reviewed = replace(preview, credential_acknowledgements=issues)
    destination = tmp_path / "archives"
    destination.mkdir(mode=0o700)
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as session:
        if entry == "preflight":
            with pytest.raises(ValueError, match="credential_material_excluded"):
                _preview_mobile_credential_issues(
                    session, reviewed, inventory.scope_digest, cancel=Event()
                )
        else:
            with (
                pytest.raises(ValueError, match="credential_material_excluded"),
                _captured_mobile(
                    session,
                    journal,
                    reviewed,
                    inventory.scope_digest,
                    destination=destination,
                    password=b"test-private-mobile-rollback",
                    cancel=Event(),
                ),
            ):
                pytest.fail("manual issues hid excluded material")
        with journal._locked(exclusive=False) as fd:
            records = journal._records(fd)
        assert not any(
            row.event == "mobile_capture_verified"
            and row.evidence["purpose"] == "canonical"
            for row in records
        )
        assert backend.values == original_keys and backend.deleted == []
