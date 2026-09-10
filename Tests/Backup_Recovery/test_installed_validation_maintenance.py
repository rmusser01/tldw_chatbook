"""Installed validation retains native maintenance while checking private copies."""

import json
import select
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_installed_restore_validation import (
    published,
    sqlite_publication,
)
from tldw_chatbook.Backup_Recovery.control_records import admission_authority

_WRITER = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
authority = Admission(Path(sys.argv[1]))
print('ready', flush=True)
with authority.normal(('profile',)):
    Path(sys.argv[2]).write_text('admitted writer')
print('finished', flush=True)
"""


@pytest.mark.parametrize("kind", ["raw", "sqlite"])
@pytest.mark.parametrize("failure", [False, True])
def test_actual_validation_keeps_native_writer_blocked_until_callback_retires(
    tmp_path, monkeypatch, kind, failure
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import private_sqlite

    if kind == "sqlite":
        candidate, plan, journal = sqlite_publication(tmp_path)
    else:
        candidate, plan, journal, _, _ = published(tmp_path)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("profile", tuple(dict(plan.destinations).values()))
    main_thread = threading.get_ident()
    children = []
    marker = tmp_path / "writer-finished"
    callbacks = []

    def barrier(path):
        path = Path(path)
        assert candidate in path.parents
        assert path not in dict(plan.restore).values()
        assert threading.get_ident() != main_thread
        assert getattr(storage._local, "maintenance_session", None) is None
        assert getattr(storage._local, "preview_scope", None) is not None
        callbacks.append(path)
        if children:
            return
        child = subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(authority.control_root), str(marker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        assert select.select([child.stdout], [], [], 10)[0]
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            child.communicate(timeout=0.2)
        assert not marker.exists()
        if failure:
            raise OSError("injected callback failure")

    if kind == "sqlite":
        original = private_sqlite.open_recovery_validation

        def checked(owner, path, *, writable):
            barrier(path)
            assert not writable
            return original(owner, path, writable=writable)

        monkeypatch.setattr(private_sqlite, "open_recovery_validation", checked)
    else:
        original = storage._consume_recovery_file

        def checked(owner, path, **kwargs):
            barrier(path)
            return original(owner, path, **kwargs)

        monkeypatch.setattr(storage, "_consume_recovery_file", checked)

    try:
        with authority.maintenance(("profile",), 2) as session:
            session._check()
            if failure:
                with pytest.raises((OSError, ValueError), match="callback|unavailable"):
                    journal.validate_installed(candidate, plan)
            else:
                journal.validate_installed(candidate, plan)
            session._check()
            assert callbacks
            assert children[0].poll() is None
            assert not marker.exists()
            assert not list(candidate.glob("installed-check-*"))
            events = [
                json.loads(path.read_bytes())["event"]
                for path in journal.root.glob("[0-9]*.json")
            ]
            assert ("installed_validated" in events) is not failure
        stdout, stderr = children[0].communicate(timeout=10)
        assert children[0].returncode == 0, stderr
        assert "finished" in stdout
        assert marker.read_text() == "admitted writer"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def _shared_case(root, *, held):
    """Actual study asset and quiz history share the installed core DB file."""
    from threading import Event

    from Tests.Backup_Recovery.test_domain_owners import context, study_store
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Backup_Recovery.publication import publish_candidate
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters

    generator = study_store.__wrapped__(root)
    source, _asset = next(generator)
    try:
        config = context(source, "chachanotes_db_path")
        config["paths"] = {"data_dir": str(root / "source-data")}
        replica = FileNotesReplica(user_data_dir(config) / "file_notes.sqlite")
        replica.close()
        owners = (core_adapters()[0], *recovery_adapters())
        entries = tuple(owner.discover(config)[0] for owner in owners)
        assert len({item.shared_group for item in entries}) == 1
        peers = tuple(
            item
            for owner in install_adapters()
            if owner.owner_id
            in {"notes.file_notes", "notes.sync_bindings", "chat.attachments"}
            for item in owner.discover(config)
        )
        archive, doc = _shared_archive(root, config, (*entries, *peers))
        destination = root / "new-data" / "restored"
        destinations = {
            row["logical_id"]: destination
            for row in doc["directories"]
            if row["parent_id"] is None
        }
        destinations["profile:fixture:paths.data_dir"] = root / "new-data"
        plan = plan_restore(
            archive,
            mode="isolated",
            destinations=destinations,
            target=None,
            profile_names={"fixture": "restored"},
        )
        (root / "control").mkdir(mode=0o700)
        journal = Journal(root / "control", "shared-study")
        candidate = stage_restore(
            archive, plan, root / "work", Event(), journal=journal
        )
        restored = dict(plan.restore)
        assert len({restored[item.logical_id] for item in entries}) == 1
        bootstrap = root / "bootstrap"
        selectors = tuple(path for _, path in plan.selectors)
        register_pending(
            bootstrap,
            journal.operation_id,
            ("profile",),
            journal.root.parent,
            selectors,
        )
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=bootstrap,
            namespaces=("profile",),
            selectors=selectors,
            generation="shared",
        )
        publish_candidate(candidate, plan, journal, None)
        if held:
            authority = admission_authority(bootstrap)
            authority.register(
                "profile", tuple(dict.fromkeys(dict(plan.destinations).values()))
            )
            with authority.maintenance(("profile",), 2) as session:
                journal.validate_installed(candidate, plan)
                session._check()
        else:
            journal.validate_installed(candidate, plan)
        assert all(
            restored[item.logical_id].samefile(restored[entries[0].logical_id])
            for item in entries
        )
        assert journal.recover() == "recovery_required"
    finally:
        generator.close()


def test_interrupted_wait_retires_callback_before_cleanup_or_native_release(
    tmp_path, monkeypatch
):
    from concurrent.futures import Future

    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    candidate, plan, journal, _, _ = published(tmp_path)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("profile", tuple(dict(plan.destinations).values()))
    entered, release, retired = (threading.Event() for _ in range(3))
    original = storage._consume_recovery_file

    def blocked(owner, path, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            assert Path(path).is_file()
            return original(owner, path, **kwargs)
        finally:
            retired.set()

    def interrupted(_future, timeout=None):
        assert entered.wait(5)
        timer.start()
        raise KeyboardInterrupt("interrupted installed validation wait")

    timer = threading.Timer(0.2, release.set)
    monkeypatch.setattr(storage, "_consume_recovery_file", blocked)
    monkeypatch.setattr(Future, "result", interrupted)
    try:
        with authority.maintenance(("profile",), 2) as session:
            with pytest.raises(KeyboardInterrupt, match="interrupted installed"):
                journal.validate_installed(candidate, plan)
            session._check()
            assert retired.is_set()
            assert not list(candidate.glob("installed-check-*"))
            assert all(
                json.loads(path.read_bytes())["event"] != "installed_validated"
                for path in journal.root.glob("[0-9]*.json")
            )
    finally:
        release.set()
        timer.cancel()


@pytest.mark.parametrize("held", [False, True])
def test_actual_study_quiz_core_aliases_validate_as_one_private_file(tmp_path, held):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "shared",
        str(held),
        script="""
import sys
from pathlib import Path
from Tests.Backup_Recovery.test_installed_validation_maintenance import _shared_case
_shared_case(Path.home(), held=sys.argv[2] == 'True')
print('retired and reopened')
""",
    )


def _shared_archive(root, config, entries):
    import shutil
    import zipfile
    from threading import Event

    import toml

    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.capture import _manifest_for
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, Inventory
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    selector = config[DISCOVERY_CONTEXT_KEY].config_path
    selector.write_text(
        toml.dumps(
            {
                key: value
                for key, value in config.items()
                if key != DISCOVERY_CONTEXT_KEY
            }
        )
    )
    entries = (*config_adapter().discover(config), *entries)
    inventory = Inventory(entries, True, "shared-fixture", ())
    stage = root / "serialized"
    (stage / "payload").mkdir(mode=0o700, parents=True)
    staged, aliases = [], {}
    for index, item in enumerate(entries):
        assert item.status == "included"
        path = stage / "payload" / str(index)
        shutil.copyfile(item.path, path)
        staged.append((item, path))
        if item.shared_group:
            aliases.setdefault(item.shared_group, []).append(item.logical_id)
    manifest = _manifest_for(
        inventory,
        staged,
        aliases,
        {
            "root": stage,
            "versions": {},
            "cancel": Event(),
            "mode": "exclude",
            "encrypted": False,
            "limits": ArchiveLimits(),
        },
        (),
    )
    archive_path = root / "shared.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", manifest)
        for _, path in staged:
            archive.write(path, path.relative_to(stage))
    return acquire(
        archive_path, root / "sealed", ArchiveLimits(), None, Event()
    ), json.loads(manifest)
