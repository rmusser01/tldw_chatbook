"""Imported history has native custody without importing execution authority."""

from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


def test_native_history_capability_preserves_bytes_and_retires(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import MaintenanceSession

    assert hasattr(MaintenanceSession, "mobile_history_files"), (
        "native history publication needs its own exact maintenance capability"
    )
    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            with session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files:
                associations = files.publish()
                assert associations
                assert any(row.sha256 for row in associations)
                assert files.resources == []
                for row in associations:
                    if row.sha256 is not None:
                        assert (files.location / row.location).read_bytes() == (
                            Path(row.original_root) / row.relative_path
                        ).read_bytes()
                assert any(
                    path.read_bytes() == b'{"event":"private retained artifact"}\n'
                    for path in files.location.rglob("events.jsonl")
                )
            with pytest.raises(ValueError, match="inactive"):
                files.check()


@pytest.mark.parametrize(
    "phase", ["before_stage", "before_publication", "after_publication"]
)
def test_history_interruption_recovers_only_witnessed_tree(
    tmp_path, monkeypatch, phase
):
    from tldw_chatbook.Backup_Recovery import native_files
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _resume_captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, authority = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        publish, create = (
            native_files.publish_new,
            native_files.create_private_directory,
        )

        def fail_publish(source, target, **kwargs):
            if source.name.startswith(".mobile-history-"):
                if phase == "after_publication":
                    publish(source, target, **kwargs)
                raise OSError("injected history interruption")
            return publish(source, target, **kwargs)

        def fail_create(target, **kwargs):
            if target.name.startswith(".mobile-history-"):
                raise OSError("injected history interruption")
            return create(target, **kwargs)

        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files,
        ):
            with monkeypatch.context() as fault:
                fault.setattr(
                    native_files,
                    "create_private_directory"
                    if phase == "before_stage"
                    else "publish_new",
                    fail_create if phase == "before_stage" else fail_publish,
                )
                with pytest.raises(OSError, match="injected history"):
                    files.publish()
            with journal._locked(exclusive=False) as fd:
                records = journal._records(fd)
            last = records[-1]
            assert last.event == (
                "mobile_history_intended"
                if phase == "before_stage"
                else "mobile_history_staged"
            )
            location, members = files.location, files.members
    with (
        authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh,
        _resume_captured_mobile(
            fresh,
            journal,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured,
    ):
        with fresh.mobile_history_files(
            captured, password=b"test-private-mobile-rollback", cancel=Event()
        ) as files:
            assert files.publish() == members
            assert files.location == location
            assert files.publish() == members
            with journal._locked(exclusive=False) as fd:
                rows = journal._records(fd)
            verified = rows[-1]
            assert verified.event == "mobile_history_verified"
            if phase != "before_stage":
                assert (
                    verified.evidence["artifact"]["inode"]
                    == last.evidence["artifact"]["inode"]
                )
        with captured.owner("db.agent_runs") as runs, runs.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM mobile_consolidation_batches"
                ).fetchone()[0]
                == 0
            )


@pytest.mark.parametrize("phase", ["pre_witness", "staged", "published"])
def test_history_retains_unknown_and_foreign_objects(tmp_path, monkeypatch, phase):
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import native_files
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files,
        ):
            if phase == "pre_witness":
                original = native_files.create_private_directory

                def failed(target, **kwargs):
                    original(target, **kwargs)
                    if target.name.startswith(".mobile-history-"):
                        raise OSError("injected pre-witness")

                seam = "create_private_directory"
            else:
                original = native_files.publish_new

                def failed(source, target, **kwargs):
                    if source.name.startswith(".mobile-history-"):
                        if phase == "published":
                            original(source, target, **kwargs)
                        raise OSError("injected publication")
                    return original(source, target, **kwargs)

                seam = "publish_new"
            with monkeypatch.context() as fault:
                fault.setattr(native_files, seam, failed)
                with pytest.raises(OSError, match="injected"):
                    files.publish()
            with journal._locked(exclusive=False) as fd:
                intent = next(
                    row.evidence
                    for row in journal._records(fd)
                    if row.event == "mobile_history_intended"
                )
            target = Path(
                intent["publication"] if phase == "published" else intent["temporary"]
            )
            if phase != "pre_witness":
                target.rename(target.with_name(target.name + "-retained-original"))
                target.mkdir(mode=0o700)
            marker = target / "foreign"
            marker.write_bytes(b"must retain")
            with pytest.raises((ValueError, FileExistsError)):
                files.publish()
            assert marker.read_bytes() == b"must retain"


def test_copied_foreign_changed_and_retired_history_capabilities_refuse(
    tmp_path, monkeypatch
):
    import copy
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            for owner, capture in (
                (copy.copy(session), captured),
                (session, copy.copy(captured)),
            ):
                with (
                    pytest.raises((ValueError, RuntimeError)),
                    owner.mobile_history_files(
                        capture,
                        password=b"test-private-mobile-rollback",
                        cancel=Event(),
                    ),
                ):
                    pytest.fail("copied scope admitted")
            with session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files:
                with pytest.raises(ValueError, match="inactive"):
                    copy.copy(files).publish()
                with (
                    pytest.raises((ValueError, RuntimeError)),
                    captured.owner("db.agent_runs"),
                ):
                    pytest.fail("overlapping DB scope admitted")
                original = files.location
                files.location = Path("/private/tmp/outside-history")
                with pytest.raises(ValueError, match="changed"):
                    files.publish()
                files.location = original
                assert files.publish()
            with pytest.raises(ValueError, match="inactive"):
                files.publish()
        with (
            pytest.raises(ValueError),
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ),
        ):
            pytest.fail("retired lifecycle admitted")


@pytest.mark.parametrize("existing", [False, True])
def test_canonical_history_capture_covers_absence_or_existing_tree(
    tmp_path, monkeypatch, existing
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture as fixture
    from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _read_mobile_archive,
    )
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        resolve_archive_groups,
        selected_absent_target_ids,
    )

    old_tree = []
    if existing:
        original = fixture.seed_canonical_console

        def seed(stores):
            result = original(stores)
            leaf = (
                stores.chat_path.parents[2]
                / "data"
                / "shared"
                / "agent_history"
                / "imports"
                / ("a" * 64)
                / ("b" * 64)
                / ("c" * 64)
                / "events.jsonl"
            )
            leaf.parent.mkdir(parents=True, mode=0o700)
            leaf.write_bytes(b"existing imported history\n")
            leaf.chmod(0o600)
            old_tree.append(leaf)
            return result

        monkeypatch.setattr(fixture, "seed_canonical_console", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            with _read_mobile_archive(
                captured.canonical.path, b"test-private-mobile-rollback", Event()
            ) as archive:
                document = ArchiveManifest.model_validate_json(archive.manifest_bytes)
            with session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files:
                files.publish()
                location = files.location
            with session._discovery_reads(private_sqlite=True):
                current = discover((config,))
                after = _preview_selected_mobile(
                    config, config, review_destination=False
                )
            assert after.source_fingerprint == preview.source_fingerprint
            native = [
                row
                for row in current.items
                if row.owner == "agents.history"
                and row.path
                and (row.path == location or location in row.path.parents)
            ]
            assert any(row.path.name == "events.jsonl" for row in native)
            config_id = next(
                row.logical_id for row in document.files if row.owner_id == "config"
            )
            if existing:
                import hashlib

                assert old_tree[0].read_bytes() == b"existing imported history\n"
                assert any(
                    row.owner_id == "agents.history"
                    and row.sha256
                    == hashlib.sha256(b"existing imported history\n").hexdigest()
                    for row in document.files
                )
                assert any(row.path == old_tree[0] for row in current.items)
            else:
                scope = resolve_archive_groups(document, None)
                absent = selected_absent_target_ids(
                    document, scope, current, ((config_id, config),), ()
                )
                assert {row.logical_id for row in native} <= absent
                assert any(
                    row.owner_id == "agents.history"
                    and row.status == "unused"
                    and row.logical_id in absent
                    for row in document.producer_inventory
                )


_COLD_HISTORY = r"""
import os,sys
from pathlib import Path
from threading import Event
assert os.environ.get("TLDW_TEST_CONFIG_ROOT")
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap,profile_paths,credentials,native_files
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
profile_paths.default_config_path=lambda:Path(sys.argv[3])
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
class EmptyIsolatedKeyring:
    def get_password(self,*args): return None
credentials._credential_store=lambda:KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
install_adapters()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _resume_captured_mobile
journal=_journal(Path(sys.argv[2]),"mobile-test")
phase=sys.argv[5]
publish=native_files.publish_new
def crash(source,target,**kwargs):
    if source.name.startswith(".mobile-history-"):
        if phase=="after": publish(source,target,**kwargs)
        os._exit(71)
    return publish(source,target,**kwargs)
if phase in ("before","after"):
    native_files.publish_new=crash
with admission_authority(Path(sys.argv[1])).maintenance(("bootstrap.unbound","mobile.profile"),2) as session:
    with _resume_captured_mobile(session,journal,destination=Path(sys.argv[4]),password=b'test-private-mobile-rollback',cancel=Event()) as captured:
        with session.mobile_history_files(captured,password=b'test-private-mobile-rollback',cancel=Event()) as files:
            members=files.publish()
            assert any((files.location/row.location).read_bytes()==b'{"event":"private retained artifact"}\n' for row in members if row.sha256)
"""


def test_process_death_preserves_staged_inode_through_cold_native_resume(
    tmp_path, monkeypatch
):
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, authority = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            captured.check()
    arguments = [
        str(authority.control_root.parent),
        str(journal.root.parent),
        str(tmp_path / "unused-default-config" / "config.toml"),
        str(output),
    ]
    inode = None
    for phase in ("before", "after", "finish"):
        child = subprocess.run(
            [sys.executable, "-c", _COLD_HISTORY, *arguments, phase],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert child.returncode == (0 if phase == "finish" else 71), (
            child.stdout + child.stderr
        )
        with journal._locked(exclusive=False) as fd:
            records = journal._records(fd)
        staged = next(row for row in records if row.event == "mobile_history_staged")
        if inode is None:
            inode = staged.evidence["artifact"]["inode"]
        assert staged.evidence["artifact"]["inode"] == inode
        assert records[-1].event == (
            "mobile_history_verified" if phase == "finish" else "mobile_history_staged"
        )
        if phase == "finish":
            assert records[-1].evidence["artifact"]["inode"] == inode


def test_only_verified_member_has_bounded_private_history_read(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files,
        ):
            member = next(row for row in files.members if row.sha256)
            assert hasattr(files, "read_member"), (
                "verified file custody needs an exact private read contract"
            )
            with pytest.raises(ValueError, match="unverified"):
                files.read_member(member.logical_id)
            files.publish()
            assert files.read_member(member.logical_id, offset=1, limit=5) == b'"even'
            with pytest.raises(ValueError, match="member"):
                files.read_member("../../foreign")
            for arguments in ({"offset": -1}, {"limit": 2**21}):
                with pytest.raises(ValueError, match="range"):
                    files.read_member(member.logical_id, **arguments)
            path = files.location / member.location
            path.write_bytes(b"changed")
            with pytest.raises(ValueError, match="changed"):
                files.read_member(member.logical_id)


@pytest.mark.parametrize(
    "phase", ["copy", "publication", "qualification", "observe_before", "observe_after"]
)
def test_failed_history_descriptor_close_retains_actual_fd_and_maintenance(
    tmp_path, monkeypatch, phase
):
    import subprocess
    import sys
    import textwrap

    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, authority = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            captured.check()
    prefix, body = _COLD_HISTORY.split("with admission_authority(", 1)
    fault = r"""
from tldw_chatbook.Backup_Recovery import storage_admission, mobile_history_files
actual_close=storage_admission.os.close
failed=[]
publication_fd=[]
payload_fd={}
publishing=[]
qualifying=[]
observing=[]
published=[]
actual_qualified=native_files.qualified_for
actual_observe=mobile_history_files.observe_artifact
def record_qualified(*args,**kwargs):
    qualifying.append(True)
    try: return actual_qualified(*args,**kwargs)
    finally: qualifying.pop()
def record_observe(path,**kwargs):
    observing.append(path)
    try: return actual_observe(path,**kwargs)
    finally: observing.pop()
native_files.qualified_for=record_qualified
mobile_history_files.observe_artifact=record_observe
actual_open=storage_admission.os.open
actual_publish=native_files.publish_new
def record_open(path,*args,**kwargs):
    fd=actual_open(path,*args,**kwargs)
    if publishing and str(path)==publishing[-1]: publication_fd.append(fd)
    if args and args[0] & os.O_WRONLY and args[0] & os.O_CREAT:
        info=os.fstat(fd)
        payload_fd[fd]=(info.st_dev,info.st_ino)
    return fd
def record_publish(source,target,**kwargs):
    publishing.append(source.name)
    try:
        result=actual_publish(source,target,**kwargs)
        published.append(target)
        return result
    finally: publishing.pop()
storage_admission.os.open=record_open
native_files.publish_new=record_publish
def fail_close(fd):
    if failed:
        assert fd != failed[0], "native descriptor close was retried"
        return actual_close(fd)
    scope=getattr(storage_admission._local,"mobile_history_scope",None)
    if ((phase=="copy" and scope is not None and payload_fd.get(fd)==(os.fstat(fd).st_dev,os.fstat(fd).st_ino) and any(fd in resource.fds for resource in scope.resources))
        or (phase=="publication" and fd in publication_fd)
        or (phase=="qualification" and publishing and qualifying)
        or (phase=="observe_before" and observing and observing[-1].name.startswith(".mobile-history-"))
        or (phase=="observe_after" and observing and observing[-1] in published)):
        assert not failed, "native descriptor close was retried"
        failed.append(fd)
        raise OSError("injected actual history descriptor close failure")
    return actual_close(fd)
storage_admission.os.close=fail_close
"""
    script = (
        prefix
        + "\ntry:\n"
        + textwrap.indent(
            "with admission_authority("
            + body.replace(
                "            members=files.publish()",
                textwrap.indent(fault, "            ")
                + "\n            members=files.publish()",
            ),
            "    ",
        )
        + r"""
except Exception as error:
    assert str(error) in {"capture_resources_not_retired", "injected actual history descriptor close failure", "qualification_unavailable"},repr(error)
else:
    raise AssertionError("uncertain retirement released maintenance")
assert len(failed)==1,failed
assert os.fstat(failed[0]).st_size>0
if phase=="copy":
    assert os.fstat(failed[0]).st_size==len(b'{"event":"private retained artifact"}\n')
assert storage_admission._failed_capture_holds
# The actual exclusive gate remains held, not merely a failed-close flag.
import subprocess
probe=r"from Tests.real_profile_guard import install; install(); from pathlib import Path; from tldw_chatbook.Backup_Recovery.control_records import admission_authority; import sys; a=admission_authority(Path(sys.argv[1])); c=a.maintenance(('bootstrap.unbound','mobile.profile'),0.1); c.__enter__()"
child=subprocess.run([sys.executable,"-c",probe,sys.argv[1]],capture_output=True,text=True,check=False,timeout=15)
assert child.returncode!=0,child.stdout+child.stderr
assert "maintenance_busy" in child.stderr or "admission_timeout" in child.stderr,child.stderr
os._exit(73)
"""
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(authority.control_root.parent),
            str(journal.root.parent),
            str(tmp_path / "unused-default-config" / "config.toml"),
            str(output),
            phase,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert child.returncode == 73, child.stdout + child.stderr
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:
        fresh._check()


def test_native_imported_history_refuses_missing_run_owner(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_inventory_unused_owners import _AGENTS

    script = _AGENTS.replace(
        "elif route=='retained':",
        "elif route=='imported':\n root=data/'agent_history'/'imports'/('a'*64)/('b'*64)/('c'*64);root.mkdir(parents=True)\n (root/'events.jsonl').write_bytes(b'retained imported history')\nelif route=='retained':",
    )
    _run(tmp_path, "imported", "unused", script=script)


@pytest.mark.parametrize("replacement", ["bytes", "inode"])
def test_history_scope_rechecks_native_selector_before_publication(
    tmp_path, monkeypatch, replacement
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files,
        ):
            before = config.read_bytes()
            if replacement == "inode":
                config.rename(config.with_name("retained-original-config.toml"))
                config.write_bytes(before)
                config.chmod(0o600)
            else:
                config.write_bytes(before + b"\n# changed after capability issuance\n")
            with pytest.raises((ValueError, RuntimeError), match="selector|enrollment"):
                files.publish()
            assert not files.location.exists()


@pytest.mark.parametrize("alias", [False, True])
def test_existing_destination_or_alias_never_grants_publication(
    tmp_path, monkeypatch, alias
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with (
            _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=output,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured,
            session.mobile_history_files(
                captured, password=b"test-private-mobile-rollback", cancel=Event()
            ) as files,
        ):
            files.location.parent.mkdir(mode=448, parents=True)
            foreign = tmp_path / "foreign-history"
            foreign.mkdir(mode=448)
            (foreign / "keep").write_bytes(b"foreign")
            if alias:
                files.location.symlink_to(foreign, target_is_directory=True)
            else:
                files.location.mkdir(mode=448)
                (files.location / "keep").write_bytes(b"foreign")
            with pytest.raises(ValueError, match="occupied"):
                files.publish()
            assert (foreign / "keep").read_bytes() == b"foreign"
            if not alias:
                assert (files.location / "keep").read_bytes() == b"foreign"


def test_history_custody_preserves_native_nofollow_with_real_system_alias(tmp_path):
    import os
    import stat
    import sys

    from tldw_chatbook.Backup_Recovery import native_files
    from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors

    root = tmp_path / "private-history-parent"
    root.mkdir(mode=0o700)
    custody = _CaptureFileDescriptors(None)
    with native_files.pinned_directory(root, _custody=custody) as fd:
        assert os.fstat(fd).st_ino == root.stat().st_ino
    assert custody.fds == []
    custody.retire()

    local_alias = tmp_path / "alias"
    local_alias.symlink_to(root, target_is_directory=True)
    aliases = [local_alias]
    system = Path("/var")
    canonical = Path("/private/var")
    if sys.platform == "darwin":
        info = system.lstat()
        assert stat.S_ISLNK(info.st_mode) and info.st_uid == 0
        assert root.is_relative_to(canonical)
        system_alias = system / root.relative_to(canonical)
        assert system_alias.resolve() == root
        aliases.append(system_alias)
    for alias in aliases:
        custody = _CaptureFileDescriptors(None)
        try:
            with (
                pytest.raises((OSError, ValueError)),
                native_files.pinned_directory(alias, _custody=custody),
            ):
                pytest.fail("native no-follow accepted an alias")
        finally:
            custody.retire()
        assert custody.fds == []
