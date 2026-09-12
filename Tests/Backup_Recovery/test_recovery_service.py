"""App-owned work uses the real reader and keeps immutable visible results."""

from dataclasses import replace
from threading import Event, Thread

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)
from Tests.Backup_Recovery.test_archive_writer import captured
from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import crypto, recovery_service, replacement
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.recovery_copies import delete_recovery_copy
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService


def test_service_inspects_real_archive_without_claiming_restore(tmp_path):
    archive = tmp_path / "source.tldw-backup.zip"
    written = write_archive(captured(tmp_path), archive, password=None, cancel=Event())
    service = RecoveryService(tmp_path / "control")
    try:
        operation = service.start_inspection(archive, password=None)
        result = service.wait(operation, timeout=15)
        assert result["state"] == "succeeded"
        assert result["phase"] == "archive_verified"
        assert result["result"]["sha256"] == written.digest
        assert not result["result"].get("restoration_validated", False)
        with pytest.raises(TypeError):
            result["state"] = "changed"
        with pytest.raises(TypeError):
            result["result"]["sha256"] = "changed"
        assert service.inspection(operation).path.is_file()
        summary = service.summary(operation)
        assert summary["archive_verified"] and summary["format_version"] == 1
        assert "relocations" not in summary
        with pytest.raises(TypeError):
            summary["roots"][0]["logical_id"] = "changed"
        assert service.current() is result
    finally:
        service.close()
    assert archive.is_file()


def test_service_sanitizes_reader_failure_without_losing_source(tmp_path):
    archive = tmp_path / "invalid.tldw-backup.zip"
    archive.write_bytes(b"private invalid archive text")
    service = RecoveryService(tmp_path / "control")
    try:
        operation = service.start_inspection(archive, password=None)
        result = service.wait(operation, timeout=15)
        assert result["state"] == "failed"
        assert result["issues"] == ("backup_operation_failed",)
        assert "private invalid" not in repr(result)
    finally:
        service.close()
    assert archive.read_bytes() == b"private invalid archive text"


def test_service_preserves_exact_nonsecret_capture_review_for_retry(tmp_path):
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired

    service = RecoveryService(tmp_path / "control")
    issues = ("credential_format_unreadable", "credential_missing:config:test-slot")

    def require_review(operation, cancel):
        raise CaptureReviewRequired(issues)

    try:
        operation = service._start("backup", require_review)
        state = service.wait(operation, timeout=15)
        assert state["issues"] == ("review_required",)
        assert state["review_issues"] == issues
    finally:
        service.close()


def test_app_close_keeps_real_ciphertext_held_until_inspection_worker_returns(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, _, _ = case
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        control = tmp_path / "control"
        original = replacement.replace(
            plan,
            candidate,
            control_root=control,
            rollback_password=b"rollback",
            cancel=Event(),
        )
        service = RecoveryService(control)
        entered, release, closed = Event(), Event(), Event()
        acquire = recovery_service.archive_reader.acquire

        def held_acquire(*args, **kwargs):
            archive = acquire(*args, **kwargs)
            entered.set()
            assert release.wait(10)
            return archive

        monkeypatch.setattr(recovery_service.archive_reader, "acquire", held_acquire)
        operation = service.start_copy_inspection(original, password=b"rollback")
        assert entered.wait(15)
        initial = service.status(operation)
        with pytest.raises(ValueError, match="recovery_operation_running"):
            service.start_copy_inspection(original, password=b"rollback")
        service.cancel(operation)
        assert not initial["cancellation_requested"]
        assert service.status(operation)["cancellation_requested"]

        def close():
            service.close()
            closed.set()

        closing = Thread(target=close)
        closing.start()
        try:
            assert not closed.wait(0.1)
            with pytest.raises(ValueError, match="recovery_copy_held"):
                delete_recovery_copy(control, original, user_selected=True)
        finally:
            release.set()
            closing.join(15)
        assert closed.is_set()
        assert service.status(operation)["state"] == "cancelled"
        assert not tuple(control.glob("inspection-*"))
        assert service.recovery_copies()[0].status == "verified"
        with pytest.raises(ValueError, match="recovery_service_closed"):
            service.start_copy_inspection(original, password=b"rollback")


_LIVE_BACKUP = r"""
import asyncio,json,os,sqlite3,sys,threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import archive_writer,archive_reader,storage_admission
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
async def main():
 app=TldwCli();home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
 service=RecoveryService(default_control_root())
 from Tests.Backup_Recovery.test_restore_plan import sealed
 warm=sealed(home)
 inspected=service.start_inspection(warm.path,password=None)
 assert (await asyncio.to_thread(service.wait,inspected,timeout=15))['state']=='succeeded'
 entered,release=threading.Event(),threading.Event()
 writer=archive_writer.write_archive
 def held_writer(*args,**kwargs):
  entered.set();assert release.wait(15);return writer(*args,**kwargs)
 archive_writer.write_archive=held_writer
 monitoring=asyncio.create_task(monitor_app(app))
 try:
  note=app.chachanotes_db.add_note('Before backup','Captured native value.')
  options={'staging_parent':home}
  destination=home/'service.tldw-backup.zip'
  details=await asyncio.to_thread(service.preview_backup_details,(selector,),options=options,destination=destination)
  preview=details['inventory']
  assert details['capacity'] and all(row['sufficient'] for row in details['capacity'])
  assert preview.complete,preview.issues
  operation=service.start_backup((selector,),preview.scope_digest,destination,options=options,password=None)
  async with asyncio.timeout(65):
   while not entered.is_set():
    assert service.status(operation)['state']=='running',dict(service.status(operation))
    await asyncio.sleep(.01)
  assert service.status(operation)['phase']=='packaging' and not destination.exists()
  for _ in range(500):
   if storage_admission._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage_admission._pause is None
  after=app.chachanotes_db.add_note('After capture','Ordinary writer resumed.')
  release.set()
  result=await asyncio.to_thread(service.wait,operation,timeout=30)
  assert result['state']=='succeeded' and result['phase']=='archive_verified',dict(result)
  assert result['result']['complete'] and result['result']['path']==str(destination)
  assert not result['result'].get('restoration_validated',False)
  acquired=archive_reader.acquire(destination,home/'readback',ArchiveLimits(),None,threading.Event())
  manifest=json.loads(acquired.manifest_bytes)
  member=next(row for row in manifest['files'] if row['owner_id']=='db.chachanotes.primary')
  import zipfile
  payload=home/'readback.sqlite'
  with zipfile.ZipFile(acquired.path) as archive:payload.write_bytes(archive.read(member['payload']))
  with closing(sqlite3.connect(payload)) as db:
   assert db.execute('SELECT content FROM notes WHERE id=?',(note,)).fetchone()[0]=='Captured native value.'
   assert db.execute('SELECT 1 FROM notes WHERE id=?',(after,)).fetchone() is None
  assert not blocked_attempts()
 finally:
  release.set();archive_writer.write_archive=writer
  await asyncio.to_thread(service.close)
  monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
print('retired and reopened')
"""


def test_service_actual_live_backup_resumes_native_writes_before_packaging(
    tmp_path, native_package
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "service",
        "backup",
        script=_LIVE_BACKUP,
        timeout=110,
        installed_package=native_package,
    )


_SERVICE_ISOLATED = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor
base=Path.home()
ambient=Path(os.environ['TLDW_CONFIG_PATH']);ambient.write_bytes(b'broken = [current config')
before=ambient.read_bytes()
def config_manifest(doc):
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(base,mutate=config_manifest,data=b'[general]\nusers_name="original"\n')
dest=base/'destinations';dest.mkdir(mode=0o700)
control=base/'custom-control';service=RecoveryService(control)
register=ProfileCatalog.register
finish=None
if sys.argv[1]=='interrupted':
 def interrupted(self,*args):
  register(self,*args)
  raise OSError('fixture after actual catalog registration')
 ProfileCatalog.register=interrupted
elif sys.argv[1]=='before_prepared':
 from tldw_chatbook.Backup_Recovery import isolated_restore
 finish=isolated_restore._finish_isolated
 def before_prepared(*args):
  args[-1].set()
  return finish(*args)
 isolated_restore._finish_isolated=before_prepared
try:
 inspection=service.start_inspection(archive.path,password=None)
 assert service.wait(inspection,timeout=15)['state']=='succeeded'
 plan=service.preview_restore(inspection,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
 operation=service.start_restore(inspection,plan)
 state=service.wait(operation,timeout=30)
 if sys.argv[1] in ('interrupted','before_prepared'):
  assert state['state']=='recovery_required',dict(state)
  journal_operation=service.pending_operations()[0]['operation_id']
  service.close();service=RecoveryService(control)
  ProfileCatalog.register=register
  if finish is not None:isolated_restore._finish_isolated=finish
  assert service.status(journal_operation)['actions']==('finish',)
  recovery=service.start_recovery(journal_operation,action='finish')
  state=service.wait(recovery,timeout=30)
  assert state['state']=='succeeded' and state['phase']=='committed',dict(state)
  profile=service.profiles()[0]['profile_id']
 else:
  assert state['state']=='succeeded' and state['phase']=='restoration_validated',dict(state)
  profile=state['result']['profile_id']
 config,data=ProfileCatalog(control).resolve(profile)
 assert config==dest/'config'/'config.toml' and data==dest/'data'
 import tomllib
 assert tomllib.loads(config.read_text())['general']['users_name']=='recovered'
 assert _launch_descriptor(profile,control).profile_id==profile
 assert service.profiles()[0]['profile_id']==profile
 assert service.profiles()[0]['status']=='restoration_validated'
 assert ambient.read_bytes()==before
finally:service.close()
assert _launch_descriptor(profile,control).profile_id==profile
assert archive.path.is_file() and not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ("isolated", "interrupted", "before_prepared"))
def test_service_real_isolated_restore_survives_worker_shutdown(tmp_path, route):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, route, "service", script=_SERVICE_ISOLATED)


@pytest.mark.parametrize("established", (False, True))
def test_service_replaces_real_stored_data_and_retains_verified_originals(
    tmp_path, monkeypatch, helper_resource_root, established
):
    import sqlite3
    from contextlib import closing

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, prepared_plan, _, _, source, selector = case
        old_config = selector.read_bytes()
        acknowledged = ("credential_format_unreadable",)
        if established:
            import hashlib
            import tomllib

            from Tests.Backup_Recovery.test_replacement import _credential_candidate
            from tldw_chatbook.Backup_Recovery import bootstrap, credentials
            from tldw_chatbook.Backup_Recovery.credential_policies import (
                GENERATION_KEYRINGS,
            )

            candidate, prepared_plan, _, _, _ = _credential_candidate(
                case, tmp_path, monkeypatch
            )

            replacement.replace(
                replace(
                    prepared_plan,
                    acknowledged_credential_issues=("credential_format_unreadable",),
                ),
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"first",
                cancel=Event(),
            )
            assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
            configured = tomllib.loads(selector.read_text())
            reviewed = []
            for section, keyring_service, names in GENERATION_KEYRINGS:
                for name in names:
                    if name in configured.get(section, {}):
                        credentials._capture_record(
                            {
                                "kind": "generation",
                                "file": "payload/"
                                + hashlib.sha256(b"profile:profile:config").hexdigest(),
                                "service": keyring_service,
                                "username": name,
                                "remappable": False,
                            },
                            [],
                            reviewed,
                        )
            acknowledged = tuple(reviewed)
        service = RecoveryService(tmp_path / "control")
        try:
            inspection = service.start_inspection(
                tmp_path
                / ("incoming-credentials.age" if established else "replacement.zip"),
                password=b"incoming" if established else None,
            )
            assert service.wait(inspection, timeout=15)["state"] == "succeeded"
            plan = service.preview_restore(
                inspection,
                mode="replace",
                destinations=dict(
                    (*prepared_plan.destinations, *prepared_plan.selectors)
                ),
                target=prepared_plan.target,
                profile_names=dict(prepared_plan.profile_names),
                acknowledged_credential_issues=acknowledged,
            )
            operation = service.start_restore(
                inspection, plan, rollback_password=b"rollback"
            )
            result = service.wait(operation, timeout=40)
            assert result["state"] == "succeeded", dict(result)
            assert result["result"]["restoration_validated"]
            with closing(sqlite3.connect(source)) as db:
                assert (
                    db.execute(
                        "SELECT query FROM research_runs WHERE id='kept'"
                    ).fetchone()[0]
                    == "old"
                )
            assert selector.read_bytes() != old_config
            copy = next(
                row
                for row in service.recovery_copies()
                if row.operation_id == result["result"]["journal_operation_id"]
            )
            assert copy.status == "verified" and not copy.pending_operation
            assert copy.operation_id == result["result"]["journal_operation_id"]
        finally:
            service.close()
        assert copy.path.is_file()


def test_service_interrupted_native_replacement_can_recover_after_close(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import bootstrap, publication

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _, template, _, _, source, selector = case
        originals = {path: path.read_bytes() for path in (source, selector)}
        service = RecoveryService(tmp_path / "control")
        retire = publication._retire

        def interrupted(item):
            retire(item)
            raise InterruptedError("fixture after native retirement")

        monkeypatch.setattr(publication, "_retire", interrupted)
        try:
            inspected = service.start_inspection(
                tmp_path / "replacement.zip", password=None
            )
            service.wait(inspected, timeout=15)
            plan = service.preview_restore(
                inspected,
                mode="replace",
                destinations=dict(template.destinations),
                target=template.target,
                profile_names=dict(template.profile_names),
                acknowledged_credential_issues=("credential_format_unreadable",),
            )
            operation = service.start_restore(
                inspected, plan, rollback_password=b"rollback"
            )
            state = service.wait(operation, timeout=40)
            assert state["state"] == "recovery_required", dict(state)
            pending = service.pending_operations()
            assert len(pending) == 1
            journal_operation = pending[0]["operation_id"]
        finally:
            service.close()
            monkeypatch.setattr(publication, "_retire", retire)
        restarted = RecoveryService(tmp_path / "control")
        try:
            before = restarted.status(journal_operation)
            assert (
                before["state"] == "recovery_required"
                and "rollback" in before["actions"]
            )
            operation = restarted.start_recovery(
                journal_operation, action="rollback", rollback_password=b"rollback"
            )
            state = restarted.wait(operation, timeout=40)
            assert state["state"] == "succeeded" and state["result"]["rolled_back"], (
                dict(state)
            )
            assert all(path.read_bytes() == value for path, value in originals.items())
            assert not restarted.pending_operations()
            assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
            assert restarted.status(journal_operation)["state"] == "succeeded"
        finally:
            restarted.close()


def test_service_explains_actual_encrypted_unlock_failure(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source = tmp_path / "encrypted.tldw-backup.zip.age"
    write_archive(
        captured(tmp_path), source, password=b"actual-password", cancel=Event()
    )
    service = RecoveryService(tmp_path / "control")
    try:
        missing = service.start_inspection(source, password=None)
        assert service.wait(missing, timeout=15)["issues"] == ("password_required",)
        wrong = service.start_inspection(source, password=b"incorrect-password")
        assert service.wait(wrong, timeout=15)["issues"] == ("archive_unlock_failed",)
        correct = service.start_inspection(source, password=b"actual-password")
        assert service.wait(correct, timeout=15)["state"] == "succeeded"
        assert b"actual-password" not in repr(service.current()).encode()
    finally:
        service.close()
    assert source.is_file()


def test_service_later_rollback_uses_reviewed_copy_and_keeps_new_safety_copy(
    tmp_path, monkeypatch, helper_resource_root
):
    from Tests.Backup_Recovery.test_later_rollback import _completed

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        original,
        before_restore,
    ):
        service = RecoveryService(tmp_path / "control")
        try:
            plan = service.preview_rollback(
                original, old_password=b"original", target=case[1].target
            )
            operation = service.start_rollback(
                original, plan, old_password=b"original", new_password=b"new"
            )
            state = service.wait(operation, timeout=40)
            assert state["state"] == "succeeded", dict(state)
            assert state["result"]["restoration_validated"]
            assert case[-1].read_bytes() == before_restore
            copies = service.recovery_copies()
            assert len(copies) == 2 and all(row.status == "verified" for row in copies)
        finally:
            service.close()


def test_service_recovers_actual_interrupted_activation_pair(
    tmp_path, monkeypatch, helper_resource_root
):
    from Tests.Backup_Recovery.test_replacement_recovery import _child, _pending_rows
    from tldw_chatbook.Backup_Recovery import bootstrap

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        selector = case[-1]
        before = selector.read_bytes()
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "finish",
            "activation_pair",
            expected=91,
        )
        operation = _pending_rows(tmp_path)[0]["operation_id"]
        service = RecoveryService(tmp_path / "control")
        try:
            with pytest.raises(ValueError, match="activation_update_pending"):
                bootstrap._records(tmp_path / "bootstrap")
            assert service.pending_operations()[0]["operation_id"] == operation
            assert "rollback" in service.status(operation)["actions"]
            running = service.start_recovery(
                operation, action="rollback", rollback_password=b"rollback"
            )
            state = service.wait(running, timeout=40)
            assert state["state"] == "succeeded" and state["result"]["rolled_back"], (
                dict(state)
            )
            assert selector.read_bytes() == before
            assert not service.pending_operations()
            assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        finally:
            service.close()


@pytest.mark.parametrize("boundary", ("pending", "credential_review"))
def test_service_aborts_only_untouched_pre_safety_replacement(
    tmp_path, monkeypatch, helper_resource_root, boundary
):
    from tldw_chatbook.Backup_Recovery import control_records

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _, original, _, _, source, selector = case
        before = {path: path.read_bytes() for path in (source, selector)}
        if boundary == "pending":
            register = control_records.register_pending

            def interrupted(*args, **kwargs):
                register(*args, **kwargs)
                raise InterruptedError("fixture after actual pending registration")

            monkeypatch.setattr(control_records, "register_pending", interrupted)
        service = RecoveryService(tmp_path / "control")
        try:
            inspection = service.start_inspection(
                tmp_path / "replacement.zip", password=None
            )
            assert service.wait(inspection, timeout=15)["state"] == "succeeded"
            plan = service.preview_restore(
                inspection,
                mode="replace",
                destinations=dict((*original.destinations, *original.selectors)),
                target=original.target,
                profile_names=dict(original.profile_names),
            )
            operation = service.start_restore(
                inspection, plan, rollback_password=b"rollback"
            )
            state = service.wait(operation, timeout=40)
            assert state["state"] == "recovery_required", dict(state)
            if boundary == "credential_review":
                assert state["review_issues"] == ("credential_format_unreadable",)
            pending = service.pending_operations()[0]["operation_id"]
        finally:
            service.close()
        restarted = RecoveryService(tmp_path / "control")
        try:
            assert restarted.status(pending)["actions"] == ("abort",)
            operation = restarted.start_recovery(pending, action="abort")
            state = restarted.wait(operation, timeout=40)
            assert state["state"] == "succeeded" and state["result"]["aborted"], dict(
                state
            )
            assert not state["result"]["restoration_validated"]
            assert all(path.read_bytes() == content for path, content in before.items())
            assert not restarted.pending_operations()
            assert restarted.status(pending)["result"]["aborted"]
        finally:
            restarted.close()


@pytest.mark.parametrize("damage", ("version", "missing_version", "relative_selector"))
def test_recovery_pending_discovery_refuses_malformed_fixed_record(
    tmp_path, monkeypatch, damage
):
    import json

    from tldw_chatbook.Backup_Recovery import bootstrap

    root = tmp_path / "bootstrap"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    record = {
        "version": 1,
        "operation_id": "test",
        "namespaces": ["scope"],
        "selectors": [str(tmp_path / "config.toml")],
        "control_root": str(tmp_path / "control"),
    }
    if damage == "version":
        record["version"] = True
    elif damage == "missing_version":
        record.pop("version")
    else:
        record["selectors"] = ["relative"]
    path = root / ("pending-" + bootstrap._key("test") + ".json")
    content = json.dumps(record).encode()
    path.write_bytes(content)
    path.chmod(0o600)
    service = RecoveryService(tmp_path / "control")
    try:
        with pytest.raises(ValueError):
            service.pending_operations()
        assert path.read_bytes() == content
    finally:
        service.close()
