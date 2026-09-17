"""Explicit empty placement containers do not grant directory replacement."""

import os

import pytest

from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore, recheck_targets


def _archive(tmp_path, *, synthetic=True, external=False):
    def shape(doc):
        doc["directories"][0]["synthetic"] = synthetic
        if external:
            doc["owners"][0]["owner_id"] = "external.files"
            doc["files"][0]["owner_id"] = "external.files"

    return sealed(tmp_path, mutate=shape)


def test_explicit_empty_synthetic_container_has_no_restore_or_metadata(tmp_path):
    archive = _archive(tmp_path)
    container = tmp_path / "selected"
    container.mkdir(mode=0o700)
    before = container.stat()
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": container}, target=None
    )
    assert dict(plan.restore) == {"file": container / "note.txt"}
    assert dict(plan.destinations) == {"root": container}
    assert plan.retire == () and plan.containers == ()
    assert "root" not in {key for key, _, _ in plan.metadata}
    recheck_targets(plan)
    after = container.stat()
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
    )
    assert list(container.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    [
        "concrete",
        "file",
        "symlink",
        "dangling",
        "public",
        "occupied",
        "payload",
        "external",
        "data_selector",
        "foreign_owner",
    ],
)
def test_existing_destinations_remain_rejected(tmp_path, monkeypatch, kind):
    archive = _archive(
        tmp_path, synthetic=kind != "concrete", external=kind == "external"
    )
    container = tmp_path / "selected"
    if kind == "file":
        container.write_bytes(b"existing")
    elif kind in {"symlink", "dangling"}:
        target = tmp_path / "other"
        if kind == "symlink":
            target.mkdir(mode=0o700)
        container.symlink_to(target, target_is_directory=True)
    else:
        container.mkdir(mode=0o700)
        if kind == "public":
            container.chmod(0o755)
        if kind in {"occupied", "payload"}:
            (
                container / ("note.txt" if kind == "payload" else "unrelated")
            ).write_bytes(b"preserve")
    destinations = {"root": container}
    if kind == "data_selector":
        destinations["profile:profile:paths.data_dir"] = container
    if kind == "foreign_owner":
        actual = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: actual + 1)
    with pytest.raises((ValueError, OSError)):
        plan_restore(archive, mode="isolated", destinations=destinations, target=None)
    if kind in {"occupied", "payload"}:
        assert next(container.iterdir()).read_bytes() == b"preserve"


@pytest.mark.parametrize("change", ["new_file", "replacement", "link", "mode"])
def test_existing_container_change_invalidates_review(tmp_path, change):
    archive = _archive(tmp_path)
    container = tmp_path / "selected"
    container.mkdir(mode=0o700)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": container}, target=None
    )
    if change == "new_file":
        (container / "arrived").write_bytes(b"new user data")
    elif change == "mode":
        container.chmod(0o755)
    else:
        held = container.with_name("original")
        container.rename(held)
        if change == "replacement":
            container.mkdir(mode=0o700)
        else:
            container.symlink_to(held, target_is_directory=True)
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)
    if change == "new_file":
        assert (container / "arrived").read_bytes() == b"new user data"


_NATIVE = r"""
import json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import bootstrap,staging
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
from tldw_chatbook.Backup_Recovery.control_records import _existing_admission_authority
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
base=Path.home();mode=sys.argv[1]
ambient=Path(os.environ['TLDW_CONFIG_PATH']);ambient.write_text('broken = [current config')
def shape(doc):
 doc['directories'][0]['synthetic']=True
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(base,mutate=shape,data=b'[general]\nusers_name="original"\n')
parent=base/'.config'/'selected-container';parent.mkdir(parents=True,mode=0o700)
before=parent.stat();dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':parent,'profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
assert dict(plan.restore)=={'profile:profile:config':parent/'config.toml'}
assert 'root' not in {key for key,_,_ in plan.metadata}
cancel=threading.Event();staged=[]
if mode=='cancel':
 original=staging.stage_restore
 def cancel_after_stage(*args,**kwargs):
  candidate=original(*args,**kwargs)
  assert (candidate/'candidate.json').is_file()
  staged.append(candidate);cancel.set();return candidate
 staging.stage_restore=cancel_after_stage
try:profile=restore_isolated(archive,plan,base/'control',cancel)
except InterruptedError:
 assert mode=='cancel' and staged and list(parent.iterdir())==[]
 assert not (dest/'data').exists()
else:
 assert mode=='success'
 entry=_launch_descriptor(profile,base/'control')
 assert entry.config==str(parent/'config.toml') and Path(entry.config).is_file()
 assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
 authority=_existing_admission_authority(bootstrap.default_bootstrap_root())
 try:authority.register('forbidden-parent',(parent.parent,))
 except AdmissionError as error:assert str(error)=='control_root_overlaps_target'
 else:raise AssertionError('Bootstrap ancestor was admitted')
assert (parent.stat().st_dev,parent.stat().st_ino,parent.stat().st_mode)==(before.st_dev,before.st_ino,before.st_mode)
assert parent.stat().st_mtime_ns!=0
assert ambient.read_text()=='broken = [current config'
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("mode", ["success", "cancel"])
def test_native_isolated_container_retains_identity(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, mode, "container", script=_NATIVE)
